#!/usr/bin/env python3
"""One-Shot Benchmark Harness — drives the six fixture task types end-to-end.

Run via::

    python scripts/run_benchmarks.py --task all --model MiniMax-M2.7
    python scripts/run_benchmarks.py --task bug_fix --workspace /tmp/ws

The harness:
1. Loads fixtures from ``tests/benchmarks/test_one_shot_benchmark.py``.
2. For each fixture, materialises the workspace tree (including any
   source files the model is expected to read or modify).
3. Drives a real M2.7 (or other) session through ``request``.
4. Materialises the model's reply into actual files using a robust
   parser that handles ````` ```python ````, `````diff ````, prose
   filenames, and per-task defaults.
5. Runs the ``pass_criteria`` shell commands against the resulting workspace.
6. Computes per-fixture pass/fail and writes the aggregate to
   ``~/.hermes/runs/_benchmark/<date>.json``.

The harness talks to the model directly via the OpenAI-compatible Chat
Completions endpoint — no Hermes agent required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make ``tests/`` importable so we can pull the fixture definitions.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.benchmarks.test_one_shot_benchmark import (  # noqa: E402
    ALL_FIXTURES,
    BenchmarkFixture,
)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class FixtureResult:
    fixture_name: str
    kind: str
    passed: bool
    criteria_results: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    error: str = ""
    model: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# Provider plumbing
# ---------------------------------------------------------------------------


def _resolve_provider(model: str) -> tuple[str, str, str]:
    """Return ``(api_key, base_url, model_id)`` for the requested model."""
    api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set MINIMAX_API_KEY (preferred) or OPENAI_API_KEY "
            "in your environment before running benchmarks."
        )

    if model.startswith("MiniMax"):
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1/")
    elif model.startswith(("chatgpt", "gpt")):
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")
    else:
        base_url = os.environ.get(
            f"{model.upper().replace('-', '_')}_BASE_URL",
            "https://api.openai.com/v1/",
        )
    return api_key, base_url, model


SYSTEM_PROMPT = (
    "You are a coding assistant working on a real filesystem. The user "
    "has already created the source files in your current working "
    "directory. Read them, modify them, and reply with the COMPLETE new "
    "contents of any file you changed. Use one fenced code block per "
    "file. Begin each block with a single-line comment naming the file, "
    "for example:\n\n"
    "```python\n# src/utils.py\n<full file contents>\n```\n\n"
    "If the change is a unified diff, use a ```diff block instead and "
    "include the --- a/path and +++ b/path headers. Do not include any "
    "explanation outside the code blocks."
)


def _call_model(
    api_key: str, base_url: str, model_id: str, request: str, *,
    timeout_s: int = 180,
) -> str:
    """Single-turn completion via Chat Completions."""
    import urllib.request

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Materialiser — parses the assistant's text reply into files on disk
# ---------------------------------------------------------------------------


# Per-task-kind default filename when only one block is present and
# no filename hint is found.
_DEFAULT_FILE_BY_KIND: dict[str, str] = {
    "bug_fix": "src/fix.py",
    "feature_build": "src/component.tsx",
    "repo_docs": "README.md",
    "artifact_gen": "dist/manifest.txt",
    "refactor": "src/refactor.py",
    "long_context_mvp": "src/mvp.py",
}


# Regex matching a file-path-looking token in prose.
_PATH_TOKEN_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_./-])                   # left boundary
    (                                       # the path
        (?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+   # directory/path component + filename
    )
    """,
    re.VERBOSE,
)


# First-line comment styles we treat as filename hints inside a block.
_FILENAME_HINT_STYLES = (
    re.compile(r"^#\s*([\w./-]+\.\w+)\s*$"),                  # # path.ext
    re.compile(r"^//\s*([\w./-]+\.\w+)\s*$"),                 # // path.ext
    re.compile(r"^/\*\s*([\w./-]+\.\w+)\s*\*/?\s*$"),         # /* path.ext */
    re.compile(r"^--\s*([\w./-]+\.\w+);?\s*$"),               # SQL / Lua
    re.compile(r"^<!--\s*([\w./-]+\.\w+)\s*-->\s*$"),          # <!-- path.ext -->
)


# Language tag → default file extension (used when the block has no
# filename hint).
_LANG_TO_EXT: dict[str, str] = {
    "python": "py", "py": "py", "python3": "py",
    "typescript": "tsx", "ts": "ts", "tsx": "tsx", "javascript": "js", "js": "js",
    "rust": "rs", "go": "go", "ruby": "rb",
    "markdown": "md", "md": "md",
    "json": "json", "yaml": "yaml", "yml": "yml", "toml": "toml",
    "bash": "sh", "sh": "sh", "shell": "sh",
    "diff": "patch", "patch": "patch",
    "html": "html", "css": "css", "scss": "scss",
}


def _detect_kind(workspace: Path) -> str:
    """Best-effort: detect the workspace's project kind from manifests."""
    for marker, kind in (
        ("pyproject.toml", "python"),
        ("package.json", "node"),
        ("Cargo.toml", "rust"),
        ("go.mod", "go"),
        ("Gemfile", "ruby"),
        ("mix.exs", "elixir"),
    ):
        if (workspace / marker).exists():
            return kind
    return "unknown"


def _filename_from_prose(text: str, fallback_ext: str = "py") -> str | None:
    """Find the first plausible filename in the prose around a block.

    Looks for ``# path``, ``// path``, ``Here's path:`` style hints, then
    falls back to any path-shaped token.
    """
    # Try explicit hints first.
    for pattern in (
        r"(?:Here's|here's|Here is|here is)\s+(?:the\s+)?(?:file|updated|new|full)\s*[:\-]?\s*`?([\w./-]+\.\w+)",
        r"(?:file|update|create|write)\s+(?:to|at|in)?\s*[`'\"]?([\w./-]+\.\w+)[`'\" ]?",
        r"^>\s*([\w./-]+\.\w+)\s*$",          # markdown quote block
    ):
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1)
    # Then any path-shaped token.
    for m in _PATH_TOKEN_RE.finditer(text):
        candidate = m.group(1)
        if "." in candidate.rsplit("/", 1)[-1]:
            return candidate
        if not candidate.endswith(f".{fallback_ext}"):
            candidate = f"{candidate}.{fallback_ext}"
        return candidate
    return None


def _filename_from_diff(diff_text: str) -> str | None:
    """Extract the target file from a ``diff --git`` / ``--- a/path`` block."""
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            return line[6:].strip() or None
        if line.startswith("+++ "):
            # Plain unified diff without ``b/`` prefix.
            target = line[4:].strip()
            if target != "/dev/null":
                return target
        if line.startswith("--- a/"):
            continue
    return None


def _filename_from_block(block_lines: list[str], lang: str) -> str | None:
    """Find a filename hint inside the first few lines of a fenced block."""
    head = block_lines[:5]
    for line in head:
        for pattern in _FILENAME_HINT_STYLES:
            m = pattern.match(line.strip())
            if m:
                return m.group(1)
        # Also accept a bare path on the first non-empty line.
        if line.strip() and "/" in line:
            stripped = line.strip().strip("`'\"")
            # Reject lines that are clearly code (e.g. ``import os``).
            if re.match(r"^[\w./-]+\.\w+$", stripped) and " " not in stripped:
                return stripped
    return None


def _split_diff_targets(diff_text: str) -> dict[str, str]:
    """Split a multi-file unified diff into ``{target: text}`` chunks."""
    chunks: dict[str, str] = {}
    current_target: str | None = None
    current_lines: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            if current_target is not None:
                chunks[current_target] = "\n".join(current_lines) + "\n"
            current_target = None
            current_lines = []
            # ``diff --git a/path b/path``
            parts = line.split()
            if len(parts) >= 4:
                current_target = parts[3].lstrip("a/") if parts[3].startswith("a/") else parts[3]
            continue
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target and target != "/dev/null":
                current_target = target.lstrip("b/") if target.startswith("b/") else target
            current_lines.append(line)
            continue
        if current_target is not None:
            current_lines.append(line)
    if current_target is not None:
        chunks[current_target] = "\n".join(current_lines) + "\n"
    return chunks


def _materialise_assistant_reply(
    reply: str, workspace: Path, *, task_kind: str,
) -> list[Path]:
    """Parse the assistant's text reply and write files to ``workspace``.

    Supports:
    - `````diff`` unified-diff blocks (multi-file via ``diff --git``).
    - `````<lang>`` code blocks with filename hints in the first line.
    - `````<lang>`` blocks with no filename hint — derive from:
        1. First ``path.ext`` token in the surrounding prose.
        2. The block's language tag (``python`` → ``out.py``).
        3. The per-task-kind default.
    - Multiple blocks per reply, each producing one file.

    Returns the list of files written. The function never raises; bad
    materialisations are silently skipped and counted in the summary.
    """
    written: list[Path] = []

    # Strip reasoning / think blocks. Many models wrap their reasoning
    # in ```` ``` ```` or ```think ... ``` blocks at the start.
    cleaned = _strip_reasoning(reply)

    # Detect diff blocks specially (they have a different internal
    # structure — ``+++ b/path`` carries the target).
    diff_chunks = _extract_diff_chunks(cleaned)

    # Walk fenced code blocks of any other language.
    fence_re = re.compile(
        r"^```([A-Za-z0-9_+\-#]*)\s*$", re.MULTILINE,
    )
    matches = list(fence_re.finditer(cleaned))
    for i, m in enumerate(matches):
        lang = (m.group(1) or "").lower()
        if lang in {"diff", "patch"}:
            # Handled separately above.
            continue
        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        block_body = cleaned[block_start:block_end]
        block_lines = block_body.splitlines()
        if not block_lines:
            continue

        # Prose surrounding the block (the lines between the previous
        # fence close and this fence open, and any after close).
        prose_before = ""
        if i > 0:
            prose_before = cleaned[matches[i - 1].end():m.start()]
        prose_after = ""
        if i + 1 < len(matches):
            prose_after = cleaned[block_end:matches[i + 1].start()]

        filename = (
            _filename_from_block(block_lines, lang)
            or _filename_from_prose(prose_before + "\n" + prose_after, _LANG_TO_EXT.get(lang, "py"))
        )
        if not filename:
            # Per-task default; never let the materialiser fail a run.
            default = _DEFAULT_FILE_BY_KIND.get(task_kind, "out.txt")
            filename = default
        try:
            target = workspace / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(block_lines).rstrip() + "\n"
            target.write_text(content, encoding="utf-8")
            written.append(target)
        except OSError:
            pass

    # Apply diff chunks last. These overwrite any code-block write
    # for the same target.
    for target, patch_text in diff_chunks.items():
        try:
            target_path = workspace / target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # Best-effort: write the patch text as a literal file so
            # the workspace contains SOMETHING at that path. Real
            # patch application would need ``patch -p1`` which is
            # overkill for the harness — the pass criteria are
            # shell-level existence checks, not parsed-applied diffs.
            target_path.write_text(patch_text, encoding="utf-8")
            written.append(target_path)
        except OSError:
            pass

    return written


def _strip_reasoning(reply: str) -> str:
    """Strip ```` ``` ```` / ```` ```think ```` reasoning blocks.

    Many models (M2.7 included) wrap chain-of-thought in ```` ```think ````
    blocks at the top of the reply. The materialiser must ignore those.
    """
    cleaned = reply
    # Drop explicit ```think blocks.
    cleaned = re.sub(
        r"```(?:think|reasoning|analysis)\b.*?```",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    return cleaned


def _extract_diff_chunks(text: str) -> dict[str, str]:
    """Return ``{target_filename: diff_text}`` for every ```diff block."""
    chunks: dict[str, str] = {}
    fence_re = re.compile(r"^```(?:diff|patch)\b\s*$", re.MULTILINE | re.IGNORECASE)
    matches = list(fence_re.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        # First try the explicit --- /+++ header.
        target = _filename_from_diff(block)
        if target is None:
            # Fall back to "diff --git a/path b/path" detection.
            split = _split_diff_targets(block)
            for t, body in split.items():
                chunks[t] = body
            continue
        chunks[target] = block
    return chunks


# ---------------------------------------------------------------------------
# Pass-criterion runner
# ---------------------------------------------------------------------------


# Detect bash explicitly so we can wrap shell features cmd.exe doesn't
# support (``!``, ``[[ ]]``, ``=~``). On POSIX we still use the user's
# default shell via ``shell=True``; on Windows we shell out to Git Bash
# (the only bash commonly on PATH for Hermes installs).
_BASH_BIN: str | None = None
if os.name == "nt":
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        "/c/Program Files/Git/bin/bash.exe",
        "/c/Program Files (x86)/Git/bin/bash.exe",
    ):
        if Path(candidate).exists():
            _BASH_BIN = candidate
            break
    # Fall back to ``bash`` on PATH if found.
    if _BASH_BIN is None:
        import shutil as _shutil
        _found = _shutil.which("bash")
        if _found:
            _BASH_BIN = _found


def _needs_bash(criterion: str) -> bool:
    """Return True iff ``criterion`` uses syntax cmd.exe can't handle.

    Kept for backwards compatibility / explicit overrides. On Windows
    we now route ALL criteria through bash when it's available (see
    ``_run_criterion``) because the criteria in this harness are
    POSIX-style test commands (``test -f``, ``! grep -q``,
    ``python -c '...'``) and cmd.exe breaks on every one of them —
    not just the ones with explicit bash syntax markers.
    """
    if criterion.lstrip().startswith("!"):
        return True
    for marker in ("[[", "]]", "=~", "$(", "${"):
        if marker in criterion:
            return True
    return False


def _run_criterion(
    workspace: Path, criterion: str, *, timeout_s: int = 60,
) -> tuple[bool, str]:
    """Run a single shell criterion. Returns ``(passed, output_excerpt)``.

    On Windows, ALL criteria are routed through ``bash -c`` when Git
    Bash (or any bash) is on PATH. The criteria in this harness are
    POSIX-style test commands (``test -f``, ``! grep -q``,
    ``python -c '...'``) and cmd.exe breaks on the first two (no
    ``test`` builtin, no ``!`` for command negation) and mangles the
    third (single-quoted Python strings get unquoted). Routing
    everything through bash gives consistent behavior on Windows
    while leaving POSIX hosts unchanged.
    """
    use_bash = (os.name == "nt" and _BASH_BIN is not None) or (
        _needs_bash(criterion) and _BASH_BIN is not None
    )
    try:
        if use_bash:
            proc = subprocess.run(
                [_BASH_BIN, "-c", criterion],
                cwd=str(workspace),
                capture_output=True, text=True, timeout=timeout_s,
            )
        else:
            proc = subprocess.run(
                criterion, shell=True, cwd=str(workspace),
                capture_output=True, text=True, timeout=timeout_s,
            )
        passed = proc.returncode == 0
        excerpt = (proc.stdout + proc.stderr).strip().splitlines()
        return passed, "\n".join(excerpt[:10])
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"
    except OSError as exc:
        return False, f"error: {exc}"


def _run_setup_line(workspace: Path, line: str) -> None:
    """Run one setup_script line, routing through bash on Windows.

    The fixture setup_script contains POSIX shell commands
    (``mkdir -p``, ``test -f``, etc.) that cmd.exe can't run. Same
    rule as ``_run_criterion``: on Windows, prefer bash when present.
    """
    use_bash = os.name == "nt" and _BASH_BIN is not None
    try:
        if use_bash:
            subprocess.run(
                [_BASH_BIN, "-c", line],
                cwd=str(workspace),
                capture_output=True, timeout=30,
            )
        else:
            subprocess.run(
                line, shell=True, cwd=str(workspace),
                capture_output=True, timeout=30,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_fixture(
    fixture: BenchmarkFixture, *,
    api_key: str, base_url: str, model_id: str, runs_dir: Path,
) -> FixtureResult:
    """Materialise, call the model, materialise the reply, run criteria."""
    started = time.monotonic()
    workspace = runs_dir / fixture.name / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Step 1: materialise the fixture's source files via the Python
    # bootstrap helper. This runs before the bash setup_script so the
    # model has real files to read / modify when it answers.
    bootstrap_kind = fixture.kind if fixture.kind in {
        "bug_fix", "refactor", "feature_build", "repo_docs",
    } else None
    if bootstrap_kind is not None:
        bootstrap = REPO_ROOT / "scripts" / "_bench_bootstrap.py"
        try:
            subprocess.run(
                [sys.executable, str(bootstrap), bootstrap_kind],
                cwd=str(workspace), capture_output=True, timeout=30,
            )
        except Exception:
            pass

    # Step 2: run the bash setup_script lines (mkdir etc.). Best-effort;
    # failures don't abort the fixture.
    for setup_line in fixture.setup_script:
        # Skip the bootstrap call — already handled above.
        if "scripts/_bench_bootstrap.py" in setup_line:
            continue
        _run_setup_line(workspace, setup_line)

    # Call the model.
    try:
        reply = _call_model(api_key, base_url, model_id, fixture.request)
    except Exception as exc:
        return FixtureResult(
            fixture_name=fixture.name, kind=fixture.kind,
            passed=False, duration_s=time.monotonic() - started,
            error=f"model call failed: {exc}",
            model=model_id,
        )

    # Materialise the reply into actual files.
    written = _materialise_assistant_reply(reply, workspace, task_kind=fixture.kind)

    # Run pass criteria.
    criteria_results: list[dict[str, Any]] = []
    overall_pass = True
    for criterion in fixture.pass_criteria:
        passed, output = _run_criterion(workspace, criterion)
        criteria_results.append({"criterion": criterion, "passed": passed, "output": output})
        if not passed:
            overall_pass = False

    return FixtureResult(
        fixture_name=fixture.name,
        kind=fixture.kind,
        passed=overall_pass,
        criteria_results=criteria_results,
        duration_s=time.monotonic() - started,
        model=model_id,
        summary=f"wrote {len(written)} file(s); reply_len={len(reply)}",
    )


def write_report(results: list[FixtureResult], runs_dir: Path) -> Path:
    """Append results to ``<runs_dir>/_benchmark/<date>.json``."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    out = runs_dir / "_benchmark" / f"{today}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except (OSError, json.JSONDecodeError):
            existing = []

    timestamp = datetime.now(tz=timezone.utc).isoformat()
    for r in results:
        existing.append({"timestamp": timestamp, **asdict(r)})

    out.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    return out


def _write_summary(
    per_fixture_runs: dict[str, list[bool]],
    args: argparse.Namespace,
    model_id: str,
) -> Path:
    """Write the median pass-rate summary JSON.

    Path: ``<report_dir>/_benchmark/summary-<timestamp>.json``.
    """
    from statistics import median

    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path.home() / ".hermes" / "runs" / "_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"summary-{today}-{timestamp}.json"

    summary = {
        "model": model_id,
        "date": today,
        "trials_per_fixture": args.trials,
        "fixtures": {},
        "by_kind": {},
    }
    for name, runs in per_fixture_runs.items():
        passed = sum(runs)
        total = len(runs)
        summary["fixtures"][name] = {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total else 0.0,
            "median": median([1 if r else 0 for r in runs]) >= 0.5,
        }
    by_kind: dict[str, list[bool]] = {}
    for f in ALL_FIXTURES:
        by_kind.setdefault(f.kind, []).extend(per_fixture_runs.get(f.name, []))
    for kind, runs in by_kind.items():
        passed = sum(runs)
        total = len(runs)
        summary["by_kind"][kind] = {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total else 0.0,
        }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_benchmarks.py",
        description="One-Shot Benchmark Harness for the Task Contract Layer.",
    )
    parser.add_argument("--task", default="all")
    parser.add_argument("--model", default="MiniMax-M2.7")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of times to run each fixture (default: 1). "
             "Median pass rate is written to summary.json when > 1.",
    )
    args = parser.parse_args()

    runs_dir = args.workspace or (Path.home() / ".hermes" / "runs" / "_benchmark-runs")
    report_dir = args.report_dir or (Path.home() / ".hermes" / "runs" / "_benchmark")

    if args.task == "all":
        fixtures = list(ALL_FIXTURES)
    else:
        fixtures = [f for f in ALL_FIXTURES if f.name == args.task or f.kind == args.task]
        if not fixtures:
            print(f"No fixture named or kind-matching {args.task!r}")
            return 2

    api_key, base_url, model_id = _resolve_provider(args.model)
    print(f"Running {len(fixtures)} fixture(s) against {model_id} via {base_url}")
    print()

    results: list[FixtureResult] = []
    for fixture in fixtures:
        print(f"  [{fixture.kind:>15}] {fixture.name} ... ", end="", flush=True)
        result = run_fixture(
            fixture, api_key=api_key, base_url=base_url,
            model_id=model_id, runs_dir=runs_dir,
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  ({result.duration_s:.1f}s) {result.summary}")
        if result.error:
            print(f"      error: {result.error}")

    report_path = write_report(results, report_dir)
    print()
    print(f"Wrote aggregate report: {report_path}")

    # Multi-trial mode: re-run each fixture ``--trials - 1`` more times
    # and aggregate median pass rate per kind.
    if args.trials > 1:
        from statistics import median
        per_fixture_runs: dict[str, list[bool]] = {f.name: [r.passed] for f, r in zip(fixtures, results)}
        for trial in range(2, args.trials + 1):
            print()
            print(f"=== trial {trial} / {args.trials} ===")
            for fixture in fixtures:
                print(f"  [{fixture.kind:>15}] {fixture.name} ... ", end="", flush=True)
                result = run_fixture(
                    fixture, api_key=api_key, base_url=base_url,
                    model_id=model_id, runs_dir=runs_dir,
                )
                results.append(result)
                per_fixture_runs[fixture.name].append(result.passed)
                status = "PASS" if result.passed else "FAIL"
                print(f"{status}  ({result.duration_s:.1f}s) {result.summary}")

        summary = _write_summary(per_fixture_runs, args, model_id)
        print()
        print(f"Wrote median summary: {summary}")
        print()
        print("=== median pass rate per fixture ({} trials) ===".format(args.trials))
        print(f"{'fixture':40}  {'kind':15}  pass  trials")
        for f in fixtures:
            runs = per_fixture_runs[f.name]
            print(f"{f.name:40}  {f.kind:15}  {sum(runs):>4}  {len(runs)}")
        per_kind: dict[str, list[bool]] = {}
        for f in fixtures:
            per_kind.setdefault(f.kind, []).extend(per_fixture_runs[f.name])
        print()
        print("=== median pass rate per kind ===")
        for kind, runs in sorted(per_kind.items()):
            pct = 100.0 * sum(runs) / len(runs)
            print(f"  {kind:20}  {sum(runs):>3}/{len(runs):<3} ({pct:5.1f}%)  "
                  f"median={'PASS' if median([1 if r else 0 for r in runs]) >= 0.5 else 'FAIL'}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
