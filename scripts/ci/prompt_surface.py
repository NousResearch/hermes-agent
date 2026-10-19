#!/usr/bin/env python3
"""Render the model-facing prompt surface of a checkout, deterministically.

The surface is what a model reads before the first user message on a stock install: the system
prompt as ``agent.system_prompt.build_system_prompt_parts`` assembles it and the ``tools[]`` array
the agent sends. A one-line code change can rewrite either without the diff showing it (a guidance
constant moves tiers, a toolset gains a tool, a dynamic rewriter edits a description). This renders
both through the production code into ``tests/fixtures/prompt_surface/``, which is committed:

* ``tests/ci/test_prompt_surface.py`` fails when the committed render is stale, so every change
  to the surface lands as a reviewable diff of real prompt text and tool schemas;
* ``.github/workflows/prompt-surface-diff.yml`` posts the per-surface summary on the PR;
* a repository ruleset requires a hermes-agent-core approval for that directory.

    python scripts/ci/prompt_surface.py render            # regenerate the committed snapshot
    python scripts/ci/prompt_surface.py diff BASE HEAD --output FILE   # Markdown summary

``render`` runs in a throwaway ``HERMES_HOME`` (stock config, bundled skills synced) with the clock,
host OS, network, tool availability and machine paths pinned, so a laptop and the CI runner produce
the same bytes. ``diff`` is stdlib-only: the PR job runs it on two snapshot directories.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "tests" / "fixtures" / "prompt_surface"
MANIFEST = "surfaces.json"
# Same rough estimate as agent.model_metadata.CHARS_PER_TOKEN; kept local so ``diff`` needs no deps.
CHARS_PER_TOKEN = 4

MODELS = {
    "claude": "anthropic/claude-sonnet-4.5",
    "gpt": "openai/gpt-5",
    "gemini": "google/gemini-2.5-pro",
    "hermes": "nousresearch/hermes-4-405b",
}
# (surface, platform, model family, cwd is a code workspace). Model names gate guidance blocks
# and the patch dialect, so the matrix runs on cli; one row per other surface catches its hint
# and toolset; ``cli-code`` adds the coding posture every repo session gets.
SURFACES = (
    ("cli.claude", "cli", "claude", False),
    ("cli.gpt", "cli", "gpt", False),
    ("cli.gemini", "cli", "gemini", False),
    ("cli.hermes", "cli", "hermes", False),
    ("cli-code.claude", "cli", "claude", True),
    ("desktop.claude", "desktop", "claude", False),
    ("telegram.claude", "telegram", "claude", False),
    ("discord.claude", "discord", "claude", False),
    ("cron.claude", "cron", "claude", False),
)
# Desktop resolves toolsets in tui_gateway (client-surface toolsets folded in); the gateway, cron
# and classic CLI read the platform toolsets directly.
TUI_GATEWAY_PLATFORMS = ("desktop",)
FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
# ``Conversation started:`` reads the stamp embedded in the session id.
SESSION_ID = FIXED_NOW.strftime("%Y%m%d_%H%M%S") + "_prompt_surface"


# ── render (runs inside the child process) ───────────────────────────────────

def _pin_environment() -> None:
    """Pin every input that differs between machines; each one moved bytes in a trial render."""
    def refuse(*_a, **_k):
        raise OSError("network disabled while rendering the prompt surface")
    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]
    sys.platform = "linux"  # skills index ``platforms:`` gate

    import hermes_time
    hermes_time.now = lambda: FIXED_NOW  # type: ignore[assignment]
    # Every registered tool counts as available, as on a fully configured install: check_fns
    # read credentials, binaries and the tool store, which vary by host and would hide schemas.
    import tools.registry as registry_module
    registry_module._check_fn_cached = lambda _fn: True  # type: ignore[assignment]
    from agent import skill_utils
    skill_utils._ENV_DETECT_CACHE.update({"docker": False, "s6": False})
    # The toolchain probe line reports the host's python/node/pip; it is environment, not code.
    from tools import env_probe
    env_probe.get_environment_probe_line = lambda **_k: ""  # type: ignore[assignment]


class _ImportFailures(logging.Handler):
    """A tool module that fails to import drops its tools from the render silently."""
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.failures: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Could not import tool module"):
            self.failures.append(record.getMessage())


def _scrub(text: str, home: Path) -> str:
    import platform as _platform
    for raw, token in sorted({
        str(home / ".hermes"): "<HERMES_HOME>",
        str(home): "<HOME>",
        f"{_platform.system()} ({_platform.release()})": "<HOST_OS>",
    }.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(raw, token)
    return text


def _enabled_toolsets(platform: str) -> list[str]:
    if platform in TUI_GATEWAY_PLATFORMS:
        from tui_gateway.server import _load_enabled_toolsets
        return list(_load_enabled_toolsets(platform) or [])
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    return sorted(_get_platform_tools(load_config(), platform))


def _render_surface(home: Path, platform: str, model: str, code: bool) -> tuple[dict[str, str], dict]:
    from hermes_cli.config import load_config, save_config
    from agent.system_prompt import build_system_prompt_parts
    from run_agent import AIAgent

    cfg = load_config()
    cfg["model"] = {**(cfg.get("model") or {}), "default": model}
    save_config(cfg)  # tool schemas that track the main model read config, not the agent
    cwd = home / ("code" if code else "plain")
    os.chdir(cwd)
    agent = AIAgent(
        model=model, api_key="inspect-only", base_url="https://openrouter.ai/api/v1",
        quiet_mode=True, save_trajectories=False, platform=platform, skip_context_files=True,
        session_id=SESSION_ID, enabled_toolsets=_enabled_toolsets(platform),
    )
    tiers = {tier: _scrub(text, home) for tier, text in build_system_prompt_parts(agent).items()}
    tools = json.loads(_scrub(json.dumps(agent.tools), home))
    return tiers, {t["function"]["name"]: t["function"] for t in tools}


def _render_child(out: Path) -> int:
    home = Path(os.environ["HOME"]).resolve()
    (home / "plain").mkdir(parents=True, exist_ok=True)
    (home / "code").mkdir(parents=True, exist_ok=True)
    (home / "code" / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    failures = _ImportFailures()
    logging.getLogger().addHandler(failures)
    _pin_environment()
    from tools.skills_sync import sync_skills
    sync_skills(quiet=True)

    prompts, tools_by_surface = {}, {}
    for surface, platform, family, code in SURFACES:
        prompts[surface], tools_by_surface[surface] = _render_surface(home, platform, MODELS[family], code)
    if failures.failures:
        print("\n".join(failures.failures), file=sys.stderr)
        return 1
    _write_snapshot(out, prompts, tools_by_surface)
    return 0


def _write_snapshot(out: Path, prompts: dict[str, dict[str, str]], tools_by_surface: dict[str, dict]) -> None:
    """One text file per surface prompt; each tool schema once (``tools/<name>.json``), with a
    ``tools/<name>.<surface>.json`` copy only where a surface serves a different schema, so a
    description edit is one hunk in review rather than one per surface."""
    (out / "prompts").mkdir(parents=True)
    (out / "tools").mkdir()
    manifest: dict[str, dict] = {}
    for surface, tiers in prompts.items():
        text = "\n\n".join(f"===== {tier} =====\n{tiers[tier]}" for tier in ("stable", "context", "volatile"))
        (out / "prompts" / f"{surface}.txt").write_text(text + "\n", encoding="utf-8")
        tools = tools_by_surface[surface]
        manifest[surface] = {
            "prompt_tokens": {tier: len(tiers[tier]) // CHARS_PER_TOKEN for tier in tiers},
            "tool_tokens": len(json.dumps(tools, sort_keys=True)) // CHARS_PER_TOKEN,
            "tools": sorted(tools),
        }
    for name in sorted({n for tools in tools_by_surface.values() for n in tools}):
        variants = {s: json.dumps(t[name], indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                    for s, t in tools_by_surface.items() if name in t}
        common = Counter(variants.values()).most_common(1)[0][0]
        (out / "tools" / f"{name}.json").write_text(common, encoding="utf-8")
        for surface, text in variants.items():
            if text != common:
                (out / "tools" / f"{name}.{surface}.json").write_text(text, encoding="utf-8")
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render(out: Path) -> int:
    """Render into *out* from a fresh child process and home, so nothing from the caller's
    environment, config or imported modules reaches the bytes."""
    with tempfile.TemporaryDirectory(prefix="prompt-surface-") as tmp:
        home = Path(tmp) / "home"
        home.mkdir()
        staging = Path(tmp) / "out"
        env = {
            "PATH": os.environ.get("PATH", ""), "HOME": str(home), "HERMES_HOME": str(home / ".hermes"),
            "TZ": "UTC", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONPATH": str(ROOT),
            "TMPDIR": tmp, "PYTHONDONTWRITEBYTECODE": "1",
        }
        for key in ("HERMES_RUNTIME_DIR", "SYSTEMROOT"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "_child", str(staging)],
            env=env, cwd=tmp, capture_output=True, text=True, encoding="utf-8", timeout=600, check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-6000:])
            return proc.returncode
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(staging, out)
    print(f"rendered {len(SURFACES)} surfaces into {out}")
    return 0


# ── diff (stdlib only) ───────────────────────────────────────────────────────

def _signed(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)


def _load(snapshot: Path) -> tuple[dict, dict[str, str], dict[str, dict]]:
    """(manifest, prompts by surface, tools by surface) with each surface's tool schemas resolved."""
    if not (snapshot / MANIFEST).exists():
        return {}, {}, {}
    manifest = json.loads((snapshot / MANIFEST).read_text(encoding="utf-8-sig"))
    prompts = {s: (snapshot / "prompts" / f"{s}.txt").read_text(encoding="utf-8-sig") for s in manifest}
    tools: dict[str, dict] = {}
    for surface, entry in manifest.items():
        tools[surface] = {}
        for name in entry["tools"]:
            variant = snapshot / "tools" / f"{name}.{surface}.json"
            path = variant if variant.exists() else snapshot / "tools" / f"{name}.json"
            tools[surface][name] = json.loads(path.read_text(encoding="utf-8-sig"))
    return manifest, prompts, tools


def _tok(obj) -> int:
    return len(obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)) // CHARS_PER_TOKEN


def _tool_changes(base: dict, head: dict) -> list[str]:
    lines = [f"- `{n}` **added** ({_tok(head[n])} tok)" for n in sorted(head.keys() - base.keys())]
    lines += [f"- `{n}` **removed** ({_tok(base[n])} tok)" for n in sorted(base.keys() - head.keys())]
    for name in sorted(base.keys() & head.keys()):
        b, h = base[name], head[name]
        if b == h:
            continue
        bits: list[str] = []
        if b.get("description") != h.get("description"):
            bits.append(f"description {_signed(_tok(h.get('description', '')) - _tok(b.get('description', '')))} tok")
        bp = (b.get("parameters") or {}).get("properties") or {}
        hp = (h.get("parameters") or {}).get("properties") or {}
        if added := sorted(hp.keys() - bp.keys()):
            bits.append("param added: " + ", ".join(f"`{p}`" for p in added))
        if removed := sorted(bp.keys() - hp.keys()):
            bits.append("param removed: " + ", ".join(f"`{p}`" for p in removed))
        if edited := sorted(p for p in bp.keys() & hp.keys() if bp[p] != hp[p]):
            bits.append("param changed: " + ", ".join(f"`{p}`" for p in edited))
        if (b.get("parameters") or {}).get("required") != (h.get("parameters") or {}).get("required"):
            bits.append("required list changed")
        lines.append(f"- `{name}`: " + ("; ".join(bits) or "schema changed"))
    return lines


def _hunks(base: str, head: str, limit: int) -> str:
    lines = [ln for ln in difflib.unified_diff(base.splitlines(), head.splitlines(), lineterm="", n=1)
             if not ln.startswith(("---", "+++"))]
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more lines: see the Files tab"]
    return "\n".join(lines)


def summarize(base_dir: Path, head_dir: Path, line_limit: int = 80, char_limit: int = 30_000) -> str:
    """Markdown summary of base→head; ``""`` when the surfaces are identical. Capped at
    *char_limit* (a PR comment holds 65,536 characters and shares that with other jobs)."""
    bm, bp, bt = _load(base_dir)
    hm, hp, ht = _load(head_dir)
    if not bm and hm:
        return (f"Prompt-surface snapshot introduced: {len(hm)} surfaces "
                f"({', '.join(sorted(hm))}). Later PRs get a per-surface diff here.\n")
    out: list[str] = []
    rows = []
    for s in sorted(bm.keys() | hm.keys()):
        if bp.get(s) == hp.get(s) and bt.get(s) == ht.get(s):
            continue
        b_prompt, h_prompt = _tok(bp.get(s, "")), _tok(hp.get(s, ""))
        b_tools, h_tools = _tok(bt.get(s, {})), _tok(ht.get(s, {}))
        rows.append(f"| {s} | {b_prompt}→{h_prompt} ({_signed(h_prompt - b_prompt)}) | "
                    f"{len(bt.get(s, {}))}→{len(ht.get(s, {}))} tools, {b_tools}→{h_tools} ({_signed(h_tools - b_tools)}) |")
    if not rows:
        return ""
    out += ["| surface | system prompt tok | tools |", "|---|---|---|", *rows, ""]

    by_diff: dict[str, list[str]] = {}
    for s in sorted(bp.keys() | hp.keys()):
        if bp.get(s) != hp.get(s):
            by_diff.setdefault(_hunks(bp.get(s, ""), hp.get(s, ""), line_limit), []).append(s)
    if by_diff:
        out += ["#### System prompt", ""]
        for body, surfaces in by_diff.items():
            out += [f"<details><summary>{', '.join(surfaces)}</summary>", "", "```diff", body, "```", "</details>", ""]

    by_change: dict[str, list[str]] = {}
    for s in sorted(bt.keys() | ht.keys()):
        if bt.get(s) != ht.get(s):
            by_change.setdefault("\n".join(_tool_changes(bt.get(s, {}), ht.get(s, {}))), []).append(s)
    if by_change:
        out += ["#### Tool schemas", ""]
        for body, surfaces in by_change.items():
            out += [f"**{', '.join(surfaces)}**", "", body, ""]
    text = "\n".join(out).rstrip() + "\n"
    if len(text) > char_limit:
        text = text[:char_limit].rsplit("\n", 1)[0] + (
            "\n\n... summary truncated: the full change is in tests/fixtures/prompt_surface/ (Files tab).\n")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="regenerate the committed snapshot (default) or --out DIR")
    r.add_argument("--out", type=Path, default=SNAPSHOT)
    d = sub.add_parser("diff", help="Markdown summary of two snapshot directories")
    d.add_argument("base", type=Path)
    d.add_argument("head", type=Path)
    d.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("_child")
    c.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "render":
        return render(args.out.resolve())
    if args.cmd == "_child":
        return _render_child(args.out)
    text = summarize(args.base, args.head)
    args.output.write_text(text, encoding="utf-8")
    print("prompt surface: changed" if text else "prompt surface: unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
