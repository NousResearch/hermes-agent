"""Runtime smoke diagnostics for ``hermes smoke``.

The default command is read-only and side-effect-light: it runs local/status
checks, prints a compact report, and does not create artifacts or persistent
Hermes chat sessions unless explicitly requested.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List


@dataclass
class SmokeResult:
    name: str
    ok: bool
    detail: str
    elapsed_s: float | None = None
    artifact: str | None = None


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _hermes_cmd(explicit: str | None = None) -> List[str]:
    """Return the Hermes CLI command, with a source-tree fallback.

    ``explicit`` is supplied by argparse/configured callers.  We intentionally
    avoid a non-secret ``HERMES_*`` environment override here; Hermes policy
    keeps non-secret behavior in CLI flags/config rather than ad-hoc env vars.
    """
    if explicit:
        return [explicit]
    installed = shutil.which("hermes")
    if installed:
        return [installed]
    return [sys.executable, "-m", "hermes_cli.main"]


def _run(cmd: List[str], timeout: int = 120, env: dict[str, str] | None = None) -> tuple[int, str, str, float]:
    start = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=env)
    return proc.returncode, proc.stdout, proc.stderr, time.monotonic() - start


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _maybe_write(artifact_dir: Path | None, filename: str, text: str) -> str | None:
    if artifact_dir is None:
        return None
    path = artifact_dir / filename
    _write(path, text)
    return str(path)


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def smoke_profile(profile: str, artifact_dir: Path | None, *, cli: str | None = None, timeout: int = 180) -> SmokeResult:
    """Run an opt-in real chat smoke for one profile.

    This creates a persistent Hermes session by design, so callers must only
    invoke it when the user requested ``--chat``.
    """
    expected = f"SMOKE_{profile}_OK"
    cmd = [
        *_hermes_cmd(cli),
        "--profile",
        profile,
        "chat",
        "-Q",
        "--max-turns",
        "1",
        "--source",
        "smoke",
        "-q",
        f"Antworte exakt: {expected}",
    ]
    safe_profile = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in profile)
    try:
        rc, out, err, elapsed = _run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out_path = _maybe_write(artifact_dir, f"profile-{safe_profile}.stdout", _timeout_text(exc.stdout))
        _maybe_write(artifact_dir, f"profile-{safe_profile}.stderr", _timeout_text(exc.stderr))
        return SmokeResult(f"profile:{profile}", False, f"timeout after {timeout}s", None, out_path)
    out_path = _maybe_write(artifact_dir, f"profile-{safe_profile}.stdout", out)
    _maybe_write(artifact_dir, f"profile-{safe_profile}.stderr", err)
    got = " ".join(out.split())
    ok = rc == 0 and got == expected
    detail = f"rc={rc}; stdout={got[:120]!r}"
    if not ok and expected in got:
        detail += "; expected token present but output not exact"
    return SmokeResult(f"profile:{profile}", ok, detail, elapsed, out_path)


def smoke_command(
    name: str,
    cmd: List[str],
    expected_substring: str | None,
    artifact_dir: Path | None,
    timeout: int = 120,
) -> SmokeResult:
    try:
        rc, out, err, elapsed = _run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out_path = _maybe_write(artifact_dir, f"{name}.stdout", _timeout_text(exc.stdout))
        _maybe_write(artifact_dir, f"{name}.stderr", _timeout_text(exc.stderr))
        return SmokeResult(name, False, f"timeout after {timeout}s", None, out_path)
    out_path = _maybe_write(artifact_dir, f"{name}.stdout", out)
    _maybe_write(artifact_dir, f"{name}.stderr", err)
    combined = out + err
    ok = rc == 0 and (expected_substring is None or expected_substring in combined)
    return SmokeResult(name, ok, f"rc={rc}", elapsed, out_path)


def _openrouter_credits() -> SmokeResult:
    try:
        from hermes_cli.config import get_env_value
        import urllib.request

        key = get_env_value("OPENROUTER_API_KEY") or ""
        if not key:
            return SmokeResult("openrouter-credits", False, "OPENROUTER_API_KEY missing")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {key}", "User-Agent": "hermes-smoke"},
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed HTTPS endpoint
            data = json.load(resp)
        elapsed = time.monotonic() - start
        credits = (data or {}).get("data") or {}
        total = float(credits.get("total_credits") or 0)
        usage = float(credits.get("total_usage") or 0)
        remaining = total - usage
        return SmokeResult("openrouter-credits", True, f"remaining_usd={remaining:.2f}", elapsed)
    except Exception as exc:  # pragma: no cover - network/env dependent
        return SmokeResult("openrouter-credits", False, f"{type(exc).__name__}: {exc}")


def _resolve_artifact_dir(artifact_dir: str | Path | None, *, write_artifacts: bool) -> Path | None:
    if artifact_dir:
        return Path(artifact_dir).expanduser().resolve()
    if write_artifacts:
        return Path(tempfile.gettempdir()) / f"hermes-smoke-{_now_stamp()}"
    return None


def run_smoke(
    profiles: Iterable[str] = ("default", "cheap", "lab"),
    *,
    artifact_dir: str | Path | None = None,
    write_artifacts: bool = False,
    chat: bool = False,
    include_credits: bool = False,
    cli: str | None = None,
) -> dict[str, Any]:
    artifacts = _resolve_artifact_dir(artifact_dir, write_artifacts=write_artifacts)
    if artifacts is not None:
        artifacts.mkdir(parents=True, exist_ok=True)

    results: List[SmokeResult] = []
    base = _hermes_cmd(cli)
    results.append(smoke_command("version", base + ["--version"], "Hermes Agent", artifacts, timeout=60))
    results.append(smoke_command("auth-list", base + ["auth", "list"], None, artifacts, timeout=120))
    results.append(smoke_command("tools-list", base + ["tools", "list"], None, artifacts, timeout=120))
    results.append(smoke_command("doctor", base + ["doctor"], "Hermes Doctor", artifacts, timeout=240))
    results.append(smoke_command("gateway-status", base + ["gateway", "status"], None, artifacts, timeout=120))
    results.append(smoke_command("cron-status", base + ["cron", "status"], None, artifacts, timeout=120))
    results.append(smoke_command("context-audit", base + ["context", "audit", "--cwd", str(Path.cwd())], "Context audit", artifacts, timeout=120))

    if include_credits:
        results.append(_openrouter_credits())

    if chat:
        for profile in profiles:
            results.append(smoke_profile(profile, artifacts, cli=cli))

    payload = {
        "artifact_dir": str(artifacts) if artifacts is not None else None,
        "results": [asdict(r) for r in results],
        "ok": all(r.ok for r in results),
    }
    if artifacts is not None:
        _write(artifacts / "summary.json", json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def render_smoke_report(data: dict[str, Any]) -> str:
    artifact_line = data["artifact_dir"] if data.get("artifact_dir") else "not written (use --write-artifacts or --output-dir)"
    lines = ["Hermes smoke report", f"  artifacts: {artifact_line}", ""]
    for item in data["results"]:
        mark = "OK" if item["ok"] else "FAIL"
        elapsed = "" if item.get("elapsed_s") is None else f" ({item['elapsed_s']:.2f}s)"
        lines.append(f"  {mark:4} {item['name']:<24} {item['detail']}{elapsed}")
    lines.append("")
    lines.append(f"Overall: {'OK' if data['ok'] else 'FAIL'}")
    return "\n".join(lines)


def cmd_smoke(args: Any) -> None:
    profiles_raw = getattr(args, "profiles", "default,cheap,lab") or ""
    profiles = [p.strip() for p in profiles_raw.split(",") if p.strip()]
    chat = bool(getattr(args, "chat", False)) and not bool(getattr(args, "skip_chat", False))
    data = run_smoke(
        profiles=profiles,
        artifact_dir=getattr(args, "output_dir", None),
        write_artifacts=getattr(args, "write_artifacts", False),
        chat=chat,
        include_credits=getattr(args, "credits", False),
        cli=getattr(args, "cli", None),
    )
    if getattr(args, "json", False):
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render_smoke_report(data))
    raise SystemExit(0 if data["ok"] else 1)
