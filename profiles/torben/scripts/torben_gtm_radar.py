from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.gtm_radar_adapter import (
    DEFAULT_MAGNUS_RADAR_PATH,
    build_torben_gtm_radar_adapter,
    load_magnus_gtm_radar,
    write_gtm_radar_adapter_artifacts,
)

DEFAULT_MAGNUS_ROOT = Path("/Users/ericfreeman/magnus")
DEFAULT_MAGNUS_HERMES_HOME = Path("/Users/ericfreeman/.hermes/profiles/magnus")
DEFAULT_MAGNUS_REFRESH_TIMEOUT_SECONDS = 480


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return default


def _refresh_magnus_enabled(*, preview: bool) -> bool:
    explicit = os.getenv("TORBEN_GTM_REFRESH_MAGNUS")
    if explicit is not None:
        return _truthy(explicit)
    if preview:
        return _truthy(os.getenv("TORBEN_GTM_REFRESH_MAGNUS_PREVIEW"))
    return True


def _newsletter_files_from_env() -> list[Path]:
    raw_values = [
        os.getenv("TORBEN_GTM_NEWSLETTER_FILE", ""),
        os.getenv("TORBEN_GTM_NEWSLETTER_FILES", ""),
    ]
    files: list[Path] = []
    seen: set[str] = set()
    for raw in raw_values:
        for value in str(raw or "").split(os.pathsep):
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            files.append(Path(text))
    if _truthy(os.getenv("TORBEN_GTM_INCLUDE_DEFAULT_NEWSLETTER", "1")):
        default_path = get_hermes_home() / "state" / "torben-morning-brief-inbox-context-latest.json"
        default_text = str(default_path)
        if default_path.exists() and default_text not in seen:
            files.append(default_path)
    return files


def _magnus_radar_command(
    *,
    hours_back: int | None = None,
    scanner_max_items: int | None = None,
    min_score: int | None = None,
    newsletter_files: list[Path] | None = None,
    dry_run: bool = False,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/cron_gtm_intelligence_radar.py",
        "--json",
        "--no-delivery-context",
    ]
    if hours_back is not None:
        command.extend(["--hours-back", str(hours_back)])
    if scanner_max_items is not None:
        command.extend(["--max-items", str(scanner_max_items)])
    if min_score is not None:
        command.extend(["--min-score", str(min_score)])
    for file_path in newsletter_files or []:
        command.extend(["--newsletter-file", str(file_path)])
    if dry_run:
        command.extend(["--dry-run", "--include-seen", "--no-mark-seen", "--no-persist"])
    return command


def _extract_json_object(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Magnus radar did not return JSON") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Magnus radar JSON was not an object")
    return payload


def _refresh_magnus_radar(*, preview: bool) -> dict:
    root = Path(os.getenv("TORBEN_GTM_MAGNUS_ROOT") or DEFAULT_MAGNUS_ROOT)
    hermes_home = Path(os.getenv("TORBEN_GTM_MAGNUS_HERMES_HOME") or DEFAULT_MAGNUS_HERMES_HOME)
    timeout_seconds = _env_int("TORBEN_GTM_MAGNUS_TIMEOUT_SECONDS", DEFAULT_MAGNUS_REFRESH_TIMEOUT_SECONDS)
    dry_run = preview or _truthy(os.getenv("TORBEN_GTM_MAGNUS_DRY_RUN"))
    command = _magnus_radar_command(
        hours_back=_optional_env_int("TORBEN_GTM_MAGNUS_HOURS_BACK"),
        scanner_max_items=_optional_env_int("TORBEN_GTM_MAGNUS_MAX_ITEMS"),
        min_score=_optional_env_int("TORBEN_GTM_MAGNUS_MIN_SCORE"),
        newsletter_files=_newsletter_files_from_env(),
        dry_run=dry_run,
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env.setdefault("UV_PROJECT_ENVIRONMENT", "venv")
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = round(time.monotonic() - started, 3)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    payload = _extract_json_object(stdout)
    summary = {
        "status": "success" if result.returncode == 0 and payload.get("success") else "failed",
        "profile": "magnus",
        "command": command,
        "dry_run": dry_run,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "generated_at": payload.get("generated_at"),
        "success": bool(payload.get("success")),
        "scanned_count": payload.get("scanned_count", 0),
        "finding_count": len(payload.get("findings") or []),
        "public_actions_taken": payload.get("public_actions_taken", 0),
        "external_mutations": payload.get("external_mutations", 0),
    }
    if result.returncode != 0 or not payload.get("success"):
        summary["stderr"] = stderr[-600:]
        summary["stdout"] = stdout[-600:]
        raise RuntimeError(
            "Magnus GTM radar refresh failed "
            f"(returncode={result.returncode}, success={bool(payload.get('success'))})"
        )
    return summary


def _optional_env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def main() -> int:
    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    radar_path = Path(os.getenv("TORBEN_GTM_RADAR_PATH") or DEFAULT_MAGNUS_RADAR_PATH)
    preview = _truthy(os.getenv("TORBEN_GTM_RADAR_PREVIEW"))
    max_items = int(os.getenv("TORBEN_GTM_RADAR_MAX_ITEMS", "3"))

    try:
        source_refresh = None
        if _refresh_magnus_enabled(preview=preview):
            source_refresh = _refresh_magnus_radar(preview=preview)
        radar = load_magnus_gtm_radar(radar_path)
        payload = build_torben_gtm_radar_adapter(
            radar,
            ledger=ActionLedger(state_dir / "torben-action-ledger.json"),
            state_path=state_dir / "torben-gtm-radar-adapter-state.json",
            max_items=max_items,
            mark_delivered=not preview,
            stage_actions=not preview,
        )
        if source_refresh:
            payload["source_refresh"] = source_refresh
    except Exception as exc:  # noqa: BLE001
        payload = {
            "task": "torben_gtm_radar_adapter",
            "wakeAgent": True,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:300],
            },
            "public_actions_taken": 0,
            "external_mutations": 0,
            "text": (
                "Torben / GTM Radar\n\n"
                "Magnus GTM radar refresh failed before it could produce a useful brief.\n"
                f"Reason: {type(exc).__name__}: {str(exc)[:180]}\n"
                "Nothing has been posted, replied to, scheduled, or sent.\n"
            ),
        }

    write_gtm_radar_adapter_artifacts(
        payload,
        json_path=state_dir / "torben-gtm-radar-latest.json",
        text_path=state_dir / "torben-gtm-radar-latest.txt",
    )
    if payload.get("wakeAgent") and payload.get("text"):
        print(str(payload["text"]), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
