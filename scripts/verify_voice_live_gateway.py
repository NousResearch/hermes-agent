#!/usr/bin/env python3
"""Verify a running local Hermes gateway is using the voice-native stack.

This is the live-service counterpart to ``verify_voice_local_stack.py``. It is
intentionally read-mostly: it inspects the systemd user service, verifies the
running process environment points at the expected checkout, confirms imports
resolve from that checkout, and checks the local WhatsApp bridge health.

Pass ``--run-tts-smoke`` to also generate one live-config TTS reply and verify
that Hermes returns a WhatsApp-ready Ogg/Opus voice-note file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from verify_voice_local_stack import audit_live_hermes_root


DEFAULT_SERVICE = "hermes-gateway.service"
DEFAULT_BRIDGE_URL = "http://127.0.0.1:3000"
DEFAULT_TTS_TEXT = "Hermes live voice gateway smoke."
IMPORT_MODULES = (
    "hermes_cli.main",
    "tools.tts_tool",
    "tools.tirith_security",
    "gateway.platforms.whatsapp_cloud",
    "gateway.platforms.whatsapp",
)


def resolve_executable(value: str, *, label: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} is not executable: {path}")
        return os.path.abspath(os.path.expanduser(value))

    found = shutil.which(value)
    if not found:
        raise SystemExit(f"{label} not found on PATH: {value}")
    return found


def run_command(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def parse_systemctl_show(output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def get_service_state(service: str, *, timeout: float) -> dict[str, str]:
    completed = run_command(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "Environment",
            "-p",
            "DropInPaths",
            "--no-pager",
        ],
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"systemctl show failed for {service}: {detail}")
    return parse_systemctl_show(completed.stdout)


def parse_systemd_environment(value: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for part in shlex.split(value):
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        env[key] = raw
    return env


def parse_proc_environ(data: bytes) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in data.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode(
            "utf-8", errors="replace"
        )
    return env


def read_process_env(pid: int) -> dict[str, str]:
    path = Path("/proc") / str(pid) / "environ"
    try:
        return parse_proc_environ(path.read_bytes())
    except OSError as exc:
        raise SystemExit(f"failed to read process environment for PID {pid}: {exc}") from exc


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_service_state(state: dict[str, str]) -> int:
    if state.get("ActiveState") != "active":
        raise SystemExit(f"gateway service is not active: {state}")
    try:
        pid = int(state.get("MainPID") or "0")
    except ValueError as exc:
        raise SystemExit(f"gateway service MainPID is invalid: {state.get('MainPID')!r}") from exc
    if pid <= 0:
        raise SystemExit(f"gateway service has no running MainPID: {state}")
    return pid


def validate_env_points_at_root(
    env: dict[str, str],
    root: Path,
    *,
    label: str,
    bridge_bin_dir: Path | None = None,
) -> dict[str, Any]:
    root_text = str(root.resolve())
    pythonpath = env.get("PYTHONPATH", "")
    if root_text not in pythonpath.split(os.pathsep):
        raise SystemExit(f"{label} PYTHONPATH does not include {root_text}: {pythonpath!r}")

    result: dict[str, Any] = {
        "PYTHONPATH": pythonpath,
    }
    if bridge_bin_dir is not None:
        path = env.get("PATH", "")
        bridge_bin_text = str(bridge_bin_dir.resolve())
        if bridge_bin_text not in path.split(os.pathsep):
            raise SystemExit(
                f"{label} PATH does not include bridge bin dir "
                f"{bridge_bin_text}: {path!r}"
            )
        result["bridge_bin_on_path"] = True
    return result


def import_smoke(
    *,
    python_bin: str,
    live_root: Path,
    hermes_home: Path,
    timeout: float,
) -> dict[str, str]:
    code = """
import importlib
import inspect
import json

modules = {}
for name in %r:
    module = importlib.import_module(name)
    modules[name] = inspect.getfile(module)
print(json.dumps(modules, sort_keys=True))
""" % (IMPORT_MODULES,)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(live_root)
    env["HERMES_HOME"] = str(hermes_home)
    completed = subprocess.run(
        [python_bin, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
        cwd=str(live_root),
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"import smoke failed: {detail}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"import smoke did not return JSON: {completed.stdout!r}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("import smoke JSON root must be an object")

    for module, file_path in parsed.items():
        if not path_is_under(Path(str(file_path)), live_root):
            raise SystemExit(
                f"module {module} resolved outside live root {live_root}: {file_path}"
            )
    return {str(key): str(value) for key, value in parsed.items()}


def get_bridge_health(url: str, *, timeout: float) -> dict[str, Any]:
    target = url.rstrip("/") + "/health"
    try:
        with urlopen(target, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise SystemExit(f"bridge health request failed for {target}: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"bridge health returned invalid JSON: {body!r}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("bridge health JSON root must be an object")
    return parsed


def validate_bridge_health(health: dict[str, Any], *, require_connected: bool) -> None:
    if require_connected and health.get("status") != "connected":
        raise SystemExit(f"bridge is not connected: {health}")


def parse_ffprobe_json(output: str) -> dict[str, str]:
    data = json.loads(output)
    streams = data.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("ffprobe output has no streams")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise ValueError("ffprobe stream must be an object")
    return {
        "codec_name": str(stream.get("codec_name") or ""),
        "sample_rate": str(stream.get("sample_rate") or ""),
        "channels": str(stream.get("channels") or ""),
    }


def probe_audio(path: Path, *, ffprobe_bin: str, timeout: float) -> dict[str, str]:
    completed = run_command(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"ffprobe failed for {path}: {detail}")
    try:
        return parse_ffprobe_json(completed.stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ffprobe output invalid for {path}: {completed.stdout!r}") from exc


def run_tts_smoke(
    *,
    python_bin: str,
    live_root: Path,
    hermes_home: Path,
    platform: str,
    text: str,
    ffprobe_bin: str,
    timeout: float,
) -> dict[str, Any]:
    code = """
import json
from tools import tts_tool
print(json.dumps(json.loads(tts_tool.text_to_speech_tool(%r)), sort_keys=True))
""" % (text,)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(live_root)
    env["HERMES_HOME"] = str(hermes_home)
    env["HERMES_SESSION_PLATFORM"] = platform
    completed = subprocess.run(
        [python_bin, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
        cwd=str(live_root),
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"TTS smoke failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"TTS smoke returned invalid JSON: {completed.stdout!r}") from exc
    if result.get("success") is not True:
        raise SystemExit(f"TTS smoke reported failure: {result}")
    if result.get("voice_compatible") is not True:
        raise SystemExit(f"TTS smoke was not voice-compatible: {result}")
    media_tag = str(result.get("media_tag") or "")
    if not media_tag.startswith("[[audio_as_voice]]\nMEDIA:"):
        raise SystemExit(f"TTS smoke media tag is not a voice note: {media_tag!r}")
    audio_path = Path(str(result.get("file_path") or ""))
    if not audio_path.is_file():
        raise SystemExit(f"TTS smoke file does not exist: {audio_path}")
    probe = probe_audio(audio_path, ffprobe_bin=ffprobe_bin, timeout=timeout)
    if probe != {"codec_name": "opus", "sample_rate": "48000", "channels": "1"}:
        raise SystemExit(f"TTS smoke audio is not mono 48 kHz Opus: {probe}")
    return {
        "result": result,
        "probe": probe,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--live-hermes-root", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path, default=Path("~/.hermes"))
    parser.add_argument("--python-bin", default="~/.hermes/hermes-agent/venv/bin/python")
    parser.add_argument("--bridge-url", default=DEFAULT_BRIDGE_URL)
    parser.add_argument("--ffprobe-bin", default=os.environ.get("FFPROBE_BIN", "ffprobe"))
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--allow-disconnected-bridge", action="store_true")
    parser.add_argument(
        "--skip-bridge-health",
        action="store_true",
        help=(
            "Skip the local Baileys bridge /health check. Use this for "
            "Cloud-API-only gateway deployments that do not run the Node "
            "WhatsApp bridge."
        ),
    )
    parser.add_argument("--run-tts-smoke", action="store_true")
    parser.add_argument("--tts-platform", default="whatsapp")
    parser.add_argument("--tts-text", default=DEFAULT_TTS_TEXT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    live_root = args.live_hermes_root.expanduser().resolve()
    hermes_home = args.hermes_home.expanduser().resolve()
    python_bin = resolve_executable(args.python_bin, label="Hermes Python")
    ffprobe_bin = (
        resolve_executable(args.ffprobe_bin, label="ffprobe")
        if args.run_tts_smoke
        else ""
    )
    bridge_bin_dir = live_root / "scripts" / "whatsapp-bridge" / "node_modules" / ".bin"

    checks: dict[str, Any] = {}
    live_root_audit = audit_live_hermes_root(live_root)
    if live_root_audit.get("success") is not True:
        raise SystemExit(
            "live root audit failed:\n- "
            + "\n- ".join(str(item) for item in live_root_audit.get("failures", []))
        )
    checks["live_root"] = live_root_audit

    service_state = get_service_state(args.service, timeout=args.timeout)
    pid = validate_service_state(service_state)
    unit_env = parse_systemd_environment(service_state.get("Environment", ""))
    checks["systemd_unit"] = {
        "service": args.service,
        "pid": pid,
        "drop_in_paths": service_state.get("DropInPaths", ""),
        "env": validate_env_points_at_root(
            unit_env,
            live_root,
            label="systemd unit",
            bridge_bin_dir=bridge_bin_dir,
        ),
    }

    process_env = read_process_env(pid)
    checks["running_process"] = {
        "pid": pid,
        "env": validate_env_points_at_root(
            process_env,
            live_root,
            label="running gateway process",
            bridge_bin_dir=bridge_bin_dir,
        ),
    }

    checks["imports"] = import_smoke(
        python_bin=python_bin,
        live_root=live_root,
        hermes_home=hermes_home,
        timeout=args.timeout,
    )

    if args.skip_bridge_health:
        checks["bridge_health"] = {
            "success": True,
            "skipped": True,
            "reason": "--skip-bridge-health was provided",
        }
    else:
        bridge_health = get_bridge_health(args.bridge_url, timeout=args.timeout)
        validate_bridge_health(
            bridge_health,
            require_connected=not args.allow_disconnected_bridge,
        )
        checks["bridge_health"] = bridge_health

    if args.run_tts_smoke:
        checks["tts_smoke"] = run_tts_smoke(
            python_bin=python_bin,
            live_root=live_root,
            hermes_home=hermes_home,
            platform=args.tts_platform,
            text=args.tts_text,
            ffprobe_bin=ffprobe_bin,
            timeout=args.timeout,
        )

    print(
        json.dumps(
            {
                "success": True,
                "live_hermes_root": str(live_root),
                "hermes_home": str(hermes_home),
                "checks": checks,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
