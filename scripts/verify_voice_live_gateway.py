#!/usr/bin/env python3
"""Verify a running local Hermes gateway is using the voice-native stack.

This is the live-service counterpart to ``verify_voice_local_stack.py``. It is
intentionally read-mostly: it inspects the systemd user service, verifies the
running process environment points at the expected checkout, confirms imports
resolve from that checkout, and checks the local WhatsApp bridge health.

Pass ``--run-tts-smoke`` to also generate one live-config TTS reply and verify
that Hermes returns a WhatsApp-ready Ogg/Opus voice-note file.

Pass ``--voice-bin`` with ``--calling-sidecar-url`` to compare the running
sidecar's machine-readable contract with the installed ``voice stream-contract``.
Pass ``--sidecar-service`` to also verify the systemd user unit that runs the
local WebRTC sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from verify_voice_local_stack import audit_live_hermes_root


DEFAULT_SERVICE = "hermes-gateway.service"
DEFAULT_BRIDGE_URL = "http://127.0.0.1:3000"
DEFAULT_TTS_TEXT = "Hermes live voice gateway smoke."
CALLING_TTS_STREAM_ENV = "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND"
CONTRACT_COMPARE_KEYS = (
    "contract",
    "version",
    "status",
    "summary",
    "audio",
    "voice_surfaces",
    "endpoints",
    "payloads",
)
REQUIRED_VOICE_SURFACES = {
    "completed_voice_note": {
        "output": "audio/ogg; codecs=opus",
        "transport": "completed_file",
    },
    "streamed_voice_note": {
        "output": "audio/ogg; codecs=opus",
        "transport": "daemon_stream_encoded_file",
    },
    "raw_outbound_pcm": {
        "output": "pcm_s16le",
        "transport": "stdout_pcm_frames",
    },
    "raw_inbound_pcm": {
        "input": "pcm_s16le",
        "transport": "stdin_pcm_frames",
    },
    "file_transcription_smoke": {
        "input": "audio_file",
        "transport": "decoded_file_to_daemon_frames",
    },
}
REQUIRED_ENDPOINTS = {
    "contract": ("GET", "/contract"),
    "health": ("GET", "/health"),
    "offer": ("POST", "/offer"),
    "call_status": ("GET", "/calls/{call_id}"),
    "receive_audio": ("GET", "/calls/{call_id}/audio"),
    "send_audio": ("POST", "/calls/{call_id}/audio"),
    "clear_audio": ("POST", "/calls/{call_id}/audio/clear"),
    "close_call": ("POST", "/calls/{call_id}/close"),
}
REQUIRED_PAYLOADS = (
    "offer_request",
    "offer_response",
    "call_state",
    "call_status_response",
    "close_call_response",
    "send_audio_request",
    "send_audio_response",
    "clear_audio_response",
    "receive_audio_response",
    "audio_shape",
    "error_response",
)
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


def parse_exec_start_argv(exec_start: str) -> list[str]:
    match = re.search(r"argv\[\]=(.*?)(?:\s;\s|\s\})", exec_start)
    if match:
        return shlex.split(match.group(1))
    return shlex.split(exec_start)


def option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    prefix = f"{option}="
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif arg.startswith(prefix):
            values.append(arg.removeprefix(prefix))
    return values


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
            "-p",
            "ExecStart",
            "-p",
            "WorkingDirectory",
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


def validate_service_state(state: dict[str, str], *, label: str = "gateway service") -> int:
    if state.get("ActiveState") != "active":
        raise SystemExit(f"{label} is not active: {state}")
    try:
        pid = int(state.get("MainPID") or "0")
    except ValueError as exc:
        raise SystemExit(f"{label} MainPID is invalid: {state.get('MainPID')!r}") from exc
    if pid <= 0:
        raise SystemExit(f"{label} has no running MainPID: {state}")
    return pid


def normalized_path_text(value: str) -> str:
    return str(Path(value).expanduser().resolve())


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


def validate_calling_sidecar_env(env: dict[str, str], expected_url: str) -> dict[str, Any]:
    expected = expected_url.rstrip("/")
    configured = str(env.get("WHATSAPP_CLOUD_CALLING_SIDECAR_URL") or "").rstrip("/")
    if configured != expected:
        raise SystemExit(
            "running gateway process WHATSAPP_CLOUD_CALLING_SIDECAR_URL "
            f"does not match {expected!r}: {configured!r}"
        )

    stream_command = str(env.get(CALLING_TTS_STREAM_ENV) or "")
    missing = [
        token
        for token in ("--raw-output", "{input_path}", "{sample_rate}", "{frame_ms}")
        if token not in stream_command
    ]
    if missing:
        raise SystemExit(
            f"running gateway process {CALLING_TTS_STREAM_ENV} is missing "
            f"{', '.join(missing)}: {stream_command!r}"
        )

    return {
        "url": configured,
        "tts_stream_command": stream_command,
        "tts_stream_timeout": env.get(
            "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT",
            "",
        ),
    }


def validate_sidecar_service_state(
    state: dict[str, str],
    *,
    service: str,
    voice_bin: str | None,
    voice_repo: Path | None,
    sidecar_url: str | None,
) -> dict[str, Any]:
    pid = validate_service_state(state, label=service)
    env = parse_systemd_environment(state.get("Environment", ""))
    result: dict[str, Any] = {
        "service": service,
        "pid": pid,
    }

    if voice_bin is not None:
        configured_voice_bin = str(env.get("VOICE_BIN") or "")
        if not configured_voice_bin:
            raise SystemExit(f"{service} does not set VOICE_BIN")
        if normalized_path_text(configured_voice_bin) != normalized_path_text(voice_bin):
            raise SystemExit(
                f"{service} VOICE_BIN does not match {voice_bin!r}: "
                f"{configured_voice_bin!r}"
            )
        result["voice_bin"] = configured_voice_bin

    exec_start = str(state.get("ExecStart") or "")
    exec_argv = parse_exec_start_argv(exec_start)

    if sidecar_url:
        parsed_url = urlparse(sidecar_url)
        expected_host = parsed_url.hostname
        expected_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
        if not expected_host:
            raise SystemExit(f"sidecar URL has no host: {sidecar_url!r}")
        host_candidates = {expected_host}
        if expected_host == "localhost":
            host_candidates.add("127.0.0.1")
        elif expected_host == "127.0.0.1":
            host_candidates.add("localhost")

        host_values = set(option_values(exec_argv, "--host"))
        port_values = set(option_values(exec_argv, "--port"))
        has_host = bool(host_candidates & host_values)
        has_port = str(expected_port) in port_values
        if not has_host or not has_port:
            raise SystemExit(
                f"{service} ExecStart does not bind expected sidecar URL "
                f"{sidecar_url}: {exec_start!r}"
            )
        result["sidecar_url"] = sidecar_url.rstrip("/")
        result["bind"] = {
            "host": expected_host,
            "port": expected_port,
        }

    if voice_repo is not None:
        expected_root = voice_repo.expanduser().resolve()
        working_directory = str(state.get("WorkingDirectory") or "")
        if not working_directory:
            raise SystemExit(f"{service} does not set WorkingDirectory")
        if Path(working_directory).expanduser().resolve() != expected_root:
            raise SystemExit(
                f"{service} WorkingDirectory does not match {expected_root}: "
                f"{working_directory!r}"
            )

        sidecar_path = expected_root / "examples" / "webrtc-sidecar" / "sidecar.py"
        if str(sidecar_path) not in exec_start:
            raise SystemExit(
                f"{service} ExecStart does not reference expected sidecar "
                f"{sidecar_path}: {exec_start!r}"
            )
        result["working_directory"] = working_directory
        result["sidecar_path"] = str(sidecar_path)

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


def get_json_url(target: str, *, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(target, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise SystemExit(f"request failed for {target}: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"request returned invalid JSON for {target}: {body!r}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"JSON root must be an object for {target}")
    return parsed


def get_bridge_health(url: str, *, timeout: float) -> dict[str, Any]:
    target = url.rstrip("/") + "/health"
    return get_json_url(target, timeout=timeout)


def get_calling_sidecar_contract(url: str, *, timeout: float) -> dict[str, Any]:
    contract = get_json_url(url.rstrip("/") + "/contract", timeout=timeout)
    validate_calling_sidecar_contract(contract)
    return contract


def validate_calling_sidecar_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("contract") != "voice.webrtc_sidecar":
        raise SystemExit(
            "calling sidecar contract id is not voice.webrtc_sidecar: "
            f"{contract.get('contract')!r}"
        )
    audio = contract.get("audio")
    if not isinstance(audio, dict):
        raise SystemExit("calling sidecar contract missing audio object")
    expected_audio = {
        "sample_rate": 48_000,
        "channels": 1,
        "frame_ms": 20,
        "encoding": "pcm_s16le",
        "frame_bytes": 1_920,
    }
    mismatches = {
        key: audio.get(key)
        for key, expected in expected_audio.items()
        if audio.get(key) != expected
    }
    if mismatches:
        raise SystemExit(
            "calling sidecar audio contract does not match Hermes WebRTC PCM "
            f"shape: {mismatches}"
        )
    required_sections = validate_required_contract_sections(contract, audio=audio)
    return {
        "contract": str(contract.get("contract")),
        "version": contract.get("version"),
        "audio": {key: audio.get(key) for key in expected_audio},
        "required_sections": required_sections,
    }


def required_mapping(contract: dict[str, Any], section: str) -> dict[str, Any]:
    value = contract.get(section)
    if not isinstance(value, dict):
        raise SystemExit(f"calling sidecar contract missing {section} object")
    return value


def validate_required_contract_sections(
    contract: dict[str, Any],
    *,
    audio: dict[str, Any],
) -> dict[str, list[str]]:
    surfaces = required_mapping(contract, "voice_surfaces")
    endpoints = required_mapping(contract, "endpoints")
    payloads = required_mapping(contract, "payloads")

    validate_voice_surface_contracts(surfaces, audio=audio)
    validate_endpoint_contracts(endpoints)
    missing_payloads = [name for name in REQUIRED_PAYLOADS if name not in payloads]
    if missing_payloads:
        raise SystemExit(
            "calling sidecar contract payloads missing keys: "
            + ", ".join(missing_payloads)
        )

    return {
        "voice_surfaces": list(REQUIRED_VOICE_SURFACES),
        "endpoints": list(REQUIRED_ENDPOINTS),
        "payloads": list(REQUIRED_PAYLOADS),
    }


def validate_voice_surface_contracts(
    surfaces: dict[str, Any],
    *,
    audio: dict[str, Any],
) -> None:
    missing = [name for name in REQUIRED_VOICE_SURFACES if name not in surfaces]
    if missing:
        raise SystemExit(
            "calling sidecar contract voice_surfaces missing keys: "
            + ", ".join(missing)
        )

    for name, expected in REQUIRED_VOICE_SURFACES.items():
        surface = surfaces.get(name)
        if not isinstance(surface, dict):
            raise SystemExit(
                f"calling sidecar contract voice_surfaces.{name} must be an object"
            )
        for field, expected_value in expected.items():
            if surface.get(field) != expected_value:
                raise SystemExit(
                    "calling sidecar contract voice_surfaces."
                    f"{name}.{field} must be {expected_value!r}: "
                    f"{surface.get(field)!r}"
                )
        command = str(surface.get("command") or "")
        if not command:
            raise SystemExit(
                f"calling sidecar contract voice_surfaces.{name}.command is empty"
            )

    raw_frame_bytes = audio.get("frame_bytes")
    for name in ("raw_outbound_pcm", "raw_inbound_pcm"):
        surface = surfaces[name]
        if surface.get("frame_bytes") != raw_frame_bytes:
            raise SystemExit(
                f"calling sidecar contract voice_surfaces.{name}.frame_bytes "
                f"must match audio.frame_bytes: {surface.get('frame_bytes')!r}"
            )


def validate_endpoint_contracts(endpoints: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_ENDPOINTS if name not in endpoints]
    if missing:
        raise SystemExit(
            "calling sidecar contract endpoints missing keys: "
            + ", ".join(missing)
        )

    for name, (method, path) in REQUIRED_ENDPOINTS.items():
        endpoint = endpoints.get(name)
        if not isinstance(endpoint, dict):
            raise SystemExit(
                f"calling sidecar contract endpoints.{name} must be an object"
            )
        if endpoint.get("method") != method or endpoint.get("path") != path:
            raise SystemExit(
                f"calling sidecar contract endpoints.{name} must be "
                f"{method} {path}: {endpoint!r}"
            )


def get_calling_sidecar_status(url: str, *, timeout: float) -> dict[str, Any]:
    base = url.rstrip("/")
    contract = get_calling_sidecar_contract(base, timeout=timeout)
    health = get_json_url(base + "/health", timeout=timeout)
    if health.get("ok") is not True:
        raise SystemExit(f"calling sidecar health is not ok: {health}")
    return {
        "contract": validate_calling_sidecar_contract(contract),
        "health": {
            "ok": health.get("ok"),
            "sessions": health.get("sessions"),
            "call_ids": health.get("call_ids"),
        },
    }


def load_voice_stream_contract(voice_bin: str, *, timeout: float) -> dict[str, Any]:
    completed = run_command([voice_bin, "stream-contract"], timeout=timeout)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"voice stream-contract failed: {detail}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"voice stream-contract returned invalid JSON: {completed.stdout!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise SystemExit("voice stream-contract JSON root must be an object")
    validate_calling_sidecar_contract(parsed)
    return parsed


def compare_voice_and_sidecar_contracts(
    *,
    voice_contract: dict[str, Any],
    sidecar_contract: dict[str, Any],
) -> dict[str, Any]:
    mismatches = [
        key
        for key in CONTRACT_COMPARE_KEYS
        if voice_contract.get(key) != sidecar_contract.get(key)
    ]
    if mismatches:
        raise SystemExit(
            "voice stream-contract does not match running sidecar /contract "
            f"for: {', '.join(mismatches)}"
        )
    validated = validate_calling_sidecar_contract(voice_contract)
    return {
        "success": True,
        "contract": validated["contract"],
        "version": validated["version"],
        "audio": validated["audio"],
        "matched_keys": list(CONTRACT_COMPARE_KEYS),
        "required_sections": validated["required_sections"],
    }


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
    parser.add_argument(
        "--sidecar-service",
        help=(
            "Optional systemd user service name for the WebRTC sidecar. When "
            "provided, verify it is active and points at --voice-bin / "
            "--voice-repo."
        ),
    )
    parser.add_argument("--live-hermes-root", type=Path, required=True)
    parser.add_argument("--voice-repo", type=Path)
    parser.add_argument("--hermes-home", type=Path, default=Path("~/.hermes"))
    parser.add_argument("--python-bin", default="~/.hermes/hermes-agent/venv/bin/python")
    parser.add_argument("--bridge-url", default=DEFAULT_BRIDGE_URL)
    parser.add_argument(
        "--calling-sidecar-url",
        help=(
            "Optional expected local WhatsApp Calling sidecar URL. When set, "
            "the verifier requires the running gateway process to be configured "
            "with this URL and validates the sidecar /contract and /health."
        ),
    )
    parser.add_argument(
        "--voice-bin",
        help=(
            "Optional voice binary. With --calling-sidecar-url, compare "
            "`voice stream-contract` with the running sidecar /contract."
        ),
    )
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
    voice_bin = (
        resolve_executable(args.voice_bin, label="voice binary")
        if args.voice_bin
        else None
    )
    voice_repo = args.voice_repo.expanduser().resolve() if args.voice_repo else None
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

    if args.calling_sidecar_url:
        checks["calling_sidecar"] = {
            "env": validate_calling_sidecar_env(
                process_env,
                args.calling_sidecar_url,
            ),
            "sidecar": get_calling_sidecar_status(
                args.calling_sidecar_url,
                timeout=args.timeout,
            ),
        }
        if voice_bin:
            checks["voice_sidecar_contract"] = compare_voice_and_sidecar_contracts(
                voice_contract=load_voice_stream_contract(
                    voice_bin,
                    timeout=args.timeout,
                ),
                sidecar_contract=get_calling_sidecar_contract(
                    args.calling_sidecar_url,
                    timeout=args.timeout,
                ),
            )
    elif voice_bin and not args.sidecar_service:
        raise SystemExit("--voice-bin requires --calling-sidecar-url or --sidecar-service")

    if args.sidecar_service:
        checks["sidecar_service"] = validate_sidecar_service_state(
            get_service_state(args.sidecar_service, timeout=args.timeout),
            service=args.sidecar_service,
            voice_bin=voice_bin,
            voice_repo=voice_repo,
            sidecar_url=args.calling_sidecar_url,
        )

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
