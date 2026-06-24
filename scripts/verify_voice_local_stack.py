#!/usr/bin/env python3
"""Validate the local Hermes + voice stack.

By default this aggregate preflight runs the existing focused verifiers without
touching the live gateway:

1. The local Hermes CLI entrypoint starts with an isolated HERMES_HOME.
2. The voice checkout's own WhatsApp contract verifier proves the installed
   voice binary can emit Ogg/Opus and WebRTC-shaped PCM frames.
3. The local Baileys bridge keeps Ogg/Opus as native voice notes.
4. The WhatsApp Cloud adapter uploads Ogg/Opus with voice-note MIME.
5. Hermes command-provider TTS returns WhatsApp-ready Ogg/Opus from voice.
6. Hermes command-provider STT transcribes through voice stream-transcribe.
7. Hermes' stream command path returns raw 48 kHz mono 20 ms pcm_s16le frames.
8. Hermes' WhatsApp Calling control plane accepts a synthetic SDP offer.
9. Hermes' WhatsApp Calling control plane can answer through a real local sidecar.
10. Optionally, the voice WebRTC sidecar full-duplex smoke passes locally.

By default the script creates a temporary Hermes home and removes it after a
passing or failing run. Pass --keep-home to inspect the generated config.
Pass --live-hermes-root to also prove the checkout used by a local gateway has
the voice-native WhatsApp and sidecar surfaces from this branch.
Pass --run-live-gateway to also inspect the running local gateway and require
the live WebRTC sidecar to answer a real SDP offer with ready_for_accept=true.
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
import tempfile
from typing import Any


DEFAULT_COMMAND_TEXT = "Hermes local voice command preflight."
DEFAULT_COMMAND_STT_TEXT = "hello world"
DEFAULT_VOICE_CONTRACT_TEXT = "Hermes voice contract preflight."
DEFAULT_STREAM_TEXT = "Hermes local voice stream preflight."
DEFAULT_FULL_DUPLEX_INBOUND_TEXT = "hello world"
DEFAULT_FULL_DUPLEX_OUTBOUND_TEXT = "Hello from Hermes through the local voice sidecar."
DEFAULT_FULL_DUPLEX_MAX_QUEUED_TX_MS = 1_000
DEFAULT_WHATSAPP_BRIDGE_MEDIA_TIMEOUT = 15.0
DEFAULT_WHATSAPP_CLOUD_WEBHOOK_TIMEOUT = 15.0
DEFAULT_WHATSAPP_CLOUD_VOICE_TIMEOUT = 15.0

LIVE_ROOT_REQUIREMENTS = (
    {
        "path": "tools/tts_tool.py",
        "description": "command-provider voice-compatible Opus routing",
        "tokens": (
            "voice_compatible",
            "libopus",
            "-application",
            "voip",
        ),
    },
    {
        "path": "tools/transcription_tools.py",
        "description": "command-provider STT routing",
        "tokens": (
            "stt.providers",
            "_transcribe_command_stt",
            "transcribe_audio",
        ),
    },
    {
        "path": "gateway/platforms/whatsapp_cloud.py",
        "description": "WhatsApp Cloud voice notes and calling sidecar client",
        "tokens": (
            "calling_sidecar_url",
            "voice.webrtc_sidecar",
            "_send_calling_sidecar_tts_stream_command",
            "_clear_calling_sidecar_audio",
            "-application",
            "voip",
        ),
    },
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def script_path(name: str) -> Path:
    path = repo_root() / "scripts" / name
    if not path.is_file():
        raise SystemExit(f"script not found: {path}")
    return path


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


def default_hermes_bin() -> str:
    local = repo_root() / ".venv" / "bin" / "hermes"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return "hermes"


def default_voice_repo() -> Path | None:
    configured = os.environ.get("VOICE_REPO")
    return Path(configured) if configured else None


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_json_object(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    text = stdout.strip()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if text[index + end :].strip():
            continue
        if not isinstance(parsed, dict):
            raise ValueError("JSON root must be an object")
        return parsed
    raise ValueError("command did not print a JSON object")


def run_process(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def audit_live_hermes_root(root: Path) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    failures: list[str] = []
    checked: list[dict[str, Any]] = []

    if not resolved.is_dir():
        failures.append(f"live Hermes root is not a directory: {resolved}")
        return {
            "success": False,
            "root": str(resolved),
            "checked": checked,
            "failures": failures,
        }

    for requirement in LIVE_ROOT_REQUIREMENTS:
        rel_path = str(requirement["path"])
        required_tokens = tuple(str(token) for token in requirement["tokens"])
        path = resolved / rel_path
        check = {
            "path": rel_path,
            "description": requirement["description"],
            "required_tokens": list(required_tokens),
        }
        if not path.is_file():
            check["present"] = False
            failures.append(f"{rel_path} is missing")
            checked.append(check)
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [token for token in required_tokens if token not in text]
        check["present"] = True
        check["missing_tokens"] = missing
        if missing:
            failures.append(f"{rel_path} is missing tokens: {', '.join(missing)}")
        checked.append(check)

    return {
        "success": not failures,
        "root": str(resolved),
        "checked": checked,
        "failures": failures,
    }


def require_live_hermes_root(root: Path) -> dict[str, Any]:
    result = audit_live_hermes_root(root)
    if result["success"] is not True:
        raise SystemExit(
            "live Hermes root is missing voice-native integration surfaces:\n- "
            + "\n- ".join(result["failures"])
        )
    return result


def run_plain_step(
    name: str,
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
    include_stdout_lines: bool = False,
) -> dict[str, Any]:
    completed = run_process(command, timeout=timeout, env=env)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"{name} failed with exit code {completed.returncode}"
            + (f": {detail[:1000]}" if detail else "")
        )
    first_line = next(
        (line for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    result: dict[str, Any] = {
        "success": True,
        "command": shell_join(command),
        "stdout_first_line": first_line,
    }
    if include_stdout_lines:
        result["stdout_lines"] = completed.stdout.splitlines()
    return result


def run_json_step(
    name: str,
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str],
) -> dict[str, Any]:
    completed = run_process(command, timeout=timeout, env=env)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"{name} failed with exit code {completed.returncode}"
            + (f": {detail[:1000]}" if detail else "")
        )
    try:
        result = parse_json_object(completed.stdout)
    except ValueError as exc:
        raise SystemExit(f"{name} did not return JSON: {exc}") from exc
    if result.get("success") is not True:
        raise SystemExit(
            f"{name} reported failure: "
            + json.dumps(result, ensure_ascii=False, indent=2)
        )
    return {
        "success": True,
        "command": shell_join(command),
        "result": result,
    }


def child_env(*, hermes_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    root = str(repo_root())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else root
    )
    return env


def command_tts_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
    hermes_home: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("verify_voice_command_tts.py")),
        "--voice-bin",
        voice_bin,
        "--hermes-home",
        str(hermes_home),
        "--force",
        "--provider",
        args.provider,
        "--voice",
        args.voice,
        "--speed",
        args.speed,
        "--timeout",
        f"{args.tts_timeout:g}",
    ]
    if args.keep_home:
        command.append("--keep-home")
    command.extend(["--text", args.command_text])
    return command


def command_stt_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
    hermes_home: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("verify_voice_command_stt.py")),
        "--voice-bin",
        voice_bin,
        "--hermes-home",
        str(hermes_home),
        "--force",
        "--provider",
        args.stt_provider,
        "--voice",
        args.voice,
        "--speed",
        args.speed,
        "--timeout",
        f"{args.stt_timeout:g}",
        "--generate-timeout",
        f"{args.tts_timeout:g}",
    ]
    if args.keep_home:
        command.append("--keep-home")
    command.extend(["--text", args.command_stt_text])
    for word in args.stt_expect_word:
        command.extend(["--expect-word", word])
    return command


def stream_tts_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("verify_voice_stream_tts.py")),
        "--voice-bin",
        voice_bin,
        "--voice",
        args.voice,
        "--speed",
        args.speed,
        "--timeout",
        f"{args.stream_timeout:g}",
        "--text",
        args.stream_text,
    ]
    if args.stream_command_template:
        command.extend(["--command-template", args.stream_command_template])
    return command


def resolve_voice_contract_script(args: argparse.Namespace) -> Path | None:
    if args.skip_voice_contract:
        return None
    if args.voice_repo is None:
        return None

    script = (
        args.voice_repo.expanduser().resolve()
        / "scripts"
        / "verify_whatsapp_voice_contract.sh"
    )
    if not script.is_file():
        raise SystemExit(
            f"voice WhatsApp contract verifier not found: {script}; "
            "update the voice checkout or pass --skip-voice-contract"
        )
    if not os.access(script, os.X_OK):
        raise SystemExit(f"voice WhatsApp contract verifier is not executable: {script}")
    return script


def voice_contract_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
    script: Path,
) -> list[str]:
    command = [
        str(script),
        "--voice-bin",
        voice_bin,
        "--text",
        args.voice_contract_text,
        "--require-daemon",
        "--run-stt-smoke",
    ]
    return command


def whatsapp_bridge_media_payload_command(*, node_bin: str) -> list[str]:
    test_path = repo_root() / "scripts" / "whatsapp-bridge" / "media-payload.test.mjs"
    if not test_path.is_file():
        raise SystemExit(f"WhatsApp bridge media-payload test not found: {test_path}")
    return [node_bin, "--test", str(test_path)]


def whatsapp_cloud_voice_note_command() -> list[str]:
    return [sys.executable, str(script_path("verify_voice_whatsapp_cloud_voice_note.py"))]


def whatsapp_cloud_webhook_command() -> list[str]:
    return [sys.executable, str(script_path("verify_voice_whatsapp_cloud_webhook.py"))]


def calling_control_plane_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(script_path("verify_voice_whatsapp_calling_control_plane.py")),
        "--timeout",
        f"{args.calling_control_plane_timeout:g}",
    ]


def calling_live_sidecar_command(
    args: argparse.Namespace,
    *,
    voice_repo: Path,
) -> list[str]:
    return [
        args.webrtc_python_bin or sys.executable,
        str(script_path("verify_voice_whatsapp_calling_live_sidecar.py")),
        "--voice-repo",
        str(voice_repo),
        "--timeout",
        f"{args.calling_live_sidecar_timeout:g}",
    ]


def resolve_voice_repo_for_full_duplex(args: argparse.Namespace) -> Path | None:
    if args.skip_full_duplex:
        return None
    if args.voice_repo is None:
        raise SystemExit(
            "full-duplex validation needs --voice-repo or VOICE_REPO; "
            "pass --skip-full-duplex to validate only command and stream TTS"
        )
    voice_repo = args.voice_repo.expanduser().resolve()
    smoke = voice_repo / "examples" / "webrtc-sidecar" / "full_duplex_loopback_smoke.py"
    if not smoke.is_file():
        raise SystemExit(f"voice full-duplex smoke not found: {smoke}")
    return voice_repo


def full_duplex_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
    voice_repo: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("verify_voice_full_duplex_sidecar.py")),
        "--voice-repo",
        str(voice_repo),
        "--voice-bin",
        voice_bin,
        "--voice",
        args.voice,
        "--speed",
        args.speed,
        "--timeout",
        f"{args.full_duplex_timeout:g}",
        "--max-queued-tx-ms",
        str(args.full_duplex_max_queued_tx_ms),
        "--inbound-text",
        args.full_duplex_inbound_text,
        "--outbound-text",
        args.full_duplex_outbound_text,
    ]
    if args.webrtc_python_bin:
        command.extend(["--python-bin", args.webrtc_python_bin])
    for word in args.expect_word:
        command.extend(["--expect-word", word])
    return command


def live_gateway_command(
    args: argparse.Namespace,
    *,
    voice_bin: str,
    voice_repo: Path | None,
) -> list[str]:
    if args.live_hermes_root is None:
        raise SystemExit("--run-live-gateway requires --live-hermes-root")
    if not args.calling_sidecar_url:
        raise SystemExit("--run-live-gateway requires --calling-sidecar-url")
    if not args.webrtc_python_bin:
        raise SystemExit("--run-live-gateway requires --webrtc-python-bin")
    if args.run_live_gateway_calling_live_sidecar_smoke and voice_repo is None:
        raise SystemExit(
            "--run-live-gateway-calling-live-sidecar-smoke requires --voice-repo"
        )

    command = [
        sys.executable,
        str(script_path("verify_voice_live_gateway.py")),
        "--live-hermes-root",
        str(args.live_hermes_root.expanduser().resolve()),
        "--python-bin",
        args.live_gateway_python_bin,
        "--hermes-home",
        str(args.live_gateway_hermes_home.expanduser()),
        "--calling-sidecar-url",
        args.calling_sidecar_url,
        "--voice-bin",
        voice_bin,
        "--run-tts-smoke",
        "--run-sidecar-offer-smoke",
        "--webrtc-python-bin",
        args.webrtc_python_bin,
    ]
    if args.run_live_gateway_stt_smoke:
        command.extend(
            [
                "--run-stt-smoke",
                "--stt-provider",
                args.stt_provider,
                "--stt-timeout",
                f"{args.stt_timeout:g}",
            ]
        )
    if args.live_gateway_sidecar_service:
        command.extend(["--sidecar-service", args.live_gateway_sidecar_service])
        if voice_repo is not None:
            command.extend(["--voice-repo", str(voice_repo)])
    if args.live_gateway_sidecar_service and args.live_gateway_voice_daemon_service:
        command.extend(
            [
                "--voice-daemon-service",
                args.live_gateway_voice_daemon_service,
            ]
        )
    if args.run_live_gateway_calling_live_sidecar_smoke:
        command.append("--run-calling-live-sidecar-smoke")
    if args.skip_live_gateway_bridge_health:
        command.append("--skip-bridge-health")
    if args.require_live_gateway_whatsapp_cloud_readiness:
        command.append("--require-whatsapp-cloud-readiness")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=os.environ.get("VOICE_BIN", "voice"))
    parser.add_argument("--hermes-bin", default=os.environ.get("HERMES_BIN", default_hermes_bin()))
    parser.add_argument("--hermes-home", type=Path)
    parser.add_argument("--keep-home", action="store_true")
    parser.add_argument("--provider", default="kokoro")
    parser.add_argument("--stt-provider", default="voice")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--tts-timeout", type=float, default=180.0)
    parser.add_argument("--stt-timeout", type=float, default=300.0)
    parser.add_argument("--stream-timeout", type=float, default=180.0)
    parser.add_argument("--calling-control-plane-timeout", type=float, default=10.0)
    parser.add_argument("--calling-live-sidecar-timeout", type=float, default=12.0)
    parser.add_argument("--full-duplex-timeout", type=float, default=90.0)
    parser.add_argument(
        "--full-duplex-max-queued-tx-ms",
        type=int,
        default=DEFAULT_FULL_DUPLEX_MAX_QUEUED_TX_MS,
        help=(
            "Maximum outbound sidecar queue depth allowed at the end of the "
            "full-duplex smoke."
        ),
    )
    parser.add_argument("--cli-timeout", type=float, default=10.0)
    parser.add_argument("--command-text", default=DEFAULT_COMMAND_TEXT)
    parser.add_argument("--command-stt-text", default=DEFAULT_COMMAND_STT_TEXT)
    parser.add_argument("--voice-contract-text", default=DEFAULT_VOICE_CONTRACT_TEXT)
    parser.add_argument("--voice-contract-timeout", type=float, default=240.0)
    parser.add_argument(
        "--whatsapp-bridge-media-timeout",
        type=float,
        default=DEFAULT_WHATSAPP_BRIDGE_MEDIA_TIMEOUT,
    )
    parser.add_argument(
        "--whatsapp-cloud-voice-timeout",
        type=float,
        default=DEFAULT_WHATSAPP_CLOUD_VOICE_TIMEOUT,
    )
    parser.add_argument(
        "--whatsapp-cloud-webhook-timeout",
        type=float,
        default=DEFAULT_WHATSAPP_CLOUD_WEBHOOK_TIMEOUT,
    )
    parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN", "node"))
    parser.add_argument("--stream-text", default=DEFAULT_STREAM_TEXT)
    parser.add_argument("--stream-command-template")
    parser.add_argument(
        "--live-hermes-root",
        type=Path,
        help=(
            "Optional checkout used by a running local gateway. When provided, "
            "fail unless that checkout contains the voice-native WhatsApp "
            "and calling sidecar integration surfaces."
        ),
    )
    parser.add_argument("--voice-repo", type=Path, default=default_voice_repo())
    parser.add_argument("--webrtc-python-bin", default=os.environ.get("VOICE_WEBRTC_PYTHON"))
    parser.add_argument(
        "--run-live-gateway",
        action="store_true",
        help=(
            "Also run verify_voice_live_gateway.py against the installed local "
            "gateway and sidecar."
        ),
    )
    parser.add_argument(
        "--live-gateway-python-bin",
        default="~/.hermes/hermes-agent/venv/bin/python",
        help="Python interpreter used by the running Hermes gateway checkout.",
    )
    parser.add_argument(
        "--live-gateway-hermes-home",
        type=Path,
        default=Path("~/.hermes"),
        help="Hermes home used by the running local gateway.",
    )
    parser.add_argument(
        "--calling-sidecar-url",
        help="Expected base URL for the running local WebRTC sidecar.",
    )
    parser.add_argument(
        "--live-gateway-sidecar-service",
        default="voice-webrtc-sidecar.service",
        help="systemd user service name for the running WebRTC sidecar.",
    )
    parser.add_argument(
        "--live-gateway-voice-daemon-service",
        default="voiced.service",
        help=(
            "systemd user service name for the running voice daemon checked by "
            "the live gateway verifier; pass an empty string to skip it."
        ),
    )
    parser.add_argument(
        "--skip-live-gateway-bridge-health",
        action="store_true",
        help="Pass --skip-bridge-health to the live gateway verifier.",
    )
    parser.add_argument(
        "--require-live-gateway-whatsapp-cloud-readiness",
        action="store_true",
        help=(
            "Pass --require-whatsapp-cloud-readiness to the live gateway "
            "verifier."
        ),
    )
    parser.add_argument(
        "--run-live-gateway-stt-smoke",
        action="store_true",
        help="Also run the live gateway command-STT smoke.",
    )
    parser.add_argument(
        "--run-live-gateway-calling-live-sidecar-smoke",
        action="store_true",
        help=(
            "Also run the live gateway Hermes connect-path sidecar smoke "
            "against imports from --live-hermes-root."
        ),
    )
    parser.add_argument("--live-gateway-timeout", type=float, default=360.0)
    parser.add_argument("--skip-voice-contract", action="store_true")
    parser.add_argument("--skip-whatsapp-bridge-media", action="store_true")
    parser.add_argument("--skip-whatsapp-cloud-webhook", action="store_true")
    parser.add_argument("--skip-whatsapp-cloud-voice", action="store_true")
    parser.add_argument("--skip-command-stt", action="store_true")
    parser.add_argument("--skip-calling-control-plane", action="store_true")
    parser.add_argument("--skip-calling-live-sidecar", action="store_true")
    parser.add_argument("--skip-full-duplex", action="store_true")
    parser.add_argument(
        "--full-duplex-inbound-text",
        default=DEFAULT_FULL_DUPLEX_INBOUND_TEXT,
    )
    parser.add_argument(
        "--full-duplex-outbound-text",
        default=DEFAULT_FULL_DUPLEX_OUTBOUND_TEXT,
    )
    parser.add_argument(
        "--stt-expect-word",
        action="append",
        default=None,
        help="word expected in the command-STT transcript; repeatable",
    )
    parser.add_argument(
        "--expect-word",
        action="append",
        default=None,
        help="word expected in the inbound full-duplex transcript; repeatable",
    )
    args = parser.parse_args()
    if args.full_duplex_max_queued_tx_ms < 0:
        parser.error("--full-duplex-max-queued-tx-ms must be non-negative")
    if args.expect_word is None:
        args.expect_word = ["hello", "world"]
    if args.stt_expect_word is None:
        args.stt_expect_word = ["hello", "world"]
    return args


def main() -> int:
    args = parse_args()
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    hermes_bin = resolve_executable(args.hermes_bin, label="Hermes CLI")
    node_bin = (
        None
        if args.skip_whatsapp_bridge_media
        else resolve_executable(args.node_bin, label="Node.js")
    )
    voice_contract_script = resolve_voice_contract_script(args)
    voice_repo = resolve_voice_repo_for_full_duplex(args)

    temporary_home = args.hermes_home is None
    hermes_home = (
        Path(tempfile.mkdtemp(prefix="hermes-voice-local-stack."))
        if temporary_home
        else args.hermes_home.expanduser().resolve()
    )
    hermes_home.mkdir(parents=True, exist_ok=True)

    env = child_env(hermes_home=hermes_home)

    checks: dict[str, Any] = {}
    try:
        if args.live_hermes_root is not None:
            checks["live_hermes_root"] = require_live_hermes_root(
                args.live_hermes_root
            )
        checks["hermes_cli"] = run_plain_step(
            "Hermes CLI",
            [hermes_bin, "--help"],
            timeout=args.cli_timeout,
            env=env,
        )
        if voice_contract_script is None:
            checks["voice_contract"] = {
                "success": True,
                "skipped": True,
                "reason": (
                    "--skip-voice-contract was provided"
                    if args.skip_voice_contract
                    else "--voice-repo or VOICE_REPO was not provided"
                ),
            }
        else:
            checks["voice_contract"] = run_plain_step(
                "voice WhatsApp contract verifier",
                voice_contract_command(
                    args,
                    voice_bin=voice_bin,
                    script=voice_contract_script,
                ),
                timeout=args.voice_contract_timeout,
                env=env,
                include_stdout_lines=True,
            )
        if node_bin is None:
            checks["whatsapp_bridge_media_payload"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-whatsapp-bridge-media was provided",
            }
        else:
            checks["whatsapp_bridge_media_payload"] = run_plain_step(
                "WhatsApp bridge media-payload verifier",
                whatsapp_bridge_media_payload_command(node_bin=node_bin),
                timeout=args.whatsapp_bridge_media_timeout,
                env=env,
                include_stdout_lines=True,
            )
        if args.skip_whatsapp_cloud_voice:
            checks["whatsapp_cloud_voice_note"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-whatsapp-cloud-voice was provided",
            }
        else:
            checks["whatsapp_cloud_voice_note"] = run_json_step(
                "WhatsApp Cloud voice-note verifier",
                whatsapp_cloud_voice_note_command(),
                timeout=args.whatsapp_cloud_voice_timeout,
                env=env,
            )
        if args.skip_whatsapp_cloud_webhook:
            checks["whatsapp_cloud_webhook"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-whatsapp-cloud-webhook was provided",
            }
        else:
            checks["whatsapp_cloud_webhook"] = run_json_step(
                "WhatsApp Cloud webhook verifier",
                whatsapp_cloud_webhook_command(),
                timeout=args.whatsapp_cloud_webhook_timeout,
                env=env,
            )
        checks["command_tts"] = run_json_step(
            "command TTS verifier",
            command_tts_command(args, voice_bin=voice_bin, hermes_home=hermes_home),
            timeout=args.tts_timeout + 30,
            env=env,
        )
        if args.skip_command_stt:
            checks["command_stt"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-command-stt was provided",
            }
        else:
            checks["command_stt"] = run_json_step(
                "command STT verifier",
                command_stt_command(
                    args,
                    voice_bin=voice_bin,
                    hermes_home=hermes_home,
                ),
                timeout=args.stt_timeout + args.tts_timeout + 30,
                env=env,
            )
        checks["stream_tts"] = run_json_step(
            "stream TTS verifier",
            stream_tts_command(args, voice_bin=voice_bin),
            timeout=args.stream_timeout + 30,
            env=env,
        )
        if args.skip_calling_control_plane:
            checks["calling_control_plane"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-calling-control-plane was provided",
            }
        else:
            checks["calling_control_plane"] = run_json_step(
                "WhatsApp Calling control-plane verifier",
                calling_control_plane_command(args),
                timeout=args.calling_control_plane_timeout + 5,
                env=env,
            )
        if args.skip_calling_live_sidecar:
            checks["calling_live_sidecar"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-calling-live-sidecar was provided",
            }
        elif voice_repo is None:
            checks["calling_live_sidecar"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-full-duplex was provided",
            }
        else:
            checks["calling_live_sidecar"] = run_json_step(
                "WhatsApp Calling live-sidecar verifier",
                calling_live_sidecar_command(args, voice_repo=voice_repo),
                timeout=args.calling_live_sidecar_timeout + 15,
                env=env,
            )
        if voice_repo is None:
            checks["full_duplex_sidecar"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-full-duplex was provided",
            }
        else:
            checks["full_duplex_sidecar"] = run_json_step(
                "full-duplex sidecar verifier",
                full_duplex_command(args, voice_bin=voice_bin, voice_repo=voice_repo),
                timeout=args.full_duplex_timeout + 15,
                env=env,
            )
        if args.run_live_gateway:
            checks["live_gateway"] = run_json_step(
                "live gateway verifier",
                live_gateway_command(
                    args,
                    voice_bin=voice_bin,
                    voice_repo=voice_repo,
                ),
                timeout=args.live_gateway_timeout,
                env=env,
            )

        print(
            json.dumps(
                {
                    "success": True,
                    "hermes_home": str(hermes_home),
                    "retained": bool(args.keep_home or not temporary_home),
                    "voice_bin": voice_bin,
                    "hermes_bin": hermes_bin,
                    "node_bin": node_bin,
                    "voice_repo": str(voice_repo) if voice_repo is not None else None,
                    "checks": checks,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        if temporary_home and not args.keep_home:
            shutil.rmtree(hermes_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
