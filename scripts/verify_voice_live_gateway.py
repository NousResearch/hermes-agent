#!/usr/bin/env python3
"""Verify a running local Hermes gateway is using the voice-native stack.

This is the live-service counterpart to ``verify_voice_local_stack.py``. It is
intentionally read-mostly: it inspects the systemd user service, verifies the
running process environment points at the expected checkout, confirms imports
resolve from that checkout, and checks the local WhatsApp bridge health.

Pass ``--run-tts-smoke`` to also generate one live-config TTS reply and verify
that Hermes returns a WhatsApp-ready Ogg/Opus voice-note file.

Pass ``--run-stt-smoke`` with ``--voice-bin`` to also generate one WAV fixture
and verify that the live Hermes config transcribes it through the configured
voice command STT provider.

Pass ``--voice-bin`` with ``--calling-sidecar-url`` to compare the running
sidecar's machine-readable contract with the installed ``voice stream-contract``.
Pass ``--sidecar-service`` to also verify the systemd user unit that runs the
local WebRTC sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
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
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from verify_voice_local_stack import audit_live_hermes_root


DEFAULT_SERVICE = "hermes-gateway.service"
DEFAULT_BRIDGE_URL = "http://127.0.0.1:3000"
DEFAULT_TTS_TEXT = "Hermes live voice gateway smoke."
DEFAULT_STT_TEXT = "hello world"
DEFAULT_STT_EXPECT_WORDS = ("hello", "world")
CALLING_TTS_STREAM_ENV = "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND"
WHATSAPP_CLOUD_REQUIRED_ENV = (
    "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
    "WHATSAPP_CLOUD_ACCESS_TOKEN",
    "WHATSAPP_CLOUD_APP_SECRET",
    "WHATSAPP_CLOUD_VERIFY_TOKEN",
)
DEFAULT_WHATSAPP_CLOUD_WEBHOOK_PORT = 8090
DEFAULT_WHATSAPP_CLOUD_WEBHOOK_PATH = "/whatsapp/webhook"
DEFAULT_WHATSAPP_CLOUD_VERIFY_CHALLENGE = "hermes-local-cloud-verify-smoke"
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
REQUIRED_PAYLOAD_FIELDS = {
    "call_state": (
        "ready_for_accept",
        "readiness",
        "queued_tx_bytes",
        "queued_tx_ms",
        "max_tx_queue_bytes",
        "max_tx_queue_ms",
        "queued_rx_bytes",
        "queued_rx_ms",
        "max_rx_queue_bytes",
        "max_rx_queue_ms",
    ),
    "send_audio_response": (
        "accepted_bytes",
        "accepted_ms",
        "queued_tx_bytes",
        "queued_tx_ms",
        "max_tx_queue_bytes",
        "max_tx_queue_ms",
    ),
    "clear_audio_response": (
        "dropped_tx_bytes",
        "dropped_tx_ms",
        "queued_tx_bytes",
        "queued_tx_ms",
        "max_tx_queue_bytes",
        "max_tx_queue_ms",
    ),
    "receive_audio_response": (
        "returned_bytes",
        "returned_ms",
        "queued_rx_bytes",
        "queued_rx_ms",
        "max_rx_queue_bytes",
        "max_rx_queue_ms",
        "pcm_s16le_base64",
    ),
}
IMPORT_MODULES = (
    "hermes_cli.main",
    "tools.tts_tool",
    "tools.transcription_tools",
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
            "After",
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


def parse_dotenv_text(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = raw_value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        env[key] = value
    return env


def load_hermes_env_file(hermes_home: Path) -> dict[str, str]:
    env_path = hermes_home / ".env"
    if not env_path.is_file():
        raise SystemExit(f"Hermes env file not found: {env_path}")
    return parse_dotenv_text(env_path.read_text(encoding="utf-8", errors="replace"))


def configured_value(
    key: str,
    *,
    file_env: dict[str, str],
    process_env: dict[str, str],
) -> tuple[str, str]:
    process_value = str(process_env.get(key) or "").strip()
    if process_value:
        return process_value, "process"
    file_value = str(file_env.get(key) or "").strip()
    if file_value:
        return file_value, "env_file"
    return "", "missing"


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {
        "changeme",
        "change-me",
        "placeholder",
        "example",
        "your-token",
        "your_token",
        "your_verify_token",
        "verify_token",
        "token",
    }:
        return True
    return any(token in lowered for token in ("<", ">", "paste_", "replace_"))


def whatsapp_cloud_field_status(
    key: str,
    value: str,
) -> tuple[dict[str, Any], str | None]:
    status: dict[str, Any] = {"present": bool(value)}
    if not value:
        return status, f"{key} is not configured"
    if _looks_like_placeholder(value):
        return status, f"{key} still looks like a placeholder"

    if key == "WHATSAPP_CLOUD_PHONE_NUMBER_ID":
        if not value.isdigit():
            return status, f"{key} must be numeric"
        if 10 <= len(value) <= 12:
            return (
                status,
                f"{key} looks like a phone number; use Meta's Phone Number ID",
            )
        if len(value) < 13 or len(value) > 20:
            return status, f"{key} should be 13-20 digits"
        return {**status, "source_shape": "meta_phone_number_id"}, None

    if key == "WHATSAPP_CLOUD_ACCESS_TOKEN":
        if not value.startswith("EAA"):
            return status, f"{key} should start with EAA"
        if len(value) < 100:
            return status, f"{key} looks too short for a Meta access token"
        return {**status, "source_shape": "meta_access_token"}, None

    if key == "WHATSAPP_CLOUD_APP_SECRET":
        if not re.fullmatch(r"[0-9a-fA-F]{32}", value):
            return status, f"{key} should be exactly 32 hex characters"
        return {**status, "source_shape": "meta_app_secret"}, None

    if key == "WHATSAPP_CLOUD_VERIFY_TOKEN":
        if len(value) < 16:
            return status, f"{key} should be at least 16 characters"
        return {**status, "source_shape": "webhook_verify_token"}, None

    return status, f"unsupported WhatsApp Cloud readiness field: {key}"


def validate_whatsapp_cloud_field(key: str, value: str) -> dict[str, Any]:
    status, failure = whatsapp_cloud_field_status(key, value)
    if failure is not None:
        raise SystemExit(failure)
    return status


def truthy_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def whatsapp_cloud_authorization_status(
    *,
    file_env: dict[str, str],
    process_env: dict[str, str],
) -> tuple[dict[str, Any], str | None]:
    allowed_users, allowed_source = configured_value(
        "WHATSAPP_CLOUD_ALLOWED_USERS",
        file_env=file_env,
        process_env=process_env,
    )
    allow_all, allow_all_source = configured_value(
        "WHATSAPP_CLOUD_ALLOW_ALL_USERS",
        file_env=file_env,
        process_env=process_env,
    )
    allow_all_enabled = truthy_env(allow_all)
    allowed_count = len([part for part in allowed_users.split(",") if part.strip()])
    status = {
        "allowed_users_configured": allowed_count > 0,
        "allowed_users_count": allowed_count,
        "allowed_users_source": allowed_source if allowed_count > 0 else "missing",
        "allow_all_users": allow_all_enabled,
        "allow_all_users_source": allow_all_source if allow_all else "missing",
    }
    if allowed_count == 0 and not allow_all_enabled:
        return (
            status,
            "WhatsApp Cloud recipient authorization is not configured; set "
            "WHATSAPP_CLOUD_ALLOWED_USERS or WHATSAPP_CLOUD_ALLOW_ALL_USERS",
        )
    return status, None


def validate_whatsapp_cloud_authorization(
    *,
    file_env: dict[str, str],
    process_env: dict[str, str],
) -> dict[str, Any]:
    status, failure = whatsapp_cloud_authorization_status(
        file_env=file_env,
        process_env=process_env,
    )
    if failure is not None:
        raise SystemExit(failure)
    return status


def validate_whatsapp_cloud_readiness(
    *,
    hermes_home: Path,
    process_env: dict[str, str],
) -> dict[str, Any]:
    file_env = load_hermes_env_file(hermes_home)
    fields: dict[str, Any] = {}
    failures: list[str] = []
    for key in WHATSAPP_CLOUD_REQUIRED_ENV:
        value, source = configured_value(
            key,
            file_env=file_env,
            process_env=process_env,
        )
        field, failure = whatsapp_cloud_field_status(key, value)
        field["source"] = source
        fields[key] = field
        if failure is not None:
            failures.append(failure)

    authorization, auth_failure = whatsapp_cloud_authorization_status(
        file_env=file_env,
        process_env=process_env,
    )
    if auth_failure is not None:
        failures.append(auth_failure)

    if failures:
        raise SystemExit(
            "WhatsApp Cloud readiness failed:\n- " + "\n- ".join(failures)
        )

    return {
        "env_file": str((hermes_home / ".env").resolve()),
        "required_fields": fields,
        "authorization": authorization,
        "_private": {
            "file_env": file_env,
            "process_env": process_env,
            "phone_number_id": configured_value(
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
                file_env=file_env,
                process_env=process_env,
            )[0],
        },
    }


def remove_private_readiness_data(readiness: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in readiness.items() if key != "_private"}


def whatsapp_cloud_health_url_from_env(
    *,
    file_env: dict[str, str],
    process_env: dict[str, str],
) -> str:
    raw_host = configured_value(
        "WHATSAPP_CLOUD_WEBHOOK_HOST",
        file_env=file_env,
        process_env=process_env,
    )[0]
    host = raw_host or "127.0.0.1"
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"

    raw_port = configured_value(
        "WHATSAPP_CLOUD_WEBHOOK_PORT",
        file_env=file_env,
        process_env=process_env,
    )[0]
    if raw_port:
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise SystemExit(
                f"WHATSAPP_CLOUD_WEBHOOK_PORT is not an integer: {raw_port!r}"
            ) from exc
        if port <= 0 or port > 65535:
            raise SystemExit(
                f"WHATSAPP_CLOUD_WEBHOOK_PORT is out of range: {raw_port!r}"
            )
    else:
        port = DEFAULT_WHATSAPP_CLOUD_WEBHOOK_PORT

    return f"http://{host}:{port}/health"


def whatsapp_cloud_webhook_url_from_env(
    *,
    file_env: dict[str, str],
    process_env: dict[str, str],
) -> str:
    health_url = whatsapp_cloud_health_url_from_env(
        file_env=file_env,
        process_env=process_env,
    )
    parsed = urlparse(health_url)
    path = configured_value(
        "WHATSAPP_CLOUD_WEBHOOK_PATH",
        file_env=file_env,
        process_env=process_env,
    )[0] or DEFAULT_WHATSAPP_CLOUD_WEBHOOK_PATH
    if not path.startswith("/"):
        path = "/" + path
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def validate_whatsapp_cloud_health(
    health: dict[str, Any],
    *,
    expected_phone_number_id: str,
    expect_calling_sidecar: bool,
) -> dict[str, Any]:
    if health.get("status") != "ok":
        raise SystemExit(
            f"WhatsApp Cloud health status is not ok: {health.get('status')!r}"
        )
    if health.get("platform") != "whatsapp_cloud":
        raise SystemExit(
            "WhatsApp Cloud health endpoint reported wrong platform: "
            f"{health.get('platform')!r}"
        )
    if str(health.get("phone_number_id") or "") != expected_phone_number_id:
        raise SystemExit(
            "WhatsApp Cloud health phone_number_id does not match configured "
            "WHATSAPP_CLOUD_PHONE_NUMBER_ID"
        )
    if health.get("verify_token_configured") is not True:
        raise SystemExit("WhatsApp Cloud health reports verify token is not configured")
    if health.get("app_secret_configured") is not True:
        raise SystemExit("WhatsApp Cloud health reports app secret is not configured")
    if expect_calling_sidecar:
        if health.get("calling_sidecar_configured") is not True:
            raise SystemExit(
                "WhatsApp Cloud health reports calling sidecar is not configured"
            )
        if health.get("calling_sidecar_tts_stream_configured") is not True:
            raise SystemExit(
                "WhatsApp Cloud health reports calling sidecar TTS stream is "
                "not configured"
            )

    return {
        "status": str(health.get("status")),
        "platform": str(health.get("platform")),
        "webhook_path": str(health.get("webhook_path") or ""),
        "verify_token_configured": health.get("verify_token_configured") is True,
        "app_secret_configured": health.get("app_secret_configured") is True,
        "calling_sidecar_configured": health.get("calling_sidecar_configured") is True,
        "calling_sidecar_contract_loaded": (
            health.get("calling_sidecar_contract_loaded") is True
        ),
        "calling_sidecar_tts_stream_configured": (
            health.get("calling_sidecar_tts_stream_configured") is True
        ),
    }


def check_whatsapp_cloud_health(
    *,
    readiness: dict[str, Any],
    health_url: str | None,
    timeout: float,
    expect_calling_sidecar: bool,
) -> dict[str, Any]:
    private = readiness.get("_private")
    if not isinstance(private, dict):
        raise SystemExit("WhatsApp Cloud readiness private data missing")
    file_env = private.get("file_env")
    process_env = private.get("process_env")
    expected_phone_number_id = str(private.get("phone_number_id") or "")
    if not isinstance(file_env, dict) or not isinstance(process_env, dict):
        raise SystemExit("WhatsApp Cloud readiness env data missing")
    target = health_url or whatsapp_cloud_health_url_from_env(
        file_env={str(k): str(v) for k, v in file_env.items()},
        process_env={str(k): str(v) for k, v in process_env.items()},
    )
    health = get_json_url(target, timeout=timeout)
    return {
        "url": target,
        **validate_whatsapp_cloud_health(
            health,
            expected_phone_number_id=expected_phone_number_id,
            expect_calling_sidecar=expect_calling_sidecar,
        ),
    }


def get_text_url(target: str, *, timeout: float, label: str) -> tuple[int, str]:
    try:
        with urlopen(target, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
    except URLError as exc:
        raise SystemExit(f"request failed for {label}: {exc}") from exc
    return status, body


def post_json_url(
    target: str,
    *,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
    label: str,
) -> tuple[int, str]:
    request = Request(target, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
    except URLError as exc:
        raise SystemExit(f"request failed for {label}: {exc}") from exc
    return status, response_body


def check_whatsapp_cloud_verify_handshake(
    *,
    readiness: dict[str, Any],
    webhook_url: str | None,
    challenge: str,
    timeout: float,
) -> dict[str, Any]:
    private = readiness.get("_private")
    if not isinstance(private, dict):
        raise SystemExit("WhatsApp Cloud readiness private data missing")
    file_env = private.get("file_env")
    process_env = private.get("process_env")
    if not isinstance(file_env, dict) or not isinstance(process_env, dict):
        raise SystemExit("WhatsApp Cloud readiness env data missing")
    verify_token = configured_value(
        "WHATSAPP_CLOUD_VERIFY_TOKEN",
        file_env={str(k): str(v) for k, v in file_env.items()},
        process_env={str(k): str(v) for k, v in process_env.items()},
    )[0]
    if not verify_token:
        raise SystemExit("WHATSAPP_CLOUD_VERIFY_TOKEN is not configured")

    target_base = webhook_url or whatsapp_cloud_webhook_url_from_env(
        file_env={str(k): str(v) for k, v in file_env.items()},
        process_env={str(k): str(v) for k, v in process_env.items()},
    )
    query = urlencode(
        {
            "hub.mode": "subscribe",
            "hub.verify_token": verify_token,
            "hub.challenge": challenge,
        }
    )
    separator = "&" if "?" in target_base else "?"
    status, body = get_text_url(
        f"{target_base}{separator}{query}",
        timeout=timeout,
        label=target_base,
    )
    if status != 200:
        raise SystemExit(f"WhatsApp Cloud verify handshake returned HTTP {status}")
    if body != challenge:
        raise SystemExit("WhatsApp Cloud verify handshake did not echo the challenge")
    return {
        "url": target_base,
        "status": status,
        "challenge_echoed": True,
    }


def whatsapp_cloud_status_payload(*, phone_number_id: str) -> dict[str, Any]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "local-cloud-readiness",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15555550100",
                                "phone_number_id": phone_number_id,
                            },
                            "statuses": [
                                {
                                    "id": "wamid.local-cloud-readiness",
                                    "status": "delivered",
                                    "timestamp": "1760000000",
                                    "recipient_id": "15555550101",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def check_whatsapp_cloud_signed_post(
    *,
    readiness: dict[str, Any],
    webhook_url: str | None,
    timeout: float,
) -> dict[str, Any]:
    private = readiness.get("_private")
    if not isinstance(private, dict):
        raise SystemExit("WhatsApp Cloud readiness private data missing")
    file_env = private.get("file_env")
    process_env = private.get("process_env")
    phone_number_id = str(private.get("phone_number_id") or "")
    if not isinstance(file_env, dict) or not isinstance(process_env, dict):
        raise SystemExit("WhatsApp Cloud readiness env data missing")
    app_secret = configured_value(
        "WHATSAPP_CLOUD_APP_SECRET",
        file_env={str(k): str(v) for k, v in file_env.items()},
        process_env={str(k): str(v) for k, v in process_env.items()},
    )[0]
    if not app_secret:
        raise SystemExit("WHATSAPP_CLOUD_APP_SECRET is not configured")
    target_base = webhook_url or whatsapp_cloud_webhook_url_from_env(
        file_env={str(k): str(v) for k, v in file_env.items()},
        process_env={str(k): str(v) for k, v in process_env.items()},
    )
    body = json.dumps(
        whatsapp_cloud_status_payload(phone_number_id=phone_number_id),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    signature = hmac.new(
        app_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    status, _body = post_json_url(
        target_base,
        body=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": f"sha256={signature}",
        },
        timeout=timeout,
        label=target_base,
    )
    if status != 200:
        raise SystemExit(f"WhatsApp Cloud signed webhook POST returned HTTP {status}")
    return {
        "url": target_base,
        "status": status,
        "payload": "status_delivery_receipt",
        "signature_accepted": True,
    }


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
    voice_daemon_service: str | None,
) -> dict[str, Any]:
    pid = validate_service_state(state, label=service)
    env = parse_systemd_environment(state.get("Environment", ""))
    result: dict[str, Any] = {
        "service": service,
        "pid": pid,
    }
    after_units = set(str(state.get("After") or "").split())
    if "voice-daemon.service" in after_units:
        raise SystemExit(
            f"{service} still orders after deprecated voice-daemon.service; "
            "rerun install_voice_local_stack.py so it uses voiced.service"
        )
    if voice_daemon_service and voice_daemon_service not in after_units:
        raise SystemExit(
            f"{service} does not order after {voice_daemon_service}"
        )
    result["after"] = sorted(after_units)

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


def validate_voice_daemon_service_state(
    state: dict[str, str],
    *,
    service: str,
    voice_bin: str | None,
) -> dict[str, Any]:
    pid = validate_service_state(state, label=service)
    result: dict[str, Any] = {
        "service": service,
        "pid": pid,
    }

    exec_start = str(state.get("ExecStart") or "")
    exec_argv = parse_exec_start_argv(exec_start)
    if "daemon" not in exec_argv or "start" not in exec_argv:
        raise SystemExit(f"{service} ExecStart is not a voice daemon start command")
    if voice_bin is not None and (
        not exec_argv
        or normalized_path_text(exec_argv[0]) != normalized_path_text(voice_bin)
    ):
        raise SystemExit(
            f"{service} ExecStart does not reference expected voice binary "
            f"{voice_bin!r}: {exec_start!r}"
        )
    result["exec_start"] = exec_start
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
    validate_payload_contracts(payloads)

    return {
        "voice_surfaces": list(REQUIRED_VOICE_SURFACES),
        "endpoints": list(REQUIRED_ENDPOINTS),
        "payloads": list(REQUIRED_PAYLOADS),
    }


def validate_payload_contracts(payloads: dict[str, Any]) -> None:
    for payload_name, required_fields in REQUIRED_PAYLOAD_FIELDS.items():
        payload = payloads.get(payload_name)
        if not isinstance(payload, dict):
            raise SystemExit(
                f"calling sidecar contract payloads.{payload_name} must be an object"
            )
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            raise SystemExit(
                "calling sidecar contract payloads."
                f"{payload_name} missing fields: "
                + ", ".join(missing_fields)
            )


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


def run_calling_sidecar_offer_smoke(
    *,
    python_bin: str,
    sidecar_url: str,
    call_id: str,
    timeout: float,
) -> dict[str, Any]:
    """POST a real local SDP offer to a running sidecar and require readiness."""
    code = r"""
import asyncio
import json
import sys

from aiohttp import ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription


async def wait_for_ice_complete(pc, timeout):
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange():
        if pc.iceGatheringState == "complete":
            done.set()

    await asyncio.wait_for(done.wait(), timeout=timeout)


async def main():
    sidecar_url = sys.argv[1].rstrip("/")
    call_id = sys.argv[2]
    timeout = float(sys.argv[3])
    pc = RTCPeerConnection()
    close_body = None
    try:
        pc.addTransceiver("audio", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await wait_for_ice_complete(pc, timeout)
        async with ClientSession() as session:
            response = await session.post(
                f"{sidecar_url}/offer",
                json={
                    "call_id": call_id,
                    "type": pc.localDescription.type,
                    "sdp": pc.localDescription.sdp,
                },
                timeout=timeout,
            )
            body = await response.json()
            if response.status == 200:
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=body["sdp"], type=body["type"])
                )
                close = await session.post(
                    f"{sidecar_url}/calls/{call_id}/close",
                    timeout=timeout,
                )
                close_body = await close.json()
            print(json.dumps({
                "status": response.status,
                "body": body,
                "close": close_body,
            }, sort_keys=True))
    finally:
        await pc.close()


asyncio.run(main())
"""
    completed = subprocess.run(
        [python_bin, "-c", code, sidecar_url, call_id, f"{timeout:g}"],
        capture_output=True,
        text=True,
        timeout=timeout + 10,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"calling sidecar offer smoke failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"calling sidecar offer smoke returned invalid JSON: {completed.stdout!r}"
        ) from exc
    if not isinstance(result, dict):
        raise SystemExit("calling sidecar offer smoke JSON root must be an object")
    if result.get("status") != 200:
        raise SystemExit(f"calling sidecar offer smoke returned non-200: {result}")

    body = result.get("body")
    if not isinstance(body, dict):
        raise SystemExit(f"calling sidecar offer smoke body is not an object: {result}")
    state = body.get("state")
    if not isinstance(state, dict):
        raise SystemExit(f"calling sidecar offer response missing state: {body}")
    readiness = state.get("readiness")
    if not isinstance(readiness, dict):
        raise SystemExit(f"calling sidecar readiness checks missing: {state}")
    failed = sorted(str(key) for key, value in readiness.items() if value is not True)
    if failed:
        raise SystemExit(
            "calling sidecar readiness checks failed: " + ", ".join(failed)
        )
    if state.get("ready_for_accept") is not True:
        raise SystemExit(f"calling sidecar was not ready for accept: {state}")

    close = result.get("close")
    if not isinstance(close, dict) or close.get("closed") is not True:
        raise SystemExit(f"calling sidecar offer smoke did not close cleanly: {result}")
    return {
        "success": True,
        "call_id": call_id,
        "ready_for_accept": state["ready_for_accept"],
        "readiness": readiness,
        "close": close,
    }


def run_calling_live_sidecar_smoke(
    *,
    python_bin: str,
    live_root: Path,
    hermes_home: Path,
    voice_repo: Path,
    sidecar_url: str | None,
    timeout: float,
) -> dict[str, Any]:
    """Run the Hermes connect-path smoke against imports from live_root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(live_root)
    env["HERMES_HOME"] = str(hermes_home)
    command = [
        python_bin,
        str(
            Path(__file__).resolve().parent
            / "verify_voice_whatsapp_calling_live_sidecar.py"
        ),
        "--voice-repo",
        str(voice_repo),
        "--timeout",
        f"{timeout:g}",
    ]
    if sidecar_url:
        command.extend(["--sidecar-url", sidecar_url])
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout + 15,
        check=False,
        env=env,
        cwd=str(live_root),
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"calling live-sidecar smoke failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"calling live-sidecar smoke returned invalid JSON: {completed.stdout!r}"
        ) from exc
    if not isinstance(result, dict):
        raise SystemExit("calling live-sidecar smoke JSON root must be an object")
    if result.get("success") is not True:
        raise SystemExit(f"calling live-sidecar smoke reported failure: {result}")
    actions = result.get("graph_actions")
    if not isinstance(actions, list) or actions[:2] != ["pre_accept", "accept"]:
        raise SystemExit(
            f"calling live-sidecar smoke did not pre_accept then accept: {result}"
        )
    if result.get("sidecar_ready_for_accept") is not True:
        raise SystemExit(f"calling live-sidecar smoke sidecar was not ready: {result}")
    readiness = result.get("sidecar_readiness")
    if not isinstance(readiness, dict):
        raise SystemExit(
            f"calling live-sidecar smoke did not report readiness checks: {result}"
        )
    failed_readiness = sorted(
        str(key) for key, value in readiness.items() if value is not True
    )
    if failed_readiness:
        raise SystemExit(
            "calling live-sidecar smoke readiness checks failed: "
            + ", ".join(failed_readiness)
        )
    statuses = result.get("webhook_statuses")
    if not isinstance(statuses, dict) or statuses.get("connect") != 200 or statuses.get(
        "terminate"
    ) != 200:
        raise SystemExit(
            f"calling live-sidecar smoke did not use signed webhooks cleanly: {result}"
        )
    try:
        outbound_bytes = int(result.get("outbound_webrtc_bytes") or 0)
        inbound_bytes = int(result.get("inbound_drain_bytes") or 0)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"calling live-sidecar smoke returned invalid byte counts: {result}"
        ) from exc
    if outbound_bytes <= 0 or inbound_bytes <= 0:
        raise SystemExit(f"calling live-sidecar smoke did not move audio: {result}")
    clear_audio = result.get("clear_audio")
    if not isinstance(clear_audio, dict):
        raise SystemExit(
            f"calling live-sidecar smoke did not report clear audio telemetry: {result}"
        )
    if clear_audio.get("skipped"):
        raise SystemExit(
            f"calling live-sidecar smoke skipped clear audio endpoint: {result}"
        )
    try:
        queued_after_clear = int(clear_audio.get("queued_tx_bytes"))
        dropped_tx_bytes = int(clear_audio.get("dropped_tx_bytes"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"calling live-sidecar smoke returned invalid clear audio telemetry: {result}"
        ) from exc
    if queued_after_clear != 0 or dropped_tx_bytes < 0:
        raise SystemExit(
            f"calling live-sidecar smoke did not clear outbound audio: {result}"
        )
    close = result.get("sidecar_close")
    if not isinstance(close, dict) or close.get("closed") is not True:
        raise SystemExit(
            f"calling live-sidecar smoke did not close sidecar cleanly: {result}"
        )
    return result


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


def run_stt_smoke(
    *,
    python_bin: str,
    live_root: Path,
    hermes_home: Path,
    voice_bin: str,
    provider: str,
    text: str,
    expect_words: list[str],
    stt_timeout: float,
    generate_timeout: float,
    process_timeout: float,
) -> dict[str, Any]:
    script = live_root / "scripts" / "verify_voice_command_stt.py"
    if not script.is_file():
        raise SystemExit(f"STT smoke verifier not found in live root: {script}")

    command = [
        python_bin,
        str(script),
        "--use-existing-config",
        "--hermes-home",
        str(hermes_home),
        "--voice-bin",
        voice_bin,
        "--provider",
        provider,
        "--timeout",
        f"{stt_timeout:g}",
        "--generate-timeout",
        f"{generate_timeout:g}",
        "--text",
        text,
    ]
    for word in expect_words:
        command.extend(["--expect-word", word])

    completed = run_command(command, timeout=process_timeout)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"STT smoke failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"STT smoke returned invalid JSON: {completed.stdout!r}") from exc
    if result.get("success") is not True:
        raise SystemExit(f"STT smoke reported failure: {result}")
    if result.get("provider") != provider:
        raise SystemExit(
            f"STT smoke provider mismatch: expected {provider!r}, "
            f"got {result.get('provider')!r}"
        )
    return result


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
    parser.add_argument(
        "--voice-daemon-service",
        default="voiced.service",
        help=(
            "systemd user service name for the voice daemon required by "
            "live-call voice stream commands; pass an empty string to skip."
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
    parser.add_argument(
        "--require-whatsapp-cloud-readiness",
        action="store_true",
        help=(
            "Require non-placeholder WhatsApp Cloud credentials and recipient "
            "authorization in HERMES_HOME/.env or the running process "
            "environment. Secret values are never printed."
        ),
    )
    parser.add_argument(
        "--whatsapp-cloud-health-url",
        help=(
            "Override the local WhatsApp Cloud health URL used with "
            "--require-whatsapp-cloud-readiness. Defaults to "
            "http://127.0.0.1:8090/health, or the configured webhook port."
        ),
    )
    parser.add_argument(
        "--skip-whatsapp-cloud-health",
        action="store_true",
        help=(
            "With --require-whatsapp-cloud-readiness, only validate local "
            "configuration shape and skip the running Cloud webhook /health "
            "probe."
        ),
    )
    parser.add_argument(
        "--whatsapp-cloud-webhook-url",
        help=(
            "Override the local WhatsApp Cloud webhook URL used with "
            "--require-whatsapp-cloud-readiness. Defaults to "
            "http://127.0.0.1:8090/whatsapp/webhook, or the configured "
            "webhook port/path."
        ),
    )
    parser.add_argument(
        "--skip-whatsapp-cloud-verify",
        action="store_true",
        help=(
            "With --require-whatsapp-cloud-readiness, skip the local Meta "
            "GET subscription verify-token handshake probe."
        ),
    )
    parser.add_argument(
        "--skip-whatsapp-cloud-signed-post",
        action="store_true",
        help=(
            "With --require-whatsapp-cloud-readiness, skip the local signed "
            "webhook POST probe."
        ),
    )
    parser.add_argument(
        "--whatsapp-cloud-verify-challenge",
        default=DEFAULT_WHATSAPP_CLOUD_VERIFY_CHALLENGE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--run-tts-smoke", action="store_true")
    parser.add_argument("--tts-platform", default="whatsapp")
    parser.add_argument("--tts-text", default=DEFAULT_TTS_TEXT)
    parser.add_argument("--run-stt-smoke", action="store_true")
    parser.add_argument("--stt-provider", default="voice")
    parser.add_argument("--stt-text", default=DEFAULT_STT_TEXT)
    parser.add_argument("--stt-timeout", type=float, default=300.0)
    parser.add_argument("--stt-generate-timeout", type=float, default=180.0)
    parser.add_argument(
        "--run-sidecar-offer-smoke",
        action="store_true",
        help=(
            "POST a real aiortc SDP offer to --calling-sidecar-url and require "
            "state.ready_for_accept before closing the smoke call."
        ),
    )
    parser.add_argument(
        "--run-calling-live-sidecar-smoke",
        action="store_true",
        help=(
            "Run the Hermes WhatsApp Calling connect-path smoke against imports "
            "from --live-hermes-root, using a real in-process voice sidecar and "
            "a fake Graph /calls endpoint."
        ),
    )
    parser.add_argument(
        "--webrtc-python-bin",
        help=(
            "Python with aiortc/aiohttp installed for sidecar/WebRTC smokes."
        ),
    )
    parser.add_argument(
        "--sidecar-offer-call-id",
        default="live-gateway-readiness-smoke",
        help="call_id to use for --run-sidecar-offer-smoke",
    )
    parser.add_argument(
        "--stt-expect-word",
        action="append",
        default=None,
        help="word expected in the live STT smoke transcript; repeatable",
    )
    args = parser.parse_args()
    if args.stt_expect_word is None:
        args.stt_expect_word = list(DEFAULT_STT_EXPECT_WORDS)
    return args


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
    webrtc_python_bin = (
        resolve_executable(args.webrtc_python_bin, label="WebRTC Python")
        if args.webrtc_python_bin
        else None
    )
    ffprobe_bin = (
        resolve_executable(args.ffprobe_bin, label="ffprobe")
        if args.run_tts_smoke
        else ""
    )
    if args.run_stt_smoke and not voice_bin:
        raise SystemExit("--run-stt-smoke requires --voice-bin")
    if args.run_sidecar_offer_smoke:
        if not args.calling_sidecar_url:
            raise SystemExit("--run-sidecar-offer-smoke requires --calling-sidecar-url")
        if not webrtc_python_bin:
            raise SystemExit("--run-sidecar-offer-smoke requires --webrtc-python-bin")
    if args.run_calling_live_sidecar_smoke:
        if not voice_repo:
            raise SystemExit(
                "--run-calling-live-sidecar-smoke requires --voice-repo"
            )
        if not webrtc_python_bin:
            raise SystemExit(
                "--run-calling-live-sidecar-smoke requires --webrtc-python-bin"
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
    cloud_readiness: dict[str, Any] | None = None
    if args.require_whatsapp_cloud_readiness:
        cloud_readiness = validate_whatsapp_cloud_readiness(
            hermes_home=hermes_home,
            process_env=process_env,
        )
        checks["whatsapp_cloud_readiness"] = remove_private_readiness_data(
            cloud_readiness
        )
        if args.skip_whatsapp_cloud_health:
            checks["whatsapp_cloud_health"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-whatsapp-cloud-health was provided",
            }
        else:
            checks["whatsapp_cloud_health"] = check_whatsapp_cloud_health(
                readiness=cloud_readiness,
                health_url=args.whatsapp_cloud_health_url,
                timeout=args.timeout,
                expect_calling_sidecar=bool(args.calling_sidecar_url),
            )
        if args.skip_whatsapp_cloud_verify:
            checks["whatsapp_cloud_verify_handshake"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-whatsapp-cloud-verify was provided",
            }
        else:
            checks["whatsapp_cloud_verify_handshake"] = (
                check_whatsapp_cloud_verify_handshake(
                    readiness=cloud_readiness,
                    webhook_url=args.whatsapp_cloud_webhook_url,
                    challenge=args.whatsapp_cloud_verify_challenge,
                    timeout=args.timeout,
                )
            )
        if args.skip_whatsapp_cloud_signed_post:
            checks["whatsapp_cloud_signed_post"] = {
                "success": True,
                "skipped": True,
                "reason": "--skip-whatsapp-cloud-signed-post was provided",
            }
        else:
            checks["whatsapp_cloud_signed_post"] = check_whatsapp_cloud_signed_post(
                readiness=cloud_readiness,
                webhook_url=args.whatsapp_cloud_webhook_url,
                timeout=args.timeout,
            )

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
        if args.run_sidecar_offer_smoke:
            checks["calling_sidecar_offer_smoke"] = run_calling_sidecar_offer_smoke(
                python_bin=str(webrtc_python_bin),
                sidecar_url=args.calling_sidecar_url,
                call_id=args.sidecar_offer_call_id,
                timeout=args.timeout,
            )
    elif voice_bin and not args.sidecar_service:
        raise SystemExit("--voice-bin requires --calling-sidecar-url or --sidecar-service")

    if args.run_calling_live_sidecar_smoke:
        checks["calling_live_sidecar_smoke"] = run_calling_live_sidecar_smoke(
            python_bin=str(webrtc_python_bin),
            live_root=live_root,
            hermes_home=hermes_home,
            voice_repo=voice_repo,
            sidecar_url=args.calling_sidecar_url,
            timeout=args.timeout,
        )

    if args.sidecar_service:
        if args.voice_daemon_service:
            checks["voice_daemon_service"] = validate_voice_daemon_service_state(
                get_service_state(args.voice_daemon_service, timeout=args.timeout),
                service=args.voice_daemon_service,
                voice_bin=voice_bin,
            )
        checks["sidecar_service"] = validate_sidecar_service_state(
            get_service_state(args.sidecar_service, timeout=args.timeout),
            service=args.sidecar_service,
            voice_bin=voice_bin,
            voice_repo=voice_repo,
            sidecar_url=args.calling_sidecar_url,
            voice_daemon_service=args.voice_daemon_service,
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
    if args.run_stt_smoke:
        checks["stt_smoke"] = run_stt_smoke(
            python_bin=python_bin,
            live_root=live_root,
            hermes_home=hermes_home,
            voice_bin=str(voice_bin),
            provider=args.stt_provider,
            text=args.stt_text,
            expect_words=args.stt_expect_word,
            stt_timeout=args.stt_timeout,
            generate_timeout=args.stt_generate_timeout,
            process_timeout=args.stt_timeout + args.stt_generate_timeout + 30,
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
