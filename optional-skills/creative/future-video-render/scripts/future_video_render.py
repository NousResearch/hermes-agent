#!/usr/bin/env python3
"""Submit, poll, and cancel Future Video Studio agent renders.

This helper intentionally uses only the Python standard library so it can run in
minimal OpenClaw environments.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Iterable


DEFAULT_BASE_URL = "https://app.future.video"
DEFAULT_POLL_INTERVAL_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 1800.0
API_KEY_ENV = "FVS_AGENT_API_KEY"
BASE_URL_ENV = "FVS_AGENT_BASE_URL"
CUSTOM_HOST_ENV = "FVS_ALLOW_CUSTOM_AGENT_HOST"
ALLOWED_AGENT_HOSTS = {"app.future.video"}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except RenderToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Future Video Studio render helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Submit a render request")
    add_common_connection_args(submit)
    submit.add_argument("--request-file", required=True, help="Path to request JSON")
    submit.add_argument(
        "--file",
        action="append",
        default=[],
        help="Upload file path. Repeat for multiple assets.",
    )
    submit.add_argument(
        "--poll",
        action="store_true",
        help="Poll until the job reaches a terminal or review state.",
    )
    submit.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL_SECONDS})",
    )
    submit.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Overall poll timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    submit.add_argument(
        "--write-json",
        help="Optional path for writing the last JSON response.",
    )
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the submission target without sending the request.",
    )
    submit.set_defaults(func=handle_submit)

    status = subparsers.add_parser("status", help="Check render status")
    add_common_connection_args(status)
    add_lookup_args(status)
    status.add_argument("--write-json", help="Optional path for writing the JSON response.")
    status.set_defaults(func=handle_status)

    cancel = subparsers.add_parser("cancel", help="Cancel a render")
    add_common_connection_args(cancel)
    add_lookup_args(cancel)
    cancel.add_argument("--write-json", help="Optional path for writing the JSON response.")
    cancel.set_defaults(func=handle_cancel)

    return parser


def add_common_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key",
        help=f"Agent API key. Defaults to ${API_KEY_ENV}.",
    )
    parser.add_argument(
        "--base-url",
        help=f"App origin or /api/agent base. Defaults to ${BASE_URL_ENV} or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--allow-custom-host",
        action="store_true",
        help=(
            "Allow a non-production FVS API host. Use only for trusted local or staging "
            f"backends; can also be set with ${CUSTOM_HOST_ENV}=1."
        ),
    )


def add_lookup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-id", help="Render project id, e.g. proj_api_123")
    parser.add_argument("--status-url", help="Full status URL from a previous response")
    parser.add_argument("--cancel-url", help="Full cancel URL from a previous response")


def handle_submit(args: argparse.Namespace) -> int:
    api_key = resolve_api_key(args.api_key)
    agent_base = normalize_agent_base(
        args.base_url or os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL,
        allow_custom_host=resolve_allow_custom_host(args),
    )
    request_path = Path(args.request_file)
    if not request_path.is_file():
        raise RenderToolError(f"request file not found: {request_path}")

    request_json = read_request_json(request_path)
    payload = json.loads(request_json)
    file_paths = [Path(p) for p in args.file]
    for path in file_paths:
        if not path.is_file():
            raise RenderToolError(f"upload file not found: {path}")

    validate_asset_filenames(payload, file_paths)

    if args.dry_run:
        preview = {
            "endpoint": f"{agent_base}/renders",
            "request_file": str(request_path),
            "files": [str(path) for path in file_paths],
            "asset_count": len(payload.get("assets", []) or []),
        }
        emit_json(preview, args.write_json)
        return 0

    body, content_type = encode_multipart(
        fields=[("request_json", request_json)],
        file_fields=[("files", path) for path in file_paths],
    )

    response = request_json_api(
        url=f"{agent_base}/renders",
        method="POST",
        api_key=api_key,
        agent_base=agent_base,
        body=body,
        content_type=content_type,
    )

    if args.poll:
        response = poll_job(
            initial_response=response,
            api_key=api_key,
            agent_base=agent_base,
            poll_interval=max(1.0, args.poll_interval),
            timeout=max(1.0, args.timeout),
        )

    emit_json(response, args.write_json)
    return 0


def handle_status(args: argparse.Namespace) -> int:
    api_key = resolve_api_key(args.api_key)
    agent_base = normalize_agent_base(
        args.base_url or os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL,
        allow_custom_host=resolve_allow_custom_host(args),
    )
    url = resolve_status_url(args, agent_base)
    response = request_json_api(url=url, method="GET", api_key=api_key, agent_base=agent_base)
    emit_json(response, args.write_json)
    return 0


def handle_cancel(args: argparse.Namespace) -> int:
    api_key = resolve_api_key(args.api_key)
    agent_base = normalize_agent_base(
        args.base_url or os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL,
        allow_custom_host=resolve_allow_custom_host(args),
    )
    url = resolve_cancel_url(args, agent_base)
    response = request_json_api(url=url, method="POST", api_key=api_key, agent_base=agent_base)
    emit_json(response, args.write_json)
    return 0


def resolve_api_key(explicit_value: str | None) -> str:
    value = explicit_value or os.getenv(API_KEY_ENV)
    if not value:
        raise RenderToolError(
            f"missing API key; pass --api-key or set {API_KEY_ENV}"
        )
    return value.strip()


def resolve_allow_custom_host(args: argparse.Namespace) -> bool:
    env_value = str(os.getenv(CUSTOM_HOST_ENV) or "").strip().lower()
    return bool(args.allow_custom_host or env_value in {"1", "true", "yes", "on"})


def normalize_agent_base(base_url: str, *, allow_custom_host: bool = False) -> str:
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        raise RenderToolError("base URL is empty")
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RenderToolError("base URL must be an absolute http(s) URL")
    validate_agent_origin(parsed, allow_custom_host=allow_custom_host)
    if cleaned.endswith("/api/agent"):
        return cleaned
    return f"{cleaned}/api/agent"


def validate_agent_origin(parsed: urllib.parse.ParseResult, *, allow_custom_host: bool) -> None:
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host in ALLOWED_AGENT_HOSTS:
        return
    if allow_custom_host:
        return
    allowed = ", ".join(sorted(ALLOWED_AGENT_HOSTS))
    raise RenderToolError(
        f"refusing to send an API key to {parsed.geturl()}; expected HTTPS host in {allowed}. "
        f"For trusted local or staging FVS backends, pass --allow-custom-host or set {CUSTOM_HOST_ENV}=1."
    )


def detect_upload_mime_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    return None


def guess_upload_content_type(path: Path, data: bytes) -> str:
    return detect_upload_mime_type(data) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def resolve_status_url(args: argparse.Namespace, agent_base: str) -> str:
    if args.status_url:
        return validate_render_url(args.status_url.strip(), agent_base=agent_base, kind="status")
    if args.project_id:
        return f"{agent_base}/renders/{args.project_id.strip()}"
    raise RenderToolError("provide --project-id or --status-url")


def resolve_cancel_url(args: argparse.Namespace, agent_base: str) -> str:
    if args.cancel_url:
        return validate_render_url(args.cancel_url.strip(), agent_base=agent_base, kind="cancel")
    if args.project_id:
        return f"{agent_base}/renders/{args.project_id.strip()}/cancel"
    raise RenderToolError("provide --project-id or --cancel-url")


def validate_render_url(url: str, *, agent_base: str, kind: str) -> str:
    validated_url = validate_api_request_url(url, agent_base=agent_base)
    parsed = urllib.parse.urlparse(validated_url)
    agent_path = urllib.parse.urlparse(agent_base).path.rstrip("/")
    expected_prefix = f"{agent_path}/renders/"
    if not parsed.path.startswith(expected_prefix):
        raise RenderToolError(f"{kind} URL must be under {expected_prefix}")
    if kind == "cancel" and not parsed.path.endswith("/cancel"):
        raise RenderToolError("cancel URL must end with /cancel")
    return validated_url


def validate_asset_filenames(payload: dict, file_paths: list[Path]) -> None:
    assets = payload.get("assets")
    if not assets:
        return
    upload_names = {path.name for path in file_paths}
    missing = []
    for asset in assets:
        filename = str((asset or {}).get("filename") or "").strip()
        if filename and filename not in upload_names:
            missing.append(filename)
    if missing:
        raise RenderToolError(
            "assets[].filename entries must match uploaded file basenames; missing uploads for "
            + ", ".join(sorted(missing))
        )


def read_request_json(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def encode_multipart(
    *,
    fields: Iterable[tuple[str, str]],
    file_fields: Iterable[tuple[str, Path]],
) -> tuple[bytes, str]:
    boundary = f"----fvs-{uuid.uuid4().hex}"
    lines: list[bytes] = []

    for name, value in fields:
        lines.extend(
            [
                f"--{boundary}".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"),
                b"",
                value.encode("utf-8"),
            ]
        )

    for field_name, path in file_fields:
        data = path.read_bytes()
        content_type = guess_upload_content_type(path, data)
        lines.extend(
            [
                f"--{boundary}".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"'
                ).encode("utf-8"),
                f"Content-Type: {content_type}".encode("utf-8"),
                b"",
                data,
            ]
        )

    lines.append(f"--{boundary}--".encode("utf-8"))
    lines.append(b"")
    body = b"\r\n".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def request_json_api(
    *,
    url: str,
    method: str,
    api_key: str,
    agent_base: str,
    body: bytes | None = None,
    content_type: str | None = None,
) -> dict:
    url = validate_api_request_url(url, agent_base=agent_base)
    headers = {
        "X-FVS-Agent-Key": api_key,
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = extract_error_detail(raw)
        raise RenderToolError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RenderToolError(f"network error while calling {url}: {exc.reason}") from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RenderToolError(f"non-JSON response from {url}") from exc


def validate_api_request_url(url: str, *, agent_base: str) -> str:
    parsed = urllib.parse.urlparse(url)
    base = urllib.parse.urlparse(agent_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RenderToolError("API URL must be an absolute http(s) URL")
    if (parsed.scheme, parsed.netloc.lower()) != (base.scheme, base.netloc.lower()):
        raise RenderToolError(
            f"refusing to send an API key to {parsed.scheme}://{parsed.netloc}; "
            f"expected {base.scheme}://{base.netloc}"
        )
    base_path = base.path.rstrip("/")
    if not parsed.path.startswith(f"{base_path}/"):
        raise RenderToolError(f"API URL path must stay under {base_path}/")
    return urllib.parse.urlunparse(parsed)


def extract_error_detail(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return "empty error response"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    return text


def poll_job(
    *,
    initial_response: dict,
    api_key: str,
    agent_base: str,
    poll_interval: float,
    timeout: float,
) -> dict:
    response = initial_response
    status_url = response.get("status_url")
    if not status_url and response.get("project_id"):
        status_url = f"{agent_base}/renders/{response['project_id']}"
    if not status_url:
        return response

    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_terminal_response(response):
            return response
        time.sleep(poll_interval)
        response = request_json_api(
            url=str(status_url),
            method="GET",
            api_key=api_key,
            agent_base=agent_base,
        )
    raise RenderToolError("poll timeout exceeded before the render reached a terminal state")


def is_terminal_response(response: dict) -> bool:
    status = str(response.get("status") or "").strip().lower()
    current_stage = str(response.get("current_stage") or "").strip().lower()
    is_running = bool(response.get("is_running"))
    if status in {"completed", "failed"}:
        return True
    if current_stage == "halted_for_review":
        return True
    if not is_running and status not in {"queued", "running"}:
        return True
    return False


def emit_json(payload: dict, output_path: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output_path:
        Path(output_path).write_text(text + "\n", encoding="utf-8")


class RenderToolError(RuntimeError):
    pass


if __name__ == "__main__":
    sys.exit(main())
