"""
Hermes Companion Direct Connection HTTP(S) Server & Pairing Assistant
--------------------------------------------------------------------
Provides a zero-cloud direct HTTP sync transport for the Hermes Companion iOS app
to stream telemetry (location, health, history, places) and Hermes Post inbox letters
directly to a Hermes host machine (macOS, Linux, Windows), without requiring Apple iCloud.

Authentication:
    Bearer token (configured or auto-generated in ~/.hermes/hermes-companion/server_config.json)

Endpoints:
    GET  /api/v1/health
    PUT  /api/v1/telemetry/{filename}
    GET  /api/v1/places
    PUT  /api/v1/places
    GET  /api/v1/inbox/pending
    POST /api/v1/inbox/messages
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

try:
    from server.qr import QRCode, generate_qr_ascii
except ImportError:
    from qr import QRCode, generate_qr_ascii  # type: ignore[no-redef]

DEFAULT_PORT: int = 8787
DEFAULT_HOST: str = "0.0.0.0"
SERVER_VERSION: str = "1.0.0"
CONFIG_FILE: Path = Path.home() / ".hermes" / "hermes-companion" / "server_config.json"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_storage_directory() -> Path:
    """Return default storage directory for companion data."""
    env = os.environ.get("HERMES_COMPANION_STORAGE_DIR", "").strip()
    if env:
        return Path(env).expanduser()

    # If running on macOS with iCloud container active, use that so local readers share state
    icloud_docs = Path.home() / "Library/Mobile Documents/iCloud~com~hermes~HermesCompanion/Documents"
    if icloud_docs.is_dir():
        return icloud_docs

    # Default to cross-platform standard location
    path = Path.home() / ".hermes" / "hermes-companion" / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_or_create_token(token_arg: Optional[str] = None) -> str:
    """Retrieve configured token or generate and persist a new one on first run."""
    if token_arg and token_arg.strip():
        return token_arg.strip()

    env_token = os.environ.get("HERMES_COMPANION_TOKEN", "").strip()
    if env_token:
        return env_token

    if CONFIG_FILE.is_file():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            token = data.get("token", "").strip()
            if token:
                return token
        except Exception:
            pass

    # Generate cryptographically secure token
    token = f"hermes_sec_{secrets.token_urlsafe(32)}"
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        config_data = {
            "token": token,
            "port": DEFAULT_PORT,
            "created_at": utcnow_iso(),
            "version": SERVER_VERSION,
        }
        CONFIG_FILE.write_text(json.dumps(config_data, indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except Exception:
            pass
    except Exception as exc:
        print(f"Warning: Could not save token to {CONFIG_FILE}: {exc}", file=sys.stderr)

    return token


def get_local_ip() -> str:
    """Attempt to discover the host's primary outbound IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually transmit packets; connects to discover routing interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def atomic_write_json(file_path: Path, data: Any) -> None:
    """Atomically write JSON data to file_path using temp file and rename."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp_path = file_path.with_name(f".{file_path.name}.tmp_{os.getpid()}_{time.time_ns()}")
    try:
        tmp_path.write_text(serialized, encoding="utf-8")
        tmp_path.replace(file_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def touch_sync_marker(storage_dir: Path) -> None:
    """Touch .sync_marker.json in storage directory and threads directory."""
    marker_data = {
        "updated_at": utcnow_iso(),
        "source": "direct_http_server",
        "timestamp": time.time(),
    }
    for sub in [storage_dir, storage_dir / "threads"]:
        try:
            sub.mkdir(parents=True, exist_ok=True)
            atomic_write_json(sub / ".sync_marker.json", marker_data)
        except Exception:
            pass


class CompanionRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, request: Any, client_address: Any, server: Any) -> None:
        self.storage_dir: Path = getattr(server, "storage_dir", default_storage_directory())
        self.auth_token: str = getattr(server, "auth_token", "")
        super().__init__(request, client_address, server)

    def log_message(self, format: str, *args: Any) -> None:
        """Formatted timestamped console log."""
        sys.stderr.write(f"[{utcnow_iso()}] [DIRECT-SERVER] {self.address_string()} - {format % args}\n")
        sys.stderr.flush()

    def send_json(self, status_code: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> Optional[Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return None
        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except Exception:
            return None

    def verify_auth(self) -> bool:
        auth_header = self.headers.get("Authorization", "").strip()
        if not auth_header.startswith("Bearer "):
            self.send_json(HTTPStatus.UNAUTHORIZED, {
                "error": "Unauthorized",
                "message": "Missing or invalid 'Authorization: Bearer <token>' header"
            })
            return False

        token = auth_header[len("Bearer "):].strip()
        if not secrets.compare_digest(token, self.auth_token):
            self.send_json(HTTPStatus.UNAUTHORIZED, {
                "error": "Unauthorized",
                "message": "Invalid Bearer token"
            })
            return False
        return True

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/v1/health":
            if not self.verify_auth():
                return
            self.send_json(HTTPStatus.OK, {
                "status": "ok",
                "service": "hermes-companion-direct",
                "version": SERVER_VERSION,
                "timestamp": utcnow_iso(),
            })
            return

        if path == "/api/v1/places":
            if not self.verify_auth():
                return
            places_file = self.storage_dir / "places.json"
            places: List[Any] = []
            if places_file.is_file():
                try:
                    loaded = json.loads(places_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        places = loaded
                except Exception:
                    pass
            self.send_json(HTTPStatus.OK, places)
            return

        if path == "/api/v1/inbox/pending":
            if not self.verify_auth():
                return
            self.handle_get_pending_inbox()
            return

        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found", "path": path})

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/v1/telemetry/"):
            if not self.verify_auth():
                return
            filename = path[len("/api/v1/telemetry/"):]
            self.handle_put_telemetry(filename)
            return

        if path == "/api/v1/places":
            if not self.verify_auth():
                return
            self.handle_put_places()
            return

        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found", "path": path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/v1/inbox/messages":
            if not self.verify_auth():
                return
            self.handle_post_inbox_message()
            return

        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found", "path": path})

    # MARK: - Route Handlers

    def handle_put_telemetry(self, filename: str) -> None:
        allowed = {
            "latest_location.json",
            "location_history.json",
            "latest_health.json",
            "places.json",
        }
        if filename not in allowed:
            self.send_json(HTTPStatus.BAD_REQUEST, {
                "error": "Bad Request",
                "message": f"Unsupported telemetry filename: {filename}. Must be one of {sorted(allowed)}"
            })
            return

        payload = self.read_json_body()
        if payload is None:
            self.send_json(HTTPStatus.BAD_REQUEST, {
                "error": "Bad Request",
                "message": "Invalid or missing JSON payload"
            })
            return

        # Structure validation
        if filename == "latest_location.json":
            if not isinstance(payload, dict) or "latitude" not in payload or "longitude" not in payload:
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {
                    "error": "Unprocessable Entity",
                    "message": "latest_location.json must contain 'latitude' and 'longitude'"
                })
                return
        elif filename == "location_history.json":
            if not isinstance(payload, dict) or "records" not in payload or not isinstance(payload["records"], list):
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {
                    "error": "Unprocessable Entity",
                    "message": "location_history.json must contain a 'records' list"
                })
                return
        elif filename == "latest_health.json":
            if not isinstance(payload, dict) or "timestamp" not in payload:
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {
                    "error": "Unprocessable Entity",
                    "message": "latest_health.json must contain 'timestamp'"
                })
                return
        elif filename == "places.json":
            if not isinstance(payload, list):
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {
                    "error": "Unprocessable Entity",
                    "message": "places.json must be a JSON array of place objects"
                })
                return

        target_file = self.storage_dir / filename
        try:
            atomic_write_json(target_file, payload)
            touch_sync_marker(self.storage_dir)
            self.send_json(HTTPStatus.OK, {
                "status": "ok",
                "filename": filename,
                "written_at": utcnow_iso(),
            })
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": "Internal Server Error",
                "message": f"Could not write file: {exc}"
            })

    def handle_put_places(self) -> None:
        payload = self.read_json_body()
        if not isinstance(payload, list):
            self.send_json(HTTPStatus.BAD_REQUEST, {
                "error": "Bad Request",
                "message": "Payload must be a JSON array of places"
            })
            return

        target_file = self.storage_dir / "places.json"
        try:
            atomic_write_json(target_file, payload)
            touch_sync_marker(self.storage_dir)
            self.send_json(HTTPStatus.OK, {
                "status": "ok",
                "count": len(payload),
                "written_at": utcnow_iso(),
            })
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": "Internal Server Error",
                "message": str(exc)
            })

    def handle_get_pending_inbox(self) -> None:
        """Scan storage_dir / threads/ and return threads and messages with agent updates."""
        threads_dir = self.storage_dir / "threads"
        if not threads_dir.is_dir():
            self.send_json(HTTPStatus.OK, {"status": "ok", "count": 0, "threads": []})
            return

        results: List[Dict[str, Any]] = []
        for item in sorted(threads_dir.iterdir()):
            if not item.is_dir() or item.name.startswith("."):
                continue
            meta_file = item / "meta.json"
            if not meta_file.is_file():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Read chronological messages
            msg_dir = item / "messages"
            messages: List[Dict[str, Any]] = []
            if msg_dir.is_dir():
                for msg_file in sorted(msg_dir.iterdir()):
                    if msg_file.suffix == ".json" and not msg_file.name.startswith("."):
                        try:
                            msg_data = json.loads(msg_file.read_text(encoding="utf-8"))
                            messages.append(msg_data)
                        except Exception:
                            pass

            thread_record = {
                "id": meta.get("id", item.name),
                "subject": meta.get("subject", "Memo"),
                "target_profile": meta.get("target_profile", "default"),
                "status": meta.get("status", "pending_agent"),
                "created_at": meta.get("created_at", utcnow_iso()),
                "updated_at": meta.get("updated_at", utcnow_iso()),
                "message_count": len(messages),
                "last_snippet": meta.get("last_snippet", ""),
                "messages": messages,
            }
            results.append(thread_record)

        self.send_json(HTTPStatus.OK, {
            "status": "ok",
            "count": len(results),
            "threads": results,
            "checked_at": utcnow_iso(),
        })

    def handle_post_inbox_message(self) -> None:
        """Ingest new user letters/memos sent from iOS Post tab."""
        payload = self.read_json_body()
        if not isinstance(payload, dict):
            self.send_json(HTTPStatus.BAD_REQUEST, {
                "error": "Bad Request",
                "message": "Payload must be a JSON object"
            })
            return

        body_text = str(payload.get("body", "")).strip()
        if not body_text:
            self.send_json(HTTPStatus.BAD_REQUEST, {
                "error": "Bad Request",
                "message": "Letter body cannot be empty"
            })
            return

        now = utcnow_iso()
        timestamp_clean = now.replace(":", "").replace("-", "")

        thread_id = str(payload.get("thread_id", "")).strip()
        subject = str(payload.get("subject", "")).strip()
        target_profile = str(payload.get("target_profile", "default")).strip() or "default"
        message_id = str(payload.get("message_id", "")).strip() or f"msg_{secrets.token_hex(6)}"

        threads_dir = self.storage_dir / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)

        is_new_thread = False
        if not thread_id:
            # Generate new thread ID
            is_new_thread = True
            thread_id = f"th_{timestamp_clean}_{secrets.token_hex(3)}"
            if not subject:
                subject = "New Letter"

        thread_folder = threads_dir / thread_id
        messages_folder = thread_folder / "messages"
        thread_folder.mkdir(parents=True, exist_ok=True)
        messages_folder.mkdir(parents=True, exist_ok=True)

        meta_file = thread_folder / "meta.json"
        existing_meta: Dict[str, Any] = {}
        if meta_file.is_file():
            try:
                existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        if not subject:
            subject = str(existing_meta.get("subject", "Memo"))

        # Determine sequence index
        existing_messages = [f for f in messages_folder.iterdir() if f.suffix == ".json" and not f.name.startswith(".")]
        next_seq = len(existing_messages) + 1

        # Write message file: <seq>_user_<timestamp>_<msg_id>.json
        msg_filename = f"{next_seq:03d}_user_{timestamp_clean}_{message_id}.json"
        msg_file = messages_folder / msg_filename

        message_obj = {
            "id": message_id,
            "thread_id": thread_id,
            "sender": "user",
            "timestamp": now,
            "body": body_text,
        }
        atomic_write_json(msg_file, message_obj)

        # Update metadata
        meta_obj = {
            "id": thread_id,
            "subject": subject,
            "target_profile": target_profile,
            "status": "pending_agent",
            "created_at": existing_meta.get("created_at", now),
            "updated_at": now,
            "message_count": next_seq,
            "last_snippet": body_text[:120],
        }
        atomic_write_json(meta_file, meta_obj)

        touch_sync_marker(self.storage_dir)

        self.send_json(HTTPStatus.OK, {
            "status": "ok",
            "thread_id": thread_id,
            "message_id": message_id,
            "sequence": next_seq,
            "written_at": now,
        })


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: Optional[str] = None,
    storage_dir: Optional[Path] = None,
) -> None:
    """Start the companion HTTP service."""
    auth_token = get_or_create_token(token)
    data_dir = storage_dir or default_storage_directory()
    data_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((host, port), CompanionRequestHandler)
    server.auth_token = auth_token  # type: ignore[attr-defined]
    server.storage_dir = data_dir    # type: ignore[attr-defined]

    print("=" * 60)
    print("Hermes Companion Direct Connection HTTP Daemon")
    print("=" * 60)
    print(f"Version:      {SERVER_VERSION}")
    print(f"Listening on: http://{host}:{port}")
    print(f"Local IP:     http://{get_local_ip()}:{port}")
    print(f"Storage dir:  {data_dir}")
    print(f"Bearer Token: {auth_token[:15]}...{auth_token[-4:]}")
    print("=" * 60)
    print("Run 'python3 companion.py --pair' to view QR pairing code.")
    print("Press Ctrl+C to terminate.")
    print("-" * 60)
    sys.stdout.flush()

    def shutdown_signal(sig: int, frame: Any) -> None:
        print(f"\n[{utcnow_iso()}] Received signal {sig}; shutting down server...")
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGINT, shutdown_signal)
    signal.signal(signal.SIGTERM, shutdown_signal)

    try:
        server.serve_forever()
    except Exception as exc:
        print(f"Server error: {exc}", file=sys.stderr)
    finally:
        server.server_close()
        print("Hermes Companion HTTP daemon terminated cleanly.")


def run_pairing_assistant(
    port: int = DEFAULT_PORT,
    token: Optional[str] = None,
    host_override: Optional[str] = None,
) -> None:
    """Output pairing payload and ASCII QR code for terminal scanning."""
    auth_token = get_or_create_token(token)
    ip = host_override or get_local_ip()
    endpoint = f"http://{ip}:{port}"

    pairing_payload = {
        "endpoint": endpoint,
        "token": auth_token,
    }
    payload_json = json.dumps(pairing_payload, separators=(",", ":"))

    print()
    print("=" * 60)
    print("Hermes Companion — iOS Direct Connection Pairing Assistant")
    print("=" * 60)
    print()
    print("Scan this QR code from the Hermes Companion iOS app:")
    print("Settings -> Sync Provider -> Direct Connection -> Scan Pairing QR Code")
    print()

    # Try native Swift QR generator on macOS first if available for pixel-perfect quality
    rendered_qr: Optional[str] = None
    if sys.platform == "darwin" and shutil.which("swift"):
        try:
            swift_script = """
import CoreImage
import Foundation
let raw = CommandLine.arguments[1]
let filter = CIFilter(name: "CIQRCodeGenerator")!
filter.setValue(raw.data(using: .utf8)!, forKey: "inputMessage")
filter.setValue("M", forKey: "inputCorrectionLevel")
let output = filter.outputImage!
let extent = output.extent
let width = Int(extent.width)
let height = Int(extent.height)
var bitmap = [UInt8](repeating: 0, count: width * height * 4)
let context = CIContext()
context.render(output, toBitmap: &bitmap, rowBytes: width * 4, bounds: extent, format: .RGBA8, colorSpace: nil)
for y in stride(from: 0, to: height, by: 2) {
    var line = ""
    for x in 0..<width {
        let topDark = bitmap[(y * width + x) * 4] < 128
        let botDark = (y + 1 < height) ? (bitmap[((y + 1) * width + x) * 4] < 128) : false
        if topDark && botDark { line += " " }
        else if !topDark && !botDark { line += "█" }
        else if topDark && !botDark { line += "▄" }
        else { line += "▀" }
    }
    print(line)
}
"""
            res = subprocess.run(["swift", "-e", swift_script, payload_json], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                rendered_qr = res.stdout.strip()
        except Exception:
            pass

    if not rendered_qr:
        # Fall back to pure-Python generator
        try:
            rendered_qr = generate_qr_ascii(payload_json)
        except Exception as exc:
            print(f"Notice: Terminal QR generation skipped ({exc})")

    if rendered_qr:
        print(rendered_qr)
        print()

    print("-" * 60)
    print("Direct Connection Configuration Parameters:")
    print(f"  • Endpoint URL: {endpoint}")
    print(f"  • Bearer Token: {auth_token}")
    print(f"  • Raw JSON:     {payload_json}")
    print("-" * 60)
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes Companion Direct Connection HTTP Daemon")
    parser.add_argument("--serve", action="store_true", help="Start the HTTP server")
    parser.add_argument("--pair", action="store_true", help="Display pairing QR code and token")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host address to bind (default {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default {DEFAULT_PORT})")
    parser.add_argument("--token", help="Pre-shared Bearer token (generated if not provided)")
    parser.add_argument("--storage-dir", type=Path, help="Storage directory for telemetry and inbox files")

    args = parser.parse_args(argv)

    if args.pair or (len(sys.argv) > 1 and sys.argv[1] == "pair"):
        run_pairing_assistant(port=args.port, token=args.token)
        return 0

    # Default action or --serve
    run_server(host=args.host, port=args.port, token=args.token, storage_dir=args.storage_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
