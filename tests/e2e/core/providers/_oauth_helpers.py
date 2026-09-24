"""Hermetic subprocess harness for the OAuth E2E tests (fake HOME, tagged tree).

Every ``hermes`` child runs with an allowlisted environment (no inherited
credentials or ``HERMES_*``), ``HOME``/``HERMES_HOME`` under ``tmp_path``, and
a unique ``OAUTH_E2E_TAG`` so cleanup signals only this test's processes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
TAG_VAR = "OAUTH_E2E_TAG"

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")
_PASSTHROUGH_ENV = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ"})


@dataclass
class FakeHome:
    root: Path
    home: Path
    hermes_home: Path
    tag: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def auth_path(self) -> Path:
        return self.hermes_home / "auth.json"

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        import pwd  # POSIX-only; the suite is Linux-gated

        real_root = Path(pwd.getpwuid(os.getuid()).pw_dir, ".hermes").resolve()
        fixture = self.hermes_home.resolve()
        assert fixture != real_root and fixture.parent != real_root / "profiles", (
            f"fixture HERMES_HOME {fixture} is the real install's live home")
        env = {
            k: v for k, v in os.environ.items()
            if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_ENV_SUFFIXES)
        }
        env.update({
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home),
            "PYTHONPATH": str(REPO_ROOT), "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb",
            TAG_VAR: self.tag,
            # The child's ~/.hermes/state.db IS the tmp home's db; see parity/_helpers.py.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        })
        env.update(extra or {})
        return env

    def write_config(self, cfg: dict[str, Any]) -> None:
        base = {"updates": {"check": False}, "display": {"compact": True}}
        base.update(cfg)
        (self.hermes_home / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    def write_auth(self, store: dict[str, Any]) -> None:
        self.auth_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
        self.auth_path.chmod(0o600)

    def read_auth(self) -> dict[str, Any]:
        return json.loads(self.auth_path.read_text(encoding="utf-8"))


def make_home(root: Path) -> FakeHome:
    home = root / "home"
    (home / ".hermes").mkdir(parents=True, exist_ok=True)
    fh = FakeHome(root=root, home=home, hermes_home=home / ".hermes")
    fh.write_config({})
    return fh


def hermes_argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "hermes_cli.main", *args]


def run_hermes(fh: FakeHome, args: list[str], *, extra_env: dict[str, str] | None = None,
               timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(hermes_argv(*args), env=fh.env(extra_env), cwd=str(fh.root), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout)


def spawn_hermes(fh: FakeHome, args: list[str], *, extra_env: dict[str, str] | None = None,
                 log: Path) -> subprocess.Popen:
    out = open(log, "w", encoding="utf-8")  # noqa: SIM115 - closed when the child is reaped
    try:
        return subprocess.Popen(hermes_argv(*args), env=fh.env(extra_env), cwd=str(fh.root),
                                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, text=True)
    finally:
        out.close()


def kill_tagged(tag: str) -> None:
    """SIGKILL every live process carrying ``OAUTH_E2E_TAG=<tag>`` (this test's tree only)."""
    needle = f"{TAG_VAR}={tag}".encode()
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                env = fh.read()
        except OSError:
            continue
        if needle in env.split(b"\0"):
            try:
                os.kill(int(entry), signal.SIGKILL)
            except OSError:
                pass


def wait_until(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        time.sleep(interval)


# ---- minimal Anthropic Messages endpoint -----------------------------------
#
# Just enough of the Messages API for a text turn (JSON and SSE). The request
# record keeps the bearer so a test can prove which access token each call
# carried; ``decide`` maps a record to ("text", str) | ("error", status, type, msg)
# | ("hold", threading.Event, next_decision).


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class MessagesServer:
    MODEL = "claude-sonnet-4-5"

    def __init__(self, decide: Callable[[dict[str, Any]], tuple]) -> None:
        self.decide = decide
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None

    @property
    def base_url(self) -> str:
        assert self._httpd is not None
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/anthropic"

    def main_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.requests if r["body"].get("tools")]

    def start(self) -> "MessagesServer":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - model listing probes
                self._json(200, {"data": [{"id": outer.MODEL, "type": "model", "display_name": outer.MODEL}],
                                 "has_more": False})

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
                auth = self.headers.get("Authorization", "")
                record = {"path": self.path, "body": body, "bearer": auth.removeprefix("Bearer ").strip(),
                          "x_api_key": self.headers.get("x-api-key", ""), "at": time.monotonic()}
                with outer._lock:
                    outer.requests.append(record)
                decision = outer.decide(record)
                while decision[0] == "hold":
                    decision[1].wait(120)
                    decision = decision[2]
                if decision[0] == "error":
                    _, status, etype, msg = decision
                    return self._json(status, {"type": "error", "error": {"type": etype, "message": msg}})
                text = decision[1]
                if body.get("stream"):
                    return self._stream(text)
                self._json(200, {"id": "msg_fake", "type": "message", "role": "assistant", "model": outer.MODEL,
                                 "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                                 "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}})

            def _stream(self, text: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                msg = {"id": "msg_fake", "type": "message", "role": "assistant", "model": outer.MODEL,
                       "content": [], "stop_reason": None, "stop_sequence": None,
                       "usage": {"input_tokens": 10, "output_tokens": 1}}
                for chunk in (
                    _sse("message_start", {"type": "message_start", "message": msg}),
                    _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                 "content_block": {"type": "text", "text": ""}}),
                    _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": text}}),
                    _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    _sse("message_delta", {"type": "message_delta",
                                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                           "usage": {"output_tokens": 5}}),
                    _sse("message_stop", {"type": "message_stop"}),
                ):
                    self.wfile.write(chunk)
                self.wfile.flush()
                self.close_connection = True

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        return Handler
