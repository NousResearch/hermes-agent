#!/usr/bin/env python3
"""Visual companion publication utility."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import html
import hmac
import json
import os
import re
import secrets
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html.parser import HTMLParser
from pathlib import Path


_CHOICE_ATTR_RE = re.compile(
    r"\bdata-choice\s*=\s*(?:\"[^\"]+\"|'[^']+')",
    re.IGNORECASE,
)
_CSS_URL_RE = re.compile(
    r"url\s*\(\s*(?P<quote>['\"]?)(?P<value>.*?)(?P=quote)\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_UNSAFE_FRAGMENT_PATTERNS = (
    (
        re.compile(r"<(?:script|iframe|object|embed|base|link|meta)\b", re.IGNORECASE),
        "active HTML elements",
    ),
    (re.compile(r"\son[a-z0-9_-]+\s*=", re.IGNORECASE), "inline event handlers"),
    (re.compile(r"javascript\s*:", re.IGNORECASE), "javascript URLs"),
    (
        re.compile(
            r"\b(?:src|href|action)\s*=\s*['\"]\s*(?:https?:)?//",
            re.IGNORECASE,
        ),
        "remote asset URLs",
    ),
    (
        re.compile(r"url\s*\(\s*['\"]?\s*(?:https?:)?//", re.IGNORECASE),
        "remote CSS URLs",
    ),
    (re.compile(r"@import\b", re.IGNORECASE), "CSS imports"),
)

_NAVIGATION_URL_ATTRIBUTES = {"action", "formaction", "href", "xlink:href"}
_RESOURCE_URL_ATTRIBUTES = {"poster", "src"}

_COOKIE_NAME_PREFIX = "hermes_visual_companion_"
_MAX_FRAGMENT_BYTES = 1024 * 1024
_MAX_REQUEST_BYTES = 32 * 1024
_MAX_BOOTSTRAP_REQUEST_BYTES = 1024
_MAX_FEEDBACK_CHARS = 2000
_MAX_CHOICE_ID_CHARS = 128
_MAX_CHOICE_LABEL_CHARS = 256
_MAX_ROUND_ID_CHARS = 128
_PREVIEW_LAUNCHER_FILENAME = "open-preview.html"
_POLL_SECONDS = 0.05
_SESSION_LOCK_TIMEOUT_SECONDS = 3.0
_SERVER_LEASE_TIMEOUT_SECONDS = 0.5


class VisualCompanionError(ValueError):
    """A user-facing companion configuration or content error."""


class StalePageError(VisualCompanionError):
    """A choice came from a superseded visual-companion round."""


class RoundAlreadySelectedError(VisualCompanionError):
    """A round already emitted a different choice event."""


class _ChoiceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.choices: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        choice_metadata_attributes: set[str] = set()
        for name, raw_value in attrs:
            attribute = name.lower()
            if attribute.startswith("on"):
                raise VisualCompanionError("fragment contains disallowed inline event handlers.")
            if attribute in {"data-choice", "data-label"}:
                if attribute in choice_metadata_attributes:
                    raise VisualCompanionError(
                        f"duplicate choice metadata attribute: {attribute}"
                    )
                choice_metadata_attributes.add(attribute)
            value = (raw_value or "").strip()
            if not value:
                continue
            if attribute in _NAVIGATION_URL_ATTRIBUTES and not value.startswith("#"):
                raise VisualCompanionError(
                    f"fragment contains disallowed URL in {attribute} attribute."
                )
            if attribute in _RESOURCE_URL_ATTRIBUTES and not value.lower().startswith("data:"):
                raise VisualCompanionError(
                    f"fragment contains disallowed URL in {attribute} attribute."
                )
            if attribute in {"srcdoc", "srcset"}:
                raise VisualCompanionError(f"fragment contains disallowed {attribute} attribute.")

        values = dict(attrs)
        choice_id = (values.get("data-choice") or "").strip()
        if not choice_id:
            return
        if choice_id in self.choices:
            raise VisualCompanionError(f"duplicate data-choice value: {choice_id}")
        self.choices[choice_id] = (values.get("data-label") or choice_id).strip() or choice_id


def _choices_from_fragment(fragment: str) -> dict[str, str]:
    parser = _ChoiceParser()
    parser.feed(fragment)
    parser.close()
    return parser.choices


def _validated_round_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisualCompanionError("round_id must be a non-empty string.")
    normalized = value.strip()
    if len(normalized) > _MAX_ROUND_ID_CHARS:
        raise VisualCompanionError(
            f"round_id must be {_MAX_ROUND_ID_CHARS} characters or fewer."
        )
    return normalized


def _atomic_write_text(path: Path, contents: str) -> None:
    temp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            tmp_file.write(contents)

        temp_path.replace(path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _render_preview_launcher(bootstrap_url: str, bootstrap_token: str) -> str:
    """Build a private local-file handoff without exposing the capability to the agent."""
    parsed = urllib.parse.urlsplit(bootstrap_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.query
        or not bootstrap_token
    ):
        raise VisualCompanionError("preview launcher target must use loopback HTTP.")
    action = html.escape(bootstrap_url, quote=True)
    token = html.escape(bootstrap_token, quote=True)
    origin = html.escape(f"{parsed.scheme}://{parsed.netloc}", quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="referrer" content="no-referrer">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; form-action {origin}">
  <title>Opening visual companion</title>
</head>
<body>
  <p>Opening visual companion…</p>
  <form id="companion-bootstrap" method="post" action="{action}">
    <input type="hidden" name="key" value="{token}">
  </form>
  <script>document.getElementById('companion-bootstrap').submit();</script>
</body>
</html>
"""


def _try_advisory_lock(descriptor: int) -> bool:
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _release_advisory_lock(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _exclusive_lease(lock_path: Path, timeout: float, timeout_message: str):
    """Hold a cross-process lease that the OS releases if its process exits."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
            os.fsync(descriptor)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        while not acquired:
            acquired = _try_advisory_lock(descriptor)
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise VisualCompanionError(timeout_message)
            time.sleep(_POLL_SECONDS)

        owner = json.dumps({"pid": os.getpid()}, separators=(",", ":")).encode()
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, owner)
        os.ftruncate(descriptor, len(owner))
        os.fsync(descriptor)
        yield
    finally:
        if acquired:
            _release_advisory_lock(descriptor)
        os.close(descriptor)


@contextmanager
def _session_lock(session_dir: Path):
    with _exclusive_lease(
        session_dir / ".session.lock",
        _SESSION_LOCK_TIMEOUT_SECONDS,
        "timed out waiting for the companion session lock.",
    ):
        yield


@contextmanager
def _server_lease(session_dir: Path):
    with _exclusive_lease(
        session_dir / ".server.lock",
        _SERVER_LEASE_TIMEOUT_SECONDS,
        "the companion session already has a running server.",
    ):
        yield


def _next_page_version(round_json_path: Path) -> int:
    if not round_json_path.exists():
        return 1

    round_data = json.loads(round_json_path.read_text(encoding="utf-8"))
    if not isinstance(round_data, dict):
        raise ValueError("round.json must contain an object.")

    page_version = round_data.get("page_version")
    if type(page_version) is not int:
        raise ValueError("round.json must include integer page_version.")

    return page_version + 1


def validate_fragment(fragment: str) -> None:
    """Reject content that could execute code, fetch remotely, or cannot be selected."""
    if len(fragment.encode("utf-8")) > _MAX_FRAGMENT_BYTES:
        raise VisualCompanionError("fragment exceeds the 1 MiB limit.")
    if not _CHOICE_ATTR_RE.search(fragment):
        raise VisualCompanionError("fragment must include at least one non-empty data-choice attribute.")

    normalized_fragment = html.unescape(fragment)
    for pattern, description in _UNSAFE_FRAGMENT_PATTERNS:
        if pattern.search(normalized_fragment):
            raise VisualCompanionError(f"fragment contains disallowed {description}.")
    for match in _CSS_URL_RE.finditer(normalized_fragment):
        target = match.group("value").strip()
        if target and not target.lower().startswith("data:") and not target.startswith("#"):
            raise VisualCompanionError("fragment contains disallowed remote CSS URLs.")

    choices = _choices_from_fragment(fragment)
    if not 2 <= len(choices) <= 4:
        raise VisualCompanionError("fragment must include two to four selectable data-choice values.")
    for choice_id, label in choices.items():
        if len(choice_id) > _MAX_CHOICE_ID_CHARS:
            raise VisualCompanionError(
                f"data-choice values must be {_MAX_CHOICE_ID_CHARS} characters or fewer."
            )
        if len(label) > _MAX_CHOICE_LABEL_CHARS:
            raise VisualCompanionError(
                f"choice labels must be {_MAX_CHOICE_LABEL_CHARS} characters or fewer."
            )


def _read_json_object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VisualCompanionError(f"{path.name} must contain a JSON object.")
    return data


def _active_round(session_dir: Path) -> tuple[dict, str]:
    """Read a consistent metadata+fragment pair published via a version pointer."""
    metadata = _read_json_object(session_dir / "round.json")
    page_version = metadata.get("page_version")
    if type(page_version) is not int:
        raise VisualCompanionError("round.json must include integer page_version.")
    metadata["round_id"] = _validated_round_id(metadata.get("round_id"))
    versioned_fragment = session_dir / ".pages" / f"page-{page_version}.html"
    if not versioned_fragment.is_file():
        raise VisualCompanionError("the active round's published page is missing.")
    fragment = versioned_fragment.read_text(encoding="utf-8")
    validate_fragment(fragment)
    return metadata, fragment


def _read_events(session_dir: Path) -> list[dict]:
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        return []

    events = []
    for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VisualCompanionError(
                f"events.jsonl contains invalid JSON on line {line_number}."
            ) from exc
        if (
            not isinstance(event, dict)
            or type(event.get("cursor")) is not int
            or event["cursor"] <= 0
        ):
            raise VisualCompanionError(f"events.jsonl line {line_number} is not a choice event.")
        events.append(event)
    return events


def wait_for_choice(session_dir: Path, after: int, timeout: float) -> dict | None:
    if after < 0:
        raise VisualCompanionError("--after must be zero or greater.")
    if timeout < 0:
        raise VisualCompanionError("--timeout must be zero or greater.")

    deadline = time.monotonic() + timeout
    while True:
        with _session_lock(session_dir):
            events = _read_events(session_dir)
        for event in events:
            if event["cursor"] > after:
                return event
        if time.monotonic() >= deadline:
            return None
        time.sleep(min(_POLL_SECONDS, max(0.0, deadline - time.monotonic())))


_BOARD_HELPER_SCRIPT = r"""
document.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const card = target.closest('[data-choice]');
  if (!card || card.getAttribute('aria-disabled') === 'true') return;
  event.preventDefault();
  const choiceId = card.getAttribute('data-choice');
  if (!choiceId) return;
  parent.postMessage({type: 'hermes-visual-choice', choice_id: choiceId}, '*');
});

window.addEventListener('message', (event) => {
  if (event.source !== parent) return;
  const message = event.data;
  if (!message || message.type !== 'hermes-visual-control') return;
  document.querySelectorAll('[data-choice]').forEach((node) => {
    node.toggleAttribute('data-selected', node.getAttribute('data-choice') === message.selected);
    node.setAttribute(
      'aria-pressed',
      node.getAttribute('data-choice') === message.selected ? 'true' : 'false',
    );
    if (message.disabled) node.setAttribute('aria-disabled', 'true');
    else node.removeAttribute('aria-disabled');
    if ('disabled' in node) node.disabled = Boolean(message.disabled);
  });
});
"""


_HELPER_SCRIPT = r"""
const currentVersion = __PAGE_VERSION__;
const boardFrame = document.getElementById('__BOARD_FRAME_ID__');
const statusNode = document.getElementById('companion-status');
const feedbackNode = document.getElementById('companion-feedback');
let selectionRecorded = false;
let selectionPending = false;

const setChoicesDisabled = (disabled, selected = null) => {
  boardFrame.contentWindow?.postMessage(
    {type: 'hermes-visual-control', disabled, selected},
    '*',
  );
};

const recordChoice = async (choiceId) => {
  if (selectionRecorded || selectionPending) return;
  if (typeof choiceId !== 'string' || !choiceId) return;
  const feedback = feedbackNode.value;
  selectionPending = true;
  setChoicesDisabled(true);
  feedbackNode.disabled = true;
  statusNode.textContent = 'Recording selection…';
  try {
    const response = await fetch('__choice', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Visual-Companion': 'choice'},
      body: JSON.stringify({
        choice_id: choiceId,
        page_version: currentVersion,
        feedback,
      }),
    });
    if (!response.ok) throw new Error(`selection failed (${response.status})`);
    const selected = await response.json();
    selectionPending = false;
    selectionRecorded = true;
    setChoicesDisabled(true, choiceId);
    statusNode.textContent = `Selected: ${selected.label}`;
  } catch (error) {
    selectionPending = false;
    setChoicesDisabled(false);
    feedbackNode.disabled = false;
    statusNode.textContent = error instanceof Error ? error.message : 'selection failed';
  }
};

window.addEventListener('message', (event) => {
  if (event.source !== boardFrame.contentWindow) return;
  const message = event.data;
  if (!message || message.type !== 'hermes-visual-choice') return;
  void recordChoice(message.choice_id);
});

boardFrame.srcdoc = __BOARD_DOCUMENT_JSON__;

setInterval(async () => {
  try {
    const response = await fetch('__version', {cache: 'no-store'});
    if (!response.ok) return;
    const state = await response.json();
    if (state.page_version > currentVersion) window.location.reload();
  } catch (_) {
    // The agent may be replacing or stopping the local companion.
  }
}, 800);
"""


def _render_page(session_dir: Path) -> tuple[bytes, str]:
    metadata, fragment = _active_round(session_dir)
    page_version = metadata.get("page_version")
    if type(page_version) is not int:
        raise VisualCompanionError("round.json must include integer page_version.")

    nonce = secrets.token_urlsafe(18)
    board_frame_id = f"companion-board-{secrets.token_hex(12)}"
    board_document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; "
        f"style-src 'unsafe-inline'; script-src 'nonce-{nonce}'; img-src data:; "
        "font-src data:; form-action 'none'; base-uri 'none'\">"
        "<style>[data-choice]{cursor:pointer}"
        "[data-choice]:focus-visible,[data-choice][data-selected]{outline:3px solid #d7a84b;"
        "outline-offset:4px}</style>"
        f"<script nonce=\"{nonce}\">{_BOARD_HELPER_SCRIPT}</script>"
        f"</head><body>{fragment}</body></html>"
    )
    board_document_json = (
        json.dumps(board_document, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    helper = (
        _HELPER_SCRIPT.replace("__PAGE_VERSION__", str(page_version))
        .replace("__BOARD_FRAME_ID__", board_frame_id)
        .replace("__BOARD_DOCUMENT_JSON__", board_document_json)
    )
    page = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Hermes Visual Companion</title>"
        "<style>"
        "html{height:100%;color-scheme:dark}body{height:100%;margin:0;overflow:hidden;display:grid;"
        "grid-template-rows:minmax(0,1fr) auto;background:#0b0c0f;color:#f4f2ec;"
        "font-family:Inter,ui-sans-serif,system-ui,sans-serif}"
        ".companion-board{display:block;width:100%;height:100%;border:0;background:#0b0c0f}"
        "#companion-feedback-wrap{position:relative;z-index:2;display:block;padding:14px 18px 18px;"
        "background:#0b0c0f}"
        "#companion-feedback-label{display:block;margin:0 0 7px;color:#d7d3ca;font-size:12px}"
        "#companion-feedback{display:block;width:min(720px,calc(100% - 32px));min-height:44px;resize:vertical;"
        "padding:10px 12px;border:1px solid #383b45;border-radius:10px;background:#17191f;color:#f4f2ec;"
        "font:14px/1.4 inherit}"
        "#companion-status{position:fixed;right:16px;bottom:14px;padding:8px 12px;border-radius:999px;"
        "background:#17191f;border:1px solid #383b45;color:#d7d3ca;font-size:12px;z-index:9999}"
        "</style></head><body>"
        f"<iframe id=\"{board_frame_id}\" class=\"companion-board\" sandbox=\"allow-scripts\" "
        "title=\"Visual design choices\"></iframe>"
        "<div id=\"companion-feedback-wrap\"><label id=\"companion-feedback-label\" "
        "for=\"companion-feedback\">Optional correction before choosing</label>"
        f"<textarea id=\"companion-feedback\" maxlength=\"{_MAX_FEEDBACK_CHARS}\" "
        "placeholder=\"e.g. Keep this structure, but reduce the orange accent\"></textarea></div>"
        "<div id=\"companion-status\" role=\"status\">Choose a direction</div>"
        f"<script nonce=\"{nonce}\">{helper}</script>"
        "</body></html>"
    )
    csp = (
        "default-src 'none'; "
        "style-src 'unsafe-inline'; "
        f"script-src 'nonce-{nonce}'; "
        "connect-src 'self'; frame-src 'self'; img-src data:; font-src data:; "
        "form-action 'none'; frame-ancestors 'none'; base-uri 'none'"
    )
    return page.encode("utf-8"), csp


class _VisualCompanionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        session_dir: Path,
        port: int,
        bootstrap_token: str,
        session_token: str,
        cookie_name: str,
        route_prefix: str,
    ) -> None:
        super().__init__(("127.0.0.1", port), _VisualCompanionHandler)
        self.session_dir = session_dir
        self.bootstrap_token = bootstrap_token
        self.session_token = session_token
        self.cookie_name = cookie_name
        self.route_prefix = route_prefix
        self.capability_lock = threading.Lock()
        self.capability_consumed = False
        self.event_lock = threading.Lock()

    def route(self, suffix: str = "") -> str:
        return f"{self.route_prefix}{suffix.lstrip('/')}"

    def exchange_capability(self, supplied: str) -> bool:
        with self.capability_lock:
            if self.capability_consumed or not hmac.compare_digest(
                supplied,
                self.bootstrap_token,
            ):
                return False
            self.capability_consumed = True
            return True

    def record_choice(self, choice_id: str, expected_page_version: int, feedback: str) -> dict:
        with self.event_lock:
            with _session_lock(self.session_dir):
                metadata, fragment = _active_round(self.session_dir)
                choices = _choices_from_fragment(fragment)
                if choice_id not in choices:
                    raise VisualCompanionError("choice_id is not present in the active round.")

                page_version = metadata.get("page_version")
                round_id = metadata.get("round_id")
                if type(page_version) is not int or not isinstance(round_id, str):
                    raise VisualCompanionError("round.json is missing round_id or page_version.")
                if expected_page_version != page_version:
                    raise StalePageError(
                        f"active page version is {page_version}, not {expected_page_version}."
                    )

                existing = _read_events(self.session_dir)
                prior = next(
                    (event for event in existing if event.get("page_version") == page_version),
                    None,
                )
                if prior is not None:
                    if prior.get("choice_id") == choice_id and prior.get("feedback", "") == feedback:
                        return prior
                    raise RoundAlreadySelectedError(
                        "the active round already has a recorded selection."
                    )
                cursor = max((event["cursor"] for event in existing), default=0) + 1
                event = {
                    "choice_id": choice_id,
                    "cursor": cursor,
                    "feedback": feedback,
                    "label": choices[choice_id],
                    "page_version": page_version,
                    "round_id": round_id,
                }
                events_path = self.session_dir / "events.jsonl"
                descriptor = os.open(
                    events_path,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
                if os.name != "nt":
                    os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "a", encoding="utf-8") as event_file:
                    event_file.write(
                        json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    event_file.flush()
                    os.fsync(event_file.fileno())
                return event


class _VisualCompanionHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _companion_server(self) -> _VisualCompanionServer:
        if not isinstance(self.server, _VisualCompanionServer):
            raise RuntimeError("visual companion handler attached to an unexpected server")
        return self.server

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _authorized(self) -> bool:
        server = self._companion_server()
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        supplied = cookie.get(server.cookie_name)
        return supplied is not None and hmac.compare_digest(
            supplied.value,
            server.session_token,
        )

    def _require_authorized(self) -> bool:
        if self._authorized():
            return True
        self._send_json(HTTPStatus.FORBIDDEN, {"error": "visual companion capability required"})
        return False

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        server = self._companion_server()
        if not self._require_authorized():
            return

        if parsed.path == server.route("__health"):
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path == server.route():
            page, csp = _render_page(server.session_dir)
            self._send_bytes(
                HTTPStatus.OK,
                page,
                "text/html; charset=utf-8",
                {"Content-Security-Policy": csp},
            )
            return
        if parsed.path == server.route("__version"):
            metadata = _read_json_object(server.session_dir / "round.json")
            self._send_json(
                HTTPStatus.OK,
                {
                    "page_version": metadata.get("page_version"),
                    "round_id": _validated_round_id(metadata.get("round_id")),
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        server = self._companion_server()
        if parsed.path == server.route("__bootstrap"):
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid bootstrap request"})
                return
            if content_length > _MAX_BOOTSTRAP_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "bootstrap request exceeds the size limit"},
                )
                return
            if not self.headers.get("Content-Type", "").lower().startswith(
                "application/x-www-form-urlencoded"
            ):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid bootstrap request"})
                return
            try:
                form = urllib.parse.parse_qs(
                    self.rfile.read(content_length).decode("utf-8"),
                    strict_parsing=True,
                )
                supplied = form.get("key", [""])[0]
            except (UnicodeDecodeError, ValueError):
                supplied = ""
            if not supplied or not server.exchange_capability(supplied):
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": "visual companion capability required"},
                )
                return
            nonce = secrets.token_urlsafe(18)
            route = json.dumps(server.route())
            transition = (
                "<!doctype html><meta charset=\"utf-8\">"
                "<meta name=\"referrer\" content=\"no-referrer\">"
                f"<script nonce=\"{nonce}\">location.replace({route});</script>"
            ).encode()
            self._send_bytes(
                HTTPStatus.OK,
                transition,
                "text/html; charset=utf-8",
                {
                    "Content-Security-Policy": (
                        f"default-src 'none'; script-src 'nonce-{nonce}'; base-uri 'none'"
                    ),
                    "Set-Cookie": (
                        f"{server.cookie_name}={server.session_token}; "
                        f"Path={server.route_prefix}; HttpOnly; SameSite=Lax"
                    ),
                },
            )
            return

        if not self._require_authorized():
            return

        if parsed.path == server.route("__choice"):
            if self.headers.get("X-Visual-Companion") != "choice":
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "choice request header required"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid request body size"})
                return
            if content_length > _MAX_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "request body exceeds the size limit"},
                )
                return
            try:
                payload = json.loads(self.rfile.read(content_length))
                choice_id = payload.get("choice_id") if isinstance(payload, dict) else None
                page_version = payload.get("page_version") if isinstance(payload, dict) else None
                feedback = payload.get("feedback", "") if isinstance(payload, dict) else None
                if not isinstance(choice_id, str) or not choice_id:
                    raise VisualCompanionError("choice_id must be a non-empty string.")
                if type(page_version) is not int:
                    raise VisualCompanionError("page_version must be an integer.")
                if not isinstance(feedback, str):
                    raise VisualCompanionError("feedback must be a string.")
                feedback = feedback.strip()
                if len(feedback) > _MAX_FEEDBACK_CHARS:
                    raise VisualCompanionError(
                        f"feedback must be {_MAX_FEEDBACK_CHARS} characters or fewer."
                    )
                event = self._companion_server().record_choice(choice_id, page_version, feedback)
            except (RoundAlreadySelectedError, StalePageError) as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            except (json.JSONDecodeError, VisualCompanionError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, event)
            return

        if parsed.path == server.route("__shutdown"):
            self._send_json(HTTPStatus.OK, {"status": "stopping"})
            threading.Thread(target=self._companion_server().shutdown, daemon=True).start()
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})


def publish(session_dir: Path, fragment_path: Path, round_id: str) -> int:
    round_id = _validated_round_id(round_id)
    fragment = fragment_path.read_text(encoding="utf-8")
    validate_fragment(fragment)

    session_dir.mkdir(parents=True, exist_ok=True)
    with _session_lock(session_dir):
        next_version = _next_page_version(session_dir / "round.json")
        metadata = {"round_id": round_id, "page_version": next_version}
        _atomic_write_text(session_dir / ".pages" / f"page-{next_version}.html", fragment)
        _atomic_write_text(session_dir / "current.html", fragment)
        _atomic_write_text(
            session_dir / "round.json",
            json.dumps(metadata, ensure_ascii=False),
        )

    print(json.dumps(metadata, ensure_ascii=False))
    return 0


def run_server(session_dir: Path, port: int) -> int:
    if port < 0 or port > 65535:
        raise VisualCompanionError("--port must be between 0 and 65535.")
    _active_round(session_dir)

    with _server_lease(session_dir):
        bootstrap_token = secrets.token_urlsafe(32)
        session_token = secrets.token_urlsafe(32)
        cookie_name = f"{_COOKIE_NAME_PREFIX}{secrets.token_hex(8)}"
        route_prefix = f"/visual-companion/{secrets.token_urlsafe(12)}/"
        server = _VisualCompanionServer(
            session_dir,
            port,
            bootstrap_token,
            session_token,
            cookie_name,
            route_prefix,
        )
        actual_port = server.server_address[1]
        origin = f"http://127.0.0.1:{actual_port}"
        base_url = f"{origin}{route_prefix.rstrip('/')}"
        state = {
            "base_url": base_url,
            "bootstrap_token": bootstrap_token,
            "bootstrap_url": f"{base_url}/__bootstrap",
            "cookie_name": cookie_name,
            "pid": os.getpid(),
            "port": actual_port,
            "session_token": session_token,
        }
        state_path = session_dir / "state.json"
        launcher_path = session_dir / _PREVIEW_LAUNCHER_FILENAME
        _atomic_write_text(
            launcher_path,
            _render_preview_launcher(state["bootstrap_url"], state["bootstrap_token"]),
        )
        _atomic_write_text(state_path, json.dumps(state, ensure_ascii=False, separators=(",", ":")))
        try:
            state_path.chmod(0o600)
            launcher_path.chmod(0o600)
        except OSError:
            pass

        readiness = {"pid": os.getpid(), "port": actual_port, "status": "ready"}
        print(json.dumps(readiness, separators=(",", ":")), flush=True)
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            try:
                current_state = _read_json_object(state_path)
                if current_state.get("session_token") == session_token:
                    state_path.unlink(missing_ok=True)
                    launcher_path.unlink(missing_ok=True)
            except (FileNotFoundError, json.JSONDecodeError, VisualCompanionError):
                pass
    return 0


def _request_from_state(session_dir: Path, path: str, method: str = "GET") -> dict:
    state = _read_json_object(session_dir / "state.json")
    base_url = state.get("base_url")
    cookie_name = state.get("cookie_name")
    session_token = state.get("session_token")
    if not all(isinstance(value, str) for value in (base_url, cookie_name, session_token)):
        raise VisualCompanionError("state.json is missing companion authentication state.")
    request = urllib.request.Request(
        f"{base_url}{path}",
        method=method,
        headers={"Cookie": f"{cookie_name}={session_token}"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise VisualCompanionError("companion returned an invalid response.")
    return payload


def _command_publish(args: argparse.Namespace) -> int:
    return publish(args.session_dir, args.file, args.round_id)


def _command_serve(args: argparse.Namespace) -> int:
    return run_server(args.session_dir, args.port)


def _command_wait(args: argparse.Namespace) -> int:
    event = wait_for_choice(args.session_dir, args.after, args.timeout)
    if event is None:
        print(json.dumps({"after": args.after, "status": "timeout"}, separators=(",", ":")))
        return 3
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    return 0


def _command_status(args: argparse.Namespace) -> int:
    print(json.dumps(_request_from_state(args.session_dir, "/__health"), separators=(",", ":")))
    return 0


def _command_stop(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            _request_from_state(args.session_dir, "/__shutdown", method="POST"),
            separators=(",", ":"),
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve interactive visual design choices locally.")
    commands = parser.add_subparsers(dest="command", required=True)

    publish_parser = commands.add_parser("publish", help="Publish a safe design-choice fragment.")
    publish_parser.add_argument("--session-dir", type=Path, required=True)
    publish_parser.add_argument("--file", type=Path, required=True)
    publish_parser.add_argument("--round-id", required=True)
    publish_parser.set_defaults(handler=_command_publish)

    serve_parser = commands.add_parser("serve", help="Serve the active round on loopback.")
    serve_parser.add_argument("--session-dir", type=Path, required=True)
    serve_parser.add_argument("--port", type=int, default=0)
    serve_parser.set_defaults(handler=_command_serve)

    wait_parser = commands.add_parser("wait", help="Wait for a choice newer than a cursor.")
    wait_parser.add_argument("--session-dir", type=Path, required=True)
    wait_parser.add_argument("--after", type=int, default=0)
    wait_parser.add_argument("--timeout", type=float, default=120.0)
    wait_parser.set_defaults(handler=_command_wait)

    status_parser = commands.add_parser("status", help="Check a running companion.")
    status_parser.add_argument("--session-dir", type=Path, required=True)
    status_parser.set_defaults(handler=_command_status)

    stop_parser = commands.add_parser("stop", help="Stop a running companion.")
    stop_parser.add_argument("--session-dir", type=Path, required=True)
    stop_parser.set_defaults(handler=_command_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        return args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
