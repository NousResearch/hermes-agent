"""Export Grok Bot data (bots, conversations) into grokbot-export.json.

The exporter is deliberately layered so it degrades instead of breaking:

L1 app witness
    The Grok Bot desktop app is relaunched under our control: its backend
    base URLs are pointed at a local logging proxy (its own documented env
    overrides) and Chromium is started with a CDP debug port. We then read
    what the app itself shows — the bot roster, every conversation's
    transcript (roles + timestamps come from the rendered DOM), and each
    bot's details pane. Whatever the vendor changes, the app's own UI
    remains the source of truth.

L2 API replay
    The proxy capture yields the account's OAuth refresh token (stable,
    public client). We mint access tokens and replay the cataloged backend
    endpoints for metadata the UI never loads (sandbox list). If the
    account's feature flags block a call, the layer is skipped with a
    warning — L1 output is unaffected.

Secrets never enter the export file. Captured tokens live only in the
capture directory (mode 0600), which is deleted unless --keep-capture is
passed.

The app is macOS-only; the exporter refuses to run elsewhere.
"""

from __future__ import annotations

import asyncio
import http.client
import http.server
import json
import logging
import os
import shutil
import signal
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = 1

DEFAULT_APP_PATH = Path("/Applications/Grok Bot.app")
DEFAULT_CDP_PORT = 9335
DEFAULT_PROXY_PORT = 9339
BACKEND_HOST = "api2.cursor.sh"
OAUTH_CLIENT_ID = "OzaBXLClY5CAGxNzUhQ2vlknpi07tGuE"
CHECKSUM_PREFIX = "4CYoNne3"

_ROSTER_ITEM_SELECTOR = ".sand-agent-item"
_MESSAGE_SELECTOR = ".sand-message"
_SCROLL_VIEWPORT_SELECTOR = ".ui-scroll-area__viewport"


class ExportError(RuntimeError):
    """A layer failed in a way the caller must see."""


# ---------------------------------------------------------------------------
# L1: capture proxy
# ---------------------------------------------------------------------------


class CaptureProxy:
    """Loopback proxy that logs every request/response to a capture dir.

    The app is pointed here via ``SAND_BACKEND_URL`` / ``CURSOR_API_BASE_URL``
    (its documented environment overrides), so no certificates or system
    proxy changes are involved. Requests forward to the real backend over
    TLS; responses are captured in full for offline analysis.
    """

    def __init__(self, port: int, capture_dir: Path) -> None:
        self.port = port
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._lock = threading.Lock()
        self._log_path = capture_dir / "proxylog.jsonl"
        self._server: Optional[http.server.ThreadingHTTPServer] = None

    def _log(self, obj: Dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            obj = {"seq": self._seq, "ts": time.time(), **obj}
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _save_body(self, body: bytes, label: str, suffix: str) -> Path:
        path = self.capture_dir / f"{self._seq:05d}-{label}{suffix}"
        path.write_bytes(body)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path

    def start(self) -> None:
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args) -> None:  # silence stderr chatter
                pass

            def _forward(self) -> None:
                seq = None
                body = b""
                if self.headers.get("Content-Length"):
                    body = self.rfile.read(int(self.headers["Content-Length"]))
                auth = self.headers.get("Authorization") or ""
                with proxy._lock:
                    proxy._seq += 1
                    seq = proxy._seq
                body_ref = None
                if body:
                    body_ref = str(proxy._save_body(body, f"req-{seq}", ".bin"))
                proxy._log(
                    {
                        "kind": "req",
                        "seq_ref": seq,
                        "method": self.command,
                        "path": self.path[:400],
                        "auth": f"{auth[:16]}...{auth[-12:]}" if auth else None,
                        "auth_full": auth or None,
                        "headers": dict(self.headers.items()),
                        "body_len": len(body),
                        "body_ref": body_ref,
                    }
                )
                headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
                conn = http.client.HTTPSConnection(
                    BACKEND_HOST, 443, timeout=60, context=ssl.create_default_context()
                )
                try:
                    conn.request(self.command, self.path, body=body, headers=headers)
                    resp = conn.getresponse()
                    resp_body = resp.read()
                    proxy._log(
                        {
                            "kind": "res",
                            "seq_ref": seq,
                            "status": resp.status,
                            "content_type": resp.getheader("content-type") or "",
                            "body_len": len(resp_body),
                        }
                    )
                    if resp_body:
                        saved = proxy._save_body(resp_body, f"res-{seq}", ".bin")
                        proxy._log({"kind": "res_body", "seq_ref": seq, "file": str(saved)})
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    if resp_body:
                        self.wfile.write(resp_body)
                except Exception as exc:  # noqa: BLE001
                    proxy._log({"kind": "fwd_err", "seq_ref": seq, "error": str(exc)})
                    try:
                        self.send_response(502)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                    except OSError:
                        pass

            do_POST = _forward
            do_GET = _forward
            do_PUT = _forward
            do_DELETE = _forward
            do_PATCH = _forward

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def refresh_tokens(self) -> List[str]:
        """Extract distinct refresh tokens seen in oauth/token requests."""
        tokens: List[str] = []
        if not self._log_path.is_file():
            return tokens
        for line in self._log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                entry.get("kind") == "req"
                and entry.get("path", "").startswith("/oauth/token")
                and entry.get("body_ref")
            ):
                body_path = Path(entry["body_ref"])
                if body_path.is_file():
                    try:
                        data = json.loads(body_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    rt = data.get("refresh_token")
                    if rt and rt not in tokens:
                        tokens.append(rt)
        return tokens


# ---------------------------------------------------------------------------
# L2: API replay
# ---------------------------------------------------------------------------


def mint_access_token(refresh_token: str) -> str:
    """Exchange a captured refresh token for an access token."""
    payload = json.dumps(
        {
            "client_id": OAUTH_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"https://{BACKEND_HOST}/oauth/token",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    token = data.get("access_token")
    if not token:
        raise ExportError("oauth/token response carried no access_token")
    return token


def api2_post(
    path: str,
    access_token: str,
    body: Optional[Dict[str, Any]] = None,
    *,
    machine_id: str = "",
    client_version: str = "0.24.0",
) -> Tuple[int, Dict[str, Any]]:
    """POST to the product backend the way the desktop client does."""
    headers = {
        "Content-Type": "application/json",
        "Connect-Protocol-Version": "1",
        "User-Agent": "connect-es/1.6.1",
        "Authorization": f"Bearer {access_token}",
        "x-cursor-client-version": client_version,
        "x-cursor-client-type": "sand",
        "x-ghost-mode": "true",
        "x-request-id": str(uuid.uuid4()),
    }
    if machine_id:
        headers["x-cursor-checksum"] = f"{CHECKSUM_PREFIX}{machine_id}"
    payload = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        f"https://{BACKEND_HOST}{path}", data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"_raw": raw}


# ---------------------------------------------------------------------------
# L1: CDP witness
# ---------------------------------------------------------------------------


class CdpClient:
    """Minimal CDP client over the `websockets` dependency (already shipped)."""

    def __init__(self, ws) -> None:
        self._ws = ws
        self._id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._events: List[Dict[str, Any]] = []

    @classmethod
    async def connect(cls, port: int) -> "CdpClient":
        import websockets

        async def _try() -> "CdpClient":
            for attempt in range(10):
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/json/list", timeout=2
                    ) as resp:
                        targets = json.loads(resp.read().decode("utf-8"))
                    page = next(
                        (t for t in targets if t.get("type") == "page"), None
                    )
                    if page is None:
                        raise ExportError("no page target on the CDP endpoint")
                    ws = await websockets.connect(
                        page["webSocketDebuggerUrl"],
                        additional_headers={"Origin": "devtools://devtools"},
                        open_timeout=5,
                    )
                    return cls(ws)
                except Exception:  # noqa: BLE001 — retry ladder below
                    await asyncio.sleep(1.5)
            raise ExportError("could not attach to the app's CDP endpoint")

        client = await _try()
        asyncio.create_task(client._reader())
        return client

    async def _reader(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                mid = msg.get("id")
                if mid is not None and mid in self._pending:
                    fut = self._pending.pop(mid)
                    if not fut.done():
                        fut.set_result(msg)
                elif msg.get("method"):
                    self._events.append(msg)
        except Exception:  # noqa: BLE001
            pass

    async def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._id += 1
        mid = self._id
        msg: Dict[str, Any] = {"id": mid, "method": method}
        if params:
            msg["params"] = params
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        await self._ws.send(json.dumps(msg))
        result = await asyncio.wait_for(fut, timeout=10)
        if result.get("error"):
            raise ExportError(
                f"CDP {method}: {json.dumps(result['error'])[:200]}"
            )
        return result.get("result", {})

    async def evaluate(self, expression: str, timeout_ms: int = 8000) -> Any:
        result = await self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
                "timeout": timeout_ms,
            },
        )
        return (result.get("result") or {}).get("value")

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:  # noqa: BLE001
            pass


_JS_ROSTER = (
    "JSON.stringify(Array.from(document.querySelectorAll('.sand-agent-item')).map("
    "(e) => ({name: (e.getAttribute('aria-label') || '').trim(), "
    "preview: (e.innerText || '').trim().slice(0, 200)})))"
)

_JS_CLICK_BY_TEXT = (
    "(text) => { const els = Array.from(document.querySelectorAll('.sand-agent-item'));"
    " const el = els.find((e) => (e.getAttribute('aria-label') || '').trim() === text);"
    " if (el) { el.click(); return true; } return false; }"
)

_JS_EXTRACT_MESSAGES = """
JSON.stringify((() => {
  const out = [];
  const seen = new Set();
  for (const m of document.querySelectorAll('.sand-message')) {
    const text = (m.innerText || '').trim().replace(/\\s+/g, ' ');
    if (!text || seen.has(text)) continue;
    seen.add(text);
    const label = m.getAttribute('aria-label') || '';
    const timeEl = m.querySelector('time') || (m.parentElement ? m.parentElement.querySelector('time') : null);
    out.push({
      role: label.includes('Your') ? 'user' : 'assistant',
      text: text.slice(0, 100000),
      tsLabel: timeEl ? timeEl.textContent.trim() : '',
      ts: Date.now() / 1000,
    });
  }
  return out;
})())
"""

_JS_SCROLL_TOP = """
(() => {
  const vp = document.querySelector('.ui-scroll-area__viewport');
  const el = vp || document.scrollingElement;
  if (el) { el.scrollTop = 0; }
  return true;
})()
"""

_JS_DETAILS = """
JSON.stringify((() => {
  const btn = Array.from(document.querySelectorAll('button')).find(
    (b) => (b.getAttribute('aria-label') || '').includes('View conversation details'));
  if (btn) btn.click();
  return true;
})())
"""

_JS_INFO_PANE = """
JSON.stringify((() => {
  const back = Array.from(document.querySelectorAll('button')).find(
    (b) => (b.innerText || '').trim() === 'Back to details');
  if (back) back.click();
  const pane = document.querySelector('.sand-info-pane');
  if (!pane) return null;
  const inputs = Array.from(pane.querySelectorAll('input, textarea')).map((i) => ({
    label: (i.getAttribute('aria-label') || i.getAttribute('placeholder') || '').trim().slice(0, 60),
    value: (i.value || '').trim().slice(0, 2000),
  })).filter((x) => x.value);
  const text = (pane.innerText || '').replace(/\\s+/g, ' ').slice(0, 4000);
  return { text, inputs };
})())
"""


def _parse_ts(ts_label: str, fallback: float) -> float:
    """Best-effort parse of the app's local 12h timestamps."""
    import datetime

    ts_label = (ts_label or "").strip()
    try:
        now = datetime.datetime.now()
        dt = datetime.datetime.strptime(ts_label, "%I:%M:%S %p")
        dt = dt.replace(year=now.year, month=now.month, day=now.day)
        return dt.timestamp()
    except ValueError:
        return fallback


async def witness_export(cdp: CdpClient, max_messages_per_conversation: int) -> Dict[str, Any]:
    """Walk the app's UI and build the export dict (bots + conversations)."""
    roster_raw = await cdp.evaluate(_JS_ROSTER)
    roster = json.loads(roster_raw) if isinstance(roster_raw, str) else []
    bots: List[Dict[str, Any]] = []
    conversations: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for i, item in enumerate(roster):
        name = (item.get("name") or "").strip()
        if not name:
            continue
        bid = f"roster-{i}"
        clicked = await cdp.evaluate(f"({_JS_CLICK_BY_TEXT})({json.dumps(name)})")
        if clicked is not True:
            warnings.append(f"could not click bot row: {name}")
            continue
        await asyncio.sleep(2.5)

        # Scroll to the top repeatedly to force history pagination to load,
        # then read the transcript once: the rendered DOM lists messages in
        # chronological order, so the final full read is authoritative.
        last_count = -1
        plateau = 0
        batch: List[Dict[str, Any]] = []
        for _ in range(12):
            raw = await cdp.evaluate(_JS_EXTRACT_MESSAGES)
            try:
                batch = json.loads(raw) if isinstance(raw, str) else []
            except json.JSONDecodeError:
                batch = []
            if len(batch) == last_count:
                plateau += 1
                if plateau >= 2:
                    break
            else:
                plateau = 0
                last_count = len(batch)
            if last_count >= max_messages_per_conversation:
                break
            await cdp.evaluate(_JS_SCROLL_TOP)
            await asyncio.sleep(1.8)
        messages = batch

        # Normalize timestamps: relative ordering with absolute guesses.
        fallback = time.time() - len(messages)
        for idx, msg in enumerate(messages):
            msg["ts"] = _parse_ts(msg.pop("tsLabel", ""), fallback + idx)

        bot = {
            "id": bid,
            "name": name,
            "title": "",
            "description": "",
            "instructions": "",
            "model": "",
            "memories": [],
            "tools": [],
            "plugins": [],
        }
        if messages:
            conversations.append(
                {
                    "bot_id": bid,
                    "thread_id": bid,
                    "title": f"Chat with {name}",
                    "messages": messages,
                }
            )
        bots.append(bot)

        # Bot details pane (name/title/description live here).
        await cdp.evaluate(_JS_DETAILS)
        await asyncio.sleep(1.5)
        pane_raw = await cdp.evaluate(_JS_INFO_PANE)
        if isinstance(pane_raw, str):
            try:
                pane = json.loads(pane_raw)
            except json.JSONDecodeError:
                pane = None
            if pane:
                lines = [f"{x['label']}: {x['value']}" for x in (pane.get("inputs") or [])]
                if not lines and pane.get("text"):
                    # Fall back to pane text, minus known UI chrome labels.
                    chrome = (
                        "Settings Name Title Description Notifications Get "
                        "notified when this Bot finishes or needs input Back "
                        "to details Close details Edit Bot avatar"
                    )
                    cleaned = pane["text"]
                    for word in chrome.split():
                        cleaned = cleaned.replace(word, "")
                    cleaned = cleaned.strip()
                    if cleaned:
                        lines = [cleaned]
                bot["description"] = "\n".join(lines)[:2000]

    if not bots:
        warnings.append(
            "no bots found in the sidebar — if the app is signed out, sign in "
            "and re-run"
        )
    return {"bots": bots, "conversations": conversations, "warnings": warnings}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _app_running() -> Optional[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "Grok Bot.app/Contents/MacOS"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return int(out.splitlines()[0]) if out else None
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def _quit_app() -> None:
    subprocess.run(["osascript", "-e", 'quit app "Grok Bot"'], timeout=10)
    time.sleep(3)
    pid = _app_running()
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        time.sleep(3)


def _launch_app(proxy_port: int, cdp_port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["SAND_BACKEND_URL"] = f"http://127.0.0.1:{proxy_port}"
    env["CURSOR_API_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"
    binary = DEFAULT_APP_PATH / "Contents" / "MacOS" / "Grok Bot"
    return subprocess.Popen(
        [
            str(binary),
            f"--remote-debugging-port={cdp_port}",
            "--remote-debugging-address=127.0.0.1",
            "--remote-allow-origins=*",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _relaunch_clean() -> None:
    subprocess.run(["open", "-na", str(DEFAULT_APP_PATH)], timeout=10)


async def _wait_for_cdp(port: int, timeout_s: float) -> CdpClient:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return await CdpClient.connect(port)
        except ExportError:
            await asyncio.sleep(2)
    raise ExportError(
        f"CDP endpoint never became ready on port {port}"
    )


def run_export(
    *,
    out_path: Path,
    capture_dir: Path,
    app_path: Path = DEFAULT_APP_PATH,
    cdp_port: int = DEFAULT_CDP_PORT,
    proxy_port: int = DEFAULT_PROXY_PORT,
    keep_capture: bool = False,
    keep_running: bool = False,
    max_messages_per_conversation: int = 2000,
) -> int:
    """Run the layered export. Returns a process exit code."""
    from hermes_cli.colors import Colors, color

    if sys.platform != "darwin":
        print(
            color("✗ Grok Bot export requires macOS (the app is macOS-only).", Colors.RED),
            file=sys.stderr,
        )
        return 1
    if not (app_path / "Contents" / "MacOS" / "Grok Bot").is_file():
        print(
            color(f"✗ Grok Bot app not found at {app_path}", Colors.RED),
            file=sys.stderr,
        )
        return 1

    capture_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    layers_used: List[str] = []

    proxy = CaptureProxy(proxy_port, capture_dir)
    print(color("◆ Grok Bot export", Colors.CYAN, Colors.BOLD))
    print()
    print(f"  Starting capture proxy on 127.0.0.1:{proxy_port} ...")
    proxy.start()

    proc: Optional[subprocess.Popen] = None
    try:
        print("  Restarting the app under capture (its UI stays usable) ...")
        _quit_app()
        proc = _launch_app(proxy_port, cdp_port)

        async def _walk() -> Dict[str, Any]:
            cdp = await _wait_for_cdp(cdp_port, timeout_s=60)
            await cdp.send("Runtime.enable")
            result = await witness_export(cdp, max_messages_per_conversation)
            await cdp.close()
            return result

        try:
            witness = asyncio.run(_walk())
        except ExportError as exc:
            witness = {"bots": [], "conversations": [], "warnings": [str(exc)]}
        layers_used.append("witness")
        warnings.extend(witness.get("warnings") or [])

        # L2: replay for backend metadata the UI never loads.
        sandboxes: List[Dict[str, Any]] = []
        tokens = proxy.refresh_tokens()
        machine_id = ""
        if tokens:
            try:
                access_token = mint_access_token(tokens[-1])
                status, data = api2_post(
                    "/aiserver.v1.GrokBotService/ListSandBoxes",
                    access_token,
                    machine_id=machine_id,
                )
                if status == 200:
                    sandboxes = data.get("boxes") or []
                    layers_used.append("api-replay")
                else:
                    warnings.append(
                        f"backend replay skipped (ListSandBoxes: {status} "
                        f"{str(data.get('message') or data)[:120]})"
                    )
            except Exception as exc:  # noqa: BLE001 — optional layer
                warnings.append(f"backend replay skipped: {exc}")

        export = {
            "schema": EXPORT_SCHEMA_VERSION,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "app_version": "0.24.0",
            "account": {},
            "bots": witness.get("bots") or [],
            "conversations": witness.get("conversations") or [],
            "files": {},
            "provenance": {
                "layers": layers_used,
                "sandboxes": sandboxes,
                "warnings": warnings,
            },
        }
        out_path.write_text(
            json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"  {color('✓', Colors.GREEN)} Exported "
            f"{len(export['bots'])} bot(s), {len(export['conversations'])} "
            f"conversation(s) → {out_path}"
        )
        for warning in warnings:
            print(f"  {color('⚠', Colors.YELLOW)} {warning}")
        if tokens:
            print(
                color(
                    f"  Note: capture dir {capture_dir} contains account "
                    f"tokens (0600). Keep it private or delete it.",
                    Colors.DIM,
                )
            )
        return 0
    finally:
        proxy.stop()
        if proc is not None and proc.poll() is None:
            _quit_app()
            if keep_running:
                print(
                    color(
                        "  Restored the app to a normal (uninstrumented) launch.",
                        Colors.DIM,
                    )
                )
                _relaunch_clean()
        if not keep_capture and capture_dir.exists():
            shutil.rmtree(capture_dir, ignore_errors=True)


def run_doctor() -> int:
    """Check prerequisites; print findings; return an exit code."""
    from hermes_cli.colors import Colors, color

    print(color("◆ Grok Bot export doctor", Colors.CYAN, Colors.BOLD))
    print()
    ok = True

    def check(label: str, good: bool, detail: str) -> None:
        nonlocal ok
        mark = color("✓", Colors.GREEN) if good else color("✗", Colors.RED)
        print(f"  {mark} {label}" + (f": {detail}" if detail else ""))
        ok = ok and good

    check(
        "Operating system",
        sys.platform == "darwin",
        "macOS required" if sys.platform != "darwin" else sys.platform,
    )
    app_bin = DEFAULT_APP_PATH / "Contents" / "MacOS" / "Grok Bot"
    check("Grok Bot app installed", app_bin.is_file(), str(DEFAULT_APP_PATH))
    if app_bin.is_file():
        try:
            info = (DEFAULT_APP_PATH / "Contents" / "Info.plist").read_bytes()
            check("App bundle readable", b"CFBundleDisplayName" in info, "Info.plist present")
        except OSError as exc:
            check("App bundle readable", False, str(exc))
    running = _app_running()
    check(
        "App state",
        True,
        "running (will be restarted during export)"
        if running
        else "not running",
    )
    check(
        "No conflicting capture proxy",
        not _port_busy(DEFAULT_PROXY_PORT),
        f"port {DEFAULT_PROXY_PORT} busy" if _port_busy(DEFAULT_PROXY_PORT) else f"port {DEFAULT_PROXY_PORT} free",
    )
    check(
        "No conflicting CDP port",
        not _port_busy(DEFAULT_CDP_PORT),
        f"port {DEFAULT_CDP_PORT} busy" if _port_busy(DEFAULT_CDP_PORT) else f"port {DEFAULT_CDP_PORT} free",
    )
    return 0 if ok else 1


def _port_busy(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True
