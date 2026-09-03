"""Computer runtime adapter.

Reuses the existing Hermes Chromium/CDP launch shape from
``tools.browser_tool._real_profile_cdp`` (real binary, ``--user-data-dir``,
loopback DevTools port). It does **not** call ``snapshot_real_profile``:
that path re-syncs the OS last-used profile and would destroy an
agent-owned BrowserIdentity.

Tests use ``InMemoryRuntime``. The Chromium adapter is opt-in when a
binary exists; production takeover still goes through this same handle,
not a second browser.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .models import AgentComputer, BrowserIdentity, Observation
from .pointer import jpeg_dimensions, map_screenshot_to_viewport


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimeHandle:
    computer_id: str
    identity_id: str | None
    user_data_dir: str
    cdp_loopback: str | None = None
    process_id: int | None = None
    backend: str = "in_memory"
    headed_same_host: bool = False
    last_pointer_x: float = 0.0
    last_pointer_y: float = 0.0
    screenshot_width: int = 0
    screenshot_height: int = 0
    viewport_width: int = 0
    viewport_height: int = 0


class ComputerRuntime(Protocol):
    def wake(self, computer: AgentComputer, identity: BrowserIdentity | None) -> RuntimeHandle: ...

    def observe(self, handle: RuntimeHandle) -> Observation: ...

    def act(
        self,
        handle: RuntimeHandle,
        *,
        kind: str,
        target: str = "",
        text: str = "",
        action_class: str = "",
        x: float | None = None,
        y: float | None = None,
        key: str = "",
        code: str = "",
        delta_x: float = 0,
        delta_y: float = 0,
    ) -> Observation: ...

    def sleep(self, handle: RuntimeHandle) -> None: ...

    def alive(self, handle: RuntimeHandle) -> bool: ...


@dataclass
class _Page:
    url: str = "about:blank"
    title: str = ""
    text: str = ""
    cookies: dict[str, str] = field(default_factory=dict)
    last_x: float = 0.0
    last_y: float = 0.0
    input_value: str = ""


class InMemoryRuntime:
    """Shared mutable page per identity/computer. Same object = same environment."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pages: dict[str, _Page] = {}
        self._alive: dict[str, bool] = {}

    def _key(self, computer: AgentComputer, identity: BrowserIdentity | None) -> str:
        if identity:
            return f"id:{identity.id}"
        return f"pc:{computer.id}"

    def _page(self, key: str) -> _Page:
        page = self._pages.get(key)
        if page is None:
            page = _Page()
            self._pages[key] = page
        return page

    def wake(self, computer: AgentComputer, identity: BrowserIdentity | None) -> RuntimeHandle:
        key = self._key(computer, identity)
        with self._lock:
            self._alive[key] = True
            page = self._page(key)
            if computer.workspace_url and page.url == "about:blank":
                page.url = computer.workspace_url
                page.title = computer.workspace_title
        return RuntimeHandle(
            computer_id=computer.id,
            identity_id=identity.id if identity else None,
            user_data_dir=identity.profile_ref if identity else computer.persistence_ref,
            cdp_loopback=None,
            backend="in_memory",
        )

    def observe(self, handle: RuntimeHandle) -> Observation:
        key = f"id:{handle.identity_id}" if handle.identity_id else f"pc:{handle.computer_id}"
        with self._lock:
            page = self._page(key)
            return Observation(
                url=page.url,
                title=page.title,
                text=page.text,
                fencing_epoch=0,
                controller="",
                observed_at=_now(),
                viewport_width=800,
                viewport_height=600,
            )

    def act(
        self,
        handle: RuntimeHandle,
        *,
        kind: str,
        target: str = "",
        text: str = "",
        action_class: str = "",
        x: float | None = None,
        y: float | None = None,
        key: str = "",
        code: str = "",
        delta_x: float = 0,
        delta_y: float = 0,
    ) -> Observation:
        page_key = f"id:{handle.identity_id}" if handle.identity_id else f"pc:{handle.computer_id}"
        with self._lock:
            page = self._page(page_key)
            if kind == "navigate":
                page.url = target
                page.title = target
                page.text = f"opened {target}"
            elif kind == "type":
                page.input_value = text
                page.text = (page.text + " typed").strip()
            elif kind == "click":
                page.text = (page.text + f" clicked:{target}").strip()
            elif kind == "pointer_move":
                page.last_x = float(x or 0)
                page.last_y = float(y or 0)
            elif kind == "pointer_click":
                page.last_x = float(x or 0)
                page.last_y = float(y or 0)
                if 400 <= page.last_x <= 600 and 160 <= page.last_y <= 208:
                    page.text = "owner-pixel-clicked"
                elif 16 <= page.last_x <= 216 and 160 <= page.last_y <= 208:
                    page.text = "agent-ready"
                else:
                    page.text = (page.text + f" pixel:{int(page.last_x)},{int(page.last_y)}").strip()
            elif kind == "text":
                page.input_value = text
                if text:
                    page.text = (page.text + " " + text).strip()
            elif kind == "scroll":
                page.text = (page.text + " scrolled").strip()
            elif kind == "key":
                page.text = (page.text + f" key:{key or code}").strip()
            elif kind == "set_cookie":
                # Test-only durable auth stand-in. Never returned to clients.
                page.cookies[target] = text
                page.text = (page.text + f" auth:{target}").strip()
            else:
                raise ValueError(f"unsupported computer action: {kind}")
            return Observation(
                url=page.url,
                title=page.title,
                text=page.text,
                fencing_epoch=0,
                controller="",
                observed_at=_now(),
                viewport_width=800,
                viewport_height=600,
            )

    def sleep(self, handle: RuntimeHandle) -> None:
        key = f"id:{handle.identity_id}" if handle.identity_id else f"pc:{handle.computer_id}"
        with self._lock:
            self._alive[key] = False

    def alive(self, handle: RuntimeHandle) -> bool:
        key = f"id:{handle.identity_id}" if handle.identity_id else f"pc:{handle.computer_id}"
        with self._lock:
            return bool(self._alive.get(key))

    def cookies_for_test(self, identity_id: str) -> dict[str, str]:
        with self._lock:
            return dict(self._page(f"id:{identity_id}").cookies)


class HermesChromiumRuntime:
    """Launch the host Chromium on an identity-owned user-data-dir.

    Launch flags match ``_real_profile_cdp`` (loopback debug port, no
    mock-keychain). Does not snapshot the OS default profile.
    """

    def __init__(self) -> None:
        self._procs: dict[str, Any] = {}

    def wake(self, computer: AgentComputer, identity: BrowserIdentity | None) -> RuntimeHandle:
        from hermes_cli.browser_connect import chromium_executable, detect_default_chromium

        browser = detect_default_chromium() or "chromium"
        binary = chromium_executable(browser)
        if not binary:
            raise RuntimeError("no Chromium-family binary on this host")
        user_data = identity.profile_ref if identity else computer.persistence_ref
        Path(user_data).mkdir(parents=True, exist_ok=True)
        # Import locally so unit tests never spawn Chrome.
        import os
        import subprocess
        import time

        port_file = os.path.join(user_data, "DevToolsActivePort")
        try:
            os.unlink(port_file)
        except OSError:
            pass
        user_data = str(Path(user_data).resolve())
        argv = [
            binary,
            f"--user-data-dir={user_data}",
            "--remote-debugging-port=0",
            "--remote-debugging-address=127.0.0.1",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-background-networking",
            "--disable-default-apps",
            "--force-device-scale-factor=1",
            "--window-size=800,600",
            "--headless=new",
            "about:blank",
        ]
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 30
        port = None
        while time.monotonic() < deadline:
            try:
                with open(port_file, encoding="utf-8") as fh:
                    line = fh.readline().strip()
                if line.isdigit():
                    port = int(line)
                    break
            except OSError:
                pass
            if proc.poll() is not None:
                raise RuntimeError("chromium exited during startup")
            time.sleep(0.2)
        if port is None:
            proc.terminate()
            raise RuntimeError("chromium did not expose a loopback debug port")
        self._procs[computer.id] = proc
        handle = RuntimeHandle(
            computer_id=computer.id,
            identity_id=identity.id if identity else None,
            user_data_dir=user_data,
            cdp_loopback=f"http://127.0.0.1:{port}",
            process_id=proc.pid,
            backend="hermes_chromium",
            headed_same_host=False,
        )
        try:
            loopback_cdp(
                handle,
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 800,
                    "height": 600,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
        except Exception:
            pass
        return handle

    def observe(self, handle: RuntimeHandle) -> Observation:
        page = loopback_cdp(handle, "Runtime.evaluate", {
            "expression": (
                "({url: location.href, title: document.title, "
                "text: document.body ? document.body.innerText.slice(0, 2000) : '', "
                "viewportWidth: window.innerWidth, viewportHeight: window.innerHeight, "
                "devicePixelRatio: window.devicePixelRatio || 1})"
            ),
            "returnByValue": True,
        })
        value = ((page or {}).get("result") or {}).get("value") or {}
        shot = {}
        try:
            shot = loopback_cdp(
                handle,
                "Page.captureScreenshot",
                {"format": "jpeg", "quality": 40},
            ) or {}
        except Exception:
            shot = {}
        raw_b64 = str(shot.get("data") or "")
        shot_w = shot_h = 0
        if raw_b64:
            try:
                import base64

                shot_w, shot_h = jpeg_dimensions(base64.b64decode(raw_b64))
            except Exception:
                shot_w = shot_h = 0
        vp_w = int(value.get("viewportWidth") or 0)
        vp_h = int(value.get("viewportHeight") or 0)
        handle.screenshot_width = shot_w
        handle.screenshot_height = shot_h
        handle.viewport_width = vp_w
        handle.viewport_height = vp_h
        return Observation(
            url=str(value.get("url") or ""),
            title=str(value.get("title") or ""),
            text=str(value.get("text") or ""),
            fencing_epoch=0,
            controller="",
            observed_at=_now(),
            screenshot_b64=raw_b64,
            screenshot_mime="image/jpeg" if raw_b64 else "",
            screenshot_width=shot_w,
            screenshot_height=shot_h,
            viewport_width=vp_w,
            viewport_height=vp_h,
            device_pixel_ratio=float(value.get("devicePixelRatio") or 1),
        )

    def act(self, handle: RuntimeHandle, **kwargs: Any) -> Observation:
        import time

        kind = str(kwargs.get("kind") or "")
        target = str(kwargs.get("target") or "")
        text = str(kwargs.get("text") or "")
        key = str(kwargs.get("key") or "")
        code = str(kwargs.get("code") or "")
        x = kwargs.get("x")
        y = kwargs.get("y")
        delta_x = float(kwargs.get("delta_x") or 0)
        delta_y = float(kwargs.get("delta_y") or 0)
        if kind == "navigate":
            loopback_cdp(handle, "Page.navigate", {"url": target})
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                obs = self.observe(handle)
                if target in obs.url or (obs.url and obs.url != "about:blank"):
                    return obs
                time.sleep(0.2)
        elif kind == "type":
            if target:
                loopback_cdp(
                    handle,
                    "Runtime.evaluate",
                    {
                        "expression": f"document.querySelector({target!r})?.focus()",
                        "userGesture": True,
                    },
                )
            loopback_cdp(handle, "Input.insertText", {"text": text})
        elif kind == "click":
            expr = (
                f"document.querySelector({target!r})?.click()"
                if target
                else "undefined"
            )
            loopback_cdp(
                handle,
                "Runtime.evaluate",
                {"expression": expr, "userGesture": True},
            )
        elif kind == "pointer_move":
            vx, vy = self._viewport_point(handle, x, y)
            self._mouse(handle, "mouseMoved", vx, vy)
        elif kind == "pointer_click":
            vx, vy = self._viewport_point(handle, x, y)
            self._mouse(handle, "mousePressed", vx, vy, click_count=1)
            self._mouse(handle, "mouseReleased", vx, vy, click_count=1)
        elif kind == "scroll":
            vx = handle.last_pointer_x or (handle.viewport_width / 2 if handle.viewport_width else 400)
            vy = handle.last_pointer_y or (handle.viewport_height / 2 if handle.viewport_height else 300)
            if x is not None and y is not None:
                vx, vy = self._viewport_point(handle, x, y)
            loopback_cdp(
                handle,
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseWheel",
                    "x": vx,
                    "y": vy,
                    "deltaX": delta_x,
                    "deltaY": delta_y,
                },
            )
        elif kind == "key":
            name = key or code
            loopback_cdp(
                handle,
                "Input.dispatchKeyEvent",
                {"type": "keyDown", "key": name, "code": code or name},
            )
            loopback_cdp(
                handle,
                "Input.dispatchKeyEvent",
                {"type": "keyUp", "key": name, "code": code or name},
            )
        elif kind == "text":
            loopback_cdp(handle, "Input.insertText", {"text": text})
        else:
            raise ValueError(f"unsupported computer action: {kind}")
        return self.observe(handle)

    def _viewport_point(self, handle: RuntimeHandle, x: Any, y: Any) -> tuple[float, float]:
        sx = float(x or 0)
        sy = float(y or 0)
        vx, vy = map_screenshot_to_viewport(
            sx,
            sy,
            screenshot_width=handle.screenshot_width,
            screenshot_height=handle.screenshot_height,
            viewport_width=handle.viewport_width,
            viewport_height=handle.viewport_height,
        )
        handle.last_pointer_x = vx
        handle.last_pointer_y = vy
        return vx, vy

    def _mouse(
        self,
        handle: RuntimeHandle,
        event: str,
        x: float,
        y: float,
        *,
        click_count: int = 0,
    ) -> None:
        params: dict[str, Any] = {"type": event, "x": x, "y": y}
        if event in ("mousePressed", "mouseReleased"):
            params["button"] = "left"
            params["clickCount"] = click_count or 1
        loopback_cdp(handle, "Input.dispatchMouseEvent", params)

    def alive(self, handle: RuntimeHandle) -> bool:
        import os
        import urllib.request

        proc = self._procs.get(handle.computer_id)
        if proc is None or proc.poll() is not None:
            return False
        if handle.process_id:
            try:
                os.kill(handle.process_id, 0)
            except OSError:
                return False
        if not handle.cdp_loopback:
            return False
        try:
            with urllib.request.urlopen(handle.cdp_loopback.rstrip("/") + "/json/version", timeout=1):
                return True
        except Exception:
            return False

    def sleep(self, handle: RuntimeHandle) -> None:
        """Stop the Chromium process tree and wait until it is reaped.

        Process/CDP/DOM are ephemeral. The managed user-data-dir stays.
        """
        proc = self._procs.pop(handle.computer_id, None)
        pid = handle.process_id or (proc.pid if proc is not None else None)
        if not pid:
            return
        import os
        import signal
        import time

        children: list[Any] = []
        try:
            import psutil

            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            try:
                parent.terminate()
            except psutil.Error:
                pass
            psutil.wait_procs([parent, *children], timeout=6)
            still = []
            for proc_ref in [parent, *children]:
                try:
                    if proc_ref.is_running():
                        proc_ref.kill()
                        still.append(proc_ref)
                except psutil.Error:
                    pass
            if still:
                psutil.wait_procs(still, timeout=3)
        except Exception:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        pass
        if proc is not None:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
        # Reap a possible zombie so os.kill(pid, 0) fails for callers.
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.05)


def loopback_cdp(handle: RuntimeHandle, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Send one CDP method to the loopback DevTools endpoint.

    Reuses ``tools.browser_cdp_tool._cdp_call``. The HTTP origin must stay
    127.0.0.1 — this is not a public CDP surface.
    """
    if not handle.cdp_loopback or not handle.cdp_loopback.startswith("http://127.0.0.1:"):
        raise RuntimeError("refusing CDP on a non-loopback handle")
    import asyncio
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from tools.browser_cdp_tool import _cdp_call

    import time

    base = handle.cdp_loopback.rstrip("/")
    version = None
    last_err = None
    for _ in range(20):
        try:
            with urllib.request.urlopen(f"{base}/json/version", timeout=5) as resp:
                version = json.load(resp)
            break
        except Exception as exc:
            last_err = exc
            time.sleep(0.15)
    if version is None:
        raise RuntimeError(f"chromium loopback CDP not ready: {last_err}")
    ws_url = version.get("webSocketDebuggerUrl")
    if not ws_url or not str(ws_url).startswith(("ws://127.0.0.1:", "ws://localhost:")):
        raise RuntimeError("chromium did not expose a loopback websocket")
    target_id = None
    try:
        with urllib.request.urlopen(f"{base}/json/list", timeout=5) as resp:
            pages = json.load(resp)
        if isinstance(pages, list):
            page = next((p for p in pages if p.get("type") == "page"), None)
            if page:
                target_id = page.get("id")
    except Exception:
        target_id = None

    async def _run() -> dict[str, Any]:
        return await _cdp_call(ws_url, method, params or {}, target_id, 8.0)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_run())).result(timeout=12)


def new_identity_profile_dir(root: Path, identity_id: str) -> str:
    """Durable managed dir. Never the OS default Chromium user-data-dir."""
    path = root / "identities" / identity_id
    path.mkdir(parents=True, exist_ok=True)
    (path / ".hermes-identity").write_text(identity_id, encoding="utf-8")
    return str(path)
