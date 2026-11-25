"""A recording computer_use backend and a cache reset, for tests that drive ``tools.computer_use.tool``."""
from typing import Any

from tools.computer_use import tool
from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend


def reset_backend() -> None:
    """Stop every cached backend and forget per-session capture state."""
    tool._shutdown_backend_atexit()
    tool._reset_screenshot_dedup()


def noop_stub(name: str, *params: str, result: Any = None) -> Any:
    # Positional args are folded in under *params* (declared params default to None). ``result`` may be a factory
    # of the recorded kwargs; None -> a trivial ok ActionResult.
    def method(self, *pos, **kw):
        self.calls.append((name, call := {**dict.fromkeys(params), **dict(zip(params, pos)), **kw}))
        return result(call) if callable(result) else ActionResult(ok=True, action=name) if result is None else result
    return method


class NoopBackend(ComputerUseBackend):
    """Records ``(name, kwargs)`` calls; returns trivial results. Tests patch ``tool._new_backend`` to return it."""

    def __init__(self) -> None: self.calls: list[tuple[str, dict[str, Any]]] = []
    start = stop = lambda self: None
    def is_available(self) -> bool: return True

    capture = noop_stub("capture", "mode", "app", "pid", "window_id", result=lambda kw: CaptureResult(
        mode=kw["mode"] or "som", width=1024, height=768, png_b64=None, elements=[], app=kw["app"] or "", window_title=""))
    click, drag, scroll = noop_stub("click"), noop_stub("drag"), noop_stub("scroll")
    type_text, key, set_value = noop_stub("type", "text"), noop_stub("key", "keys"), noop_stub("set_value", "value", "element")
    list_apps, list_windows = noop_stub("list_apps", result=[]), noop_stub("list_windows", result=[])
    focus_app = noop_stub("focus_app", "app", "raise_window")
