"""One module's view of ``time`` with a test-driven ``monotonic``.

Patching ``time.monotonic`` itself also moves the running event loop's clock (its timers read the
same function); this swaps only the module's ``time`` binding, so the loop keeps real time."""
import time


class ModuleClock:
    """``time`` for one module: ``monotonic()`` returns ``now``, advanced by ``step`` per read."""

    def __init__(self, now: float, step: float):
        self.now, self.step = now, step

    def monotonic(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def __getattr__(self, name):
        return getattr(time, name)


def install_module_clock(monkeypatch, module, *, now: float = 1000.0, step: float = 0.0) -> ModuleClock:
    clock = ModuleClock(now, step)
    monkeypatch.setattr(module, "time", clock)
    return clock
