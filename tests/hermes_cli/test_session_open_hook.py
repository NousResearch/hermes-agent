"""Contract tests for the ``on_session_open`` host-open lifecycle hook (#89385).

``on_session_open`` fires exactly once when a session becomes live and
addressable (CLI interactive start, TUI/desktop ``session.create`` /
``session.resume``), BEFORE the first model turn — the seam memory providers,
peer registries and external status consumers need. Distinct from
``on_session_start``, which fires per agent turn.

Guarantees pinned here:
- once per active ``(platform, session_id)`` lifecycle (idempotent re-fires no-op)
- empty/blank session ids fail closed
- finalize/reset release the id so the same durable session can later resume
  and re-fire
- the LRU cap bounds the dedup set
- hook errors do not lose the dedup registration's release accounting
"""
import threading

import pytest

from hermes_cli import plugins
from hermes_cli.plugins import notify_session_open


@pytest.fixture(autouse=True)
def _reset_session_open_state():
    """Fresh dedup state per test (module-level mutable state)."""
    plugins._session_open_seen.clear()
    yield
    plugins._session_open_seen.clear()


@pytest.fixture()
def hook_calls(monkeypatch):
    """Capture ``invoke_hook`` calls made by ``notify_session_open`` while the
    real dispatcher (and its finalize/reset release path) still runs."""
    calls: list[tuple] = []
    real_invoke = plugins.invoke_hook

    def _capture(hook_name, **kwargs):
        calls.append((hook_name, kwargs))
        return real_invoke(hook_name, **kwargs)

    monkeypatch.setattr(plugins, "invoke_hook", _capture)
    return calls


class TestSessionOpenHook:
    def test_fires_once_per_active_session(self, hook_calls):
        assert notify_session_open("sess-1", "cli") is True
        assert notify_session_open("sess-1", "cli") is False  # idempotent no-op
        assert len(hook_calls) == 1
        hook_name, kwargs = hook_calls[0]
        assert hook_name == "on_session_open"
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["platform"] == "cli"

    def test_platform_scoped_independence(self, hook_calls):
        """The same id on a different host surface is a distinct lifecycle."""
        assert notify_session_open("sess-1", "cli") is True
        assert notify_session_open("sess-1", "tui") is True
        assert len(hook_calls) == 2

    def test_blank_session_id_fails_closed(self, hook_calls):
        assert notify_session_open("", "cli") is False
        assert notify_session_open("   ", "cli") is False
        assert notify_session_open(None, "cli") is False
        assert hook_calls == []

    def test_blank_platform_defaults_to_unknown(self, hook_calls):
        assert notify_session_open("sess-1", "") is True
        assert hook_calls[0][1]["platform"] == "unknown"

    def test_finalize_releases_the_id_for_resume(self, hook_calls):
        assert notify_session_open("sess-1", "cli") is True
        plugins.invoke_hook("on_session_finalize", session_id="sess-1", platform="cli")
        # The released id can open again (resume of a closed durable session).
        assert notify_session_open("sess-1", "cli") is True
        # The second call is the release dispatch itself (on_session_finalize),
        # then the resumed open fires again.
        names = [name for name, _ in hook_calls]
        assert names == ["on_session_open", "on_session_finalize", "on_session_open"]

    def test_finalize_only_releases_matching_id(self, hook_calls):
        assert notify_session_open("sess-1", "cli") is True
        assert notify_session_open("sess-2", "cli") is True
        plugins.invoke_hook("on_session_finalize", session_id="sess-1", platform="cli")
        # sess-2 stays active; sess-1 was released.
        assert notify_session_open("sess-2", "cli") is False
        assert notify_session_open("sess-1", "cli") is True

    def test_reset_without_old_session_id_is_a_noop_release(self, hook_calls):
        """CLI reset context carries the NEW id; release must not clear it."""
        assert notify_session_open("sess-1", "cli") is True
        # Vanilla on_session_reset kwarg set has no old_session_id -> release(None) no-ops.
        plugins.invoke_hook("on_session_reset", session_id="other", platform="cli")
        assert notify_session_open("sess-1", "cli") is False

    def test_hook_error_preserves_release_accounting(self, monkeypatch):
        """A failing hook removes the dedup key so a retry can re-fire."""
        state = {"fail": True}
        attempts: list[int] = []

        def _fail_once(hook_name, **kwargs):
            if state["fail"]:
                attempts.append(1)
                state["fail"] = False
                raise RuntimeError("plugin exploded")
            return []

        monkeypatch.setattr(plugins, "invoke_hook", _fail_once)
        with pytest.raises(RuntimeError):
            notify_session_open("sess-1", "cli")
        assert len(attempts) == 1
        # Key released on the failure path: a retry fires again instead of
        # silently no-opping.
        assert notify_session_open("sess-1", "cli") is True

    def test_dedup_set_is_bounded(self, monkeypatch, hook_calls):
        plugins._SESSION_OPEN_LIMIT = 3
        try:
            for i in range(5):
                assert notify_session_open(f"sess-{i}", "cli") is True
            # Oldest keys evicted: sess-0 and sess-1 may fire again; sess-4 still held.
            assert notify_session_open("sess-4", "cli") is False
            assert notify_session_open("sess-0", "cli") is True
        finally:
            plugins._SESSION_OPEN_LIMIT = 4096

    def test_concurrent_first_fire_wins_exactly_once(self, hook_calls):
        """Parallel open of the same session: exactly one hook emission."""
        barrier = threading.Barrier(8)
        results: list[bool] = []

        def _open():
            barrier.wait()
            results.append(notify_session_open("sess-race", "cli"))

        threads = [threading.Thread(target=_open) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count(True) == 1
        assert len(hook_calls) == 1