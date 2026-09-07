"""Temporary sessions are invisible to observe-only plugin hooks.

Plugins are arbitrary code the core cannot audit, and two bundled examples
alone would durably record a temporary chat: disk-cleanup writes its
tracked.json registry from ``post_tool_call``, and nemo_relay exports a full
ATIF trajectory file from ``on_session_end``/``finalize``/``reset``. The
contract, enforced at the single dispatch point
(``hermes_cli.plugins.invoke_hook``, which ``hermes_cli.lifecycle`` routes
through):

* observe-only hooks (returns never consumed by the core) are suppressed
  entirely for a registered temporary session;
* hooks the core consumes (pre_llm_call context injection, pre_tool_call
  blocking, ...) still fire, stamped ``ephemeral=True`` so a well-behaved
  plugin can decline to record what it sees;
* deliveries without a ``session_id`` cannot be classified and pass through
  unchanged — every content-bearing hook in the core carries one.
"""

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_state import mark_session_ephemeral, unmark_session_ephemeral

TEMP_SID = "ephemeral-hook-probe-sid"
NORMAL_SID = "normal-hook-probe-sid"


@pytest.fixture()
def probe():
    """Register a recording callback for a hook name; always deregisters."""
    manager = plugins_mod.get_plugin_manager()
    registered = []
    calls = []

    def _register(hook_name):
        def _cb(**kwargs):
            calls.append((hook_name, kwargs))
            return "probe-result"

        manager._hooks.setdefault(hook_name, []).append(_cb)
        registered.append((hook_name, _cb))
        return calls

    yield _register, calls
    for hook_name, cb in registered:
        try:
            manager._hooks.get(hook_name, []).remove(cb)
        except ValueError:
            pass


@pytest.fixture()
def temp_session():
    mark_session_ephemeral(TEMP_SID)
    yield TEMP_SID
    unmark_session_ephemeral(TEMP_SID)


@pytest.mark.parametrize("hook_name", sorted(plugins_mod._EPHEMERAL_SUPPRESSED_HOOKS))
def test_observe_hooks_are_suppressed_for_temporary_sessions(
    hook_name, probe, temp_session
):
    register, calls = probe
    register(hook_name)
    results = plugins_mod.invoke_hook(hook_name, session_id=temp_session)
    assert results == []
    assert not calls, (
        f"{hook_name} was delivered for a temporary session — observe-only "
        "hooks must not see a chat the user was promised leaves no trace"
    )


@pytest.mark.parametrize("hook_name", sorted(plugins_mod._EPHEMERAL_SUPPRESSED_HOOKS))
def test_observe_hooks_still_fire_for_normal_sessions(hook_name, probe):
    register, calls = probe
    register(hook_name)
    results = plugins_mod.invoke_hook(hook_name, session_id=NORMAL_SID)
    assert results == ["probe-result"]
    assert len(calls) == 1
    assert "ephemeral" not in calls[0][1]


def test_functional_hooks_fire_with_the_ephemeral_stamp(probe, temp_session):
    register, calls = probe
    register("pre_llm_call")
    results = plugins_mod.invoke_hook(
        "pre_llm_call", session_id=temp_session, user_message="PROBE"
    )
    assert results == ["probe-result"], (
        "functional hooks must keep firing — a temporary chat that loses "
        "plugin context injection or tool blocking is degraded, not private"
    )
    assert calls[0][1].get("ephemeral") is True, (
        "kept deliveries must carry ephemeral=True so plugins can decline "
        "to record what they see"
    )


def test_functional_hooks_are_unstamped_for_normal_sessions(probe):
    register, calls = probe
    register("pre_llm_call")
    plugins_mod.invoke_hook("pre_llm_call", session_id=NORMAL_SID)
    assert "ephemeral" not in calls[0][1]


def test_delivery_without_session_id_passes_through(probe):
    # No session context means no classification: fail open. Every
    # content-bearing hook in the core passes a session_id.
    register, calls = probe
    register("on_session_end")
    results = plugins_mod.invoke_hook("on_session_end", completed=True)
    assert results == ["probe-result"]


def test_lifecycle_wrapper_applies_the_same_contract(probe, temp_session):
    from hermes_cli.lifecycle import invoke_hook as lifecycle_invoke_hook

    register, calls = probe
    register("on_session_end")
    assert lifecycle_invoke_hook("on_session_end", session_id=temp_session) == []
    assert not calls


def test_bundled_writer_registrations_stay_inside_the_suppressed_set():
    """The two known bundled disk-writers must remain fully covered.

    disk-cleanup records file paths from post_tool_call and writes its
    cleanup log from on_session_end; nemo_relay exports ATIF trajectory
    files from every session-boundary hook it registers. If one of these
    hook names leaves the suppressed set, a temporary chat becomes durable
    again through a plugin.
    """
    suppressed = plugins_mod._EPHEMERAL_SUPPRESSED_HOOKS
    disk_cleanup_writers = {"post_tool_call", "on_session_end"}
    nemo_export_paths = {"on_session_end", "on_session_finalize", "on_session_reset"}
    assert disk_cleanup_writers <= suppressed
    assert nemo_export_paths <= suppressed
