"""Contract cases for the prompt-refresh epochs on ``SessionRepository``: one counter orders profile-wide and per-session requests, a session reads the newer of the two, and a compression child keeps the epoch its parent's prompt was built at."""

from __future__ import annotations

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession


def test_no_request_reads_epoch_zero(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("pr-none", "cli"))
    assert store.sessions.prompt_refresh_epoch("pr-none") == 0
    assert store.sessions.prompt_refresh_epoch("pr-absent") == 0


def test_one_counter_orders_profile_and_session_requests(backend, scope_a):
    store = backend.open(scope_a)
    for sid in ("pr-a", "pr-b"):
        store.sessions.create(NewSession(sid, "cli"))
    assert store.sessions.request_prompt_refresh() == 1
    assert (store.sessions.prompt_refresh_epoch("pr-a"), store.sessions.prompt_refresh_epoch("pr-b")) == (1, 1)
    # A session request takes the next epoch of the shared counter and applies to that session only.
    assert store.sessions.request_prompt_refresh("pr-a") == 2
    assert (store.sessions.prompt_refresh_epoch("pr-a"), store.sessions.prompt_refresh_epoch("pr-b")) == (2, 1)
    # The next profile request is newer than every session request.
    assert store.sessions.request_prompt_refresh() == 3
    assert (store.sessions.prompt_refresh_epoch("pr-a"), store.sessions.prompt_refresh_epoch("pr-b")) == (3, 3)
    # A session request counts past the session epochs too, and a session never seen reads the profile epoch.
    assert store.sessions.request_prompt_refresh("pr-b") == 4
    assert store.sessions.request_prompt_refresh("pr-b") == 5
    assert (store.sessions.prompt_refresh_epoch("pr-a"), store.sessions.prompt_refresh_epoch("pr-b")) == (3, 5)
    assert store.sessions.prompt_refresh_epoch("pr-absent") == 3


def test_epochs_are_per_profile(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(NewSession("pr-iso", "cli"))
    store_b.sessions.create(NewSession("pr-iso", "cli"))
    store_a.sessions.request_prompt_refresh()
    store_a.sessions.request_prompt_refresh("pr-iso")
    assert store_b.sessions.prompt_refresh_epoch("pr-iso") == 0
    assert store_b.sessions.request_prompt_refresh() == 1
    assert store_a.sessions.prompt_refresh_epoch("pr-iso") == 2


def test_count_with_stored_prompt_counts_sessions_a_refresh_rebuilds(backend, scope_a, scope_b):
    store = backend.open(scope_a)
    assert store.sessions.count_with_stored_prompt() == 0
    store.sessions.create(NewSession("pr-created", "cli", system_prompt="created prompt"))
    store.sessions.create(NewSession("pr-updated", "cli"))
    store.sessions.create(NewSession("pr-bare", "cli"))
    store.sessions.update_system_prompt("pr-updated", "updated prompt")
    assert store.sessions.count_with_stored_prompt() == 2
    store.sessions.update_system_prompt("pr-updated", None)
    assert store.sessions.count_with_stored_prompt() == 1
    assert backend.open(scope_b).sessions.count_with_stored_prompt() == 0


def test_a_compression_child_keeps_the_parents_prompt_refresh_epoch(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("pr-root", "cli", model_config={"keep": 1}))
    store.sessions.patch_model_config("pr-root", {"prompt_refresh_epoch": 3})
    store.lineage.publish_compression_child(
        "pr-root", CompressionChild("pr-tip", "cli"), [NewMessage("user", "summary", timestamp=1.0)],
        require_compression_lease=False,
    )
    assert store.sessions.get("pr-tip").model_config == {"prompt_refresh_epoch": 3}
