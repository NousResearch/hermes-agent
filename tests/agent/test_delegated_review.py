"""Reusable children review skills after teardown without memory access or retained runtimes.

The fake provider boundary blocks on events; lifecycle, queueing, whitelist dispatch,
and cancellation are real. These catch the original close-vs-review race and the
lost cancellation owner when a durable child is reconstructed.
"""

import gc
import threading
import weakref
from types import SimpleNamespace

import pytest

from agent import delegated_review as delegated
from agent import review_idle_queue
from run_agent import AIAgent


class _Fork:
    def __init__(self):
        self.started = threading.Event()
        self.finish = threading.Event()
        self.released = threading.Event()
        self.interrupted = threading.Event()
        self._session_messages = []
        self._memory_enabled = self._user_profile_enabled = True
        self.allowed = {}
        self.history = None

    def run_conversation(self, *, user_message, conversation_history):
        from hermes_cli.plugins import get_pre_tool_call_block_message
        from hermes_constants import get_hermes_home
        from agent.secret_scope import current_secret_scope
        self.profile_home = get_hermes_home()
        self.secret_scope = current_secret_scope()
        self.history = conversation_history
        self.allowed = {name: get_pre_tool_call_block_message(name, {}) is None
                        for name in ("skill_manage", "memory", "propose_shared_memory", "terminal")}
        self.started.set()
        assert self.finish.wait(10), "test did not release review request"
        self._session_messages = [*conversation_history, {"role": "assistant", "content": "reviewed"}]

    def interrupt(self, message=None):
        self.interrupted.set()
        self.finish.set()

    def close(self):
        pytest.fail("ordinary child teardown must not close a running reviewer")

    def release_clients(self):
        self.released.set()


def _child(tmp_path):
    child = object.__new__(AIAgent)
    child.session_id = "compressed-child-tip"
    child._delegate_session_id = "stable-child"
    child._delegate_depth = 1
    child._delegate_resumable = True
    child._delegate_review_skills = True
    child._session_db = SimpleNamespace(db_path=tmp_path / "state.db")
    child._active_children = []
    child._active_children_lock = threading.Lock()
    return child


@pytest.fixture
def setup_review(monkeypatch):
    fork = _Fork()
    monkeypatch.setattr(delegated.review, "load_background_review_settings",
                        lambda: (True, {"defer": "never", "extra_tools": ["memory", "propose_shared_memory"]}))
    monkeypatch.setattr(delegated.review, "build_cache_parity_fork", lambda *a, **k: (fork, {}, False))
    yield fork
    fork.finish.set()
    for owner in list(delegated._owners.values()):
        owner._discard_background_review()


def test_review_survives_child_teardown_and_is_cancelled_by_stable_session(tmp_path, setup_review):
    fork = setup_review
    child = _child(tmp_path)
    history = [{"role": "user", "content": "implement workflow"}]
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    home = tmp_path / "served-profile"
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({"TEST_REVIEW_SECRET": "served"}, profile_home=str(home))
    try:
        child._spawn_background_review(history, review_memory=True, review_skills=True)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
    assert not fork.started.is_set()  # only snapshot capture during the live turn
    delegated.finish_delegated_review(child)
    try:
        assert fork.started.wait(10)
        assert fork.profile_home == home
        assert fork.secret_scope == {"TEST_REVIEW_SECRET": "served"}
        child._close_active_children(soft=False)
        child_ref = weakref.ref(child)
        del child
        gc.collect()
        assert child_ref() is None, "idle review must not retain the retired AIAgent"
        assert not fork.released.is_set()
        assert fork.allowed == {"skill_manage": True, "memory": False,
                                "propose_shared_memory": False, "terminal": False}
        assert history == [{"role": "user", "content": "implement workflow"}]
        assert fork.history == history
        # Same public ID under another profile/database has no cancellation authority.
        delegated.cancel_delegated_review("stable-child", db_path=tmp_path / "other.db")
        assert not fork.interrupted.is_set()
        delegated.cancel_delegated_review("stable-child", db_path=tmp_path / "state.db")
        assert fork.interrupted.wait(10)
        assert fork.released.wait(10)
    finally:
        fork.finish.set()


@pytest.mark.parametrize("reason", ["resume", "disabled"])
def test_deferred_review_releases_prepared_fork_without_provider_call(
    tmp_path, monkeypatch, setup_review, reason,
):
    fork = setup_review
    queue = review_idle_queue.ReviewIdleQueue()
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    monkeypatch.setattr(review_idle_queue, "QUEUE", queue)
    monkeypatch.setattr(review_idle_queue, "review_targets_managed_local", lambda *a: True)
    monkeypatch.setattr(delegated.review, "load_background_review_settings", lambda: (True, {}))
    child = _child(tmp_path)
    child._spawn_background_review([], review_skills=True)
    delegated.finish_delegated_review(child)
    assert queue.pending_count() == 1
    if reason == "resume":
        delegated.cancel_delegated_review("stable-child", db_path=tmp_path / "state.db")
        assert queue.pending_count() == 0
    else:
        monkeypatch.setattr(queue, "_quiet_for", lambda: 100)
        monkeypatch.setattr(queue, "_server_idle", lambda: True)
        item = queue._pop_dispatchable()
        monkeypatch.setattr(delegated.review, "load_background_review_settings", lambda: (False, {}))
        queue._dispatch(item)
    assert fork.released.wait(10)
    assert not fork.started.is_set()
    assert delegated._key("stable-child", tmp_path / "state.db") not in delegated._owners


def test_memory_nudge_alone_does_not_review_a_reusable_child(tmp_path, setup_review):
    child = _child(tmp_path)
    child._spawn_background_review([], review_memory=True, review_skills=False)
    delegated.finish_delegated_review(child)
    assert not setup_review.started.is_set()


def test_explicit_refine_keeps_upstream_manual_path_for_retained_child(tmp_path, monkeypatch):
    """Extension policy must not capture /refine or apply automatic-review gates to it."""
    child = _child(tmp_path)
    spawned = []
    monkeypatch.setattr(delegated.review, "load_background_review_settings", lambda: (False, {}))
    monkeypatch.setattr(child, "_spawn_background_review_now", lambda **kwargs: spawned.append(kwargs))
    history = [{"role": "user", "content": "Review this work explicitly"}]

    child._spawn_background_review(history, review_memory=True, review_skills=True, explicit=True)

    assert len(spawned) == 1
    assert spawned[0]["explicit"] and spawned[0]["review_memory"] and spawned[0]["review_skills"]
    assert spawned[0]["messages_snapshot"] == history
    assert not hasattr(child, "_delegate_pending_review")
