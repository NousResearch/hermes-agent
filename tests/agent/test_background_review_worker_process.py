"""A managed worker process never forks the post-turn review.

The worker's interpreter exits after its frames, taking a daemon review thread with it, and its
``RuntimeSessionStore`` is a receipt-based persistence facade, not the owner's SessionDB: the
same precedent as a finite gateway turn (``load_background_review_settings`` returns disabled
before any config read). The owner process keeps learning from its own turns as before.
"""

from agent import runtime_session_store
from agent.background_review import load_background_review_settings


def test_worker_process_disables_the_automatic_review(monkeypatch):
    monkeypatch.setattr(runtime_session_store, "_worker_process", True)
    assert load_background_review_settings() == (False, {})


def test_owner_process_keeps_the_automatic_review(monkeypatch):
    monkeypatch.setattr(runtime_session_store, "_worker_process", False)
    enabled, task_cfg = load_background_review_settings()
    assert enabled is True and isinstance(task_cfg, dict)
