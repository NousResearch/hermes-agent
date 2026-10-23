"""Admission lifecycle hooks turn plugin policy into a stable refusal."""

from hermes_cli import lifecycle
from hermes_cli.plugins import VALID_HOOKS


def test_admission_hooks_are_public_plugin_lifecycle_hooks():
    """Plugin manifests and shell hooks can register both live admission seams."""
    assert {"pre_turn_admission", "pre_session_resume"} <= VALID_HOOKS


def test_admission_refusal_accepts_the_existing_block_directive(monkeypatch):
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda name, **kwargs: [{"action": "block", "message": "policy refused"}],
    )

    assert lifecycle.admission_refusal("pre_turn_admission", session_id="s1") == "policy refused"


def test_admission_refusal_accepts_structured_failed_result(monkeypatch):
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda name, **kwargs: [{"failed": True, "error": "wrong session kind"}],
    )

    assert lifecycle.admission_refusal("pre_session_resume", session_id="s1") == "wrong session kind"


def test_admission_refusal_fails_closed_for_a_false_result(monkeypatch):
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *_a, **_kw: [False])

    assert lifecycle.admission_refusal("pre_turn_admission", session_id="s1") == (
        "pre_turn_admission refused the request")
