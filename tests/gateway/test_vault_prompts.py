"""The gateway can ask the user for a vault secret, and the answer stays out of history.

These pin the contract the vault tools depend on: three callbacks registered on the
thread that runs the tools, each returning the shape its caller documents, and a
"can anyone answer here?" probe that reflects whether a surface actually installed
a prompt.
"""

import pytest

from agent.vault_backends import unlock as vault_unlock
from gateway.vault_prompts import (
    build_vault_prompt_callbacks,
    clear_vault_prompt_callbacks,
    install_vault_prompt_callbacks,
)
from tools import clarify_gateway


@pytest.fixture(autouse=True)
def _clean_prompt_registration():
    """Each test starts and ends with no surface prompts, on its own thread."""
    clear_vault_prompt_callbacks()
    yield
    clear_vault_prompt_callbacks()


class _Recorder:
    """Stand-in for the blocking ask: records questions, replays canned answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.questions = []
        self.secret_flags = []
        self.last_flags = []

    def __call__(self, question, secret=False, last=True):
        self.questions.append(question)
        self.secret_flags.append(secret)
        self.last_flags.append(last)
        return self.answers.pop(0) if self.answers else ""


def test_install_registers_every_callback_the_vault_tools_read():
    install_vault_prompt_callbacks(_Recorder("x"))
    assert vault_unlock.get_unlock_prompt_callback() is not None
    assert vault_unlock.get_save_login_prompt_callback() is not None
    assert vault_unlock.get_code_prompt_callback() is not None


def test_install_is_visible_to_can_prompt_here():
    assert vault_unlock.can_prompt_here() is False
    install_vault_prompt_callbacks(_Recorder("x"))
    assert vault_unlock.can_prompt_here() is True


def test_any_single_callback_is_enough_to_answer():
    """A surface offering only one vault prompt still counts as a place a human can
    answer — the tools call this next to their own callback check, so coupling it to
    the unlock callback alone would refuse a surface that never offers an unlock."""
    vault_unlock.set_code_prompt_callback(lambda site, hint: "123456")
    assert vault_unlock.can_prompt_here() is True


def test_clear_removes_every_callback():
    install_vault_prompt_callbacks(_Recorder("x"))
    clear_vault_prompt_callbacks()
    assert vault_unlock.get_unlock_prompt_callback() is None
    assert vault_unlock.get_save_login_prompt_callback() is None
    assert vault_unlock.get_code_prompt_callback() is None
    assert vault_unlock.can_prompt_here() is False


def test_unlock_prompt_returns_the_master_password():
    ask = _Recorder("  hunter2  ")
    callbacks = build_vault_prompt_callbacks(ask)
    assert callbacks["unlock"]("onepassword", "1Password") == "hunter2"
    assert ask.secret_flags == [True], "a master password must be marked secret"


def test_unlock_prompt_blank_answer_reads_as_cancelled():
    callbacks = build_vault_prompt_callbacks(_Recorder("   "))
    assert callbacks["unlock"]("onepassword", "1Password") == ""


def test_save_login_asks_for_identifier_then_password():
    ask = _Recorder("someone@example.com", "s3cret")
    callbacks = build_vault_prompt_callbacks(ask)
    result = callbacks["save_login"]("https://example.com", "example.com")
    assert result == {"identifier": "someone@example.com", "password": "s3cret"}
    assert len(ask.questions) == 2
    # The identifier is not a secret; the password absolutely is.
    assert ask.secret_flags == [False, True]
    # Only the final question may re-arm the stream: a mid-sequence re-arm opens a
    # bubble the next card immediately closes (the clarify-batch contract).
    assert ask.last_flags == [False, True]
    # Both cards must announce the sequence, or the second reads as the first repeating.
    assert "(1/2)" in ask.questions[0] and "(2/2)" in ask.questions[1]


def test_save_login_declines_without_an_identifier():
    ask = _Recorder("")
    callbacks = build_vault_prompt_callbacks(ask)
    assert callbacks["save_login"]("https://example.com", "example.com") is None
    assert len(ask.questions) == 1, "must not ask for a password with nothing to bind it to"


def test_save_login_declines_without_a_password():
    callbacks = build_vault_prompt_callbacks(_Recorder("someone@example.com", ""))
    assert callbacks["save_login"]("https://example.com", "example.com") is None


def test_code_prompt_returns_the_code_and_marks_it_secret():
    ask = _Recorder(" 123456 ")
    callbacks = build_vault_prompt_callbacks(ask)
    assert callbacks["code"]("example.com", "") == "123456"
    assert ask.secret_flags == [True]


def test_register_marks_a_secret_prompt_and_defaults_to_public():
    secret_entry = clarify_gateway.register("cid-secret", "sess", "?", None, secret=True)
    public_entry = clarify_gateway.register("cid-public", "sess", "?", None)
    try:
        assert secret_entry.secret is True
        assert public_entry.secret is False
    finally:
        clarify_gateway.resolve_gateway_clarify("cid-secret", "x")
        clarify_gateway.resolve_gateway_clarify("cid-public", "x")


def test_single_question_prompts_are_always_the_last_question():
    """unlock/code ask once: they must re-arm, or the continuation never reopens."""
    for name, args in (("unlock", ("onepassword", "1Password")), ("code", ("example.com", "digits"))):
        ask = _Recorder("value")
        callbacks = build_vault_prompt_callbacks(ask)
        callbacks[name](*args)
        assert ask.last_flags == [True], f"{name} must be a final question"
        assert ask.secret_flags == [True], f"{name} answer is a secret"


def test_vault_ask_holds_the_rearm_for_non_final_questions():
    """The runner seam must translate `last` into `rearm`, the way the batch does."""
    from gateway.run_turn_runner import TurnRunner
    calls = []
    runner = object.__new__(TurnRunner)
    runner._ask_clarify_question = lambda q, c, m, rearm=True, secret=False: (
        calls.append({"rearm": rearm, "secret": secret}) or ("answer", True))
    assert runner._vault_ask_sync("q1", secret=False, last=False) == "answer"
    assert runner._vault_ask_sync("q2", secret=True, last=True) == "answer"
    assert calls == [{"rearm": False, "secret": False}, {"rearm": True, "secret": True}]


def test_vault_ask_returns_blank_when_unanswered():
    """A blank return is what every vault callback reads as a decline."""
    from gateway.run_turn_runner import TurnRunner
    runner = object.__new__(TurnRunner)
    runner._ask_clarify_question = lambda q, c, m, rearm=True, secret=False: ("notice", False)
    assert runner._vault_ask_sync("q") == ""
