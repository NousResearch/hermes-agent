"""Exit-code contract shared by the ``-q`` and ``-Q`` single-query paths.

Regression cover for the kanban incident (card t_4081a982): the ``-q`` branch had no
``sys.exit()`` at all, so a worker killed by a provider quota wall exited rc=0 with its
card still ``running``. The dispatcher read that as a clean exit that never touched its
task -> protocol violation -> failure counter -> circuit breaker blocked the card, while
the correct ``rate_limited`` requeue path (no failure counted) was dead code for every
normal worker. ``-Q`` already emitted the ``KANBAN_RATE_LIMIT_EXIT_CODE`` sentinel; these
tests pin that BOTH branches now route through one helper so they cannot drift again.
"""

from types import SimpleNamespace

import pytest

import cli as cli_mod
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


# ── a) helper truth table ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "result, kanban_task, expected",
    [
        # Nothing to report -> success. ``chat()`` returns None on its early-exit paths.
        (None, None, 0),
        ({}, None, 0),
        # ``failed`` is the only gate: a rate-limit note on a turn that COMPLETED is not
        # a failure and must never become the requeue sentinel.
        ({"failed": False, "failure_reason": "rate_limit"}, "t_x", 0),
        # Generic failure -> the plain 1 automation wrappers expect.
        ({"failed": True}, None, 1),
        ({"failed": True}, "t_x", 1),
        # Quota failures only become the sentinel INSIDE a kanban worker; a plain CLI
        # run/cron/wrapper keeps the 0/1 contract.
        ({"failed": True, "failure_reason": "rate_limit"}, None, 1),
        ({"failed": True, "failure_reason": "billing"}, None, 1),
        ({"failed": True, "failure_reason": "rate_limit"}, "t_x", KANBAN_RATE_LIMIT_EXIT_CODE),
        ({"failed": True, "failure_reason": "billing"}, "t_x", KANBAN_RATE_LIMIT_EXIT_CODE),
        # A real task failure inside a kanban worker still counts as a failure.
        ({"failed": True, "failure_reason": "tool_error"}, "t_x", 1),
    ],
)
def test_single_query_exit_code_truth_table(monkeypatch, result, kanban_task, expected):
    if kanban_task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task)

    assert cli_mod._single_query_exit_code(result) == expected


def test_single_query_exit_code_sentinel_is_ex_tempfail():
    """The sentinel must stay EX_TEMPFAIL — the dispatcher's classifier keys on it."""
    assert KANBAN_RATE_LIMIT_EXIT_CODE == 75


def test_single_query_exit_code_ignores_non_dict_results(monkeypatch):
    """``chat()`` returns a str; a non-dict must never be read as a failure."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    assert cli_mod._single_query_exit_code("done") == 0
    assert cli_mod._single_query_exit_code(object()) == 0


# ── b) chat() exposes the turn outcome ──────────────────────────────────────

def _make_chat_cli(run_agent, *, credentials=True):
    """Minimal object driving the REAL ``CLIChatTurnMixin.chat`` control flow.

    Only the per-turn phase helpers are stubbed; the ``turn`` lifecycle, the
    try/except/finally around them and the stash under test are the real code.
    """

    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

    class FakeChatCLI(CLIChatTurnMixin):
        def __init__(self):
            self.agent = SimpleNamespace(name="agent")
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self._secret_capture_callback = lambda *_a, **_kw: None
            self._last_turn_interrupted = False
            # Deliberately non-None so a test can only pass if chat() wrote it.
            self._last_turn_result: object = "STALE-SENTINEL"
            self.rendered = None

        # -- pre-turn phases --
        def _ensure_runtime_credentials(self):
            return credentials

        def _resolve_turn_agent_config(self, message):
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **_kwargs):
            return True

        def _chat_route_images(self, message, images):
            return message

        def _chat_expand_context_references(self, message):
            return message, None

        def _chat_stage_user_message(self, agent, message):
            self.conversation_history.append({"role": "user", "content": message})

        # -- turn phases --
        def _reset_stream_state(self):
            pass

        def _chat_setup_turn_audio(self, turn, message, voice_input):
            pass

        def _chat_run_agent(self, turn, message):
            run_agent(turn, message)

        def _chat_monitor_agent_thread(self, turn, agent_thread):
            agent_thread.join(timeout=10)
            return None

        def _chat_settle_turn(self, turn):
            pass

        def _chat_render_turn(self, turn, agent_thread, interrupt_msg):
            self.rendered = turn.result
            return "rendered-response"

        def _chat_release_turn_audio(self, turn):
            pass

    return FakeChatCLI()


def test_chat_stashes_failed_turn_result():
    """The dict ``run_conversation`` returned is reachable from outside ``chat()``."""
    failed = {"final_response": "", "failed": True, "failure_reason": "rate_limit",
              "error": "429 quota exhausted"}

    def run_agent(turn, _message):
        turn.result = failed

    cli = _make_chat_cli(run_agent)
    cli.chat("hello")

    assert cli._last_turn_result is failed


def test_chat_stashes_exception_path_result():
    """``_chat_run_agent``'s ``except`` fallback dict must land in the stash too.

    This mirrors the dict the real handler builds when ``run_conversation`` raises —
    that is exactly the shape a provider 5xx/quota wall produces, so if only the happy
    path were stashed the incident would still exit 0.
    """
    except_dict = {"final_response": "Error: 502 Bad Gateway", "messages": [],
                   "api_calls": 0, "completed": False, "failed": True,
                   "error": "502 Bad Gateway"}

    def run_agent(turn, _message):
        turn.result = except_dict

    cli = _make_chat_cli(run_agent)
    cli.chat("hello")

    assert cli._last_turn_result is except_dict


def test_chat_stashes_successful_turn_result():
    ok = {"final_response": "hi", "completed": True, "failed": False}

    def run_agent(turn, _message):
        turn.result = ok

    cli = _make_chat_cli(run_agent)
    cli.chat("hello")

    assert cli._last_turn_result is ok
    assert cli_mod._single_query_exit_code(cli._last_turn_result) == 0


def test_chat_resets_stash_before_the_turn_runs():
    """A later turn must never inherit an earlier turn's failure.

    ``chat()`` returns early (credentials refused) BEFORE a turn object exists, so the
    only thing that can clear a previous failure is the reset at the top of ``chat()``.
    Without it the next single-query run would exit non-zero on a turn that never ran.
    """
    cli = _make_chat_cli(lambda turn, _m: None, credentials=False)
    cli._last_turn_result = {"failed": True, "failure_reason": "rate_limit"}

    assert cli.chat("hello") is None
    assert cli._last_turn_result is None


def test_chat_stash_initialized_by_ui_state():
    """``-q`` never goes through ``run()``; the attribute must exist from init."""
    fake = SimpleNamespace()
    cli_mod.HermesCLI._init_ui_state(fake)  # type: ignore[arg-type]

    assert fake._last_turn_result is None


# ── shared fakes for the main() branch tests ────────────────────────────────

def _install_common_main_stubs(monkeypatch, calls):
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )


def _make_human_fake_cli(calls, turn_result):
    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = SimpleNamespace(
                print=lambda *_a, **_kw: calls.append("query-label")
            )
            self.session_id = "sq-session"
            self.agent = SimpleNamespace(session_id="sq-session", platform="cli")
            self._last_turn_result = None

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            self._last_turn_result = turn_result
            return "done"

        def _print_exit_summary(self, clear_screen=True):
            calls.append(("summary", clear_screen))

    return FakeCLI


# ── c) THE regression test: the -q branch signals the quota wall ────────────

def test_q_branch_exits_with_rate_limit_sentinel_for_kanban_worker(monkeypatch):
    """A kanban worker killed by a provider quota wall must exit 75, not 0.

    Before the fix the ``-q`` branch had no ``sys.exit`` at all: ``main()`` returned
    normally, the shell saw rc=0, and the dispatcher counted a protocol violation
    against the circuit breaker instead of requeueing the card.
    """
    calls = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_6df870d8")
    monkeypatch.setattr(
        cli_mod, "HermesCLI",
        _make_human_fake_cli(calls, {"failed": True, "failure_reason": "rate_limit"}),
    )
    _install_common_main_stubs(monkeypatch, calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == KANBAN_RATE_LIMIT_EXIT_CODE
    # The session lease must still be released while SystemExit propagates.
    assert calls[-1] == ("finalize", "sq-session")


def test_q_branch_exits_one_on_generic_failure(monkeypatch):
    calls = []
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(
        cli_mod, "HermesCLI",
        _make_human_fake_cli(calls, {"failed": True, "error": "boom"}),
    )
    _install_common_main_stubs(monkeypatch, calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 1
    assert calls[-1] == ("finalize", "sq-session")


def test_q_branch_exits_zero_on_success(monkeypatch):
    calls = []
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(
        cli_mod, "HermesCLI",
        _make_human_fake_cli(calls, {"final_response": "hi", "failed": False}),
    )
    _install_common_main_stubs(monkeypatch, calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 0
    assert calls == [
        ("claim", "cli", False),
        "query-label",
        "advisories",
        ("chat", "hello", None),
        ("summary", False),
        ("finalize", "sq-session"),
    ]


def test_q_branch_exits_zero_when_chat_left_no_result(monkeypatch):
    """``chat()`` early-returns (blocked @ reference, credential refresh) leave the
    stash at None. That is not a failure and must stay rc=0."""
    calls = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    monkeypatch.setattr(cli_mod, "HermesCLI", _make_human_fake_cli(calls, None))
    _install_common_main_stubs(monkeypatch, calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 0


def test_q_branch_interrupted_partial_turn_does_not_report_failure(monkeypatch):
    """Ctrl+C handled in-agent yields a PARTIAL turn, not a failed one.

    Pins that the new exit does not start reporting 1 for user interrupts (the
    uncaught-KeyboardInterrupt path keeps its own 130 semantics).
    """
    calls = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    monkeypatch.setattr(
        cli_mod, "HermesCLI",
        _make_human_fake_cli(
            calls, {"final_response": "partial", "partial": True, "completed": False}
        ),
    )
    _install_common_main_stubs(monkeypatch, calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 0


def test_q_branch_keyboard_interrupt_is_not_converted_to_an_exit_code(monkeypatch):
    """An uncaught KeyboardInterrupt must propagate (Python renders it as 130),
    never be swallowed and turned into the new ``sys.exit``."""
    calls = []
    FakeCLI = _make_human_fake_cli(calls, None)

    def _interrupting_chat(self, query, images=None):
        raise KeyboardInterrupt()

    FakeCLI.chat = _interrupting_chat
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    _install_common_main_stubs(monkeypatch, calls)

    with pytest.raises(KeyboardInterrupt):
        cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    # finalize still ran from the `finally` on the way out.
    assert calls[-1] == ("finalize", "sq-session")


# ── trap 1: the finally must not swallow SystemExit ─────────────────────────

def test_finalize_single_query_lets_system_exit_propagate(monkeypatch):
    """``sys.exit`` fires inside the try whose ``finally`` calls this function.

    A bare ``except:``/``except BaseException:`` anywhere on this path would eat the
    exit code and make the whole fix a silent no-op, so pin it directly.
    """
    calls = []
    fake_cli = SimpleNamespace(
        session_id="s", agent=None,
        _release_active_session=lambda: calls.append("release"),
    )
    monkeypatch.setattr(cli_mod, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli_mod, "_cleanup_done", False)
    monkeypatch.setattr(
        cli_mod, "_notify_single_query_session_finalize",
        lambda _c: calls.append("notify"),
    )
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **_kw: calls.append("cleanup"))

    def _boom(_cli):
        raise SystemExit(KANBAN_RATE_LIMIT_EXIT_CODE)

    # Simulate SystemExit crossing the finalize path via one of its own steps.
    monkeypatch.setattr(cli_mod, "_flush_one_shot_session_store", _boom)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod._finalize_single_query(fake_cli)

    assert exc_info.value.code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert "release" in calls, "the session lease must still be released"


# ── d) -Q regression: existing behavior unchanged ───────────────────────────

def _make_quiet_fake_cli(calls, result):
    def run_conversation(*, user_message, conversation_history, **_kw):
        calls.append(("run", user_message, conversation_history))
        return result

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "quiet-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id="quiet-session", platform="cli", quiet_mode=False,
                suppress_status_output=False, stream_delta_callback=object(),
                tool_gen_callback=object(), run_conversation=run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, effective_query):
            return {"signature": "same-route", "model": None, "runtime": None,
                    "request_overrides": None}

        def _init_agent(self, **_kwargs):
            return True

    return FakeCLI


@pytest.mark.parametrize(
    "result, kanban_task, expected",
    [
        ({"final_response": "hi", "failed": False}, None, 0),
        ({"final_response": "", "error": "boom", "failed": True}, None, 1),
        ({"final_response": "", "failed": True, "failure_reason": "rate_limit"}, None, 1),
        ({"final_response": "", "failed": True, "failure_reason": "rate_limit"},
         "t_x", KANBAN_RATE_LIMIT_EXIT_CODE),
        ({"final_response": "", "failed": True, "failure_reason": "billing"},
         "t_x", KANBAN_RATE_LIMIT_EXIT_CODE),
        ({"final_response": "", "failed": True, "failure_reason": "tool_error"}, "t_x", 1),
    ],
)
def test_capital_q_branch_exit_contract_unchanged(monkeypatch, result, kanban_task, expected):
    calls = []
    if kanban_task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", _make_quiet_fake_cli(calls, result))
    _install_common_main_stubs(monkeypatch, calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == expected
    assert calls[-1] == ("finalize", "quiet-session")
