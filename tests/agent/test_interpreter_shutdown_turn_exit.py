"""Interpreter-shutdown handling in the conversation loop's outer retry path.

When the process starts tearing down (TUI quit, SIGTERM, one-shot exit) while
a turn — most commonly the post-turn background-review fork's daemon thread —
is still mid-request, every further API attempt raises ``RuntimeError:
cannot schedule new futures after interpreter shutdown``.

Before the fix, the outer loop treated that as a retryable API error: it
printed an un-gated ``❌ Error during OpenAI-compatible API call #N`` line
per attempt (spamming the user's shell AFTER the TUI already exited) and
retried until the iteration budget or interpreter froze the thread.

Now the loop recognizes the shutdown signal via the shared predicate in
``tools.interpreter_shutdown`` (same class as cron delivery #55924/#58720
and concurrent tool submission), logs one warning, and abandons the turn —
no print, no traceback, no retry.
"""

from types import SimpleNamespace


def _text_response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, reasoning=None, tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


class _ShutdownThenTextCompletions:
    """First call raises the CPython shutdown error; later calls would succeed.

    The 'would succeed' part is the sabotage detector: if the loop wrongly
    retries after the shutdown signal, call #2 returns a normal response and
    the assertions on call count / failed flag below fail loudly.
    """

    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(
                "cannot schedule new futures after interpreter shutdown"
            )
        return _text_response("should never be reached")


class _AlwaysFailingCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise Exception("API down")


def _make_agent(monkeypatch, completions):
    from run_agent import AIAgent

    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])

    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="cli",
        max_iterations=4,
        quiet_mode=True,
        skip_memory=True,
    )
    agent._disable_streaming = True
    return agent


def test_shutdown_error_exits_loop_without_retry(monkeypatch, capsys):
    completions = _ShutdownThenTextCompletions()
    agent = _make_agent(monkeypatch, completions)

    result = agent.run_conversation("hello")

    # Precondition: the shutdown error actually fired on call #1.
    assert completions.calls >= 1
    # The core contract: NO retry after the shutdown signal.
    assert completions.calls == 1, (
        "conversation loop retried an API call after interpreter-shutdown "
        f"signal (made {completions.calls} calls)"
    )
    assert result["failed"] is True
    assert "shutting down" in result["final_response"]

    # And no ❌ spam on the user's terminal.
    out = capsys.readouterr()
    assert "❌" not in out.out
    assert "cannot schedule new futures" not in out.out


def test_shutdown_variant_without_interpreter_word_also_exits(monkeypatch):
    """CPython's plain-ThreadPoolExecutor variant omits 'interpreter'."""

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("cannot schedule new futures after shutdown")

    completions = _Completions()
    agent = _make_agent(monkeypatch, completions)

    result = agent.run_conversation("hello")

    assert completions.calls == 1
    assert result["failed"] is True


def test_suppress_status_output_gates_error_print_for_ordinary_api_errors(
    monkeypatch, capsys
):
    """suppress_status_output routes ❌ lines to the log, not stdout.

    This is the leak path from the field report: the background-review fork
    runs with suppress_status_output=True (which gates _vprint and the
    buffered retry trace), yet the bare print() in the outer error handler
    bypassed it.
    """
    completions = _AlwaysFailingCompletions()
    agent = _make_agent(monkeypatch, completions)
    agent.suppress_status_output = True

    result = agent.run_conversation("hello")

    # Precondition: the ordinary (non-shutdown) error path actually retried
    # up to the budget — this test exercises the print gating, not early exit.
    assert completions.calls > 1
    assert "API down" in result["final_response"]

    out = capsys.readouterr()
    assert "❌" not in out.out


def test_non_quiet_mode_still_prints_error(monkeypatch, capsys):
    completions = _AlwaysFailingCompletions()
    agent = _make_agent(monkeypatch, completions)
    agent.quiet_mode = False

    agent.run_conversation("hello")

    out = capsys.readouterr()
    assert "❌" in out.out


class TestSharedPredicate:
    def test_matches_interpreter_variant(self):
        from tools.interpreter_shutdown import interpreter_shutting_down

        exc = RuntimeError("cannot schedule new futures after interpreter shutdown")
        assert interpreter_shutting_down(exc) is True

    def test_matches_plain_executor_variant(self):
        from tools.interpreter_shutdown import interpreter_shutting_down

        exc = RuntimeError("cannot schedule new futures after shutdown")
        assert interpreter_shutting_down(exc) is True

    def test_ignores_unrelated_errors(self):
        from tools.interpreter_shutdown import interpreter_shutting_down

        assert interpreter_shutting_down(RuntimeError("boom")) is False
        assert interpreter_shutting_down(None) is False

    def test_cron_wrapper_delegates(self):
        from cron.scheduler import _interpreter_shutting_down

        exc = RuntimeError("cannot schedule new futures after shutdown")
        assert _interpreter_shutting_down(exc) is True
        assert _interpreter_shutting_down(RuntimeError("boom")) is False

    def test_tool_executor_wrapper_delegates(self):
        from agent.tool_executor import _is_interpreter_shutdown_submit_error

        # The tool-executor predicate previously matched ONLY the fuller
        # variant; via the shared home it now catches both.
        exc = RuntimeError("cannot schedule new futures after shutdown")
        assert _is_interpreter_shutdown_submit_error(exc) is True
        assert _is_interpreter_shutdown_submit_error(RuntimeError("boom")) is False
