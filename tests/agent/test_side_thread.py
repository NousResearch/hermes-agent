"""Side exploration threads (``/btw --explore``): the tool-using ``/btw`` (agent/side_thread.py).

Contracts: plain ``/btw <question>`` still parses as the tool-less question; an exploration fork
dispatches only the configured, advertised tools under its own task id (never the main session's
terminal), denies dangerous-command approvals, never writes the main transcript, and reaches the
main conversation only through an explicit promotion prompt.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from agent import side_thread
from agent.side_thread import (
    DEFAULT_EXPLORE_TOOLS,
    MAX_RUNNING_PER_SESSION,
    MAX_THREADS_PER_SESSION,
    SideThreadUnavailable,
    build_promotion_prompt,
    explore_settings,
    find_thread,
    list_threads,
    parse_btw_args,
    register_thread,
    run_side_exploration,
)


@pytest.fixture(autouse=True)
def _empty_registry():
    side_thread._registry.clear()
    yield
    side_thread._registry.clear()


class TestParseBtwArgs:
    @pytest.mark.parametrize("raw", ["what was that error?", "-x is not a flag", "explore the repo"])
    def test_plain_text_stays_a_classic_question(self, raw):
        assert parse_btw_args(raw) == side_thread.BtwRequest("ask", raw)

    def test_double_dash_asks_literally(self):
        assert parse_btw_args("-- --explore means what?") == side_thread.BtwRequest("ask", "--explore means what?")

    @pytest.mark.parametrize("raw", ["--explore find the retry logic", "-e find the retry logic"])
    def test_explore(self, raw):
        assert parse_btw_args(raw) == side_thread.BtwRequest("explore", "find the retry logic")

    def test_promote_with_id_and_note(self):
        assert parse_btw_args("--promote btw_ab12cd focus on the tests") == side_thread.BtwRequest(
            "promote", "focus on the tests", "btw_ab12cd")

    def test_promote_note_starting_with_a_hex_word_is_not_an_id(self):
        assert parse_btw_args("--promote add tests for it") == side_thread.BtwRequest("promote", "add tests for it")

    def test_list_and_stop(self):
        assert parse_btw_args("--threads").action == "list"
        assert parse_btw_args("--stop 3f9a").ref == "3f9a"


class TestRegistry:
    def test_threads_are_scoped_to_their_session(self):
        mine = register_thread("sess-a", "task a")
        register_thread("sess-b", "task b")
        assert [t.thread_id for t in list_threads("sess-a")] == [mine.thread_id]
        assert find_thread("sess-b", mine.thread_id) is None

    def test_threads_are_scoped_to_their_profile(self, tmp_path, monkeypatch):
        thread = register_thread("sess", "task")
        other = tmp_path / "other-profile"
        other.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(other))
        assert find_thread("sess", thread.thread_id) is None

    def test_lookup_by_bare_id_prefix_and_latest_finished(self):
        first = register_thread("s", "one")
        second = register_thread("s", "two")
        first._finish("done", answer="a")
        bare = first.thread_id[len("btw_"):]
        assert find_thread("s", bare) is first
        assert find_thread("s", bare[:4]) is first
        assert find_thread("s", "") is first  # the latest FINISHED thread, not the running one
        second._finish("done", answer="b")
        assert find_thread("s", "") is second

    def test_running_threads_are_capped(self):
        for i in range(MAX_RUNNING_PER_SESSION):
            register_thread("s", f"task {i}")
        with pytest.raises(SideThreadUnavailable):
            register_thread("s", "one too many")

    def test_finished_threads_are_evicted_oldest_first(self):
        threads = []
        for i in range(MAX_THREADS_PER_SESSION):
            threads.append(register_thread("s", f"t{i}"))
            threads[-1]._finish("done", answer="x")
        newest = register_thread("s", "newest")
        ids = [t.thread_id for t in list_threads("s")]
        assert threads[0].thread_id not in ids and newest.thread_id in ids
        assert len(ids) == MAX_THREADS_PER_SESSION


class TestExploreSettings:
    def test_defaults(self):
        tools, iterations = explore_settings({})
        assert tools == frozenset(DEFAULT_EXPLORE_TOOLS)
        assert iterations == side_thread.DEFAULT_MAX_ITERATIONS

    def test_configured_tools_replace_the_default_and_iterations_clamp(self):
        tools, iterations = explore_settings({"explore_tools": ["read_file", " ", 3], "explore_max_iterations": 999})
        assert tools == frozenset({"read_file"})
        assert iterations == side_thread.MAX_ITERATIONS

    def test_read_from_the_profiles_config_yaml(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "auxiliary:\n  side_question:\n    explore_tools: [search_files]\n    explore_max_iterations: 4\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert explore_settings() == (frozenset({"search_files"}), 4)


class _FakeFork:
    def __init__(self, record, *, result=None, block=None):
        self.record = record
        self.valid_tool_names = {"read_file", "terminal", "write_file", "memory"}
        self.tool_start_callback = None
        self._result = result
        self._block = block

    def run_conversation(self, *, user_message, conversation_history, task_id):
        from tools.terminal_tool import _get_approval_callback

        self.record.update(user_message=user_message, history=conversation_history, task_id=task_id,
                           approval=_get_approval_callback()("rm -rf /", "dangerous"))
        self.tool_start_callback("c1", "terminal", {"command": "rg -n retry src/"})
        if self._block is not None:
            self._block.wait(5)
        if self._result is not None:
            return self._result
        return {"final_response": "retry lives in net.py:40",
                "messages": [*conversation_history, {"role": "user", "content": user_message},
                             {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "terminal"}}]},
                             {"role": "tool", "content": "net.py:40: retry"},
                             {"role": "assistant", "content": "retry lives in net.py:40"}]}

    def _close_task_resources(self, task_id):
        self.record["closed_task"] = task_id


def _run(thread, fork, history, **kwargs):
    whitelists, released = [], []
    with patch("agent.background_review.build_cache_parity_fork", return_value=(fork, {}, False)), \
         patch("agent.background_review.release_cache_parity_fork",
               side_effect=lambda parent, f, task: released.append(task)), \
         patch("agent.side_question_request._agent_provider_refusal", return_value=None), \
         patch("agent.side_question._side_question_task_config", return_value={}), \
         patch("hermes_cli.plugins.set_thread_tool_whitelist",
               side_effect=lambda allowed, **kw: whitelists.append(allowed)):
        out = run_side_exploration(thread, object(), history, **kwargs)
    return out, whitelists, released


class TestRunSideExploration:
    HISTORY = [{"role": "user", "content": "fix the flaky test"}, {"role": "assistant", "content": "on it"},
               {"role": "user", "content": "in-flight turn"}]

    def test_explores_with_whitelisted_tools_on_its_own_task(self):
        record, progress = {}, []
        thread = register_thread("sess", "find the retry logic")
        out, whitelists, released = _run(thread, _FakeFork(record), self.HISTORY, on_progress=progress.append)

        assert out == "retry lives in net.py:40"
        # Configured ∩ advertised: write_file/memory are advertised but never dispatchable.
        assert whitelists == [{"read_file", "terminal"}]
        assert record["task_id"] == thread.thread_id == record["closed_task"]
        assert record["approval"] == "deny"
        # The in-flight tail is trimmed, the completed prefix replayed verbatim.
        assert record["history"] == self.HISTORY[:2]
        assert "find the retry logic" in record["user_message"]
        assert released == [side_thread.SIDE_THREAD_TASK]
        assert thread.status == "done" and thread.tool_calls == ["terminal"]
        assert any("rg -n retry src/" in line for line in progress)
        # Only the side transcript is kept — never the replayed main history.
        assert thread.messages[0]["role"] == "user" and "Side task" in thread.messages[0]["content"]

    def test_approval_callback_and_whitelist_are_cleared_afterwards(self):
        from hermes_cli.plugins import _thread_tool_whitelist
        from tools.terminal_tool import _get_approval_callback

        _run(register_thread("sess", "t"), _FakeFork({}), [])
        assert _get_approval_callback() is None
        assert getattr(_thread_tool_whitelist, "allowed", None) is None

    def test_no_live_parent_is_refused_and_marked_failed(self):
        thread = register_thread("sess", "t")
        with pytest.raises(SideThreadUnavailable):
            run_side_exploration(thread, None, [])
        assert thread.status == "failed"

    def test_stop_interrupts_the_running_fork(self):
        gate = threading.Event()
        fork = _FakeFork({}, block=gate,
                         result={"final_response": "partial", "messages": []})
        interrupts = []

        def fake_interrupt(agent, message=None, **kw):
            interrupts.append(agent)
            gate.set()
            return True

        thread = register_thread("sess", "t")
        holder = {}
        with patch("agent.interrupt_compat.request_hard_interrupt", side_effect=fake_interrupt):
            worker = threading.Thread(target=lambda: holder.update(out=_run(thread, fork, [])[0]))
            worker.start()
            from tests.waiting import wait_until
            assert wait_until(lambda: thread.tool_calls, timeout=5)
            assert thread.stop() is True
            worker.join(5)
        assert interrupts == [fork]
        assert thread.status == "stopped" and holder["out"] == "partial"
        assert thread.stop() is False

    def test_main_session_terminal_environment_is_left_alone(self):
        """A real fork of a real parent: the side thread's teardown releases its OWN task's
        environment, never the one keyed by the shared session id."""
        from run_agent import AIAgent
        from tools import terminal_tool

        class _Env:
            def __init__(self):
                self.cleaned = False

            def cleanup(self, **_kwargs):
                self.cleaned = True

        with patch("model_tools.get_tool_definitions", return_value=[]), \
             patch("model_tools.check_toolset_requirements", return_value={}), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):
            parent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model",
                             quiet_mode=True, skip_context_files=True, skip_memory=True, session_id="sess-main")
            parent._cached_system_prompt = "You are helpful."
            thread = register_thread("sess-main", "look around")
            main_env, side_env = _Env(), _Env()
            terminal_tool._active_environments["sess-main"] = main_env
            terminal_tool._active_environments[thread.thread_id] = side_env
            try:
                with patch.object(AIAgent, "run_conversation",
                                  return_value={"final_response": "found it", "messages": []}):
                    assert run_side_exploration(thread, parent, []) == "found it"
                assert terminal_tool._active_environments.get("sess-main") is main_env
                assert not main_env.cleaned
                assert thread.thread_id not in terminal_tool._active_environments
            finally:
                terminal_tool._active_environments.pop("sess-main", None)
                terminal_tool._active_environments.pop(thread.thread_id, None)


def test_progress_line_names_the_most_telling_argument():
    from agent.side_thread import describe_tool_call
    assert describe_tool_call("search_files", {"pattern": "def backoff", "path": "agent"}) == "search_files · def backoff"
    assert describe_tool_call("read_file", {"path": "agent/x.py", "offset": 3}) == "read_file · agent/x.py"
    assert describe_tool_call("terminal", {"command": "rg  -n\n retry"}) == "terminal · rg -n retry"


class TestPromotion:
    def test_promotion_carries_the_report_as_a_user_turn(self):
        thread = register_thread("s", "find the retry logic")
        thread.tool_calls = ["search_files", "read_file", "read_file"]
        thread._finish("done", answer="retry lives in net.py:40")
        prompt = build_promotion_prompt(thread, "now fix it")
        assert thread.thread_id in prompt and "retry lives in net.py:40" in prompt
        assert "read_file ×2" in prompt
        assert prompt.rstrip().endswith("now fix it")

    def test_running_or_empty_threads_cannot_be_promoted(self):
        thread = register_thread("s", "t")
        with pytest.raises(ValueError):
            build_promotion_prompt(thread)
        thread._finish("failed", error="boom")
        with pytest.raises(ValueError):
            build_promotion_prompt(thread)


class TestThreadCommands:
    def _done(self, task="find the retry logic", answer="net.py:40"):
        thread = register_thread("s", task)
        thread._finish("done", answer=answer)
        return thread

    def test_promote_latest_returns_the_turn_to_submit(self):
        from agent.side_thread import run_thread_command
        thread = self._done()
        result = run_thread_command("s", parse_btw_args("--promote now fix it"))
        assert thread.thread_id in result.text
        assert result.prompt == build_promotion_prompt(thread, "now fix it")

    def test_promote_refusals_submit_nothing(self):
        from agent.side_thread import run_thread_command
        assert run_thread_command("s", parse_btw_args("--promote")).prompt == ""
        running = register_thread("s", "still going")
        assert run_thread_command("s", parse_btw_args(f"--promote {running.thread_id}")).prompt == ""
        assert run_thread_command("s", parse_btw_args("--promote btw_000000")).prompt == ""

    def test_stop_and_list(self):
        from agent.side_thread import run_thread_command
        running = register_thread("s", "still going")
        listing = run_thread_command("s", parse_btw_args("--threads")).text
        assert running.thread_id in listing and "still going" in listing
        run_thread_command("s", parse_btw_args(f"--stop {running.thread_id}"))
        assert running.stop_requested
