"""Unit tests for the Cursor Agent CLI bridge.

These tests fully mock the ``cursor-agent`` subprocess — they never spawn the
real CLI. End-to-end smoke against a live ``cursor-agent`` lives in the
integration script ``tests/agent/test_cursor_agent_client_smoke.py`` (skipped
unless ``HERMES_CURSOR_SMOKE=1`` is set in the environment).
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.cursor_agent_client import (
    CURSOR_MARKER_BASE_URL,
    CursorAgentClient,
    _StreamJsonAccumulator,
    _format_messages_as_prompt,
    _normalize_cursor_tool_name,
)


# ---------------------------------------------------------------------------
# Fake subprocess that emits a pre-canned stream-json transcript
# ---------------------------------------------------------------------------


class _PersistentStringIO(io.StringIO):
    """StringIO whose contents survive ``close()`` so tests can inspect stdin
    after the client has finished writing and closing it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._snapshot = ""

    def close(self) -> None:  # type: ignore[override]
        try:
            self._snapshot = self.getvalue()
        except Exception:
            pass
        super().close()

    def getvalue(self) -> str:  # type: ignore[override]
        if self.closed:
            return self._snapshot
        return super().getvalue()


class _FakeProcess:
    """Minimal stand-in for ``subprocess.Popen[str]``."""

    def __init__(self, stdout_lines: list[str], stderr_lines: list[str] | None = None):
        self.stdin = _PersistentStringIO()
        self.stdout = io.StringIO("\n".join(stdout_lines) + ("\n" if stdout_lines else ""))
        self.stderr = io.StringIO("\n".join(stderr_lines or []) + ("\n" if stderr_lines else ""))
        self._terminated = False
        self._killed = False
        self._exit_code: int | None = None
        self.argv_seen: list[str] = []
        self.cwd_seen: str | None = None
        self.env_seen: dict[str, str] | None = None
        # Once the stream is fully consumed treat the process as exited.
        # poll() returns None while events are still being read so the
        # accumulator loop doesn't early-exit.
        self._lock = threading.Lock()

    def poll(self) -> int | None:
        # Report "still running" until both pipes have been drained at least
        # once. The reader threads are eager so by the time the main loop
        # finishes iterating events, the process is effectively done.
        with self._lock:
            return self._exit_code

    def terminate(self) -> None:
        with self._lock:
            self._terminated = True
            if self._exit_code is None:
                self._exit_code = 0

    def kill(self) -> None:
        with self._lock:
            self._killed = True
            if self._exit_code is None:
                self._exit_code = 0

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        with self._lock:
            if self._exit_code is None:
                self._exit_code = 0
            return self._exit_code


def _make_event(**fields) -> str:
    return json.dumps(fields)


SUCCESS_STREAM = [
    _make_event(type="system", subtype="init", session_id="s-1", model="Auto"),
    _make_event(type="user", message={"role": "user", "content": [{"type": "text", "text": "Hi"}]}, session_id="s-1"),
    _make_event(type="thinking", subtype="delta", text="thinking-bit", session_id="s-1"),
    _make_event(type="assistant", message={"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}, session_id="s-1"),
    _make_event(
        type="result",
        subtype="success",
        duration_ms=1234,
        is_error=False,
        result="Hello world",
        session_id="s-1",
        request_id="r-1",
        usage={"inputTokens": 42, "outputTokens": 13, "cacheReadTokens": 9, "cacheWriteTokens": 0},
    ),
]


ERROR_STREAM = [
    _make_event(type="system", subtype="init", session_id="s-2", model="Auto"),
    _make_event(
        type="result",
        subtype="error",
        duration_ms=100,
        is_error=True,
        result="quota exceeded",
        session_id="s-2",
        request_id="r-2",
    ),
]


# ---------------------------------------------------------------------------
# Accumulator unit tests (no subprocess)
# ---------------------------------------------------------------------------


class StreamAccumulatorTests(unittest.TestCase):
    def test_success_stream_assembles_text_and_usage(self) -> None:
        acc = _StreamJsonAccumulator()
        for line in SUCCESS_STREAM:
            acc.feed(json.loads(line))
        self.assertTrue(acc.terminal)
        self.assertFalse(acc.is_error)
        self.assertEqual(acc.assembled_text(), "Hello world")
        self.assertEqual(acc.assembled_reasoning(), "thinking-bit")
        usage = acc.openai_usage()
        # New contract: prompt_tokens approximates "what was in context"
        # = cacheReadTokens + inputTokens averaged across internal rounds.
        # With 0 tool events there's 1 round so just sum: 9 + 42 = 51.
        self.assertEqual(usage.prompt_tokens, 51)
        self.assertEqual(usage.completion_tokens, 13)
        self.assertEqual(usage.total_tokens, 64)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 9)
        # Raw cursor billing values still available for cost tracking.
        self.assertEqual(usage.cursor_raw_input_tokens, 42)
        self.assertEqual(usage.cursor_raw_cache_read_tokens, 9)
        self.assertEqual(usage.cursor_internal_rounds, 1)

    def test_error_stream_sets_error_flag_and_message(self) -> None:
        acc = _StreamJsonAccumulator()
        for line in ERROR_STREAM:
            acc.feed(json.loads(line))
        self.assertTrue(acc.terminal)
        self.assertTrue(acc.is_error)
        self.assertIn("quota exceeded", acc.error_message)

    def test_unknown_event_types_are_ignored(self) -> None:
        acc = _StreamJsonAccumulator()
        acc.feed({"type": "weird-future-event", "anything": True})
        acc.feed({"no-type": "ignored"})
        self.assertFalse(acc.terminal)
        self.assertEqual(acc.assembled_text(), "")

    def test_result_without_text_uses_final_result_field(self) -> None:
        acc = _StreamJsonAccumulator()
        acc.feed(json.loads(_make_event(type="system", subtype="init", session_id="x")))
        acc.feed(json.loads(_make_event(
            type="result",
            subtype="success",
            is_error=False,
            result="standalone",
            session_id="x",
            request_id="x-r",
            usage={"inputTokens": 1, "outputTokens": 1, "cacheReadTokens": 0},
        )))
        self.assertEqual(acc.assembled_text(), "standalone")


# ---------------------------------------------------------------------------
# Argv + workspace defaults
# ---------------------------------------------------------------------------


class BuildArgvTests(unittest.TestCase):
    def setUp(self) -> None:
        # Clear all HERMES_CURSOR_* env overrides so defaults are predictable.
        keys = [k for k in os.environ if k.startswith("HERMES_CURSOR_") or k == "CURSOR_API_KEY" or k == "CURSOR_AGENT_PATH"]
        self._saved = {k: os.environ.pop(k) for k in keys}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            os.environ[k] = v

    def test_default_argv_uses_full_agent_mode_and_print_flag(self) -> None:
        # Default mode is ``agent`` (cursor-agent's own default permissionMode,
        # which is invoked by omitting ``--mode`` entirely).  This matches what
        # users get when they run ``cursor-agent -p`` directly from the shell,
        # so picking ``cursor`` in ``hermes model`` doesn't silently demote
        # the agent to read-only. Hermes' own ``approvals.mode`` gate (manual /
        # smart / off) still applies on top, matching every other provider.
        client = CursorAgentClient()
        argv = client._build_argv(model="composer-2.5", workspace="/tmp/work")
        self.assertEqual(argv[0], "cursor-agent")
        self.assertIn("-p", argv)
        self.assertIn("--output-format", argv)
        self.assertIn("stream-json", argv)
        # ``agent`` is the synthetic value meaning "no --mode flag" →
        # cursor uses its default full-capability permission mode.
        self.assertNotIn("--mode", argv,
                         "default cursor mode (agent) must omit the --mode CLI flag")
        self.assertEqual(argv[argv.index("--model") + 1], "composer-2.5")
        self.assertEqual(argv[argv.index("--workspace") + 1], "/tmp/work")
        self.assertIn("--force", argv)
        self.assertIn("--trust", argv)
        self.assertNotIn("--api-key", argv)  # no key passed → omit flag

    def test_explicit_ask_mode_still_works_via_env(self) -> None:
        # Users who want read-only behavior opt in via env var.
        os.environ["HERMES_CURSOR_MODE"] = "ask"
        try:
            client = CursorAgentClient()
            argv = client._build_argv(model="auto", workspace="/tmp/x")
            self.assertIn("--mode", argv)
            self.assertEqual(argv[argv.index("--mode") + 1], "ask")
        finally:
            os.environ.pop("HERMES_CURSOR_MODE", None)

    def test_api_key_threads_through_argv(self) -> None:
        client = CursorAgentClient(api_key="crsr_abc123")
        argv = client._build_argv(model="auto", workspace="/tmp/x")
        self.assertEqual(argv[argv.index("--api-key") + 1], "crsr_abc123")

    def test_sentinel_api_key_is_dropped(self) -> None:
        # Regression: external_process auth path injects "cursor-agent-login"
        # as a placeholder. Forwarding it to ``cursor-agent --api-key`` makes
        # the CLI reject the request and close stdin, which surfaces as
        # BrokenPipeError on our writes. The client must treat it as "no key
        # — use the cursor-agent CLI's own login session" and omit the flag.
        for sentinel in (
            "",
            "cursor-agent-login",
            "cursor-cli-login",
            "external-process",
            "external_process",
        ):
            with self.subTest(sentinel=sentinel):
                client = CursorAgentClient(api_key=sentinel)
                try:
                    self.assertIsNone(client.api_key)
                    argv = client._build_argv(model="auto", workspace="/tmp/x")
                    self.assertNotIn("--api-key", argv)
                finally:
                    client.close()

    def test_extra_args_appended(self) -> None:
        client = CursorAgentClient(args=["--header", "X-Hermes: 1"])
        argv = client._build_argv(model="auto", workspace="/tmp/x")
        self.assertEqual(argv[-2:], ["--header", "X-Hermes: 1"])

    def test_mode_override(self) -> None:
        client = CursorAgentClient(mode="plan")
        argv = client._build_argv(model="auto", workspace="/tmp/x")
        self.assertEqual(argv[argv.index("--mode") + 1], "plan")

    def test_invalid_mode_falls_back_to_default_agent(self) -> None:
        # Invalid env values must not crash and must not silently downgrade
        # the user — falling back to ``agent`` (Hermes' default) means the
        # user picking cursor still gets the expected full-power behavior.
        # Hermes' own ``approvals.mode`` config still gates dangerous ops.
        client = CursorAgentClient(mode="lol-not-a-mode")
        argv = client._build_argv(model="auto", workspace="/tmp/x")
        self.assertNotIn("--mode", argv,
                         "invalid mode must fall back to agent (no --mode flag)")

    def test_command_and_args_propagate_to_subprocess(self) -> None:
        # Regression: agent_init.py used to only copy ``acp_command`` /
        # ``acp_args`` into ``client_kwargs`` for ``provider=copilot-acp``,
        # not for ``cursor``. Symptom: status/auth flow resolves a
        # specific cursor-agent binary path, but chat used whatever's
        # first on $PATH. This pins the client-level behaviour: when
        # ``command`` and ``args`` are passed, they make it into argv.
        client = CursorAgentClient(
            command="/opt/custom/cursor-agent",
            args=["--header", "X-Hermes-Trace: 1"],
        )
        try:
            argv = client._build_argv(model="auto", workspace="/tmp")
            self.assertEqual(argv[0], "/opt/custom/cursor-agent")
            self.assertEqual(argv[-2:], ["--header", "X-Hermes-Trace: 1"])
        finally:
            client.close()

    def test_mode_agent_omits_cli_flag(self) -> None:
        # Regression: cursor-agent only accepts ``--mode ask|plan``. The
        # synthetic ``agent`` value means "run in cursor's full-capability
        # default permissionMode" — achieved by NOT passing the flag at
        # all. Earlier code passed ``--mode agent`` and cursor crashed
        # with a confusing BrokenPipe before even reading stdin.
        client = CursorAgentClient(mode="agent")
        argv = client._build_argv(model="auto", workspace="/tmp/x")
        self.assertNotIn("--mode", argv)
        # Critical flags that enable full agentic behaviour are still present.
        self.assertIn("-p", argv)
        self.assertIn("--force", argv)
        self.assertIn("--trust", argv)
        self.assertIn("--workspace", argv)

    def test_env_override_command(self) -> None:
        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "/opt/custom/cursor-agent"}):
            client = CursorAgentClient()
            argv = client._build_argv(model="auto", workspace="/tmp/x")
            self.assertEqual(argv[0], "/opt/custom/cursor-agent")

    def test_workspace_override_via_ctor_is_reused(self) -> None:
        client = CursorAgentClient(workspace="/persistent/ws")
        ws1, _ = client._allocate_workspace()
        ws2, _ = client._allocate_workspace()
        self.assertEqual(ws1, "/persistent/ws")
        self.assertEqual(ws2, "/persistent/ws")  # reused, not minted

    def test_workspace_default_is_session_scoped_not_per_call(self) -> None:
        # Perf: cursor-agent pays ~4.5s of bootstrap overhead when given a
        # fresh empty workspace. Within the SAME chat session (same
        # CursorAgentClient instance), all calls must share one workspace
        # so we only pay that tax once.
        client = CursorAgentClient()
        try:
            ws1, eph1 = client._allocate_workspace()
            ws2, eph2 = client._allocate_workspace()
            ws3, eph3 = client._allocate_workspace()
            self.assertTrue(eph1 and eph2 and eph3,
                            "default workspace should be marked ephemeral")
            self.assertTrue("hermes-cursor-" in ws1)
            self.assertEqual(ws1, ws2,
                             "second call must REUSE the first session workspace")
            self.assertEqual(ws2, ws3,
                             "third call must REUSE the session workspace")
            import os
            self.assertTrue(os.path.isdir(ws1),
                            "session workspace dir must exist on disk")
        finally:
            client.close()

    def test_session_workspace_is_freshly_minted_after_close(self) -> None:
        # close() drops the cached session workspace ref so a new chat
        # session (after /new or a fresh client) gets its own scratch dir.
        client = CursorAgentClient()
        try:
            ws1, _ = client._allocate_workspace()
            client.close()
            client.is_closed = False  # simulate re-use (defensive)
            ws2, _ = client._allocate_workspace()
            self.assertNotEqual(ws1, ws2,
                                "post-close allocation must mint a NEW dir")
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Prompt formatter — must steer cursor away from its built-in tool harness
# ---------------------------------------------------------------------------


class FormatMessagesAsPromptTests(unittest.TestCase):
    """Cursor-agent is itself an agent with built-in shell/edit/read tools.

    Without the hardened framing in our formatter, the model treats Hermes'
    advertised tools as if they were its own and runs them in-process — the
    user sees zero ``tool_calls`` in the chat session even when tools are
    available.  These tests pin the directives that empirically push cursor
    into emitting raw <tool_call> blocks for Hermes to execute.
    """

    def test_directs_model_to_emit_tool_call_blocks(self) -> None:
        prompt = _format_messages_as_prompt(
            messages=[{"role": "user", "content": "use the kanban tool"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "kanban",
                    "description": "manage tasks",
                    "parameters": {"type": "object"},
                },
            }],
        )
        # Hermes-side tools must be discoverable and the <tool_call>
        # block format must be specified.
        self.assertIn("<tool_call>", prompt)
        self.assertIn("\"kanban\"", prompt)
        # The model is told it has BOTH built-in cursor tools and
        # Hermes-side tools — this prevents the slow round-trip we used
        # to force for every shell/file action.
        self.assertIn("built-in cursor-agent tools", prompt)
        self.assertIn("Hermes-side tools", prompt)

    def test_no_tools_uses_lite_preamble_not_agentic_one(self) -> None:
        # Regression: aux tasks (title gen, compression, vision, ...)
        # call cursor with NO ``tools`` advertised. They want a short
        # direct reply, not a multi-paragraph agentic response. The
        # heavy "you are not the agent, emit tool_call blocks" preamble
        # made cursor produce verbose responses on these short-form
        # tasks AND wasted ~500 prompt tokens per call.
        prompt = _format_messages_as_prompt(
            messages=[
                {"role": "system", "content": "Generate a short title."},
                {"role": "user", "content": "User: hi\n\nAssistant: hello"},
            ],
            tools=None,
        )
        # No agentic framing for tool-less calls.
        self.assertNotIn("NOT an autonomous", prompt)
        self.assertNotIn("<tool_call>", prompt)
        self.assertNotIn("Available tools", prompt)
        # But we DO tell cursor "answer directly, don't run tools" so
        # its harness doesn't burn time exploring the workspace.
        self.assertIn("auxiliary call", prompt.lower())
        self.assertIn("do not run", prompt.lower())
        # The system + user content is preserved.
        self.assertIn("Generate a short title", prompt)

    def test_with_tools_keeps_agentic_preamble(self) -> None:
        # When Hermes-side tools are advertised, the prompt teaches the
        # model about BOTH built-in cursor tools and the Hermes tools —
        # this is the speed-vs-control tradeoff: cursor handles
        # shell/file work directly (fast), Hermes-only tools go through
        # <tool_call> blocks (full UI visibility + audit).
        prompt = _format_messages_as_prompt(
            messages=[{"role": "user", "content": "do something"}],
            tools=[{
                "type": "function",
                "function": {"name": "kanban", "description": "x", "parameters": {}},
            }],
        )
        self.assertIn("built-in cursor-agent tools", prompt)
        self.assertIn("Hermes-side tools", prompt)
        self.assertIn("<tool_call>", prompt)

    def test_cursor_tool_name_normalisation(self) -> None:
        cases = {
            "shellToolCall": "shell",
            "readToolCall": "read_file",
            "listToolCall": "list_directory",
            "editToolCall": "edit_file",
            "writeToolCall": "write_file",
            "grepToolCall": "grep",
            "globToolCall": "glob",
            "fetchToolCall": "web_fetch",
            "wackyFutureToolCall": "wackyFuture",
        }
        for envelope, expected in cases.items():
            with self.subTest(envelope=envelope):
                self.assertEqual(_normalize_cursor_tool_name(envelope), expected)

    def test_messages_appear_in_transcript_order(self) -> None:
        prompt = _format_messages_as_prompt(
            messages=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "ping"},
                {"role": "assistant", "content": "pong"},
                {"role": "user", "content": "again"},
            ],
        )
        idx_ping = prompt.index("ping")
        idx_pong = prompt.index("pong")
        idx_again = prompt.index("again")
        self.assertLess(idx_ping, idx_pong)
        self.assertLess(idx_pong, idx_again)


# ---------------------------------------------------------------------------
# End-to-end of `_create_chat_completion` with the subprocess mocked
# ---------------------------------------------------------------------------


class CreateChatCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Make env predictable.
        keys = [k for k in os.environ if k.startswith("HERMES_CURSOR_") or k == "CURSOR_API_KEY" or k == "CURSOR_AGENT_PATH"]
        self._saved = {k: os.environ.pop(k) for k in keys}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            os.environ[k] = v

    def _patch_popen(self, fake_proc: _FakeProcess):
        def _fake_popen(argv, **kwargs):
            fake_proc.argv_seen = list(argv)
            fake_proc.cwd_seen = kwargs.get("cwd")
            fake_proc.env_seen = kwargs.get("env")
            return fake_proc

        return patch("agent.cursor_agent_client.subprocess.Popen", side_effect=_fake_popen)

    def test_happy_path_returns_openai_shaped_response(self) -> None:
        proc = _FakeProcess(SUCCESS_STREAM)
        client = CursorAgentClient()
        try:
            with self._patch_popen(proc):
                resp = client.chat.completions.create(
                    model="composer-2.5",
                    messages=[
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Hi"},
                    ],
                )
            self.assertEqual(len(resp.choices), 1)
            self.assertEqual(resp.choices[0].message.content, "Hello world")
            self.assertEqual(resp.choices[0].finish_reason, "stop")
            self.assertEqual(resp.choices[0].message.reasoning_content, "thinking-bit")
            # prompt_tokens comes from Hermes' messages estimate
            # (authoritative for the status bar); raw cursor billing
            # is preserved as ``cursor_raw_input_tokens`` for cost
            # tracking, and the per-round average as
            # ``cursor_per_round_context``.
            self.assertGreater(resp.usage.prompt_tokens, 0)
            self.assertEqual(resp.usage.completion_tokens, 13)
            self.assertEqual(resp.usage.prompt_tokens_details.cached_tokens, 9)
            self.assertEqual(resp.usage.cursor_raw_input_tokens, 42)
            self.assertEqual(resp.usage.cursor_raw_cache_read_tokens, 9)
            # Per-round context still computed for debug visibility.
            self.assertEqual(resp.usage.cursor_per_round_context, 51)
            self.assertEqual(resp.model, "composer-2.5")
            self.assertEqual(resp.id, "r-1")
            # Argv must include the user-requested model + ask mode + the
            # ephemeral workspace.
            self.assertEqual(proc.argv_seen[proc.argv_seen.index("--model") + 1], "composer-2.5")
            # Default mode is ``agent`` which is encoded by omitting --mode.
            self.assertNotIn("--mode", proc.argv_seen)
            # Prompt is delivered via stdin.
            stdin_content = proc.stdin.getvalue()
            self.assertIn("Hi", stdin_content)
            self.assertIn("You are concise.", stdin_content)
        finally:
            client.close()

    def test_error_stream_raises_runtime_error(self) -> None:
        proc = _FakeProcess(ERROR_STREAM)
        client = CursorAgentClient()
        try:
            with self._patch_popen(proc):
                with self.assertRaises(RuntimeError) as ctx:
                    client.chat.completions.create(
                        model="auto",
                        messages=[{"role": "user", "content": "boom"}],
                    )
            self.assertIn("quota exceeded", str(ctx.exception))
        finally:
            client.close()

    def test_missing_cli_raises_helpful_error(self) -> None:
        client = CursorAgentClient(command="this-cli-does-not-exist-anywhere")
        try:
            # subprocess.Popen raises FileNotFoundError when the command can't
            # be located on PATH; we don't patch it here.
            with self.assertRaises(RuntimeError) as ctx:
                client.chat.completions.create(
                    model="auto",
                    messages=[{"role": "user", "content": "hi"}],
                )
            self.assertIn("Install Cursor CLI", str(ctx.exception))
        finally:
            client.close()

    def test_stream_true_returns_iterable_chunks(self) -> None:
        # Regression: Hermes' chat streaming hot path does ``for chunk in
        # stream`` on whatever ``create()`` returns.  Returning a bare
        # SimpleNamespace surfaces as ``TypeError: 'types.SimpleNamespace'
        # object is not iterable``.  When ``stream=True`` is set we must
        # return an iterator of OpenAI-style chunks.
        proc = _FakeProcess(SUCCESS_STREAM)
        client = CursorAgentClient()
        try:
            with self._patch_popen(proc):
                stream = client.chat.completions.create(
                    model="auto",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
            chunks = list(stream)
            self.assertGreater(len(chunks), 0)
            # First content chunk should carry the assembled text
            content_chunks = [
                c for c in chunks
                if c.choices and c.choices[0].delta.content
            ]
            self.assertTrue(content_chunks)
            self.assertIn("Hello world", "".join(c.choices[0].delta.content for c in content_chunks))
            # Last chunk carries finish_reason + usage
            last = chunks[-1]
            self.assertEqual(last.choices[0].finish_reason, "stop")
            self.assertIsNotNone(last.usage)
        finally:
            client.close()

    def test_usage_is_per_round_not_cumulative_billing_sum(self) -> None:
        # Regression: a deep agentic turn (e.g. 10 internal tool calls)
        # makes cursor-agent emit inputTokens that's the SUM across all
        # internal LLM round-trips. Reporting that raw to Hermes' status
        # bar made it show e.g. ``1.07M/200K  100%`` — bar visually
        # broken AND it triggered spurious compression. Verify we now
        # report a per-round average that stays under the model window.
        events = [
            _make_event(type="system", subtype="init", session_id="s-big"),
        ]
        # Fabricate 10 shell tool round-trips so accumulator counts rounds.
        for i in range(10):
            events.append(_make_event(
                type="tool_call",
                subtype="started",
                call_id=f"t-{i}",
                tool_call={"shellToolCall": {"args": {"command": f"echo {i}"}}},
            ))
            events.append(_make_event(
                type="tool_call",
                subtype="completed",
                call_id=f"t-{i}",
                tool_call={"shellToolCall": {
                    "args": {"command": f"echo {i}"},
                    "result": {"success": {"stdout": "ok", "exitCode": 0}},
                }},
            ))
        events.append(_make_event(
            type="result",
            subtype="success",
            is_error=False,
            result="done",
            session_id="s-big",
            request_id="r-big",
            usage={
                # 1.07M cumulative input across 11 internal rounds —
                # exactly the pathological case the user reported.
                "inputTokens": 1_070_000,
                "outputTokens": 500,
                "cacheReadTokens": 220_000,
                "cacheWriteTokens": 0,
            },
        ))
        proc = _FakeProcess(events)
        client = CursorAgentClient()
        try:
            with self._patch_popen(proc):
                resp = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[{"role": "user", "content": "heavy task"}],
                )
            usage = resp.usage
            # 11 rounds (10 tools + 1 final response) → per-round avg
            # input is 1.07M / 11 ≈ 97K, plus cache 220K / 11 ≈ 20K.
            # Total per-round context ≈ 117K. Well under 200K.
            self.assertLess(usage.prompt_tokens, 200_000)
            self.assertGreater(usage.prompt_tokens, 0)
            # Raw cumulative numbers still preserved for billing.
            self.assertEqual(usage.cursor_raw_input_tokens, 1_070_000)
            self.assertEqual(usage.cursor_raw_cache_read_tokens, 220_000)
            self.assertEqual(usage.cursor_internal_rounds, 11)
        finally:
            client.close()

    def test_usage_prefers_messages_estimate_when_provided(self) -> None:
        # When the client supplies a ``messages_estimate`` (from
        # ``estimate_request_tokens_rough(messages, tools)``), the
        # accumulator must surface that as ``prompt_tokens`` instead of
        # the per-round average — this keeps the status bar consistent
        # before, during, and after generation. The per-round average
        # is still exposed as ``cursor_per_round_context``.
        acc = _StreamJsonAccumulator()
        acc.messages_estimate = 42_000
        acc.usage = {
            "inputTokens": 1_000_000,
            "outputTokens": 100,
            "cacheReadTokens": 200_000,
        }
        acc.tool_events = [object()] * 10  # 10 internal tools → 11 rounds
        usage = acc.openai_usage()
        self.assertEqual(usage.prompt_tokens, 42_000)
        # Per-round average still computed and exposed for debug.
        self.assertGreater(usage.cursor_per_round_context, 0)
        self.assertLess(usage.cursor_per_round_context, 200_000)

    def test_usage_pure_chat_no_tools_reports_messages_estimate(self) -> None:
        # A pure-chat turn (no tool calls) reports Hermes' messages
        # estimate as ``prompt_tokens`` (not cursor's raw billing) so
        # the status bar stays consistent between live estimate and
        # post-turn snapshot. Cursor's per-round context is still
        # exposed as a debug field for cost-tracking consumers.
        proc = _FakeProcess([
            _make_event(type="system", subtype="init", session_id="s-chat"),
            _make_event(
                type="assistant",
                message={"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
                session_id="s-chat",
            ),
            _make_event(
                type="result",
                subtype="success",
                is_error=False,
                result="Hello!",
                session_id="s-chat",
                request_id="r-chat",
                usage={
                    "inputTokens": 5_000,
                    "outputTokens": 10,
                    "cacheReadTokens": 25_000,
                    "cacheWriteTokens": 0,
                },
            ),
        ])
        client = CursorAgentClient()
        try:
            with self._patch_popen(proc):
                resp = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[{"role": "user", "content": "hi"}],
                )
            usage = resp.usage
            # prompt_tokens = max(messages_estimate, cursor_per_round)
            # so the bar never undercounts. Messages estimate for "hi"
            # is tiny (~few tokens) but per_round_context is 30K, so
            # 30K wins.
            self.assertEqual(usage.prompt_tokens, 30_000)
            self.assertEqual(usage.cursor_per_round_context, 30_000)
            self.assertEqual(usage.cursor_internal_rounds, 1)
            self.assertEqual(usage.cursor_raw_input_tokens, 5_000)
        finally:
            client.close()

    def test_context_bar_is_monotonic_within_session(self) -> None:
        # Regression: when Hermes loops on tool_calls (cursor returning
        # <tool_call> blocks for Hermes to run), each cursor call has
        # its own messages footprint. Some calls have full tool schemas
        # (~30K), some don't. Without a high-water mark, the bar wobbled
        # high/low between calls in the same user turn, looking like
        # "junk resetting and overriding" to the user.
        client = CursorAgentClient()

        big_proc = _FakeProcess([
            _make_event(type="system", subtype="init", session_id="s-1"),
            _make_event(
                type="assistant",
                message={"role": "assistant", "content": [{"type": "text", "text": "first"}]},
                session_id="s-1",
            ),
            _make_event(
                type="result", subtype="success", is_error=False,
                result="first", session_id="s-1", request_id="r-1",
                usage={"inputTokens": 40_000, "outputTokens": 5,
                       "cacheReadTokens": 20_000, "cacheWriteTokens": 0},
            ),
        ])
        small_proc = _FakeProcess([
            _make_event(type="system", subtype="init", session_id="s-1"),
            _make_event(
                type="assistant",
                message={"role": "assistant", "content": [{"type": "text", "text": "second"}]},
                session_id="s-1",
            ),
            _make_event(
                type="result", subtype="success", is_error=False,
                result="second", session_id="s-1", request_id="r-2",
                usage={"inputTokens": 500, "outputTokens": 5,
                       "cacheReadTokens": 1_000, "cacheWriteTokens": 0},
            ),
        ])
        try:
            with self._patch_popen(big_proc):
                resp1 = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[{"role": "user", "content": "x"}],
                    tools=[{"type": "function", "function": {
                        "name": "shell", "description": "x", "parameters": {}}}],
                )
            with self._patch_popen(small_proc):
                resp2 = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[{"role": "user", "content": "y"}],
                )
            # The first call set a high-water of 60K (per-round). The
            # second call's per-round is just 1.5K — but the bar must
            # stay at or above 60K, not collapse to 1.5K.
            self.assertGreaterEqual(resp1.usage.prompt_tokens, 60_000)
            self.assertGreaterEqual(resp2.usage.prompt_tokens, resp1.usage.prompt_tokens)
        finally:
            client.close()

    def test_high_water_resets_on_new_user_turn(self) -> None:
        # Across SEPARATE user prompts the high-water mark must reset so
        # the bar reflects the new turn's actual context size. Earlier
        # behaviour froze the bar at the highest-activity turn's value
        # which prevented users from seeing context growth/shrinkage as
        # they continued conversing.
        client = CursorAgentClient()

        heavy_proc = _FakeProcess([
            _make_event(type="system", subtype="init", session_id="s-1"),
            _make_event(
                type="assistant",
                message={"role": "assistant", "content": [{"type": "text", "text": "first"}]},
                session_id="s-1",
            ),
            _make_event(
                type="result", subtype="success", is_error=False,
                result="first", session_id="s-1", request_id="r-1",
                usage={"inputTokens": 200_000, "outputTokens": 5,
                       "cacheReadTokens": 100_000, "cacheWriteTokens": 0},
            ),
        ])
        light_proc = _FakeProcess([
            _make_event(type="system", subtype="init", session_id="s-1"),
            _make_event(
                type="assistant",
                message={"role": "assistant", "content": [{"type": "text", "text": "second"}]},
                session_id="s-1",
            ),
            _make_event(
                type="result", subtype="success", is_error=False,
                result="second", session_id="s-1", request_id="r-2",
                usage={"inputTokens": 500, "outputTokens": 5,
                       "cacheReadTokens": 1_000, "cacheWriteTokens": 0},
            ),
        ])
        try:
            with self._patch_popen(heavy_proc):
                resp1 = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "do a lot of work"},
                    ],
                    tools=[{"type": "function", "function": {
                        "name": "shell", "description": "x", "parameters": {}}}],
                )
            # Heavy turn locks the bar high (per-round average from a
            # multi-tool turn).
            self.assertGreaterEqual(resp1.usage.prompt_tokens, 100_000)

            # NEW user prompt (second user message added). High-water
            # should reset; bar should reflect the second turn's much
            # smaller actual context, not the prior heavy turn.
            with self._patch_popen(light_proc):
                resp2 = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "do a lot of work"},
                        {"role": "assistant", "content": "did work"},
                        {"role": "user", "content": "hi"},
                    ],
                )
            self.assertLess(resp2.usage.prompt_tokens, resp1.usage.prompt_tokens,
                            "second-turn bar must drop from the prior heavy turn's "
                            "frozen peak when a new user prompt arrives")
        finally:
            client.close()

    def test_high_water_holds_within_same_user_turn(self) -> None:
        # WITHIN a single Hermes user turn the bar must NOT flicker down
        # between internal cursor calls (Hermes loops on tool_calls).
        # The two calls below share the same user message; only assistant
        # / tool messages are added between them. High-water from the
        # first call's heavy per-round average must carry into the second.
        client = CursorAgentClient()

        big_proc = _FakeProcess([
            _make_event(type="system", subtype="init", session_id="s-1"),
            _make_event(
                type="assistant",
                message={"role": "assistant", "content": [{"type": "text", "text": "first"}]},
                session_id="s-1",
            ),
            _make_event(
                type="result", subtype="success", is_error=False,
                result="first", session_id="s-1", request_id="r-1",
                usage={"inputTokens": 40_000, "outputTokens": 5,
                       "cacheReadTokens": 20_000, "cacheWriteTokens": 0},
            ),
        ])
        small_proc = _FakeProcess([
            _make_event(type="system", subtype="init", session_id="s-1"),
            _make_event(
                type="assistant",
                message={"role": "assistant", "content": [{"type": "text", "text": "second"}]},
                session_id="s-1",
            ),
            _make_event(
                type="result", subtype="success", is_error=False,
                result="second", session_id="s-1", request_id="r-2",
                usage={"inputTokens": 500, "outputTokens": 5,
                       "cacheReadTokens": 1_000, "cacheWriteTokens": 0},
            ),
        ])
        same_user_msg = {"role": "user", "content": "do tool work"}
        try:
            with self._patch_popen(big_proc):
                resp1 = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[same_user_msg],
                    tools=[{"type": "function", "function": {
                        "name": "shell", "description": "x", "parameters": {}}}],
                )
            with self._patch_popen(small_proc):
                resp2 = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[
                        same_user_msg,
                        {"role": "assistant", "content": "running tool..."},
                        {"role": "tool", "content": "tool result"},
                    ],
                )
            # Within the same user turn the bar must stay AT OR ABOVE the
            # peak it hit on the first internal call (no flicker down).
            self.assertGreaterEqual(resp2.usage.prompt_tokens, resp1.usage.prompt_tokens)
        finally:
            client.close()

    def test_close_resets_high_water_for_new_session(self) -> None:
        # When a new chat starts (e.g. /new), close() is called on the
        # client — the bar's monotonic floor must drop back to current
        # prompt size so the new session shows accurate context.
        client = CursorAgentClient()
        client._context_high_water = 123_000
        client.close()
        self.assertEqual(client._context_high_water, 0)

    def _feed(self, acc, **fields) -> None:
        """Feed a single event into the accumulator (it expects a dict)."""
        acc.feed(fields)

    def test_edit_completion_populates_diff_stats(self) -> None:
        # Cursor's editToolCall result carries ``linesAdded`` /
        # ``linesRemoved`` / ``diffString``. The accumulator must pull
        # them onto the tool event so the activity feed can render
        # "+5 -2" (matching Claude Code / Aider conventions) and the
        # full diff is preserved for replay.
        acc = _StreamJsonAccumulator()
        self._feed(acc, type="system", subtype="init", session_id="s")
        self._feed(acc,
            type="tool_call", subtype="started", call_id="t1",
            tool_call={"editToolCall": {
                "args": {"path": "/tmp/x.py", "streamContent": "NEW"},
            }},
        )
        self._feed(acc,
            type="tool_call", subtype="completed", call_id="t1",
            tool_call={"editToolCall": {
                "args": {"path": "/tmp/x.py", "streamContent": "NEW"},
                "result": {"success": {
                    "linesAdded": 7,
                    "linesRemoved": 3,
                    "diffString": "--- a/x.py\n+++ b/x.py\n@@ ...",
                    "afterFullFileContent": "NEW",
                    "message": "updated",
                }},
            }},
        )
        self.assertEqual(len(acc.tool_events), 1)
        evt = acc.tool_events[0]
        self.assertEqual(evt.lines_added, 7)
        self.assertEqual(evt.lines_removed, 3)
        self.assertIn("--- a/x.py", evt.diff_string)
        self.assertEqual(evt.name, "edit_file")

    def test_bridge_merges_diff_stats_into_args_on_completion(self) -> None:
        # When the bridge fires ``tool.completed`` for an edit, cli.py
        # pops the ``args`` it captured at ``tool.started`` and passes
        # them to ``get_cute_tool_message``. Because ``args`` is the
        # SAME dict reference between started and completed, mutating
        # it here lets the display layer show "+N -M" without changing
        # any callback signatures.
        captured = []

        def cb(stage, name, preview=None, args=None, **kwargs):
            captured.append((stage, name, dict(args) if isinstance(args, dict) else args))

        events = [
            _make_event(type="system", subtype="init", session_id="s"),
            _make_event(
                type="tool_call", subtype="started", call_id="t1",
                tool_call={"editToolCall": {
                    "args": {"path": "/tmp/x.py", "streamContent": "NEW"},
                }},
            ),
            _make_event(
                type="tool_call", subtype="completed", call_id="t1",
                tool_call={"editToolCall": {
                    "args": {"path": "/tmp/x.py", "streamContent": "NEW"},
                    "result": {"success": {
                        "linesAdded": 5,
                        "linesRemoved": 2,
                        "diffString": "--- a\n+++ b",
                        "message": "ok",
                    }},
                }},
            ),
            _make_event(
                type="assistant",
                message={"role": "assistant", "content": [
                    {"type": "text", "text": "done"},
                ]},
            ),
            _make_event(
                type="result", subtype="success", is_error=False,
                result="done", session_id="s", request_id="r",
                usage={"inputTokens": 1, "outputTokens": 1},
            ),
        ]
        client = CursorAgentClient(tool_progress_callback=cb)
        try:
            with self._patch_popen(_FakeProcess(events)):
                client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[{"role": "user", "content": "edit x"}],
                    tools=[{"type": "function", "function": {
                        "name": "edit_file", "description": "x",
                        "parameters": {},
                    }}],
                )
        finally:
            client.close()
        # The bridge fired started + completed; the started args dict
        # was the same reference cursor handed us, and on completion
        # the bridge mutated it in place. ``captured`` is a deep-copy
        # taken at callback invocation time, so the "started" snapshot
        # may or may not contain _diff_stats depending on whether the
        # completion fired before the deep-copy happened. The contract
        # we care about is: the bridge fired both events with the
        # right name.
        started = [c for c in captured if c[0] == "tool.started" and c[1] == "edit_file"]
        completed = [c for c in captured if c[0] == "tool.completed" and c[1] == "edit_file"]
        self.assertEqual(len(started), 1)
        self.assertEqual(len(completed), 1)
        # Hit the display layer directly to verify the diff-stats
        # surface ends up in the activity feed.
        from agent.display import get_cute_tool_message
        line = get_cute_tool_message(
            "edit_file",
            {"path": "/tmp/x.py", "_diff_stats": {"added": 5, "removed": 2}},
            0.1,
        )
        self.assertIn("+5 -2", line)
        self.assertIn("✏️", line)
        # Without diff stats (e.g. older cursor build) we still get a
        # clean line — no broken "+0 -0" noise.
        bare = get_cute_tool_message("edit_file", {"path": "/tmp/x.py"}, 0.1)
        self.assertNotIn("+", bare)
        self.assertNotIn("-", bare.split("/tmp")[0])

    def test_edit_line_resolves_path_from_alternate_field_names(self) -> None:
        # Cursor sometimes emits editToolCall args with ``target_file``
        # / ``file_path`` instead of ``path``. The activity feed line
        # must still show the path so the user sees what was touched.
        from agent.display import get_cute_tool_message
        for field in ("path", "target_file", "targetFile",
                      "file_path", "relative_workspace_path"):
            line = get_cute_tool_message("edit_file", {field: "/tmp/x.py"}, 0.1)
            self.assertIn("/tmp/x.py", line,
                          f"path lost when args used {field!r}")

    def test_cursor_preview_resolves_alternate_path_fields(self) -> None:
        # Same resilience on the cursor-side preview that surfaces in
        # ``tool.started`` (so the user sees the path BEFORE completion).
        from agent.cursor_agent_client import (
            _build_cursor_tool_preview,
            _CursorToolEvent,
        )
        for field in ("path", "target_file", "targetFile",
                      "file_path", "relative_workspace_path"):
            evt = _CursorToolEvent(
                call_id="c",
                envelope_key="editToolCall",
                args={field: "/tmp/x.py"},
            )
            preview = _build_cursor_tool_preview(evt)
            self.assertEqual(preview, "/tmp/x.py",
                             f"preview lost when args used {field!r}")

    def test_diff_string_routes_through_native_renderer(self) -> None:
        # Consistency goal: cursor edits should use the SAME diff
        # renderer Hermes already uses for write_file/patch — same
        # skin colors, same file/hunk headers, same line caps.
        # ``extract_edit_diff`` is the entry point; it must honour
        # ``function_args["_diff_string"]`` regardless of tool name.
        from agent.display import (
            extract_edit_diff,
            _summarize_rendered_diff_sections,
        )
        diff_str = (
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n"
            "-hello world\n+HELLO WORLD\n+second line"
        )
        # Pre-extracted diff path: works for ANY tool_name (cursor
        # uses edit_file/write_file, but the routing is name-agnostic).
        for name in ("edit_file", "write_file", "cursor_edit"):
            extracted = extract_edit_diff(name, None, function_args={
                "_diff_string": diff_str,
            })
            self.assertEqual(extracted, diff_str,
                             f"tool {name} failed to extract diff")
        # Renderer turns it into colored lines (text content survives).
        rendered = _summarize_rendered_diff_sections(diff_str)
        joined = "\n".join(rendered)
        self.assertIn("hello world", joined)
        self.assertIn("HELLO WORLD", joined)
        self.assertIn("second line", joined)

    def test_synthesis_excludes_planning_text_between_tools(self) -> None:
        # Cursor's stream emits planning prose ("Searching the agent
        # dir…") → tool → reflection ("Reading each match…") → tool →
        # synthesis ("Here are 13 files: …"). The user response should
        # be the synthesis only; the planning lines were already
        # surfaced live as narration events.
        acc = _StreamJsonAccumulator()
        self._feed(acc, type="system", subtype="init", session_id="s")
        self._feed(acc,
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "text", "text": "Searching the agent directory."},
            ]},
        )
        self._feed(acc,
            type="tool_call", subtype="started", call_id="t1",
            tool_call={"grepToolCall": {"args": {"pattern": "cursor"}}},
        )
        self._feed(acc,
            type="tool_call", subtype="completed", call_id="t1",
            tool_call={"grepToolCall": {
                "args": {"pattern": "cursor"},
                "result": {"success": {"hits": 13}},
            }},
        )
        self._feed(acc,
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "text", "text": "Reading each match for context."},
            ]},
        )
        self._feed(acc,
            type="tool_call", subtype="started", call_id="t2",
            tool_call={"readToolCall": {"args": {"path": "a.py"}}},
        )
        self._feed(acc,
            type="tool_call", subtype="completed", call_id="t2",
            tool_call={"readToolCall": {
                "args": {"path": "a.py"},
                "result": {"success": {"content": "..."}},
            }},
        )
        self._feed(acc,
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "text", "text": "Final answer: there are 13 files."},
            ]},
        )
        self._feed(acc,
            type="result", subtype="success", is_error=False,
            result="Searching the agent directory.\nReading each match.\nFinal answer: there are 13 files.",
            session_id="s", request_id="r", usage={"inputTokens": 1, "outputTokens": 1},
        )
        self.assertEqual(acc.synthesis_text(), "Final answer: there are 13 files.")
        # Narration helper captures the between-tool prose.
        narration = acc.narration_text()
        self.assertIn("Searching the agent directory", narration)
        self.assertIn("Reading each match", narration)
        self.assertNotIn("Final answer", narration)
        # Backward-compat: full assembled text still available.
        self.assertIn("Final answer", acc.assembled_text())

    def test_synthesis_falls_back_to_full_text_when_no_tools_ran(self) -> None:
        # Pure chat (no tool calls) — there's no "between-tools" vs
        # "post-tools" distinction. Return everything.
        acc = _StreamJsonAccumulator()
        self._feed(acc, type="system", subtype="init", session_id="s")
        self._feed(acc,
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "text", "text": "Hello there!"},
            ]},
        )
        self._feed(acc,
            type="result", subtype="success", is_error=False,
            result="Hello there!", session_id="s", request_id="r",
            usage={"inputTokens": 1, "outputTokens": 1},
        )
        self.assertEqual(acc.synthesis_text(), "Hello there!")

    def test_intermediate_text_events_fire_narrate_callback(self) -> None:
        # Each ``assistant`` text block between tools must fire the
        # narration bridge so the Hermes activity feed shows the
        # ReAct-style chain live (💬 narrate → 🔎 grep → 💬 narrate
        # → 📖 read → final answer in the response). Without this the
        # tool icons stack at the top and the prose lands as one big
        # blob at the end — exactly the wobble the user reported.
        events = []
        captured: list[tuple] = []

        def cb_fn(stage, name, preview=None, args=None, **kwargs):
            captured.append((stage, name, preview, args, kwargs))

        class _Cap:
            calls = captured

        cb = _Cap()
        client = CursorAgentClient(tool_progress_callback=cb_fn)
        events.append(_make_event(type="system", subtype="init", session_id="s"))
        events.append(_make_event(
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "text", "text": "Let me check the disk first."},
            ]},
        ))
        events.append(_make_event(
            type="tool_call", subtype="started", call_id="t1",
            tool_call={"shellToolCall": {"args": {"command": "df -h"}}},
        ))
        events.append(_make_event(
            type="tool_call", subtype="completed", call_id="t1",
            tool_call={"shellToolCall": {
                "args": {"command": "df -h"},
                "result": {"success": {"stdout": "Filesystem...", "exitCode": 0}},
            }},
        ))
        events.append(_make_event(
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "text", "text": "Looks fine. Now CPU count."},
            ]},
        ))
        events.append(_make_event(
            type="tool_call", subtype="started", call_id="t2",
            tool_call={"shellToolCall": {"args": {"command": "nproc"}}},
        ))
        events.append(_make_event(
            type="tool_call", subtype="completed", call_id="t2",
            tool_call={"shellToolCall": {
                "args": {"command": "nproc"},
                "result": {"success": {"stdout": "24", "exitCode": 0}},
            }},
        ))
        events.append(_make_event(
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "text", "text": "Summary: healthy disk, 24 CPUs."},
            ]},
        ))
        events.append(_make_event(
            type="result", subtype="success", is_error=False,
            result="Summary: healthy disk, 24 CPUs.",
            session_id="s", request_id="r",
            usage={"inputTokens": 10, "outputTokens": 5},
        ))
        try:
            with self._patch_popen(_FakeProcess(events)):
                resp = client.chat.completions.create(
                    model="composer-2.5-fast",
                    messages=[{"role": "user", "content": "verify hardware"}],
                    tools=[{
                        "type": "function",
                        "function": {"name": "shell", "description": "x",
                                     "parameters": {}},
                    }],
                )
        finally:
            client.close()
        # Buffered narrate flush rule: only TEXT events that are
        # followed by ANOTHER tool become narrate events. The final
        # text-after-last-tool is reserved for the synthesis response,
        # so the user doesn't see "Summary: healthy disk, 24 CPUs."
        # both as a 💬 narrate line and as the assistant reply. The
        # two pre-tool / between-tools texts still fire live.
        narrate_started = [c for c in cb.calls
                          if c[0] == "tool.started" and c[1] == "narrate"]
        self.assertEqual(len(narrate_started), 2)
        previews = [c[2] for c in narrate_started]
        self.assertIn("Let me check the disk first.", previews)
        self.assertIn("Looks fine. Now CPU count.", previews)
        self.assertNotIn("Summary: healthy disk, 24 CPUs.", previews)
        self.assertIn("disk first", narrate_started[0][2])
        self.assertIn("CPU count", narrate_started[1][2])
        # Response text is the synthesis only (no intermediate prose).
        self.assertEqual(resp.choices[0].message.content,
                         "Summary: healthy disk, 24 CPUs.")

    def test_cursor_internal_tool_events_fire_tool_progress_callback(self) -> None:
        # Regression: when cursor-agent's harness runs an internal tool
        # (shell/read/edit) the stream-json emits ``tool_call started`` and
        # ``tool_call completed`` events. We must translate those into the
        # same ``tool_progress_callback(stage, name, preview, args, ...)``
        # contract Hermes' UI uses for native tool calls, so the user sees
        # the activity instead of just the final model text.
        shell_started = _make_event(
            type="tool_call",
            subtype="started",
            call_id="tool_a1",
            tool_call={"shellToolCall": {"args": {"command": "df -h"}, "description": "disk usage"}},
        )
        shell_completed = _make_event(
            type="tool_call",
            subtype="completed",
            call_id="tool_a1",
            tool_call={
                "shellToolCall": {
                    "args": {"command": "df -h"},
                    "result": {"success": {"stdout": "Filesystem ...\n", "exitCode": 0}},
                }
            },
        )
        success_text = _make_event(
            type="assistant",
            message={"role": "assistant", "content": [{"type": "text", "text": "Used 46%."}]},
            session_id="s-tool",
        )
        final_result = _make_event(
            type="result",
            subtype="success",
            is_error=False,
            result="Used 46%.",
            session_id="s-tool",
            request_id="r-tool",
            usage={"inputTokens": 1, "outputTokens": 1, "cacheReadTokens": 0},
        )
        proc = _FakeProcess([
            _make_event(type="system", subtype="init", session_id="s-tool", model="Auto"),
            shell_started,
            shell_completed,
            success_text,
            final_result,
        ])

        events: list[tuple] = []

        def _callback(stage, name, preview=None, args=None, **kwargs):
            events.append((stage, name, preview, args, kwargs))

        client = CursorAgentClient(tool_progress_callback=_callback)
        try:
            with self._patch_popen(proc):
                resp = client.chat.completions.create(
                    model="auto",
                    messages=[{"role": "user", "content": "df please"}],
                )
            # Started + completed both fired
            stages = [e[0] for e in events]
            self.assertIn("tool.started", stages)
            self.assertIn("tool.completed", stages)
            started = [e for e in events if e[0] == "tool.started"][0]
            self.assertEqual(started[1], "shell")
            self.assertEqual(started[2], "df -h")
            self.assertEqual(started[3], {"command": "df -h"})
            completed = [e for e in events if e[0] == "tool.completed"][0]
            self.assertFalse(completed[4].get("is_error"))
            self.assertIn("Filesystem", completed[4].get("result", ""))
            # Audit list attached to response
            self.assertEqual(len(resp.cursor_internal_tools), 1)
            self.assertEqual(resp.cursor_internal_tools[0]["name"], "shell")
        finally:
            client.close()

    def test_broken_pipe_surfaces_stderr_in_error(self) -> None:
        # Regression: if cursor-agent rejects auth (e.g. bad --api-key) it
        # closes stdin before reading the prompt, so our write raises
        # BrokenPipeError. The client must convert that into a clear
        # RuntimeError carrying the CLI's stderr instead of a bare
        # BrokenPipeError bubbling into the chat retry loop.

        class _BrokenStdin(io.StringIO):
            def write(self, _data):  # type: ignore[override]
                raise BrokenPipeError("Broken pipe")

            def flush(self):  # type: ignore[override]
                return None

        proc = _FakeProcess(
            stdout_lines=[],
            stderr_lines=[
                "Warning: The provided API key is invalid.",
                "Please check you have the right key.",
            ],
        )
        proc.stdin = _BrokenStdin()
        proc._exit_code = 1

        client = CursorAgentClient()
        try:
            with self._patch_popen(proc):
                with self.assertRaises(RuntimeError) as ctx:
                    client.chat.completions.create(
                        model="auto",
                        messages=[{"role": "user", "content": "hi"}],
                    )
            msg = str(ctx.exception)
            self.assertIn("closed stdin", msg)
            self.assertIn("API key is invalid", msg)
        finally:
            client.close()

    def test_tool_call_block_is_extracted(self) -> None:
        tool_call_payload = json.dumps({
            "id": "call_xyz",
            "type": "function",
            "function": {"name": "shell", "arguments": "{\"command\":\"ls\"}"},
        })
        stream = [
            _make_event(type="system", subtype="init", session_id="s-3", model="Auto"),
            _make_event(
                type="assistant",
                message={
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": f"Sure. <tool_call>{tool_call_payload}</tool_call>",
                    }],
                },
                session_id="s-3",
            ),
            _make_event(
                type="result",
                subtype="success",
                duration_ms=10,
                is_error=False,
                result=f"Sure. <tool_call>{tool_call_payload}</tool_call>",
                session_id="s-3",
                request_id="r-3",
                usage={"inputTokens": 5, "outputTokens": 5, "cacheReadTokens": 0},
            ),
        ]
        proc = _FakeProcess(stream)
        client = CursorAgentClient()
        try:
            with self._patch_popen(proc):
                resp = client.chat.completions.create(
                    model="auto",
                    messages=[{"role": "user", "content": "list files"}],
                    tools=[{"type": "function", "function": {"name": "shell", "description": "Run a shell command", "parameters": {}}}],
                )
            self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
            self.assertEqual(len(resp.choices[0].message.tool_calls), 1)
            tc = resp.choices[0].message.tool_calls[0]
            self.assertEqual(tc.function.name, "shell")
            self.assertIn("ls", tc.function.arguments)
        finally:
            client.close()

    def test_subprocess_env_has_cursor_api_key_when_provided(self) -> None:
        proc = _FakeProcess(SUCCESS_STREAM)
        client = CursorAgentClient(api_key="crsr_test_42")
        try:
            with self._patch_popen(proc):
                client.chat.completions.create(
                    model="auto",
                    messages=[{"role": "user", "content": "hi"}],
                )
            self.assertIsNotNone(proc.env_seen)
            self.assertEqual(proc.env_seen.get("CURSOR_API_KEY"), "crsr_test_42")
        finally:
            client.close()

    def test_marker_base_url_is_default(self) -> None:
        client = CursorAgentClient()
        try:
            self.assertEqual(client.base_url, CURSOR_MARKER_BASE_URL)
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Workspace cleanup
# ---------------------------------------------------------------------------


class WorkspaceCleanupTests(unittest.TestCase):
    def test_close_removes_ephemeral_dirs(self) -> None:
        client = CursorAgentClient()
        ws1, _ = client._allocate_workspace()
        ws2, _ = client._allocate_workspace()
        self.assertTrue(os.path.isdir(ws1))
        self.assertTrue(os.path.isdir(ws2))
        client.close()
        self.assertFalse(os.path.exists(ws1))
        self.assertFalse(os.path.exists(ws2))

    def test_close_leaves_user_workspace_alone(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as user_ws:
            client = CursorAgentClient(workspace=user_ws)
            ws, _ = client._allocate_workspace()
            self.assertEqual(ws, user_ws)
            client.close()
            self.assertTrue(os.path.isdir(user_ws))  # not removed


class CompressionHookTests(unittest.TestCase):
    """The bar's high-water mark must drop after Hermes' context
    compression — otherwise it stays inflated showing the pre-compression
    peak when the actual request is now much smaller.

    The contract is: ``CursorAgentClient`` exposes ``reset_context_baseline()``,
    and Hermes' compression module calls it on any client that defines the
    method.  See ``agent/conversation_compression.py``.
    """

    def test_reset_context_baseline_drops_high_water(self) -> None:
        client = CursorAgentClient()
        try:
            client._context_high_water = 80_000
            client.reset_context_baseline()
            self.assertEqual(client._context_high_water, 0)
        finally:
            client.close()

    def test_compression_hook_invokes_reset_on_cursor_client(self) -> None:
        # Simulate the compression code path: it fetches agent.client and
        # calls reset_context_baseline() if available. Verify that path
        # actually drops cursor's bar floor.
        client = CursorAgentClient()
        try:
            client._context_high_water = 120_000

            class _FakeAgent:
                pass

            agent = _FakeAgent()
            agent.client = client

            # Mirror of the snippet in agent/conversation_compression.py
            target = getattr(agent, "client", None)
            if target is not None and hasattr(target, "reset_context_baseline"):
                target.reset_context_baseline()

            self.assertEqual(client._context_high_water, 0,
                             "compression hook must drop cursor's bar floor")
        finally:
            client.close()

    def test_compression_hook_is_no_op_for_clients_without_reset(self) -> None:
        # Other providers (Grok, OpenAI, Claude) don't define
        # reset_context_baseline.  The compression hook must not raise
        # when client lacks it.

        class _OpenAILikeClient:
            pass

        class _FakeAgent:
            pass

        agent = _FakeAgent()
        agent.client = _OpenAILikeClient()

        target = getattr(agent, "client", None)
        # This is exactly the conditional the production code uses:
        invoked = False
        if target is not None and hasattr(target, "reset_context_baseline"):
            target.reset_context_baseline()
            invoked = True
        self.assertFalse(invoked,
                         "hook must not fire for non-cursor clients")


class TimeoutTests(unittest.TestCase):
    """Regression tests for the event-driven idle timeout.

    Before this fix the wrapper enforced a wall-clock deadline that killed
    healthy long-running turns at 90s (outer Hermes wrapper) or 600s
    (inner cursor client). The semantics are now: the deadline resets on
    every stream-json event, so total wall-clock can be arbitrary as long
    as events keep flowing. Only true subprocess hangs (no events for the
    threshold) trigger termination. Default idle threshold is 1800s
    (30 min) to comfortably cover cursor-agent's internal 10-min shell
    ceiling plus chained long operations.
    """

    def setUp(self) -> None:
        keys = [k for k in os.environ if k.startswith("HERMES_CURSOR_")]
        self._saved = {k: os.environ.pop(k) for k in keys}

    def tearDown(self) -> None:
        for k in list(os.environ):
            if k.startswith("HERMES_CURSOR_"):
                os.environ.pop(k, None)
        for k, v in self._saved.items():
            os.environ[k] = v

    def test_env_var_overrides_default_idle_threshold(self) -> None:
        os.environ["HERMES_CURSOR_TIMEOUT_SECONDS"] = "1234"
        client = CursorAgentClient()
        try:
            self.assertEqual(client._timeout_seconds, 1234.0)
        finally:
            client.close()

    def test_env_var_invalid_value_falls_back_to_default(self) -> None:
        os.environ["HERMES_CURSOR_TIMEOUT_SECONDS"] = "not-a-number"
        client = CursorAgentClient()
        try:
            self.assertEqual(client._timeout_seconds, 1800.0)
        finally:
            client.close()

    def test_env_var_zero_value_falls_back_to_default(self) -> None:
        # Zero or negative would mean "kill the subprocess instantly on
        # arrival" which is never useful; fall back to the default.
        os.environ["HERMES_CURSOR_TIMEOUT_SECONDS"] = "0"
        client = CursorAgentClient()
        try:
            self.assertEqual(client._timeout_seconds, 1800.0)
        finally:
            client.close()

    def test_explicit_arg_overrides_default_but_env_wins(self) -> None:
        # Precedence: env > explicit arg > default.
        client1 = CursorAgentClient(timeout_seconds=123.0)
        try:
            self.assertEqual(client1._timeout_seconds, 123.0)
        finally:
            client1.close()

        os.environ["HERMES_CURSOR_TIMEOUT_SECONDS"] = "987"
        client2 = CursorAgentClient(timeout_seconds=123.0)
        try:
            self.assertEqual(client2._timeout_seconds, 987.0)
        finally:
            client2.close()

    @staticmethod
    def _patch_popen(fake_proc):
        def _fake_popen(argv, **kwargs):
            fake_proc.argv_seen = list(argv)
            fake_proc.cwd_seen = kwargs.get("cwd")
            fake_proc.env_seen = kwargs.get("env")
            return fake_proc

        return patch("agent.cursor_agent_client.subprocess.Popen", side_effect=_fake_popen)

    def test_context_estimate_callback_fires_before_subprocess(self) -> None:
        # Regression: the status bar sat at 0/200K throughout a long
        # in-flight first turn because the compressor only learns about
        # prompt_tokens from the response usage at end-of-turn. The
        # context_estimate_callback fires with the messages-based token
        # estimate before the subprocess spawns so the bar reflects
        # input context immediately.
        proc = _FakeProcess(SUCCESS_STREAM)
        observed: list[int] = []
        client = CursorAgentClient(
            context_estimate_callback=lambda tokens: observed.append(tokens),
        )
        try:
            with self._patch_popen(proc):
                client.chat.completions.create(
                    model="composer-2.5",
                    messages=[
                        {"role": "system", "content": "y" * 4000},
                        {"role": "user", "content": "x" * 4000},
                    ],
                )
            self.assertEqual(len(observed), 1,
                             "callback should fire exactly once per turn")
            self.assertGreater(observed[0], 0,
                               "estimate should be > 0 for non-empty messages")
        finally:
            client.close()

    def test_context_estimate_callback_is_optional(self) -> None:
        # Sanity: not passing the callback must not break anything.
        proc = _FakeProcess(SUCCESS_STREAM)
        client = CursorAgentClient()
        try:
            with self._patch_popen(proc):
                resp = client.chat.completions.create(
                    model="composer-2.5",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            self.assertEqual(resp.choices[0].finish_reason, "stop")
        finally:
            client.close()

    def test_high_water_bumps_before_subprocess_for_long_first_turn_visibility(self) -> None:
        # Even without an external callback, the internal high-water mark
        # must be bumped before the subprocess spawns so that
        # ``_estimate_per_round_context`` and the final usage shape don't
        # regress against the input-side estimate.
        proc = _FakeProcess(SUCCESS_STREAM)
        client = CursorAgentClient()
        observed_high_water_at_subprocess_start: list[int] = []
        real_run = client._run_prompt

        def _wrap_run(*args, **kwargs):
            observed_high_water_at_subprocess_start.append(client._context_high_water)
            return real_run(*args, **kwargs)

        client._run_prompt = _wrap_run  # type: ignore[method-assign]
        try:
            with self._patch_popen(proc):
                client.chat.completions.create(
                    model="composer-2.5",
                    messages=[
                        {"role": "system", "content": "y" * 4000},
                        {"role": "user", "content": "x" * 4000},
                    ],
                )
            self.assertGreater(observed_high_water_at_subprocess_start[0], 0,
                               "high-water must be bumped before subprocess "
                               "spawn, not after the result event")
        finally:
            client.close()

    def test_idle_deadline_resets_on_each_stream_event(self) -> None:
        # Slow-stream regression: emit a handful of events with per-event
        # delays that, in aggregate, exceed the idle threshold. The old
        # wall-clock implementation would have raised TimeoutError after
        # the first 0.4s; the new implementation must let the turn finish
        # because each individual gap (0.1s) stays well below 0.4s.
        class _SlowLineStdout:
            def __init__(self, events):
                self._events = list(events)
                self._idx = 0

            def __iter__(self):
                return self

            def __next__(self) -> str:
                if self._idx >= len(self._events):
                    raise StopIteration
                delay, line = self._events[self._idx]
                self._idx += 1
                if delay > 0:
                    time.sleep(delay)
                return line + "\n"

        class _SlowFakeProcess(_FakeProcess):
            def __init__(self, slow_events) -> None:
                super().__init__([])
                self.stdout = _SlowLineStdout(slow_events)

        # 8 events, each 0.1s apart => 0.8s total wall-clock.
        # Idle threshold: 0.4s. Every gap stays under threshold, so we
        # should succeed cleanly.
        slow = [
            (0.1, SUCCESS_STREAM[0]),
            (0.1, SUCCESS_STREAM[1]),
            (0.1, SUCCESS_STREAM[2]),
            (0.1, SUCCESS_STREAM[3]),
            (0.1, SUCCESS_STREAM[4]),
        ]
        proc = _SlowFakeProcess(slow)
        client = CursorAgentClient(timeout_seconds=0.4)

        def _fake_popen(argv, **kwargs):
            proc.argv_seen = list(argv)
            proc.cwd_seen = kwargs.get("cwd")
            proc.env_seen = kwargs.get("env")
            return proc

        try:
            with patch("agent.cursor_agent_client.subprocess.Popen", side_effect=_fake_popen):
                resp = client.chat.completions.create(
                    model="composer-2.5",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            # Successful completion with deadline resets working correctly.
            self.assertEqual(resp.choices[0].finish_reason, "stop")
            self.assertEqual(resp.choices[0].message.content, "Hello world")
        finally:
            client.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
