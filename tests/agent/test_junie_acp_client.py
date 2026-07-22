"""Tests for the JetBrains Junie ACP client shim.

The client drives a Junie agent subprocess over the Agent Client Protocol
Python SDK. The behaviour-critical paths (persistent-process reuse, model/brave
forwarding, no-replay-after-dispatch, fs/permission safety, tool activity) are
exercised end-to-end against ``fake_junie_acp_agent.py`` — a real subprocess
speaking ACP via the same SDK — rather than mocked, so the SDK wiring is under
test too. Pure helpers and the OpenAI-shape conversion are unit-tested directly.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent.junie_acp_client as junie_mod
from agent.junie_acp_client import (
    JunieACPClient,
    _HermesClient,
    _choose_permission_option,
    _extract_model_ids,
    _merge_tool_update,
    _render_tool_activity,
    _resolve_args,
    _resolve_brave_override,
    _resolve_command,
    _resolve_permission_policy,
    fetch_junie_models,
)

_FAKE_AGENT = str(Path(__file__).with_name("fake_junie_acp_agent.py"))


def _read_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _make_client(cwd: str, **kwargs) -> JunieACPClient:
    """A JunieACPClient wired to the in-repo fake agent, settling fast."""
    client = JunieACPClient(
        acp_cwd=cwd,
        command=sys.executable,
        args=[_FAKE_AGENT],
        **kwargs,
    )
    client._settle_quiet_gap = 0.05
    client._settle_max = 2.0
    return client


# --------------------------------------------------------------------------- #
# Pure launch / resolution helpers                                            #
# --------------------------------------------------------------------------- #
class JunieLaunchResolutionTests(unittest.TestCase):
    def test_default_command_and_args(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_command(), "junie")
            self.assertEqual(_resolve_args(), ["--acp=true", "--skip-update-check"])

    def test_command_override(self) -> None:
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_COMMAND": "/opt/junie"}, clear=True):
            self.assertEqual(_resolve_command(), "/opt/junie")
        with patch.dict(os.environ, {"JUNIE_CLI_PATH": "/usr/bin/junie"}, clear=True):
            self.assertEqual(_resolve_command(), "/usr/bin/junie")

    def test_auth_injected_from_env(self) -> None:
        with patch.dict(os.environ, {"JUNIE_API_KEY": "perm-token"}, clear=True):
            args = _resolve_args()
        self.assertIn("--auth", args)
        self.assertEqual(args[args.index("--auth") + 1], "perm-token")

    def test_explicit_auth_arg_not_double_injected(self) -> None:
        with patch.dict(
            os.environ,
            {"HERMES_JUNIE_ACP_ARGS": "--acp=true --auth explicit", "JUNIE_API_KEY": "perm-token"},
            clear=True,
        ):
            args = _resolve_args()
        self.assertEqual(args.count("--auth"), 1)
        self.assertIn("explicit", args)
        self.assertNotIn("perm-token", args)

    def test_default_policy_is_deny(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_permission_policy(), "deny")

    def test_permission_policy_env(self) -> None:
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_PERMISSION": "allow"}, clear=True):
            self.assertEqual(_resolve_permission_policy(), "allow")
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_PERMISSION": "deny"}, clear=True):
            self.assertEqual(_resolve_permission_policy(), "deny")

    def test_brave_override_env_parsing(self) -> None:
        for on in ("on", "1", "true", "yes"):
            with patch.dict(os.environ, {"HERMES_JUNIE_ACP_BRAVE": on}, clear=True):
                self.assertIs(_resolve_brave_override(), True)
        for off in ("off", "0", "false", "no"):
            with patch.dict(os.environ, {"HERMES_JUNIE_ACP_BRAVE": off}, clear=True):
                self.assertIs(_resolve_brave_override(), False)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_resolve_brave_override())

    def test_constructor_override_beats_env(self) -> None:
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_BRAVE": "off"}, clear=True):
            client = JunieACPClient(acp_cwd="/tmp", brave_mode=True)
        self.assertIs(client._brave_override, True)

    def test_brave_default_is_no_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = JunieACPClient(acp_cwd="/tmp")
        self.assertIsNone(client._brave_override)


# --------------------------------------------------------------------------- #
# Permission-option selection (pure)                                          #
# --------------------------------------------------------------------------- #
class ChoosePermissionOptionTests(unittest.TestCase):
    _ALLOW_ONCE = [{"optionId": "yes", "name": "Allow", "kind": "allow_once"}]

    def test_deny_policy_never_selects(self) -> None:
        self.assertIsNone(_choose_permission_option(self._ALLOW_ONCE, "deny"))

    def test_allow_selects_allow_once(self) -> None:
        self.assertEqual(_choose_permission_option(self._ALLOW_ONCE, "allow"), "yes")

    def test_allow_without_allow_option_returns_none(self) -> None:
        opts = [{"optionId": "no", "name": "Deny", "kind": "reject_once"}]
        self.assertIsNone(_choose_permission_option(opts, "allow"))

    def test_allow_prefers_allow_once_over_allow_always(self) -> None:
        opts = [
            {"optionId": "always", "kind": "allow_always"},  # listed first
            {"optionId": "once", "kind": "allow_once"},
        ]
        self.assertEqual(_choose_permission_option(opts, "allow"), "once")


# --------------------------------------------------------------------------- #
# Client-side ACP callbacks (fs + permission bridge), tested directly         #
# --------------------------------------------------------------------------- #
class HermesClientHandlerTests(unittest.TestCase):
    def _handler(self, cwd: str, policy: str = "deny") -> _HermesClient:
        return _HermesClient(cwd=cwd, permission_policy=policy)

    def test_request_permission_denies_by_default(self) -> None:
        from acp.schema import PermissionOption, ToolCallUpdate

        h = self._handler("/tmp", policy="deny")
        resp = asyncio.run(h.request_permission(
            options=[PermissionOption(kind="allow_once", name="Allow", option_id="yes")],
            session_id="s",
            tool_call=ToolCallUpdate(tool_call_id="t", title="x"),
        ))
        self.assertEqual(resp.outcome.model_dump(by_alias=True)["outcome"], "cancelled")

    def test_request_permission_allow_selects_option(self) -> None:
        from acp.schema import PermissionOption, ToolCallUpdate

        h = self._handler("/tmp", policy="allow")
        resp = asyncio.run(h.request_permission(
            options=[PermissionOption(kind="allow_once", name="Allow", option_id="yes")],
            session_id="s",
            tool_call=ToolCallUpdate(tool_call_id="t", title="x"),
        ))
        dumped = resp.outcome.model_dump(by_alias=True)
        self.assertEqual(dumped["outcome"], "selected")
        self.assertEqual(dumped["optionId"], "yes")

    def test_read_text_file_redacts_sensitive_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secret = root / "config.env"
            secret.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")
            h = self._handler(str(root))
            with patch("agent.redact._REDACT_ENABLED", True):
                resp = asyncio.run(h.read_text_file(path=str(secret), session_id="s"))
        self.assertNotIn("abc123def456", resp.content)
        self.assertIn("OPENAI_API_KEY=", resp.content)

    def test_read_text_file_honors_limit_at_top(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f = root / "big.txt"
            f.write_text("".join(f"line{i}\n" for i in range(100)))
            h = self._handler(str(root))
            with patch("agent.redact._REDACT_ENABLED", False):
                resp = asyncio.run(h.read_text_file(path=str(f), session_id="s", line=1, limit=3))
        self.assertEqual(resp.content, "line0\nline1\nline2\n")

    def test_read_outside_cwd_is_rejected(self) -> None:
        from acp.exceptions import RequestError

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "workspace"
            root.mkdir()
            outside = Path(tmpdir) / "secret.txt"
            outside.write_text("nope")
            h = self._handler(str(root))
            with self.assertRaises(RequestError):
                asyncio.run(h.read_text_file(path=str(outside), session_id="s"))

    def test_write_text_file_respects_safe_root(self) -> None:
        from acp.exceptions import RequestError

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_root = root / "workspace"
            safe_root.mkdir()
            outside = root / "outside.txt"
            h = self._handler(str(root))
            with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
                with self.assertRaises(RequestError):
                    asyncio.run(h.write_text_file(content="should-not-write", path=str(outside), session_id="s"))
            self.assertFalse(outside.exists())


# --------------------------------------------------------------------------- #
# OpenAI-shape conversion (patch the ACP turn, assert the response mapping)   #
# --------------------------------------------------------------------------- #
class OpenAIShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = JunieACPClient(acp_cwd="/tmp")

    def test_completion_never_fabricates_openai_tool_calls(self) -> None:
        events: dict = {}
        _merge_tool_update(events, {
            "sessionUpdate": "tool_call", "toolCallId": "t1", "title": 'Found "*"',
            "kind": "other", "status": "pending",
        })
        _merge_tool_update(events, {
            "sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed",
            "content": [{"type": "content", "content": {"type": "text", "text": "gamma.log\n"}}],
        })
        with patch.object(self.client, "_run_prompt", return_value=("Here are the files.", "", events, None)):
            resp = self.client._create_chat_completion(model="junie-acp", messages=[])
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.message.tool_calls, [])
        self.assertEqual(choice.message.content, "Here are the files.")
        self.assertIn("Junie tool activity", choice.message.reasoning)
        self.assertIn("gamma.log", choice.message.reasoning)

    def test_usage_prefers_reported_counts(self) -> None:
        from types import SimpleNamespace
        reported = SimpleNamespace(input_tokens=123, output_tokens=45, thought_tokens=5, cached_read_tokens=10)
        with patch.object(self.client, "_run_prompt", return_value=("resp", "", {}, reported)):
            resp = self.client._create_chat_completion(model="junie-acp", messages=[{"role": "user", "content": "q"}])
        u = resp.usage
        self.assertEqual(u.prompt_tokens, 123)
        self.assertEqual(u.completion_tokens, 50)  # output + thought
        self.assertEqual(u.total_tokens, 173)
        self.assertEqual(u.prompt_tokens_details.cached_tokens, 10)

    def test_usage_falls_back_to_estimate(self) -> None:
        with patch.object(self.client, "_run_prompt", return_value=("some response text here", "", {}, None)):
            resp = self.client._create_chat_completion(
                model="junie-acp",
                messages=[{"role": "user", "content": "a fairly long user question to size the prompt"}],
            )
        u = resp.usage
        self.assertGreater(u.prompt_tokens, 0)
        self.assertGreater(u.completion_tokens, 0)
        self.assertEqual(u.total_tokens, u.prompt_tokens + u.completion_tokens)

    def test_literal_tool_call_text_is_not_parsed(self) -> None:
        poison = 'Sure — <tool_call>{"id":"x","type":"function","function":{"name":"rm","arguments":"{}"}}</tool_call> done.'
        with patch.object(self.client, "_run_prompt", return_value=(poison, "", {}, None)):
            resp = self.client._create_chat_completion(model="junie-acp", messages=[])
        choice = resp.choices[0]
        self.assertEqual(choice.message.tool_calls, [])
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIn("<tool_call>", choice.message.content)


# --------------------------------------------------------------------------- #
# Native tool-activity helpers                                                #
# --------------------------------------------------------------------------- #
class ToolActivityHelperTests(unittest.TestCase):
    def test_tool_call_then_update_merge_by_id(self) -> None:
        events: dict = {}
        _merge_tool_update(events, {
            "sessionUpdate": "tool_call", "toolCallId": "id1", "title": 'Found "*"',
            "kind": "other", "status": "pending",
        })
        _merge_tool_update(events, {
            "sessionUpdate": "tool_call_update", "toolCallId": "id1", "status": "completed",
            "content": [{"type": "content", "content": {"type": "text", "text": "alpha\ngamma.log\n"}}],
        })
        self.assertEqual(len(events), 1)
        ev = events["id1"]
        self.assertEqual(ev["status"], "completed")
        self.assertEqual(ev["kind"], "other")
        self.assertEqual(ev["title"], 'Found "*"')
        self.assertIn("gamma.log", ev["result"])

    def test_render_tool_activity_is_readable(self) -> None:
        events: dict = {}
        _merge_tool_update(events, {"sessionUpdate": "tool_call", "toolCallId": "id1",
                                    "title": "list", "kind": "other", "status": "completed",
                                    "content": [{"type": "content", "content": {"type": "text", "text": "gamma.log"}}]})
        rendered = _render_tool_activity(events)
        self.assertIn("[other]", rendered)
        self.assertIn("completed", rendered)
        self.assertIn("gamma.log", rendered)


# --------------------------------------------------------------------------- #
# End-to-end against the fake ACP agent subprocess                            #
# --------------------------------------------------------------------------- #
class JunieEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cwd = self._tmp.name
        self.log = os.path.join(self.cwd, "calls.log")

    def _client(self, **kwargs) -> JunieACPClient:
        client = _make_client(self.cwd, **kwargs)
        self.addCleanup(client.close)
        return client

    def _ask(self, client: JunieACPClient, text: str, model: str = "junie-acp"):
        return client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": text}], timeout=30
        )

    def test_process_reused_across_two_calls(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log}, clear=False):
            client = self._client()
            r1 = self._ask(client, "hi 1")
            r2 = self._ask(client, "hi 2")
        self.assertEqual(r1.choices[0].message.content, "ANSWER1")
        self.assertEqual(r2.choices[0].message.content, "ANSWER2")
        self.assertEqual(r1.choices[0].finish_reason, "stop")
        log = _read_log(self.log)
        self.assertEqual(sum(1 for e in log if e["method"] == "initialize"), 1)
        self.assertEqual(sum(1 for e in log if e["method"] == "session/new"), 2)
        self.assertEqual(sum(1 for e in log if e["method"] == "session/prompt"), 2)

    def test_dead_process_is_respawned(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_EXIT_AFTER_PROMPT": "1"}, clear=False):
            client = self._client()
            r1 = self._ask(client, "x")
            self.assertEqual(r1.choices[0].message.content, "ANSWER1")
            r2 = self._ask(client, "y")
        # Fresh process → its own counter restarts at 1.
        self.assertEqual(r2.choices[0].message.content, "ANSWER1")
        log = _read_log(self.log)
        self.assertEqual(sum(1 for e in log if e["method"] == "initialize"), 2)

    def test_real_model_is_forwarded_via_set_config_option(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log}, clear=False):
            client = self._client()
            self._ask(client, "x", model="claude-opus-4-8")
        sets = [(e.get("configId"), e.get("value")) for e in _read_log(self.log)
                if e["method"] == "session/set_config_option"]
        self.assertIn(("model", "claude-opus-4-8"), sets)

    def test_provider_sentinel_model_is_not_forwarded(self) -> None:
        for sentinel in ("junie-acp", "junie", "jetbrains-junie-acp", "junie-acp-agent", ""):
            log = os.path.join(self.cwd, f"calls_{sentinel or 'empty'}.log")
            with patch.dict(os.environ, {"FAKE_JUNIE_LOG": log}, clear=False):
                client = self._client()
                self._ask(client, "x", model=sentinel)
                client.close()
            model_sets = [e for e in _read_log(log)
                          if e["method"] == "session/set_config_option" and e.get("configId") == "model"]
            self.assertFalse(model_sets, f"sentinel {sentinel!r} should not set a model")

    def test_model_set_failure_does_not_abort_turn(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_FAIL_MODEL": "1"}, clear=False):
            client = self._client()
            resp = self._ask(client, "x", model="no-such-model")
        self.assertEqual(resp.choices[0].message.content, "ANSWER1")
        self.assertEqual(resp.choices[0].finish_reason, "stop")

    def test_brave_override_sends_set_config_option(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log}, clear=False):
            client = self._client(brave_mode=False)
            self._ask(client, "x")
        sets = [(e.get("configId"), e.get("value")) for e in _read_log(self.log)
                if e["method"] == "session/set_config_option"]
        self.assertIn(("brave_mode", False), sets)

    def test_no_replay_after_prompt_dispatch(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_FAIL_PROMPT": "1"}, clear=False):
            client = self._client()
            with self.assertRaises(RuntimeError):
                self._ask(client, "x")
        log = _read_log(self.log)
        self.assertEqual(sum(1 for e in log if e["method"] == "session/prompt"), 1)   # dispatched once
        self.assertEqual(sum(1 for e in log if e["method"] == "initialize"), 1)        # no respawn

    def test_retry_before_prompt_respawns(self) -> None:
        sentinel = os.path.join(self.cwd, "session_new.sentinel")
        with patch.dict(os.environ, {
            "FAKE_JUNIE_LOG": self.log,
            "FAKE_JUNIE_FAIL_SESSION_NEW_ONCE": "1",
            "FAKE_JUNIE_SESSION_SENTINEL": sentinel,
        }, clear=False):
            client = self._client()
            resp = self._ask(client, "x")
        self.assertEqual(resp.choices[0].message.content, "ANSWER1")
        log = _read_log(self.log)
        self.assertEqual(sum(1 for e in log if e["method"] == "initialize"), 2)  # respawned
        self.assertEqual(sum(1 for e in log if e["method"] == "session/prompt"), 1)

    def test_tool_activity_surfaced_as_reasoning_not_tool_calls(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_EMIT_TOOLCALL": "1"}, clear=False):
            client = self._client()
            resp = self._ask(client, "list files")
        choice = resp.choices[0]
        self.assertEqual(choice.message.tool_calls, [])
        self.assertIn("Junie tool activity", choice.message.reasoning or "")
        self.assertIn("gamma.log", choice.message.reasoning or "")

    def test_thought_chunk_surfaced_as_reasoning(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_EMIT_THOUGHT": "1"}, clear=False):
            client = self._client()
            resp = self._ask(client, "think")
        self.assertIn("thinking...", resp.choices[0].message.reasoning or "")

    def test_usage_from_prompt_response(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_USAGE": "1"}, clear=False):
            client = self._client()
            resp = self._ask(client, "x")
        self.assertEqual(resp.usage.prompt_tokens, 123)
        self.assertEqual(resp.usage.completion_tokens, 45)

    def test_permission_bridge_denies_by_default(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_ASK_PERMISSION": "1"}, clear=False):
            client = self._client()  # default policy: deny
            self._ask(client, "x")
        outcomes = [e["outcome"] for e in _read_log(self.log) if e["method"] == "permission_outcome"]
        self.assertTrue(outcomes)
        self.assertEqual(outcomes[0]["outcome"], "cancelled")

    def test_permission_bridge_allows_when_policy_allow(self) -> None:
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_ASK_PERMISSION": "1"}, clear=False):
            client = self._client(permission_policy="allow")
            self._ask(client, "x")
        outcomes = [e["outcome"] for e in _read_log(self.log) if e["method"] == "permission_outcome"]
        self.assertTrue(outcomes)
        self.assertEqual(outcomes[0]["outcome"], "selected")
        self.assertEqual(outcomes[0]["optionId"], "yes")

    def test_fs_read_bridge_redacts_and_sandboxes(self) -> None:
        secret = os.path.join(self.cwd, "config.env")
        Path(secret).write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")
        with patch.dict(os.environ, {"FAKE_JUNIE_LOG": self.log, "FAKE_JUNIE_READ_PATH": secret}, clear=False):
            with patch("agent.redact._REDACT_ENABLED", True):
                client = self._client()
                self._ask(client, "read it")
        reads = [e["content"] for e in _read_log(self.log) if e["method"] == "fs_read_result"]
        self.assertTrue(reads)
        self.assertIn("OPENAI_API_KEY=", reads[0])
        self.assertNotIn("abc123def456", reads[0])


# --------------------------------------------------------------------------- #
# session_update routing + cross-session filtering (handler, direct)          #
# --------------------------------------------------------------------------- #
class SessionUpdateRoutingTests(unittest.TestCase):
    def _turn(self):
        h = _HermesClient(cwd="/tmp", permission_policy="deny")
        text, reasoning, tools = [], [], {}
        h.begin_turn("current", text, reasoning, tools)
        return h, text, reasoning, tools

    @staticmethod
    def _msg(kind, content):
        return {"sessionUpdate": kind, "content": content}

    def test_stale_session_update_is_dropped(self):
        # A straggler chunk tagged with a different session must not leak in.
        h, text, _r, _t = self._turn()
        asyncio.run(h.session_update(
            session_id="OTHER", update=self._msg("agent_message_chunk", {"type": "text", "text": "leak"})))
        self.assertEqual(text, [])

    def test_matching_session_update_is_kept(self):
        h, text, _r, _t = self._turn()
        asyncio.run(h.session_update(
            session_id="current", update=self._msg("agent_message_chunk", {"type": "text", "text": "ok"})))
        self.assertEqual(text, ["ok"])

    def test_list_form_chunk_content_is_extracted(self):
        h, text, _r, _t = self._turn()
        asyncio.run(h.session_update(
            session_id="current",
            update=self._msg("agent_message_chunk", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])))
        self.assertEqual(text, ["ab"])

    def test_thought_chunk_routes_to_reasoning_not_text(self):
        h, text, reasoning, _t = self._turn()
        asyncio.run(h.session_update(
            session_id="current", update=self._msg("agent_thought_chunk", {"type": "text", "text": "hmm"})))
        self.assertEqual(reasoning, ["hmm"])
        self.assertEqual(text, [])

    def test_tool_update_routes_to_tool_events_not_text(self):
        h, text, reasoning, tools = self._turn()
        asyncio.run(h.session_update(
            session_id="current",
            update={"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "x", "kind": "other", "status": "pending"}))
        self.assertEqual(text, [])
        self.assertEqual(reasoning, [])
        self.assertIn("t1", tools)


# --------------------------------------------------------------------------- #
# Live model discovery from ACP config_options                                #
# --------------------------------------------------------------------------- #
class ModelDiscoveryTests(unittest.TestCase):
    def test_extract_model_ids_current_first(self) -> None:
        config_options = [
            {"id": "brave_mode", "type": "select", "options": [{"value": "on"}]},
            {"id": "model", "type": "select", "currentValue": "claude-fable-5",
             "options": [
                 {"value": "claude-opus-4-8", "name": "Opus 4.8"},
                 {"value": "claude-fable-5", "name": "Fable 5"},
                 {"value": "gpt-5.4", "name": "GPT-5.4"},
             ]},
        ]
        ids = _extract_model_ids(config_options)
        self.assertEqual(ids[0], "claude-fable-5")  # currentValue promoted
        self.assertEqual(set(ids), {"claude-opus-4-8", "claude-fable-5", "gpt-5.4"})

    def test_extract_model_ids_no_model_option(self) -> None:
        self.assertEqual(_extract_model_ids([{"id": "brave_mode", "options": []}]), [])

    def test_fetch_junie_models_returns_none_on_failure(self) -> None:
        async def _boom(*a, **k):
            raise RuntimeError("no junie here")

        junie_mod._model_cache.clear()
        with patch.object(junie_mod, "_afetch_junie_models", _boom), \
             patch.object(junie_mod, "_load_model_disk_cache", lambda cmd: None), \
             patch.object(junie_mod, "_save_model_disk_cache", lambda cmd, ids: None):
            self.assertIsNone(fetch_junie_models(command="junie", args=[], force_refresh=True))

    def test_fetch_junie_models_caches_in_memory(self) -> None:
        calls = {"n": 0}

        async def _ok(command, args, cwd):
            calls["n"] += 1
            return ["claude-fable-5", "gpt-5.4"]

        junie_mod._model_cache.clear()
        with patch.object(junie_mod, "_afetch_junie_models", _ok), \
             patch.object(junie_mod, "_load_model_disk_cache", lambda cmd: None), \
             patch.object(junie_mod, "_save_model_disk_cache", lambda cmd, ids: None):
            first = fetch_junie_models(command="junie", args=[], force_refresh=True)
            second = fetch_junie_models(command="junie", args=[])  # served from cache
        self.assertEqual(first, ["claude-fable-5", "gpt-5.4"])
        self.assertEqual(second, first)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
