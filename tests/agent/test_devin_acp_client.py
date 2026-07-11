"""Tests for Devin CLI ACP provider wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.devin_acp_client import ACP_MARKER_BASE_URL, DevinACPClient, _resolve_args
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
    resolve_provider,
)


class TestDevinAcpProviderRegistry(unittest.TestCase):
    def test_registry_entry(self):
        p = PROVIDER_REGISTRY["devin-acp"]
        assert p.auth_type == "external_process"
        assert p.inference_base_url == "acp://devin"

    def test_aliases(self):
        assert resolve_provider("devin") == "devin-acp"
        assert resolve_provider("devin-cli") == "devin-acp"
        assert resolve_provider("cognition-devin") == "devin-acp"


class TestDevinAcpResolve(unittest.TestCase):
    def test_status_and_creds(self):
        with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/devin"):
            with patch(
                "hermes_cli.auth._devin_local_credentials_present",
                return_value=True,
            ):
                with patch.dict("os.environ", {"HERMES_DEVIN_ACP_ARGS": "acp --debug"}, clear=False):
                    status = get_external_process_provider_status("devin-acp")
                    assert status["configured"] is True
                    assert status["cli_installed"] is True
                    assert status["auth_present"] is True
                    assert status["logged_in"] is True
                    assert status["command"] == "devin"
                    assert status["resolved_command"] == "/usr/local/bin/devin"
                    assert status["args"] == ["acp", "--debug"]
                    assert status["base_url"] == "acp://devin"

                    creds = resolve_external_process_provider_credentials("devin-acp")
                    assert creds["provider"] == "devin-acp"
                    assert creds["api_key"] == "devin-acp"
                    assert creds["base_url"] == "acp://devin"
                    assert creds["command"] == "/usr/local/bin/devin"
                    assert creds["args"] == ["acp", "--debug"]

    def test_status_cli_without_credentials_is_not_logged_in(self):
        with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/devin"):
            with patch(
                "hermes_cli.auth._devin_local_credentials_present",
                return_value=False,
            ):
                status = get_external_process_provider_status("devin-acp")
        assert status["configured"] is True
        assert status["cli_installed"] is True
        assert status["auth_present"] is False
        assert status["logged_in"] is False
        assert status.get("hint")
        assert "devin auth login" in status["hint"]


class TestDevinAcpClientDefaults(unittest.TestCase):
    def test_marker_and_defaults(self):
        assert ACP_MARKER_BASE_URL == "acp://devin"
        with patch.dict("os.environ", {"HERMES_DEVIN_ACP_ARGS": ""}, clear=False):
            # Force empty env so default path is exercised even if the host
            # shell exported HERMES_DEVIN_ACP_ARGS.
            assert _resolve_args() == ["acp"]
        client = DevinACPClient(acp_cwd="/tmp", command="devin", args=["acp"])
        assert client.api_key == "devin-acp"
        assert client.base_url == "acp://devin"
        assert client._acp_command == "devin"
        assert client._acp_args == ["acp"]

    def test_empty_args_does_not_fall_through_to_copilot_defaults(self):
        """Regression: args=[] used to become ['--acp', '--stdio'] via parent."""
        with patch.dict(
            "os.environ",
            {
                "HERMES_DEVIN_ACP_ARGS": "",
                "HERMES_COPILOT_ACP_ARGS": "",
            },
            clear=False,
        ):
            via_args = DevinACPClient(acp_cwd="/tmp", command="devin", args=[])
            via_acp_args = DevinACPClient(acp_cwd="/tmp", command="devin", acp_args=[])
            via_none = DevinACPClient(acp_cwd="/tmp", command="devin")

        for client in (via_args, via_acp_args, via_none):
            assert client._acp_args == ["acp"], client._acp_args
            assert "--acp" not in client._acp_args
            assert "--stdio" not in client._acp_args

    def test_display_name_and_install_hint_are_devin(self):
        client = DevinACPClient(acp_cwd="/tmp", command="devin", args=["acp"])
        assert client._acp_display_name == "Devin ACP"
        assert "Devin CLI" in client._install_hint
        assert "Copilot" not in client._install_hint


class TestAcpClientFactory(unittest.TestCase):
    def test_create_devin(self):
        from agent.acp_client_factory import ACP_PROVIDERS, create_acp_client, is_acp_provider

        assert "devin-acp" in ACP_PROVIDERS
        assert is_acp_provider("devin-acp") is True
        assert is_acp_provider(base_url="acp://devin") is True
        client = create_acp_client(
            provider="devin-acp", command="devin", args=["acp"], acp_cwd="/tmp"
        )
        assert isinstance(client, DevinACPClient)

    def test_create_devin_empty_args_uses_devin_defaults(self):
        from agent.acp_client_factory import create_acp_client

        with patch.dict("os.environ", {"HERMES_DEVIN_ACP_ARGS": ""}, clear=False):
            client = create_acp_client(
                provider="devin-acp", command="devin", args=[], acp_cwd="/tmp"
            )
        assert isinstance(client, DevinACPClient)
        assert client._acp_args == ["acp"]

    def test_create_devin_fills_missing_command_from_resolver(self):
        from agent.acp_client_factory import create_acp_client

        with patch(
            "hermes_cli.auth.resolve_external_process_provider_credentials",
            return_value={
                "command": "/resolved/devin",
                "args": ["acp", "--from-resolver"],
            },
        ):
            client = create_acp_client(provider="devin-acp", acp_cwd="/tmp")
        assert isinstance(client, DevinACPClient)
        assert client._acp_command == "/resolved/devin"
        assert client._acp_args == ["acp", "--from-resolver"]

    def test_devin_credentials_probe_reads_marker_not_secret(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        from hermes_cli.auth import _devin_local_credentials_present

        with tempfile.TemporaryDirectory() as td:
            cred_dir = Path(td) / "devin"
            cred_dir.mkdir()
            cred_file = cred_dir / "credentials.toml"
            cred_file.write_text(
                'windsurf_api_key = "devin-session-token$secret-value"\n',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"APPDATA": td, "XDG_CONFIG_HOME": td}, clear=False):
                # Point home-based candidates away from the real user home by
                # still using APPDATA/XDG which our probe checks first.
                assert _devin_local_credentials_present() is True

            missing = Path(td) / "empty"
            missing.mkdir()
            with patch.dict(
                "os.environ",
                {"APPDATA": str(missing), "XDG_CONFIG_HOME": str(missing)},
                clear=False,
            ):
                with patch("pathlib.Path.home", return_value=missing):
                    assert _devin_local_credentials_present() is False


class TestAcpToolLoopSession(unittest.TestCase):
    def test_tool_call_then_tool_result_reuses_session(self):
        """Assistant tool_call → tool result should continue the same ACP session."""
        from agent.copilot_acp_client import CopilotACPClient

        procs: list[_ScriptedAcpProcess] = []

        class _ToolAwareProcess(_ScriptedAcpProcess):
            def write(self, data: str) -> int:
                import json

                line = data.strip()
                if not line:
                    return 0
                req = json.loads(line)
                self.writes.append(req)
                method = req.get("method")
                req_id = req.get("id")
                if method == "initialize":
                    result = {"protocolVersion": 1}
                elif method == "session/new":
                    self.session_seq += 1
                    result = {"sessionId": f"sess-{self.session_seq}"}
                elif method == "session/prompt":
                    prompt_text = req["params"]["prompt"][0]["text"]
                    if "Tool:" in prompt_text or "tool result" in prompt_text.lower():
                        body = "file contents here"
                    else:
                        body = (
                            '<tool_call>{"id":"call_1","type":"function",'
                            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}}'
                            "</tool_call>"
                        )
                    chunk = {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": body},
                            }
                        },
                    }
                    self.stdout.push(json.dumps(chunk) + "\n")
                    result = {"stopReason": "end_turn"}
                else:
                    result = {}
                self.stdout.push(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
                return len(data)

        def _popen(*_a, **_k):
            proc = _ToolAwareProcess()
            procs.append(proc)
            return proc

        with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_popen):
            client = CopilotACPClient(
                command="fake-acp",
                args=["--stdio"],
                acp_cwd="/tmp",
            )
            client._reuse_enabled = True
            client._session_reuse_enabled = True

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            r1 = client._create_chat_completion(
                model="x",
                messages=[{"role": "user", "content": "read a.txt"}],
                tools=tools,
                timeout=5,
            )
            assert r1.choices[0].finish_reason == "tool_calls"
            tc = r1.choices[0].message.tool_calls[0]
            assert tc.function.name == "read_file"

            r2 = client._create_chat_completion(
                model="x",
                messages=[
                    {"role": "user", "content": "read a.txt"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": tc.function.arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "hello from file",
                    },
                ],
                tools=tools,
                timeout=5,
            )
            client.close()

        assert "file contents here" in (r2.choices[0].message.content or "")
        assert client._spawn_count == 1
        assert client._session_count == 1
        assert client._session_continues == 1
        methods = [w.get("method") for w in procs[0].writes]
        assert methods.count("session/new") == 1
        assert methods.count("session/prompt") == 2


class TestDevinDesktopUiSupport(unittest.TestCase):
    def test_oauth_catalog_has_hand_tuned_devin_card(self):
        from hermes_cli.web_server import _build_oauth_catalog

        cards = {p["id"]: p for p in _build_oauth_catalog()}
        assert "devin-acp" in cards
        card = cards["devin-acp"]
        assert card["flow"] == "external"
        assert card["cli_command"] == "devin auth login"
        assert card.get("status_fn") is not None
        assert "docs.devin.ai" in (card.get("docs_url") or "")

    def test_devin_acp_status_shape(self):
        from hermes_cli.web_server import _devin_acp_status

        with patch(
            "hermes_cli.auth.get_external_process_provider_status",
            return_value={
                "logged_in": True,
                "cli_installed": True,
                "auth_present": True,
                "resolved_command": "/usr/bin/devin",
                "command": "devin",
                "hint": None,
            },
        ):
            st = _devin_acp_status()
        assert st["logged_in"] is True
        assert st["source"] == "devin_cli"
        assert st["token_preview"] is None
        assert "Devin CLI" in st["source_label"]
        assert "/usr/bin/devin" in st["source_label"]

    def test_list_authenticated_includes_devin_when_logged_in(self):
        from hermes_cli.model_switch import list_authenticated_providers

        with patch(
            "hermes_cli.auth.get_external_process_provider_status",
            side_effect=lambda pid: {
                "logged_in": pid == "devin-acp",
                "cli_installed": pid == "devin-acp",
                "auth_present": pid == "devin-acp",
                "configured": pid == "devin-acp",
            },
        ):
            rows = list_authenticated_providers()
        slugs = {r.get("slug") for r in rows}
        assert "devin-acp" in slugs
        devin = next(r for r in rows if r.get("slug") == "devin-acp")
        assert "devin-acp" in (devin.get("models") or [])


class TestAcpErrorClassification(unittest.TestCase):
    def test_missing_cli_is_non_retryable(self):
        from agent.error_classifier import FailoverReason, classify_api_error

        err = RuntimeError(
            "Could not start Devin ACP command 'devin'. "
            "Install Devin CLI and run `devin auth login`."
        )
        result = classify_api_error(err, provider="devin-acp")
        assert result.retryable is False
        assert result.reason in {
            FailoverReason.format_error,
            FailoverReason.auth_permanent,
        }

    def test_wrong_argv_is_non_retryable(self):
        from agent.error_classifier import FailoverReason, classify_api_error

        err = RuntimeError(
            "Devin ACP process exited early: error: unexpected argument '--acp' found"
        )
        result = classify_api_error(err, provider="devin-acp")
        assert result.retryable is False
        assert result.reason == FailoverReason.format_error


class TestOneshotAcpWiring(unittest.TestCase):
    def test_oneshot_forwards_runtime_command_and_args(self):
        """hermes -z must pass ACP command/args into AIAgent (P0 regression)."""
        import hermes_cli.oneshot as oneshot_mod

        captured: dict = {}

        class _FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.suppress_status_output = False
                self.stream_delta_callback = None
                self.tool_gen_callback = None

            def run_conversation(self, prompt):
                return {
                    "final_response": "pong",
                    "messages": [{"role": "user", "content": prompt}],
                }

        with (
            patch.object(
                oneshot_mod,
                "resolve_runtime_provider",
                create=True,
            ),
            patch("hermes_cli.oneshot.load_config", create=True),
            patch("hermes_cli.config.load_config", return_value={"model": {}}),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "provider": "devin-acp",
                    "api_mode": "chat_completions",
                    "base_url": "acp://devin",
                    "api_key": "devin-acp",
                    "command": "/usr/bin/devin",
                    "args": ["acp"],
                    "credential_pool": None,
                },
            ),
            patch("hermes_cli.oneshot.get_fallback_chain", return_value=None),
            patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None),
            patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
            patch("run_agent.AIAgent", _FakeAgent),
        ):
            response, result = oneshot_mod._run_agent(
                "ping",
                model="devin-acp",
                provider="devin-acp",
                use_config_toolsets=False,
            )

        assert response == "pong"
        assert captured.get("acp_command") == "/usr/bin/devin"
        assert captured.get("acp_args") == ["acp"]
        assert captured.get("provider") == "devin-acp"
        assert captured.get("base_url") == "acp://devin"


class _LinePipe:
    """Thread-safe line pipe used as stdout/stderr stand-in."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._cond = __import__("threading").Condition()
        self._closed = False

    def push(self, line: str) -> None:
        with self._cond:
            self._lines.append(line)
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        with self._cond:
            while not self._lines and not self._closed:
                self._cond.wait(timeout=0.05)
            if self._lines:
                return self._lines.pop(0)
            raise StopIteration


class _ScriptedAcpProcess:
    """Minimal Popen stand-in that answers ACP initialize / session RPCs."""

    def __init__(self) -> None:
        self.stdin = self
        self.stdout = _LinePipe()
        self.stderr = _LinePipe()
        self.returncode = None
        self.writes: list[dict] = []
        self.session_seq = 0

    def write(self, data: str) -> int:
        import json

        line = data.strip()
        if not line:
            return 0
        req = json.loads(line)
        self.writes.append(req)
        method = req.get("method")
        req_id = req.get("id")
        if method == "initialize":
            result = {"protocolVersion": 1}
        elif method == "session/new":
            self.session_seq += 1
            result = {"sessionId": f"sess-{self.session_seq}"}
        elif method == "session/prompt":
            chunk = {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": f"ok-{self.session_seq}"},
                    }
                },
            }
            self.stdout.push(json.dumps(chunk) + "\n")
            result = {"stopReason": "end_turn"}
        else:
            result = {}
        resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        self.stdout.push(json.dumps(resp) + "\n")
        return len(data)

    def flush(self) -> None:
        return None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.close()
        self.stderr.close()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout=None) -> int:
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode


class TestAcpProcessReuse(unittest.TestCase):
    def test_reuses_process_across_prompts(self):
        from agent.copilot_acp_client import CopilotACPClient

        procs: list[_ScriptedAcpProcess] = []

        def _popen(*_a, **_k):
            proc = _ScriptedAcpProcess()
            procs.append(proc)
            return proc

        with patch.dict(
            "os.environ",
            {"HERMES_ACP_PROCESS_REUSE": "1", "HERMES_ACP_SESSION_REUSE": "0"},
            clear=False,
        ):
            with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_popen):
                client = CopilotACPClient(
                    command="fake-acp",
                    args=["--stdio"],
                    acp_cwd="/tmp",
                )
                client._reuse_enabled = True
                client._session_reuse_enabled = False
                r1, _ = client._run_prompt("first", timeout_seconds=5)
                r2, _ = client._run_prompt("second", timeout_seconds=5)
                client.close()

        assert r1 == "ok-1"
        assert r2 == "ok-2"
        assert client._spawn_count == 1
        assert len(procs) == 1
        methods = [w.get("method") for w in procs[0].writes]
        # initialize once; session/new + session/prompt per turn when session reuse off
        assert methods.count("initialize") == 1
        assert methods.count("session/new") == 2
        assert methods.count("session/prompt") == 2

    def test_session_continuity_sends_only_delta(self):
        from agent.copilot_acp_client import CopilotACPClient

        procs: list[_ScriptedAcpProcess] = []

        def _popen(*_a, **_k):
            proc = _ScriptedAcpProcess()
            procs.append(proc)
            return proc

        with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_popen):
            client = CopilotACPClient(
                command="fake-acp",
                args=["--stdio"],
                acp_cwd="/tmp",
            )
            client._reuse_enabled = True
            client._session_reuse_enabled = True
            m1 = [{"role": "user", "content": "hello"}]
            c1 = client._create_chat_completion(model="x", messages=m1, timeout=5)
            m2 = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": c1.choices[0].message.content},
                {"role": "user", "content": "follow up"},
            ]
            c2 = client._create_chat_completion(model="x", messages=m2, timeout=5)
            client.close()

        assert c1.choices[0].message.content == "ok-1"
        assert c2.choices[0].message.content == "ok-1"  # same session seq (no new session)
        assert client._spawn_count == 1
        assert client._session_count == 1
        assert client._session_continues == 1
        methods = [w.get("method") for w in procs[0].writes]
        assert methods.count("session/new") == 1
        assert methods.count("session/prompt") == 2
        # Second prompt body should be a continuation delta, not full history.
        prompt_bodies = [
            w["params"]["prompt"][0]["text"]
            for w in procs[0].writes
            if w.get("method") == "session/prompt"
        ]
        assert "New messages:" in prompt_bodies[1]
        assert "follow up" in prompt_bodies[1]
        assert "Conversation transcript:" not in prompt_bodies[1]

    def test_reuse_disabled_spawns_each_prompt(self):
        from agent.copilot_acp_client import CopilotACPClient

        procs: list[_ScriptedAcpProcess] = []

        def _popen(*_a, **_k):
            proc = _ScriptedAcpProcess()
            procs.append(proc)
            return proc

        with patch.dict("os.environ", {"HERMES_ACP_PROCESS_REUSE": "0"}, clear=False):
            with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_popen):
                client = CopilotACPClient(
                    command="fake-acp",
                    args=["--stdio"],
                    acp_cwd="/tmp",
                )
                # Re-read flag after env patch (constructor captured it).
                client._reuse_enabled = False
                client._session_reuse_enabled = False
                client._run_prompt("first", timeout_seconds=5)
                client._run_prompt("second", timeout_seconds=5)

        assert client._spawn_count == 2
        assert len(procs) == 2

    def test_dead_process_respawns_on_next_prompt(self):
        from agent.copilot_acp_client import CopilotACPClient

        procs: list[_ScriptedAcpProcess] = []

        def _popen(*_a, **_k):
            proc = _ScriptedAcpProcess()
            procs.append(proc)
            return proc

        with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_popen):
            client = CopilotACPClient(
                command="fake-acp",
                args=["--stdio"],
                acp_cwd="/tmp",
            )
            client._reuse_enabled = True
            client._session_reuse_enabled = True
            client._run_prompt("first", timeout_seconds=5)
            # Simulate crash between turns without going through close().
            procs[0].returncode = 1
            client._run_prompt("second", timeout_seconds=5)
            client.close()

        assert client._spawn_count == 2
        assert len(procs) == 2

    def test_interrupt_terminates_live_process(self):
        from agent.copilot_acp_client import CopilotACPClient

        procs: list[_ScriptedAcpProcess] = []

        def _popen(*_a, **_k):
            proc = _ScriptedAcpProcess()
            procs.append(proc)
            return proc

        with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_popen):
            client = CopilotACPClient(
                command="fake-acp",
                args=["--stdio"],
                acp_cwd="/tmp",
            )
            client._reuse_enabled = True
            client._run_prompt("first", timeout_seconds=5)
            assert procs[0].poll() is None
            client.interrupt()
            assert procs[0].poll() is not None

    def test_stream_true_yields_iterable_deltas(self):
        from agent.copilot_acp_client import CopilotACPClient

        def _popen(*_a, **_k):
            return _ScriptedAcpProcess()

        with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_popen):
            client = CopilotACPClient(
                command="fake-acp",
                args=["--stdio"],
                acp_cwd="/tmp",
            )
            client._reuse_enabled = True
            stream = client._create_chat_completion(
                model="devin-acp",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                timeout=5,
            )
            assert not hasattr(stream, "choices")  # must be iterable, not final response
            chunks = list(stream)
            client.close()

        contents = []
        for ch in chunks:
            if not ch.choices:
                continue
            delta = ch.choices[0].delta
            if getattr(delta, "content", None):
                contents.append(delta.content)
        assert "".join(contents) == "ok-1" or any(c == "ok-1" for c in contents)


if __name__ == "__main__":
    unittest.main()
