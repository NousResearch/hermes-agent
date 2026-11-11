"""The deprecated ``agent.copilot_acp_client`` alias keeps every listed catalog ACP plugin working.

Each test reproduces one catalog plugin's import + patch set at its pinned SHA (see
``plugin-catalog/<name>.yaml``) against a fake ACP agent: kiro-acp@02c4dd0, grok-acp@2851d66,
cursor-acp@d5b6146. kimchi-acp-provider@d6ffe62 is covered in ``test_acp_client.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

import pytest

_AGENT = r'''
import json, os, sys
log = open(os.environ["FAKE_ACP_LOG"], "a")
def send(o): print(json.dumps(o), flush=True)
for line in sys.stdin:
    msg = json.loads(line); log.write(json.dumps(msg) + "\n"); log.flush()
    m, i = msg.get("method"), msg.get("id")
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": i, "result": {"protocolVersion": 1}})
    elif m == "session/new":
        result = {"sessionId": "s1", "models": {"availableModels": [
            {"modelId": "fast[v2]", "name": "Fast"}, {"modelId": "slow", "name": "Slow"}]}}
        if os.environ.get("FAKE_ACP_CONFIG_OPTIONS"):
            result["configOptions"] = [{"id": "model", "category": "model", "options": [
                {"value": "m-a"}, {"value": "m-off", "_meta": {"copilotEnablement": "disabled"}}]}]
        send({"jsonrpc": "2.0", "id": i, "result": result})
    elif m == "session/prompt":
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"update": {
            "sessionUpdate": "agent_message_chunk", "content": {"text": "hello"}}}})
        send({"jsonrpc": "2.0", "id": i, "result": {"stopReason": "end_turn"}})
    elif i is not None:
        send({"jsonrpc": "2.0", "id": i, "result": {}})
'''


@pytest.fixture
def agent(tmp_path, monkeypatch):
    script = tmp_path / "fake_acp_agent.py"
    script.write_text(_AGENT, encoding="utf-8")
    log = tmp_path / "acp.jsonl"
    monkeypatch.setenv("FAKE_ACP_LOG", str(log))
    warnings.simplefilter("ignore", DeprecationWarning)

    def methods() -> list[str]:
        return [json.loads(line)["method"] for line in log.read_text().splitlines() if "method" in json.loads(line)]

    return [sys.executable, str(script)], methods


_HI = [{"role": "user", "content": "hi"}]


def test_kiro_acp_pinned_import_shape(agent, tmp_path):
    (command, script), _ = agent
    # kiro_acp_client.py@02c4dd0: module-top import of the marker constant next to the class.
    from agent.copilot_acp_client import ACP_MARKER_BASE_URL as _COPILAT_MARKER, CopilotACPClient

    class KiroACPClient(CopilotACPClient):
        def __init__(self, *, base_url=None, command=None, args=None, **kwargs):
            super().__init__(base_url=base_url or "acp://kiro", acp_command=command, acp_args=args, **kwargs)

        def _spawn(self):
            try:
                return super()._spawn()
            except RuntimeError as exc:
                if "Could not start Copilot ACP command" in str(exc):
                    raise RuntimeError("Could not start Kiro ACP command") from exc
                raise

    assert _COPILAT_MARKER == "acp://copilot"
    client = KiroACPClient(command=command, args=[script], acp_cwd=str(tmp_path))
    assert client.chat.completions.create(model=None, messages=_HI, timeout=30).choices[0].message.content == "hello"
    missing = KiroACPClient(command=str(tmp_path / "absent"), args=["acp"], acp_cwd=str(tmp_path))
    with pytest.raises(RuntimeError, match="Could not start Kiro ACP command"):
        missing.chat.completions.create(messages=_HI, timeout=10)


def test_grok_acp_pinned_import_and_popen_patch_shape(agent, tmp_path, monkeypatch):
    (command, script), methods = agent
    # grok_acp_client.py@2851d66: setup probe env, its own _spawn through the module's subprocess,
    # and model selection through the former module helper.
    import agent.copilot_acp_client as copilot_mod
    from agent.copilot_acp_client import (
        CopilotACPClient, _acp_supported, _build_subprocess_env, _model_selection_request)

    assert _build_subprocess_env()["HOME"]
    assert _acp_supported(command, ["stdio"]) is True  # no --acp in args: no probe
    spawned: list[list[str]] = []
    real_popen = subprocess.Popen

    def _popen(argv, *args, **kwargs):
        spawned.append(list(argv))
        return real_popen(argv, *args, **kwargs)

    # The plugin's tests patch the module path; it must reach every ACP spawn, the base class's included.
    monkeypatch.setattr("agent.copilot_acp_client.subprocess.Popen", _popen)

    class GrokACPClient(CopilotACPClient):
        def _spawn(self):
            proc = copilot_mod.subprocess.Popen(
                [self._acp_command] + self._acp_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, cwd=self._acp_cwd, env=_build_subprocess_env())
            with self._active_process_lock:
                self._active_processes.add(proc)
                self.is_closed = False
            return proc

        def _run_prompt(self, prompt_text, *, timeout_seconds, model=None):
            with self._session(timeout_seconds) as (session, request):
                if model and model != "copilot-acp" and (selection := _model_selection_request(session, model)):
                    request(*selection)
                parts: list[str] = []
                request("session/prompt", {"sessionId": session["sessionId"], "prompt": [{"type": "text", "text": prompt_text}]},
                        text_parts=parts, reasoning_parts=[])
                return "".join(parts), ""

    monkeypatch.setenv("FAKE_ACP_CONFIG_OPTIONS", "1")
    grok = GrokACPClient(command=command, args=[script], acp_cwd=str(tmp_path))
    assert grok.chat.completions.create(model="m-a", messages=_HI, timeout=30).choices[0].message.content == "hello"
    assert "session/set_config_option" in methods()
    # Former semantics: the copilot placeholder and copilotEnablement-disabled ids are never selected.
    session = {"sessionId": "s1", "configOptions": [{"id": "model", "category": "model", "options": [
        {"value": "m-a"}, {"value": "m-off", "_meta": {"copilotEnablement": "disabled"}}]}]}
    assert [_model_selection_request(session, m) for m in ("copilot-acp", "m-off")] == [None, None]

    plain = CopilotACPClient(command=command, args=[script], acp_cwd=str(tmp_path))
    plain.chat.completions.create(messages=_HI, timeout=30)
    assert spawned == [[command, script], [command, script]]


def test_cursor_acp_pinned_import_shape(agent, tmp_path):
    (command, script), methods = agent
    # cursor-acp __init__.py@d5b6146: own completion shaping on the former timeout normaliser,
    # name-based model selection over the legacy session/set_model, discovery without fs access.
    from agent.copilot_acp_client import CopilotACPClient

    class CursorACPClient(CopilotACPClient):
        def _create_chat_completion(self, *, model=None, messages=None, timeout=None, **_):
            from agent.copilot_acp_client import _effective_timeout

            text, _reasoning = self._run_prompt("prompt", timeout_seconds=_effective_timeout(timeout), model=model)
            return text

        def list_models(self, *, timeout_seconds=15.0):
            with self._session(timeout_seconds, allow_file_requests=False) as (session, _):
                return [m["modelId"].split("[")[0] for m in session["models"]["availableModels"]]

        def _run_prompt(self, prompt_text, *, timeout_seconds, model=None):
            with self._session(timeout_seconds) as (session, request):
                ids = {m["name"]: m["modelId"] for m in session["models"]["availableModels"]}
                request("session/set_model", {"sessionId": session["sessionId"], "modelId": ids[model]})
                parts: list[str] = []
                request("session/prompt", {"sessionId": session["sessionId"], "prompt": [{"type": "text", "text": prompt_text}]},
                        text_parts=parts, reasoning_parts=[])
                return "".join(parts), ""

    timeout = type("Timeout", (), {"connect": 5.0, "read": 30.0, "write": None, "pool": None})()
    client = CursorACPClient(acp_command=command, acp_args=[script], acp_cwd=str(tmp_path))
    assert client.chat.completions.create(model="Fast", messages=_HI, timeout=timeout) == "hello"
    assert client.list_models() == ["fast", "slow"]
    assert methods().count("session/set_model") == 1


def test_alias_keeps_the_former_gh_extension_diagnosis(tmp_path):
    from agent.copilot_acp_client import CopilotACPClient

    script = tmp_path / "old_gh_copilot.py"
    script.write_text("import sys; print('gh-copilot has been deprecated', file=sys.stderr); sys.exit(1)\n", encoding="utf-8")
    client = CopilotACPClient(command=sys.executable, args=[str(script)], acp_cwd=str(tmp_path))
    with pytest.raises(RuntimeError, match="deprecated `gh copilot` extension"):
        client.list_models(timeout_seconds=30)


# Every top-level name agent/copilot_acp_client.py had before the deprecation (0c771d7a5ba).
_FORMER_SURFACE = (
    "ACP_MARKER_BASE_URL", "Any", "Callable", "CopilotACPClient", "Iterator", "Path", "SimpleNamespace",
    "_ACP_PROBE_CACHE", "_DEFAULT_TIMEOUT_SECONDS", "_DEPRECATED_CLI_ERROR", "_DEPRECATION_MARKERS",
    "_DEPRECATION_REQUIRED", "_FS_HANDLERS", "_INITIALIZE_PARAMS", "_PROMPT_PREAMBLE", "_ROLE_LABELS",
    "_acp_supported", "_build_subprocess_env", "_completion_to_stream_chunks", "_effective_timeout",
    "_enabled_id_list", "_ensure_path_within_cwd", "_extract_tool_calls_from_text", "_format_messages_as_prompt",
    "_fs_read_text_file", "_fs_write_text_file", "_is_gh_copilot_deprecation_message", "_jsonrpc_error",
    "_jsonrpc_result", "_legacy_session_model_ids", "_model_config_option", "_model_selection_request",
    "_render_message_content", "_render_tool_bridge_sections", "_resolve_args", "_resolve_command",
    "_resolve_home_dir", "_session_model_ids", "contextlib", "get_nt_namespace_error", "get_read_block_error",
    "get_write_denied_error", "hermes_subprocess_env", "is_write_approval_required", "json", "logger", "logging",
    "os", "queue", "re", "redact_sensitive_text", "resolve_agent_cwd", "shlex", "subprocess", "tempfile",
    "threading", "time", "deque",
)


def test_alias_serves_its_whole_former_surface_and_warns_once(monkeypatch):
    import agent.copilot_acp_client as alias

    monkeypatch.setattr(alias, "_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        missing = [name for name in _FORMER_SURFACE if not hasattr(alias, name)]
    assert missing == []
    deprecations = [str(w.message) for w in caught if w.category is DeprecationWarning]
    assert len(deprecations) == 1 and "agent.acp_client.ACPClient" in deprecations[0]
    assert alias._session_model_ids({"models": {"availableModels": [
        {"modelId": "a"}, {"modelId": "b", "_meta": {"copilotEnablement": "disabled"}}]}}) == ["a"]
    for name in ("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH", "HERMES_COPILOT_ACP_ARGS"):
        monkeypatch.delenv(name, raising=False)
    assert (alias._resolve_command(), alias._resolve_args()) == ("copilot", ["--acp", "--stdio"])
    monkeypatch.setenv("COPILOT_CLI_PATH", "/opt/copilot")
    assert alias._resolve_command() == "/opt/copilot"
