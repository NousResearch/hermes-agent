"""Contracts of the public ACP stdio client (``agent.acp_client.ACPClient``) and its deprecated alias."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent.acp_client import ACPClient

_REPO = Path(__file__).resolve().parents[2]

_CATALOG_SERVER = """import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": 1}
    elif method == "session/new":
        result = {"sessionId": "s1", "configOptions": [{"id": "model", "category": "model", "options": [
            {"value": "open-model"},
            {"value": "policy-blocked", "_meta": {"vendorPolicy": "off"}},
            {"value": "copilot-gated", "_meta": {"copilotEnablement": "disabled"}},
        ]}]}
    elif method == "session/prompt":
        print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "hello"}}}}), flush=True)
        result = {"stopReason": "end_turn"}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


class _VendorClient(ACPClient):
    default_command = "vendor-agent"
    default_args = ("--acp",)
    command_env_vars = ("VENDOR_ACP_COMMAND", "VENDOR_CLI_PATH")
    args_env_var = "VENDOR_ACP_ARGS"
    marker_base_url = "acp://vendor"
    placeholder_model = "vendor-acp"
    label = "Vendor ACP"

    def model_entry_disabled(self, entry):
        return (entry.get("_meta") or {}).get("vendorPolicy") == "off"

    def early_exit_error(self, stderr_text):
        return "vendor CLI too old: " + stderr_text if "legacy build" in stderr_text else None


def _server(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "fake_agent.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def test_subclass_launch_fallback_reads_its_env_vars_in_order(monkeypatch, tmp_path):
    for name in ("VENDOR_ACP_COMMAND", "VENDOR_CLI_PATH", "VENDOR_ACP_ARGS"):
        monkeypatch.delenv(name, raising=False)
    client = _VendorClient(acp_cwd=str(tmp_path))
    assert (client._acp_command, client._acp_args, client.base_url, client.api_key) == (
        "vendor-agent", ["--acp"], "acp://vendor", "vendor-acp")

    monkeypatch.setenv("VENDOR_CLI_PATH", "/opt/second")
    monkeypatch.setenv("VENDOR_ACP_COMMAND", "/opt/first")
    monkeypatch.setenv("VENDOR_ACP_ARGS", "--acp --stdio")
    client = _VendorClient(acp_cwd=str(tmp_path))
    assert (client._acp_command, client._acp_args) == ("/opt/first", ["--acp", "--stdio"])
    # Explicit command/args (what runtime resolution passes) always win over the fallback.
    assert _VendorClient(command="x", args=["y"], acp_cwd=str(tmp_path))._acp_args == ["y"]


def test_model_entry_disabled_hook_owns_vendor_filtering(tmp_path):
    command, *args = _server(tmp_path, _CATALOG_SERVER)
    generic = ACPClient(command=command, args=args, acp_cwd=str(tmp_path))
    vendor = _VendorClient(command=command, args=args, acp_cwd=str(tmp_path))
    # The generic client applies no vendor _meta policy; the subclass hook does.
    assert generic.list_models(timeout_seconds=30) == ["open-model", "policy-blocked", "copilot-gated"]
    assert vendor.list_models(timeout_seconds=30) == ["open-model", "copilot-gated"]
    completion = vendor.chat.completions.create(model="vendor-acp", messages=[{"role": "user", "content": "hi"}], timeout=30)
    assert (completion.choices[0].message.content, completion.model) == ("hello", "vendor-acp")


def test_errors_carry_the_subclass_label_and_early_exit_hook(tmp_path):
    missing = _VendorClient(command=str(tmp_path / "absent"), args=["--stdio"], acp_cwd=str(tmp_path))
    with pytest.raises(RuntimeError, match=r"Could not start Vendor ACP command .*VENDOR_ACP_COMMAND/VENDOR_CLI_PATH"):
        missing.chat.completions.create(messages=[{"role": "user", "content": "x"}], timeout=10)

    command, *args = _server(tmp_path, "import sys; print('legacy build 1.0', file=sys.stderr); sys.exit(3)")
    with pytest.raises(RuntimeError, match="vendor CLI too old: legacy build 1.0"):
        _VendorClient(command=command, args=args, acp_cwd=str(tmp_path)).list_models(timeout_seconds=30)
    with pytest.raises(RuntimeError, match="ACP process exited early: legacy build 1.0"):
        ACPClient(command=command, args=args, acp_cwd=str(tmp_path)).list_models(timeout_seconds=30)


def test_malformed_vendor_model_options_fall_back_to_the_session_default(tmp_path):
    body = _CATALOG_SERVER.replace('result = {"sessionId": "s1", "configOptions": [', 'result = {"sessionId": "s1", "configOptions": 5, "x": [')
    command, *args = _server(tmp_path, body)
    completion = _VendorClient(command=command, args=args, acp_cwd=str(tmp_path)).chat.completions.create(
        model="open-model", messages=[{"role": "user", "content": "hi"}], timeout=30)
    assert completion.choices[0].message.content == "hello"


_FROZEN_SUBCLASS_PLUGIN = '''
"""Frozen shape of an out-of-tree ACP provider built on the deprecated alias (kimchi-acp-provider)."""
import os
from providers import register_provider
from providers.base import ProviderProfile
from agent.copilot_acp_client import CopilotACPClient as _Base


class _Mixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (self._acp_command or "") in ("", "copilot"):
            self._acp_command = os.environ["FROZEN_AGENT"]
            self._acp_args = [os.environ["FROZEN_SCRIPT"]]
        if self.base_url == "acp://copilot":
            self.base_url = "acp://frozen"

    def _handle_server_message(self, msg, *, process, cwd, text_parts, reasoning_parts, allow_file_requests=True):
        handled = super()._handle_server_message(msg, process=process, cwd=cwd, text_parts=text_parts,
                                                 reasoning_parts=reasoning_parts,
                                                 allow_file_requests=allow_file_requests)
        if handled and text_parts:
            text_parts[-1] = text_parts[-1].upper()
        return handled


class FrozenClient(_Mixin, _Base):
    pass


class FrozenProfile(ProviderProfile):
    def create_client(self, **kw):
        return FrozenClient(**kw)


register_provider(FrozenProfile(name="frozen-acp", base_url="acp://frozen", auth_type="external_process",
                                process_command="frozen", process_args=("--mode", "acp")))
'''

_FROZEN_DRIVER = """
import json, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import providers
    client = providers.get_provider_profile("frozen-acp").create_client()
    import agent.copilot_acp_client as alias
    alias.CopilotACPClient
from agent.acp_client import ACPClient
reply = client.chat.completions.create(model="copilot-acp", messages=[{"role": "user", "content": "hi"}], timeout=30)
print(json.dumps({
    "warnings": [str(w.message) for w in caught if w.category is DeprecationWarning and "acp_client" in str(w.message)],
    "is_acp_client": isinstance(client, ACPClient),
    "base_url": client.base_url,
    "models": client.list_models(timeout_seconds=30),
    "reply": reply.choices[0].message.content,
}))
"""


def test_deprecated_alias_keeps_a_frozen_out_of_tree_subclass_working(tmp_path):
    home = tmp_path / "hermes"
    plugin = home / "plugins" / "model-providers" / "frozen-acp"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_FROZEN_SUBCLASS_PLUGIN), encoding="utf-8")
    (plugin / "plugin.yaml").write_text("name: frozen-acp\nkind: model-provider\nversion: 1.0.0\n", encoding="utf-8")
    script = tmp_path / "fake_acp_agent.py"
    script.write_text(_CATALOG_SERVER, encoding="utf-8")
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(home), "FROZEN_AGENT": sys.executable, "FROZEN_SCRIPT": str(script),
           "PYTHONPATH": str(_REPO)}
    proc = subprocess.run([sys.executable, "-c", _FROZEN_DRIVER], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["warnings"] == [
        "agent.copilot_acp_client.CopilotACPClient is deprecated; subclass agent.acp_client.ACPClient instead."]
    assert result["is_acp_client"] and result["base_url"] == "acp://frozen"
    assert result["models"] == ["open-model", "policy-blocked"]  # the alias keeps its enablement filter
    assert result["reply"] == "HELLO"
