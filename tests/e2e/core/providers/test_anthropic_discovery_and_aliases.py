"""Anthropic route plumbing: model discovery endpoint and OAuth wire tool-name aliases.

* Discovery: with a custom Anthropic-protocol endpoint configured (``ANTHROPIC_BASE_URL``
  or ``model.base_url``), the picker catalog (``provider_model_ids``, behind ``hermes model``,
  ``/model``, the dashboard and Desktop pickers) must come from THAT endpoint's
  ``/v1/models`` — asserted by the fake seeing the GET and its relay-only id in the result.
* OAuth wire aliases: on the Claude subscription route Hermes renames tools on the wire
  (``mcp__`` prefix, ``memory`` -> ``context_notes``, ``session_search`` ->
  ``chat_history_lookup``). A real ``hermes -z`` turn must map every wire name the model
  may use back to the real tool, including names passed as tool-search arguments.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.core._pending_fixes import known_gate
from tests.e2e.core.providers._anthropic_helpers import Rig, blocks, start_rig
from tests.fakes.providers.anthropic_messages import AnthropicMessagesServer, Reply, Text, ToolUse

pytestmark = [pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-tree cleanup uses /proc"),
              pytest.mark.live_system_guard_bypass]

class DiscoveryIgnoredEndpoint(AssertionError):
    """#120844's signature: raised ONLY when the configured endpoint's catalog is not what the picker
    serves; the type ``known_gate`` accepts for the discovery cells."""


class AliasNotReverseMapped(AssertionError):
    """#120858's signature: raised ONLY when tool_describe fails to resolve the advertised wire alias;
    the type ``known_gate`` accepts for the alias cell."""


# Red on main for a tracked, open bug: key -> (the bug's own failure-message pattern, reason). Each
# gate accepts only its dedicated exception, so harness failures in a KNOWN cell stay failures.
KNOWN: dict[str, tuple[str, str]] = {
    "discovery_env": (r"configured endpoint probed=False \(native host saw \d+ GETs\); relay catalog served=False",
                      "#120844 Anthropic model discovery ignores ANTHROPIC_BASE_URL"),
    "alias_describe": (r"tool_describe did not resolve the advertised alias: .*not_found.*chat_history_lookup",
                       "#120858 wire alias chat_history_lookup not reverse-mapped in tool_describe args"),
}
RELAY_ONLY_MODEL = "claude-e2e-relay-only-7"
OAUTH_TOKEN = "sk-ant-oat01-e2e-fake-oauth-token"


@pytest.fixture
def rig(tmp_path: Path):
    made: list[Rig] = []

    def make(script, **kw) -> Rig:
        r = start_rig(tmp_path / f"r{len(made)}", script, **kw)
        made.append(r)
        return r

    yield make
    for r in made:
        r.stop()


# The CLI's own startup sequence (load ~/.hermes/.env), then the picker entry point.
_DISCOVERY_PROBE = (
    "import json; from hermes_cli.env_loader import load_hermes_dotenv; load_hermes_dotenv(); "
    "from hermes_cli.models import provider_model_ids; "
    "print('IDS=' + json.dumps(list(provider_model_ids('anthropic', force_refresh=True))))"
)


def _discover(r: Rig, relay: AnthropicMessagesServer | None, *, via: str) -> list[str]:
    if relay is None:
        extra: dict[str, str] = {}
    elif via == "config":
        r.update_config(lambda c: c["model"].__setitem__("base_url", relay.base_url))
        extra = {}
    else:
        extra = {"ANTHROPIC_BASE_URL": relay.base_url.removesuffix("/anthropic")}
    proc = subprocess.run([sys.executable, "-c", _DISCOVERY_PROBE], cwd=r.project, env=r.env(extra),
                          capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("IDS="))
    return json.loads(line[4:])


def test_model_discovery_serves_the_live_native_catalog(rig) -> None:
    """Control for the override cases: with no custom endpoint the picker probes the native
    host's /v1/models (the intercepted fake) and serves its live ids, not the static table."""
    r = rig([])
    r.srv.models = [RELAY_ONLY_MODEL, "claude-e2e-native-other"]
    ids = _discover(r, None, via="native")
    assert [g for g in r.srv.gets if "/v1/models" in g["path"]], "discovery never probed the native endpoint"
    assert RELAY_ONLY_MODEL in ids, f"live catalog not served: {ids[:15]}"


@pytest.mark.parametrize("via", ["env", "config"])
def test_model_discovery_probes_the_configured_anthropic_endpoint(rig, via: str) -> None:
    relay = AnthropicMessagesServer([], models=[RELAY_ONLY_MODEL, "claude-e2e-relay-other"]).start()
    try:
        r = rig([])
        ids = _discover(r, relay, via=via)
        probed = [g["path"] for g in relay.gets if "/v1/models" in g["path"]]
        with known_gate(KNOWN, f"discovery_{via}", raises=DiscoveryIgnoredEndpoint):
            if not probed or RELAY_ONLY_MODEL not in ids:
                raise DiscoveryIgnoredEndpoint(
                    f"configured endpoint probed={bool(probed)} (native host saw {len(r.srv.gets)} GETs); "
                    f"relay catalog served={RELAY_ONLY_MODEL in ids}: {ids[:15]}")
    finally:
        relay.stop()


def _oauth_rig(rig, script) -> Rig:
    r = rig(script)
    (r.hermes_home / ".env").write_text(f"ANTHROPIC_TOKEN={OAUTH_TOKEN}\n", encoding="utf-8")
    return r


def _tool_results(body: dict) -> list[str]:
    return [json.dumps(b.get("content")) for m in body["messages"] if m["role"] == "user"
            for b in blocks(m) if b.get("type") == "tool_result"]


def test_oauth_wire_names_map_back_to_real_tools(rig) -> None:
    """The model answers with wire names (``mcp__read_file`` and the ``context_notes`` alias of
    ``memory``) in one parallel turn; both must execute as the real tools: the file content comes
    back in the tool_result and the note lands in the real MEMORY.md."""
    note = "E2E-OAUTH-ALIAS-NOTE-5d1c"
    r = _oauth_rig(rig, [])
    target = r.project / "wire.txt"
    target.write_text("WIRE-FILE-CONTENT-93", encoding="utf-8")
    r.srv.push(
        Reply([ToolUse("mcp__read_file", {"path": str(target)}),
               ToolUse("mcp__context_notes", {"action": "add", "target": "memory", "content": note})]),
        Reply([Text("ALIASES-DONE")]),
    )
    proc = r.run("-z", "Read wire.txt and remember the note.")
    assert proc.returncode == 0 and "ALIASES-DONE" in proc.stdout, proc.stderr[-2000:]
    mains = r.srv.main_requests()
    assert len(mains) == 2
    first = mains[0]
    assert first["headers"].get("authorization") == f"Bearer {OAUTH_TOKEN}", "OAuth route must send the bearer"
    wire_names = {t["name"] for t in first["body"]["tools"]}
    assert {"mcp__read_file", "mcp__context_notes"} <= wire_names, sorted(wire_names)
    assert not {"memory", "session_search", "read_file"} & wire_names, "real names leaked onto the OAuth wire"
    results = _tool_results(mains[1]["body"])
    assert any("WIRE-FILE-CONTENT-93" in res for res in results), results
    memory_md = r.hermes_home / "memories" / "MEMORY.md"
    assert memory_md.exists() and note in memory_md.read_text(encoding="utf-8"), (
        f"aliased context_notes call did not reach the memory tool: {results}")
    assert not r.srv.schema_errors(), r.srv.schema_errors()


def test_oauth_deferred_alias_resolves_through_tool_describe(rig) -> None:
    """The deferred catalog advertises ``chat_history_lookup`` (the OAuth alias of
    ``session_search``); asking ``tool_describe`` for that exact name must return its schema."""
    r = _oauth_rig(rig, [Reply([ToolUse("mcp__tool_describe", {"names": ["chat_history_lookup"]})]),
                         Reply([Text("DESCRIBED")])])
    proc = r.run("-z", "Look up how to recall past chats.")
    assert proc.returncode == 0 and "DESCRIBED" in proc.stdout, proc.stderr[-2000:]
    mains = r.srv.main_requests()
    catalog = next(t for t in mains[0]["body"]["tools"] if t["name"] == "mcp__tool_search")
    assert "chat_history_lookup" in catalog["description"], "precondition: alias advertised in the catalog"
    (result,) = _tool_results(mains[1]["body"])
    not_found = "not_found" in result and "chat_history_lookup" in result.split("not_found", 1)[1][:80]
    described = "chat_history_lookup" in result and ("parameters" in result or "input_schema" in result)
    with known_gate(KNOWN, "alias_describe", raises=AliasNotReverseMapped):
        if not_found or not described:
            raise AliasNotReverseMapped(f"tool_describe did not resolve the advertised alias: {result[:600]}")
