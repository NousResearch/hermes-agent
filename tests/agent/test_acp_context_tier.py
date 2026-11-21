"""A request's context tier rides the spawn args; an in-session model set must not run
when the session already reports the model (Copilot clears the tier on any set)."""

from __future__ import annotations

import sys

import pytest

from agent.acp_client import ACPClient


def _echo_server(tmp_path) -> str:
    """Fake ACP agent answering from its own argv; any session/set_* clears tier+model.

    A missing --context surfaces as 'no-flag' so a default-tier assertion proves the
    flag actually rode the launch args.
    """
    server = tmp_path / "server.py"
    server.write_text(
        "import json, sys\n"
        "def flag(name):\n"
        "    args = sys.argv\n"
        "    for i, a in enumerate(args):\n"
        "        if a == name and i + 1 < len(args): return args[i + 1]\n"
        "        if a.startswith(name + '='): return a.split('=', 1)[1]\n"
        "    return None\n"
        "model = flag('--model') or 'auto'; tier = flag('--context') or 'no-flag'\n"
        "for line in sys.stdin:\n"
        "    r = json.loads(line); method = r['method']\n"
        "    if method == 'session/new':\n"
        "        result = {'sessionId': 's', 'configOptions': [{'id': 'model', 'currentValue': model,\n"
        "            'options': [{'value': model}, {'value': 'test-model'}]}]}\n"
        "    elif method.startswith('session/set_'):\n"
        "        tier = 'default'; model = 'test-model'; result = {}\n"
        "    elif method == 'session/prompt':\n"
        "        print(json.dumps({'method': 'session/update', 'params': {'update': {\n"
        "            'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text',\n"
        "                'text': model + ':' + tier}}}}), flush=True)\n"
        "        result = {'stopReason': 'end_turn'}\n"
        "    else: result = {}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': r['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    return str(server)


@pytest.mark.parametrize("tier", ["default", "long_context"])
def test_tier_and_model_ride_the_spawn_args(tmp_path, tier):
    server = _echo_server(tmp_path)
    args = [server]
    client = ACPClient(command=sys.executable, args=args, acp_cwd=str(tmp_path))
    result = client.chat.completions.create(
        model="test-model", messages=[{"role": "user", "content": "hi"}], timeout=15,
        extra_body={"copilot_context_tier": tier})
    assert result.choices[0].message.content == "test-model:" + tier
    assert client._acp_args == args  # another caller must not inherit this tier


@pytest.mark.parametrize("configured", [
    ["--model", "env-model", "--sync-session"],
    ["--model=env-model", "--sync-session"],
])
def test_tier_request_without_a_model_keeps_the_configured_one(tmp_path, configured):
    server = _echo_server(tmp_path)
    client = ACPClient(command=sys.executable, args=[server, *configured], acp_cwd=str(tmp_path))
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}], timeout=15,  # no model on this request
        extra_body={"copilot_context_tier": "default"})
    assert result.choices[0].message.content == "env-model:default"


def test_same_model_does_not_reissue_a_tier_clearing_set(tmp_path):
    """session/new already reports the requested model: no session/set_* may run."""
    server = tmp_path / "server.py"
    server.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    r = json.loads(line); method = r['method']\n"
        "    print(json.dumps({'method': 'session/update', 'params': {'update': {\n"
        "        'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text',\n"
        "            'text': method}}}}), flush=True) if method == 'session/prompt' else None\n"
        "    result = {'sessionId': 's', 'configOptions': [{'id': 'model',\n"
        "        'currentValue': 'test-model', 'options': [{'value': 'test-model'}]}]} \\\n"
        "        if method == 'session/new' else {'stopReason': 'end_turn'} \\\n"
        "        if method == 'session/prompt' else {}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': r['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    client = ACPClient(command=sys.executable, args=[str(server)], acp_cwd=str(tmp_path))
    result = client.chat.completions.create(model="test-model", messages=[], timeout=15)
    assert result.choices[0].message.content == "session/prompt"  # no set_config_option first


def test_invalid_tier_fails_before_starting_a_process(tmp_path):
    client = ACPClient(command=sys.executable, args=["/nonexistent-agent"], acp_cwd=str(tmp_path))
    with pytest.raises(ValueError, match="context tier"):
        client.chat.completions.create(
            messages=[], timeout=15, extra_body={"copilot_context_tier": "bogus"})
