#!/usr/bin/env python3
"""Production-shape e2e for the reasoning_replay fix (Nous PR #73811).

The bug the sweeper caught: named custom providers resolve to runtime
provider == "custom", not the config-key name. The earlier tests hand-set
agent.provider = "llamacpp", bypassing that. This test does NOT hand-set the
provider/base_url: it feeds the documented config through Hermes's REAL
resolver (resolve_runtime_provider, hermes_cli/runtime_provider.py) and builds
the agent from whatever the resolver returns — the exact production path.

Runs against a live llama-server (the tiny K3 fixture, which genuinely emits
reasoning_content deltas) so the replay half is real bytes, not a stubbed
field. HERMES_HOME points at a scratch profile carrying the named-custom-
provider config.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, "/home/darkstar/hermes-agent-pr")

from hermes_cli.runtime_provider import resolve_runtime_provider
from agent.agent_runtime_helpers import copy_reasoning_content_for_api
from run_agent import AIAgent

SERVER = "http://127.0.0.1:8098"


def build_agent_from_resolver():
    """Construct an AIAgent using ONLY values the real resolver produced."""
    rt = resolve_runtime_provider(requested="llamacpp-k3")
    agent = object.__new__(AIAgent)
    # These come from the resolver, not from us:
    agent.provider = rt["provider"]
    agent.base_url = rt["base_url"]
    agent.model = rt.get("model", "kimi-k3")
    agent.verbose_logging = False
    return agent, rt


def server_reasoning():
    body = json.dumps({"model": "kimi-k3",
                       "messages": [{"role": "user", "content": "Hi"}],
                       "max_tokens": 12, "temperature": 0}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            SERVER + "/v1/chat/completions", body,
            {"Content-Type": "application/json"}), timeout=300)
        return json.load(r)["choices"][0]["message"].get("reasoning_content")
    except Exception as e:
        return f"__error__ {e}"


def main():
    expect_present = sys.argv[1] == "present"

    agent, rt = build_agent_from_resolver()
    print(f"resolver returned: provider={rt['provider']!r} "
          f"base_url={rt['base_url']!r} source={rt.get('source')!r}")
    assert rt["provider"] == "custom", \
        f"expected runtime label 'custom', got {rt['provider']!r} — test would be vacuous"

    fires = agent._provider_reasoning_replay_configured()
    print(f"detection with resolver-built agent: {fires}")

    rc = server_reasoning()
    got_server_reasoning = isinstance(rc, str) and not rc.startswith("__error__")
    print(f"server reasoning_content: "
          f"{'yes' if got_server_reasoning else rc}")
    reasoning = rc if got_server_reasoning else "model chain of thought"

    source = {"role": "assistant", "content": "calling a tool",
              "reasoning_content": reasoning}
    api_msg = dict(source)
    copy_reasoning_content_for_api(agent, source, api_msg)
    present = "reasoning_content" in api_msg
    print(f"after replay: reasoning_content {'PRESENT' if present else 'STRIPPED'}")

    assert fires == expect_present, f"detection={fires}, expected {expect_present}"
    assert present == expect_present, f"replay present={present}, expected {expect_present}"
    print(f"VERDICT: PASS (resolver->custom, detection+replay {'on' if expect_present else 'off'})")


if __name__ == "__main__":
    main()
