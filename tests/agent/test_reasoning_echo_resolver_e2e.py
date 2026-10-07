"""Production-shape coverage for reasoning_echo: real resolver + real init read.

Salvaged from #73811 per the consolidation triage on #76503 and adapted to this
PR's per-active-provider `model.reasoning_echo` design.

Every existing reasoning_echo test hand-sets `agent._reasoning_echo_flag` (and
`provider`/`base_url`). None drives the REAL config path that `init_agent` uses:

    agent._reasoning_echo_flag = bool(
        (load_config_readonly().get("model") or {}).get("reasoning_echo"))

...which is wrapped in `except Exception: False`, so if that read ever breaks the
feature silently dies and no current test catches it. This test closes that gap.

It also pins the named-custom-provider case: a provider declared under
`providers.<name>` resolves at runtime to `provider == "custom"` (see
`hermes_cli/runtime_provider.py`), and the echo flag must still take effect for
it. A future refactor keying the flag on provider *name* would reintroduce that
miss; this test is the tripwire.

Uses a temp HERMES_HOME + real `load_config_readonly` (the config cache is
path-keyed, so this is hermetic) — no live server, no hand-set flag.
"""

from __future__ import annotations

from hermes_cli.runtime_provider import resolve_runtime_provider
from agent.agent_runtime_helpers import copy_reasoning_content_for_api
from run_agent import AIAgent


def _write_home(tmp_path, monkeypatch, reasoning_echo: bool):
    """Point HERMES_HOME at a temp profile declaring a named custom provider."""
    home = tmp_path / "hermes"
    home.mkdir()
    lines = [
        "model:",
        "  default: kimi-k3",
        "  provider: llamacpp-k3",
    ]
    if reasoning_echo:
        lines.append("  reasoning_echo: true")
    lines += [
        "providers:",
        "  llamacpp-k3:",
        "    base_url: http://127.0.0.1:8098/v1",
        "    key_env: LLAMACPP_KEY",
    ]
    (home / "config.yaml").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Drop any path-keyed config cache from a prior test.
    try:
        from hermes_cli import config as _cfg
        for name in ("_CONFIG_CACHE", "_config_cache"):
            if hasattr(_cfg, name):
                getattr(_cfg, name).clear()
    except Exception:
        pass


def _agent_with_init_flag() -> AIAgent:
    """Build an agent and materialize the echo flag the way init_agent does."""
    from hermes_cli.config import load_config_readonly
    agent = object.__new__(AIAgent)
    agent._reasoning_echo_flag = bool(
        (load_config_readonly().get("model") or {}).get("reasoning_echo")
    )
    agent.provider = "custom"          # what the resolver returns for a named custom provider
    agent.base_url = "http://127.0.0.1:8098/v1"
    agent.model = "kimi-k3"
    agent.verbose_logging = False
    return agent


class TestReasoningEchoResolverE2E:
    def test_named_custom_provider_resolves_to_custom(self, tmp_path, monkeypatch):
        """The provider config path really does collapse to provider == 'custom'."""
        _write_home(tmp_path, monkeypatch, reasoning_echo=True)
        rt = resolve_runtime_provider(requested="llamacpp-k3")
        assert rt["provider"] == "custom"
        assert rt["base_url"].rstrip("/") == "http://127.0.0.1:8098/v1"

    def test_flag_materializes_from_real_config_and_echoes(self, tmp_path, monkeypatch):
        """reasoning_echo: true -> flag True via the real init read -> reasoning kept."""
        _write_home(tmp_path, monkeypatch, reasoning_echo=True)
        agent = _agent_with_init_flag()
        assert agent._reasoning_echo_flag is True
        assert agent._reasoning_echo_opt_in() is True
        assert agent._needs_thinking_reasoning_pad() is True

        source = {"role": "assistant", "content": "calling a tool",
                  "reasoning_content": "the model's chain of thought"}
        api_msg = dict(source)
        copy_reasoning_content_for_api(agent, source, api_msg)
        assert api_msg["reasoning_content"] == "the model's chain of thought"

    def test_flag_absent_strips(self, tmp_path, monkeypatch):
        """No reasoning_echo -> flag False -> reasoning_content stripped on replay."""
        _write_home(tmp_path, monkeypatch, reasoning_echo=False)
        agent = _agent_with_init_flag()
        assert agent._reasoning_echo_flag is False
        assert agent._reasoning_echo_opt_in() is False

        source = {"role": "assistant", "content": "calling a tool",
                  "reasoning_content": "should be stripped"}
        api_msg = dict(source)
        copy_reasoning_content_for_api(agent, source, api_msg)
        assert "reasoning_content" not in api_msg
