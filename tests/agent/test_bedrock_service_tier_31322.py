"""Bedrock Converse ``serviceTier`` / ``performanceConfig`` wiring (#31322).

Bedrock prices and schedules each request by service tier and can serve some
models on a latency-optimized path, but both are per-request Converse fields
that Hermes never sent — every turn ran at the account default.

These tests cover the whole chain rather than just the adapter, because the
bug was never in the field names: it was that nothing carried a configured
value to the wire.

  1. ``resolve_bedrock_tier_config()`` reads ``bedrock.service_tier`` /
     ``bedrock.latency`` out of a real config.yaml under an isolated
     ``HERMES_HOME`` and shapes them as Converse structures.
  2. Bad values are dropped, not forwarded — Bedrock answers an unknown enum
     with a ValidationException that would fail every turn.
  3. ``build_api_kwargs`` → ``BedrockTransport.build_kwargs`` →
     ``build_converse_kwargs`` carries the resolved structures into the exact
     dict handed to ``client.converse(**kwargs)``.
  4. The field shapes match the botocore service model for
     ``bedrock-runtime.Converse`` and ``ConverseStream``.
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def isolated_home(monkeypatch):
    """Temp HERMES_HOME so config reads never touch the developer's real one."""
    test_home = tempfile.mkdtemp(prefix="hermes_test_31322_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    monkeypatch.setenv("HERMES_HOME", hermes_home)
    yield hermes_home
    shutil.rmtree(test_home, ignore_errors=True)


def _write_config(home: str, text: str) -> None:
    with open(os.path.join(home, "config.yaml"), "w") as fp:
        fp.write(text)


def _bedrock_agent(service_tier=None, performance_config=None):
    """Minimal agent stub that reaches the ``bedrock_converse`` branch."""
    from agent.transports.bedrock import BedrockTransport

    transport = BedrockTransport()
    return SimpleNamespace(
        api_mode="bedrock_converse",
        provider="bedrock",
        model="us.amazon.nova-pro-v1:0",
        tools=None,
        max_tokens=4096,
        _bedrock_region="us-east-2",
        _bedrock_guardrail_config=None,
        _bedrock_service_tier=service_tier,
        _bedrock_performance_config=performance_config,
        _get_transport=lambda: transport,
    )


# ---------------------------------------------------------------------------
# 1. config.yaml → resolved Converse structures
# ---------------------------------------------------------------------------

class TestResolveFromConfig:

    def test_reads_both_keys(self, isolated_home):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, "bedrock:\n  service_tier: flex\n  latency: optimized\n")
        tier, perf = resolve_bedrock_tier_config()
        assert tier == {"type": "flex"}
        assert perf == {"latency": "optimized"}

    def test_absent_block_resolves_to_none(self, isolated_home):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, "bedrock:\n  region: us-east-2\n")
        assert resolve_bedrock_tier_config() == (None, None)

    def test_empty_strings_resolve_to_none(self, isolated_home):
        """The shipped default is ``""`` — it must not become a wire field."""
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, 'bedrock:\n  service_tier: ""\n  latency: ""\n')
        assert resolve_bedrock_tier_config() == (None, None)

    def test_keys_are_independent(self, isolated_home):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, "bedrock:\n  latency: optimized\n")
        assert resolve_bedrock_tier_config() == (None, {"latency": "optimized"})

    @pytest.mark.parametrize("tier", ["priority", "default", "flex", "reserved"])
    def test_every_documented_tier_survives(self, isolated_home, tier):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, f"bedrock:\n  service_tier: {tier}\n")
        assert resolve_bedrock_tier_config()[0] == {"type": tier}

    def test_case_and_whitespace_tolerated(self, isolated_home):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, 'bedrock:\n  service_tier: "  FLEX  "\n')
        assert resolve_bedrock_tier_config()[0] == {"type": "flex"}

    def test_explicit_config_dict_skips_the_disk_read(self):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        tier, perf = resolve_bedrock_tier_config({"bedrock": {"service_tier": "priority"}})
        assert (tier, perf) == ({"type": "priority"}, None)


# ---------------------------------------------------------------------------
# 2. Invalid values are dropped, never forwarded
# ---------------------------------------------------------------------------

class TestInvalidValuesAreDropped:

    def test_unknown_tier_dropped_with_warning(self, isolated_home, caplog):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, "bedrock:\n  service_tier: cheapest\n")
        with caplog.at_level("WARNING"):
            tier, _ = resolve_bedrock_tier_config()
        assert tier is None
        assert "cheapest" in caplog.text

    def test_unknown_latency_dropped_with_warning(self, isolated_home, caplog):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, "bedrock:\n  latency: fastest\n")
        with caplog.at_level("WARNING"):
            _, perf = resolve_bedrock_tier_config()
        assert perf is None
        assert "fastest" in caplog.text

    def test_a_bad_tier_does_not_take_a_good_latency_with_it(self, isolated_home):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _write_config(isolated_home, "bedrock:\n  service_tier: nope\n  latency: optimized\n")
        assert resolve_bedrock_tier_config() == (None, {"latency": "optimized"})

    def test_non_string_values_do_not_raise(self):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        assert resolve_bedrock_tier_config(
            {"bedrock": {"service_tier": 5, "latency": ["optimized"]}}
        ) == (None, None)

    def test_missing_config_module_does_not_raise(self, isolated_home):
        """A config read failure must leave the request at the account default."""
        from unittest.mock import patch

        import hermes_cli.config as cfg
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        with patch.object(cfg, "load_config_readonly", side_effect=RuntimeError("boom")):
            assert resolve_bedrock_tier_config() == (None, None)


# ---------------------------------------------------------------------------
# 3. Resolved config reaches the dict passed to client.converse()
# ---------------------------------------------------------------------------

class TestConfigReachesTheRequest:

    def test_end_to_end_config_to_request(self, isolated_home):
        """The seam the adapter unit tests miss: config.yaml → converse kwargs."""
        from agent.bedrock_adapter import resolve_bedrock_tier_config
        from agent.chat_completion_helpers import build_api_kwargs

        _write_config(isolated_home, "bedrock:\n  service_tier: flex\n  latency: optimized\n")
        tier, perf = resolve_bedrock_tier_config()
        agent = _bedrock_agent(service_tier=tier, performance_config=perf)

        kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

        assert kwargs["serviceTier"] == {"type": "flex"}
        assert kwargs["performanceConfig"] == {"latency": "optimized"}

    def test_unset_config_sends_neither_field(self, isolated_home):
        from agent.bedrock_adapter import resolve_bedrock_tier_config
        from agent.chat_completion_helpers import build_api_kwargs

        _write_config(isolated_home, "bedrock:\n  region: us-east-2\n")
        tier, perf = resolve_bedrock_tier_config()
        agent = _bedrock_agent(service_tier=tier, performance_config=perf)

        kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

        assert "serviceTier" not in kwargs
        assert "performanceConfig" not in kwargs

    def test_agent_without_the_attributes_still_builds(self):
        """Agents built before this change (or by other call paths) must not
        crash on the new getattr lookups."""
        from agent.chat_completion_helpers import build_api_kwargs

        agent = _bedrock_agent()
        del agent._bedrock_service_tier
        del agent._bedrock_performance_config

        kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

        assert "serviceTier" not in kwargs
        assert "performanceConfig" not in kwargs

    def test_transport_forwards_both_params(self):
        from agent.transports.bedrock import BedrockTransport

        kwargs = BedrockTransport().build_kwargs(
            model="us.amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            service_tier={"type": "reserved"},
            performance_config={"latency": "standard"},
        )
        assert kwargs["serviceTier"] == {"type": "reserved"}
        assert kwargs["performanceConfig"] == {"latency": "standard"}

    def test_adapter_omits_falsy_structures(self):
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model="us.amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            service_tier={},
            performance_config={},
        )
        assert "serviceTier" not in kwargs
        assert "performanceConfig" not in kwargs

    def test_tier_fields_do_not_disturb_guardrails(self):
        """Both features are optional top-level Converse fields; setting one
        must not shadow the other."""
        from agent.bedrock_adapter import build_converse_kwargs

        guardrail = {"guardrailIdentifier": "gr-abc123", "guardrailVersion": "1"}
        kwargs = build_converse_kwargs(
            model="us.amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            guardrail_config=guardrail,
            service_tier={"type": "flex"},
        )
        assert kwargs["guardrailConfig"] == guardrail
        assert kwargs["serviceTier"] == {"type": "flex"}


# ---------------------------------------------------------------------------
# 4. Shapes match the AWS service model
# ---------------------------------------------------------------------------

class TestWireContract:
    """Freeze the enums and structures taken from the AWS service model.

    Transcribed from the botocore service model for ``bedrock-runtime``
    (botocore 1.42.59), where ``Converse`` and ``ConverseStream`` declare
    identical members:

        serviceTier      structure, required member ``type``:
                         priority | default | flex | reserved
        performanceConfig structure, optional member ``latency``:
                         standard | optimized

    Asserted from the frozen values rather than by introspecting botocore,
    because botocore is an optional dependency here (the Bedrock suites run
    against fake ``botocore`` modules) and an ``importorskip`` check would
    silently never run. Drift in either enum is a ValidationException on every
    turn, so re-check against ``get_service_model('bedrock-runtime')`` when
    touching these sets.
    """

    def test_service_tier_enum_frozen(self):
        from agent.bedrock_adapter import _BEDROCK_SERVICE_TIERS

        assert _BEDROCK_SERVICE_TIERS == {"priority", "default", "flex", "reserved"}

    def test_latency_enum_frozen(self):
        from agent.bedrock_adapter import _BEDROCK_LATENCY_MODES

        assert _BEDROCK_LATENCY_MODES == {"standard", "optimized"}

    def test_service_tier_is_a_structure_not_a_bare_string(self):
        """``serviceTier`` takes ``{"type": tier}`` — a bare string is rejected."""
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        tier, _ = resolve_bedrock_tier_config({"bedrock": {"service_tier": "flex"}})
        assert tier == {"type": "flex"}

    def test_latency_is_nested_under_performance_config(self):
        from agent.bedrock_adapter import resolve_bedrock_tier_config

        _, perf = resolve_bedrock_tier_config({"bedrock": {"latency": "optimized"}})
        assert perf == {"latency": "optimized"}


# ---------------------------------------------------------------------------
# 5. The shipped default keeps behaviour unchanged
# ---------------------------------------------------------------------------

def test_default_config_ships_both_keys_empty():
    """Present so users can discover them; empty so nothing changes on upgrade."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    bedrock_defaults = DEFAULT_CONFIG["bedrock"]
    assert bedrock_defaults["service_tier"] == ""
    assert bedrock_defaults["latency"] == ""
