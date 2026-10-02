"""Per-job max_tokens output cap: store contract, scheduler precedence, clamp.

A cron job may cap the output tokens any one of its turns requests. The cap
exists because a metered or weekly-budget API key is rejected by the provider
for asking for more than the budget allows, and today nothing in the job record
or the cron config can lower that number.

Contract under test:

- Job store (cron/jobs.py): the field is validated at the storage choke point.
  Only a positive int persists; booleans, non-positive, fractional and
  non-numeric values are rejected rather than reaching the wire (a bare True
  would become 1 output token). An absent field keeps the job record
  byte-identical to pre-feature behavior; an empty string clears it.
- Scheduler resolution (cron/scheduler.py::_resolve_job_max_tokens): a job
  cap beats cron.max_tokens_default; neither set yields None so the transport's
  own default is untouched. The effective cap is clamped to the provider
  profile's static default, so a cap can only TIGHTEN a request — on the
  qwen-oauth route a job asking for 131072 lands on the route's 65536 instead
  of raising it. A hand-edited garbage value warns and falls back to the
  config default rather than killing the tick.
- Wiring (_construct_cron_agent): the resolved cap reaches AIAgent as
  max_tokens, which is the value the transports put on the wire.

The clamp reads the provider profile's default_max_tokens, NOT
DEFAULT_CONTEXT_LENGTHS: that table is the model's context window and sizes the
compressor threshold, so treating it as an output cap would both be wrong and
let a cap exceed what the route already sends.
"""

import pytest

from cron.jobs import create_job, load_jobs, update_job


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate the cron store (same pattern as tests/cron/test_cron_reasoning_effort.py)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path / "cron"


def _create(**kw):
    kw.setdefault("prompt", "say hi")
    kw.setdefault("schedule", "every 1h")
    return create_job(**kw)


# A route with a static cap (mirrors plugins/model-providers/qwen-oauth) and one
# without (mirrors openrouter, whose default_max_tokens is None).
CAPPED_RUNTIME = {"provider": "qwen-oauth"}
UNCAPPED_RUNTIME = {"provider": "openrouter"}


class TestJobStoreMaxTokens:

    @pytest.mark.parametrize("raw,expected", [(32768, 32768), ("32768", 32768), ("  32768 ", 32768), (1, 1)])
    def test_positive_int_stored_normalized(self, tmp_cron_dir, raw, expected):
        job = _create(max_tokens=raw)
        assert job["max_tokens"] == expected
        # And it round-trips through the store.
        assert load_jobs()[0]["max_tokens"] == expected

    @pytest.mark.parametrize("garbage", [0, -1, "0", "-5", True, False, "abc", "32768.5", 3.5, [1]])
    def test_invalid_rejected_nothing_persisted(self, tmp_cron_dir, garbage):
        with pytest.raises(ValueError):
            _create(max_tokens=garbage)
        assert load_jobs() == []

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_empty_means_unset_key_absent(self, tmp_cron_dir, empty):
        job = _create(max_tokens=empty)
        assert "max_tokens" not in job
        assert job.get("max_tokens") is None

    def test_absent_field_record_is_byte_identical(self, tmp_cron_dir):
        """A job that never sets a cap must not gain the key at all (config fallback stays clean)."""
        job = _create()
        assert "max_tokens" not in job

    def test_update_sets_field(self, tmp_cron_dir):
        job = _create()
        updated = update_job(job["id"], {"max_tokens": 8192})
        assert updated["max_tokens"] == 8192
        assert load_jobs()[0]["max_tokens"] == 8192

    def test_update_empty_string_clears(self, tmp_cron_dir):
        job = _create(max_tokens=8192)
        updated = update_job(job["id"], {"max_tokens": ""})
        assert updated.get("max_tokens") is None

    def test_update_garbage_rejected_stored_value_untouched(self, tmp_cron_dir):
        job = _create(max_tokens=8192)
        with pytest.raises(ValueError):
            update_job(job["id"], {"max_tokens": -3})
        assert load_jobs()[0]["max_tokens"] == 8192


class TestSchedulerMaxTokensPrecedence:

    def test_job_cap_wins_over_config_default(self):
        from cron.scheduler import _resolve_job_max_tokens

        cfg = {"cron": {"max_tokens_default": 65536}}
        job = {"id": "j1", "max_tokens": 32768}
        assert _resolve_job_max_tokens(job, cfg, CAPPED_RUNTIME, "qwen3-max") == 32768

    def test_config_default_applies_when_job_sets_none(self):
        from cron.scheduler import _resolve_job_max_tokens

        cfg = {"cron": {"max_tokens_default": 32768}}
        assert _resolve_job_max_tokens({"id": "j1"}, cfg, CAPPED_RUNTIME, "qwen3-max") == 32768
        # An explicit null on the job falls through to config, not to the transport.
        assert _resolve_job_max_tokens({"id": "j1", "max_tokens": None}, cfg, CAPPED_RUNTIME, "qwen3-max") == 32768

    def test_nothing_set_leaves_transport_default_untouched(self):
        """The pre-feature path: None means the agent keeps max_tokens=None."""
        from cron.scheduler import _resolve_job_max_tokens

        for cfg in ({}, {"cron": {}}, {"cron": {"max_tokens_default": 0}}, {"cron": {"max_tokens_default": ""}}):
            assert _resolve_job_max_tokens({"id": "j1"}, cfg, CAPPED_RUNTIME, "qwen3-max") is None

    def test_cap_clamped_to_provider_default(self):
        """A cap can only tighten: 131072 on a 65536-default route lands on 65536."""
        from cron.scheduler import _resolve_job_max_tokens

        job = {"id": "j1", "max_tokens": 131072}
        assert _resolve_job_max_tokens(job, {}, CAPPED_RUNTIME, "qwen3-max") == 65536

    def test_cap_below_provider_default_survives(self):
        from cron.scheduler import _resolve_job_max_tokens

        job = {"id": "j1", "max_tokens": 16384}
        assert _resolve_job_max_tokens(job, {}, CAPPED_RUNTIME, "qwen3-max") == 16384

    def test_uncapped_route_takes_the_cap_verbatim(self):
        """No profile default = nothing to clamp against, so the operator's number is used."""
        from cron.scheduler import _resolve_job_max_tokens

        job = {"id": "j1", "max_tokens": 131072}
        assert _resolve_job_max_tokens(job, {}, UNCAPPED_RUNTIME, "qwen3-max") == 131072

    def test_garbage_stored_value_falls_back_to_config(self, tmp_cron_dir):
        """A hand-edited jobs.json must not kill the tick: warn, then use the config default."""
        from cron.scheduler import _resolve_job_max_tokens

        cfg = {"cron": {"max_tokens_default": 32768}}
        assert _resolve_job_max_tokens({"id": "j1", "max_tokens": "turbo"}, cfg, CAPPED_RUNTIME, "qwen3-max") == 32768
        # And with no config default it degrades to the transport's own behaviour.
        assert _resolve_job_max_tokens({"id": "j1", "max_tokens": -1}, {}, CAPPED_RUNTIME, "qwen3-max") is None

    def test_boolean_stored_value_is_not_a_cap(self):
        """True must never become a 1-token cap: a bool is a mis-set flag, not a number."""
        from cron.scheduler import _positive_int_or_none

        assert _positive_int_or_none(True) is None
        assert _positive_int_or_none(False) is None
        assert _positive_int_or_none("4096") == 4096

    def test_unresolvable_provider_does_not_raise(self):
        """An unknown provider name leaves no ceiling to clamp against; the cap still applies."""
        from cron.scheduler import _resolve_job_max_tokens

        job = {"id": "j1", "max_tokens": 16384}
        assert _resolve_job_max_tokens(job, {}, {"provider": "no-such-provider"}, "some-model") == 16384
        assert _resolve_job_max_tokens(job, {}, {}, "some-model") == 16384
        assert _resolve_job_max_tokens(job, {}, {"provider": None}, "some-model") == 16384


class TestConfigDefault:
    def test_cron_section_ships_the_knob_unset(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["cron"]["max_tokens_default"] == 0


class TestCapReachesTheWire:
    """The resolved cap is what the transports put in the outgoing request."""

    @pytest.mark.parametrize("cap,expected", [(None, 65536), (32768, 32768), (131072, 65536)])
    def test_qwen_oauth_route(self, cap, expected):
        from agent.transports import get_transport
        from cron.scheduler import _resolve_job_max_tokens

        resolved = _resolve_job_max_tokens(
            {"id": "j1", "max_tokens": cap}, {}, CAPPED_RUNTIME, "qwen3-max")
        kwargs = get_transport("chat_completions").build_kwargs(
            "qwen3-max", [{"role": "user", "content": "hi"}], None,
            max_tokens=resolved, reasoning_config=None,
            max_tokens_param_fn=lambda v: {"max_tokens": v}, provider_profile=None)
        assert kwargs.get("max_tokens", expected) == expected

    def test_anthropic_route_receives_the_cap(self):
        from agent.transports import get_transport
        from cron.scheduler import _resolve_job_max_tokens

        resolved = _resolve_job_max_tokens(
            {"id": "j1", "max_tokens": 16384}, {}, {"provider": "anthropic"}, "claude-sonnet-4.6")
        kwargs = get_transport("anthropic_messages").build_kwargs(
            model="claude-sonnet-4.6", messages=[{"role": "user", "content": "hi"}], tools=None,
            max_tokens=resolved, reasoning_config=None)
        assert kwargs["max_tokens"] == 16384


class TestAgentConstructionWiring:

    def test_cap_is_forwarded_to_the_agent(self, tmp_cron_dir):
        """_construct_cron_agent must pass the resolved cap as AIAgent(max_tokens=...)."""
        from cron.scheduler import _CronAgentSetup, _construct_cron_agent

        captured = {}

        class _FakeAgent:
            def __init__(self, **kw):
                captured.update(kw)

        setup = _CronAgentSetup(model="qwen3-max", runtime={"provider": "qwen-oauth"}, max_tokens=16384)
        _construct_cron_agent(_FakeAgent, {"id": "j1"}, {}, setup, workdir=None,
                              session_id="cron_j1", session_db=None)
        assert captured["max_tokens"] == 16384
        assert captured["model"] == "qwen3-max"

    def test_no_cap_stays_none(self):
        """Unset keeps the pre-feature construction exactly: max_tokens is None, not a derived number."""
        from cron.scheduler import _CronAgentSetup, _construct_cron_agent

        captured = {}

        class _FakeAgent:
            def __init__(self, **kw):
                captured.update(kw)

        setup = _CronAgentSetup(model="qwen3-max", runtime={"provider": "qwen-oauth"}, max_tokens=None)
        _construct_cron_agent(_FakeAgent, {"id": "j1"}, {}, setup, workdir=None,
                              session_id="cron_j1", session_db=None)
        assert captured["max_tokens"] is None


class TestCLIFlags:

    def _parser(self):
        import argparse

        from hermes_cli.subcommands.cron import build_cron_parser

        def _noop(args):  # pragma: no cover - only the parser shape is asserted
            return None

        parser = argparse.ArgumentParser(prog="hermes")
        build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=_noop)
        return parser

    @pytest.mark.parametrize("argv", [
        ["cron", "create", "every 1h", "say hi", "--max-tokens", "32768"],
        ["cron", "edit", "job1", "--max-tokens", "32768"],
    ])
    def test_flag_parses(self, argv):
        args = self._parser().parse_args(argv)
        assert getattr(args, "max_tokens", None) == "32768"

    def test_edit_accepts_empty_string_to_clear(self):
        args = self._parser().parse_args(["cron", "edit", "job1", "--max-tokens", ""])
        assert args.max_tokens == ""

    def test_absent_flag_is_none(self):
        args = self._parser().parse_args(["cron", "create", "every 1h", "say hi"])
        assert getattr(args, "max_tokens", None) is None
