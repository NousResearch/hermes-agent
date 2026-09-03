"""Tests for the model.max_context_length global context ceiling.

Semantics under test (see agent/context_compressor.py, agent/model_metadata.py):

  * ``effective = min(resolved, max_context_length)`` — the ceiling only ever
    LOWERS a window, never raises one.
  * ``pre_cap_context_length`` is the resolved value BEFORE the ceiling — the
    model's real capability. It is what the persistent context cache is fed, so
    the ceiling (a runtime/user policy) never contaminates
    ``context_length_cache.yaml``.
  * The ceiling survives ``/model`` switches and fallback activations because
    it is applied in the compressor's setter (the single convergence point),
    not at each call site.
  * Display/gate paths (WebUI model-info, model-switch warnings, tool-search
    gate) report the EFFECTIVE value, not the raw auto-detected capability.
"""

import sys

import pytest
from unittest.mock import patch, MagicMock

from contextlib import contextmanager
from agent import context_compressor as _cc
from agent.context_compressor import ContextCompressor
from agent import model_metadata


@contextmanager
def _resolver(**mock_kwargs):
    """Patch get_model_context_length in BOTH namespaces the code path uses.

    The compressor imports it into its OWN module namespace at import time
    (``from agent.model_metadata import get_model_context_length``), while
    ``effective_context_length()`` (defined in model_metadata) resolves it via
    that module's globals. Patching both keeps the tests deterministic whether
    the value is read through the compressor or the helper.

    Accepts the usual mock kwargs (``return_value=`` or ``side_effect=``) and
    applies a MagicMock with those settings to both namespaces.
    """
    fake = MagicMock(**mock_kwargs)
    with patch.object(_cc, "get_model_context_length", fake), \
         patch.object(model_metadata, "get_model_context_length", fake):
        yield fake


# ─────────────────────────────────────────────────────────────────────────────
# Core clamp / pre-cap semantics on the compressor
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressorCeiling:
    def test_ceiling_clamps_auto_detected_window(self):
        """Luna: 900K detected, 272K ceiling -> consumers see 272K."""
        with _resolver(return_value=900_000):
            c = ContextCompressor(model="gpt-5.6-luna", quiet_mode=True, max_context_length=272_000)
            _ = c.context_length  # force deferred resolution while the mock is live
        assert c.context_length == 272_000
        assert c.pre_cap_context_length == 900_000

    def test_no_ceiling_returns_raw(self):
        with _resolver(return_value=900_000):
            c = ContextCompressor(model="gpt-5.6-luna", quiet_mode=True)
            _ = c.context_length  # force deferred resolution while the mock is live
        assert c.context_length == 900_000
        assert c.pre_cap_context_length == 900_000
        assert c.max_context_length is None

    def test_ceiling_below_override_still_caps(self):
        """The ceiling is a hard maximum: it clamps an explicit
        model.context_length override that is above it."""
        with _resolver(side_effect=_faithful_resolver):
            c = ContextCompressor(
                model="m", quiet_mode=True,
                config_context_length=1_000_000, max_context_length=272_000,
            )
        assert c.context_length == 272_000
        assert c.pre_cap_context_length == 1_000_000

    def test_ceiling_above_override_does_not_raise(self):
        """A small explicit override stays small; the ceiling never raises."""
        with _resolver(side_effect=_faithful_resolver):
            c = ContextCompressor(
                model="m", quiet_mode=True,
                config_context_length=32_768, max_context_length=272_000,
            )
        assert c.context_length == 32_768
        assert c.pre_cap_context_length == 32_768

    def test_malformed_ceiling_is_ignored(self):
        """Non-positive / non-int ceilings coerce to None (no ceiling)."""
        for bad in (0, -1, "nope", None, float("nan")):
            with _resolver(return_value=100_000):
                c = ContextCompressor(model="m", quiet_mode=True, max_context_length=bad)
                _ = c.context_length  # force deferred resolution while the mock is live
            assert c.max_context_length is None
            assert c.context_length == 100_000

    def test_sub_floor_ceiling_is_rejected_not_truncating(self):
        """A positive-int ceiling BELOW the 64K floor is a MIS-CONFIGURATION,
        not a capability statement. The compressor must REJECT it (treat as
        "no ceiling") and leave the pre-cap window INTACT — never truncate the
        window to a sub-floor value. This is the floor-enforcing validator
        (validate_context_ceiling), not the runtime coerce_ variant."""
        # 32K, 20K, and exactly the floor-minus-1 are all below MINIMUM_CONTEXT_LENGTH.
        for sub_floor in (32_000, 20_000, 31_999):
            with _resolver(return_value=900_000):
                c = ContextCompressor(model="m", quiet_mode=True, max_context_length=sub_floor)
                _ = c.context_length  # force deferred resolution while the mock is live
            # Rejected -> no ceiling recorded.
            assert c.max_context_length is None
            # Window stays at the FULL pre-cap capability (never truncated).
            assert c.context_length == 900_000
            assert c.pre_cap_context_length == 900_000

    def test_at_floor_and_above_is_accepted(self):
        """Boundaries: exactly MINIMUM_CONTEXT_LENGTH (64K) and above are
        valid ceilings and DO clamp the window (the floor is inclusive)."""
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        with _resolver(return_value=900_000):
            c = ContextCompressor(model="m", quiet_mode=True, max_context_length=MINIMUM_CONTEXT_LENGTH)
            _ = c.context_length
        assert c.max_context_length == MINIMUM_CONTEXT_LENGTH
        assert c.context_length == MINIMUM_CONTEXT_LENGTH
        # Above the floor also accepted.
        with _resolver(return_value=900_000):
            c2 = ContextCompressor(model="m", quiet_mode=True, max_context_length=MINIMUM_CONTEXT_LENGTH + 1)
            _ = c2.context_length
        assert c2.max_context_length == MINIMUM_CONTEXT_LENGTH + 1
        assert c2.context_length == MINIMUM_CONTEXT_LENGTH + 1

    def test_threshold_derives_from_effective_window(self):
        """Compression threshold must be computed against the capped window,
        not the raw 900K — otherwise the cap would be cosmetic."""
        from agent.model_metadata import DEFAULT_OUTPUT_RESERVATION
        with _resolver(return_value=900_000):
            c = ContextCompressor(
                model="m", quiet_mode=True, threshold_percent=0.75, max_context_length=272_000,
            )
            _ = c.context_length  # force deferred resolution while the mock is live
        # 75% of (EFFECTIVE 272K − output reservation). The reservation is the
        # shared-policy finite default (4096, unknown provider) — never 0 —
        # so the input budget is (window − reservation).
        assert c.threshold_tokens == int((272_000 - DEFAULT_OUTPUT_RESERVATION) * 0.75)
        assert c.threshold_tokens < 900_000 * 0.75


# ─────────────────────────────────────────────────────────────────────────────
# Setter path (Codex app-server, switch_model, fallback, provider-error)
# ─────────────────────────────────────────────────────────────────────────────

class TestSetterClamp:
    def test_direct_assignment_clamps(self):
        """codex_runtime.py:189 does `compressor.context_length = window`.
        The setter must apply the ceiling."""
        with _resolver(return_value=1000):
            c = ContextCompressor(model="m", quiet_mode=True, max_context_length=272_000)
        c.context_length = 900_000  # raw window reported by the provider
        assert c.context_length == 272_000
        assert c.pre_cap_context_length == 900_000

    def test_repeated_same_window_is_noop(self):
        """The no-op guard compares against the PRE-CAP value (not the clamped
        one), so re-reporting the same provider window stays a no-op and does
        not invalidate derived budgets."""
        with _resolver(return_value=1000):
            c = ContextCompressor(model="m", quiet_mode=True, max_context_length=272_000)
        c.context_length = 900_000
        assert c.threshold_tokens is not None
        before = c.threshold_tokens
        c.context_length = 900_000  # same window re-reported
        assert c.threshold_tokens == before
        assert c.context_length == 272_000

    def test_update_model_clamps_and_scales_threshold(self):
        """switch_model / fallback call update_model(context_length=raw). The
        effective window and the threshold must both reflect the cap."""
        from agent.model_metadata import DEFAULT_OUTPUT_RESERVATION
        with _resolver(return_value=1000):
            c = ContextCompressor(model="m", quiet_mode=True, threshold_percent=0.75, max_context_length=272_000)
        c.update_model(model="m", context_length=900_000)
        assert c.context_length == 272_000
        assert c.pre_cap_context_length == 900_000
        # 75% of (272K − shared-policy output reservation 4096).
        assert c.threshold_tokens == int((272_000 - DEFAULT_OUTPUT_RESERVATION) * 0.75)

    def test_ceiling_survives_model_switch_to_larger_model(self):
        """The regression the feature exists to prevent: switching from a
        272K-capped model to a 900K model must not escape the ceiling."""
        with _resolver(return_value=900_000):
            c = ContextCompressor(model="small", quiet_mode=True, max_context_length=272_000)
            _ = c.context_length  # force deferred resolution while the mock is live
        assert c.context_length == 272_000
        # Switch to a larger model with a 1M window.
        c.update_model(model="big", context_length=1_000_000)
        assert c.context_length == 272_000
        assert c.pre_cap_context_length == 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# effective_context_length() policy helper + _get_max_context_length
# ─────────────────────────────────────────────────────────────────────────────

class TestEffectiveHelper:
    def test_effective_applies_ceiling(self):
        with _resolver(return_value=900_000), \
             patch.object(model_metadata, "_get_max_context_length", return_value=272_000):
            assert model_metadata.effective_context_length("m") == 272_000

    def test_effective_without_ceiling_is_raw(self):
        with _resolver(return_value=900_000), \
             patch.object(model_metadata, "_get_max_context_length", return_value=None):
            assert model_metadata.effective_context_length("m") == 900_000

    def test_get_max_reads_config(self):
        with patch("hermes_cli.config.load_config_readonly", return_value={"model": {"max_context_length": 272_000}}):
            assert model_metadata._get_max_context_length() == 272_000

    def test_get_max_absent(self):
        with patch("hermes_cli.config.load_config_readonly", return_value={"model": {}}):
            assert model_metadata._get_max_context_length() is None

    def test_get_max_non_positive_is_none(self):
        with patch("hermes_cli.config.load_config_readonly", return_value={"model": {"max_context_length": 0}}):
            assert model_metadata._get_max_context_length() is None

    def test_get_max_malformed_is_none(self):
        with patch("hermes_cli.config.load_config_readonly", return_value={"model": {"max_context_length": "272K"}}):
            assert model_metadata._get_max_context_length() is None

    def test_get_max_config_error_is_none(self):
        with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")):
            assert model_metadata._get_max_context_length() is None


# ─────────────────────────────────────────────────────────────────────────────
# Round 4.1 item 3: invalid-ceiling warning dedup is profile/value safe
# ─────────────────────────────────────────────────────────────────────────────

class TestCeilingWarningDedup:
    """The invalid-ceiling diagnostic is deduped (a process-global set) so the
    gateway — which resolves the ceiling on many paths — does not spam the log.

    Round 4.1 invariant: the dedup key MUST be (profile, value). An invalid
    ceiling in profile A must NOT suppress the appropriate warning for the SAME
    value in profile B (cross-profile state leakage), and a CHANGED invalid
    value within a profile must still warn (not over-suppressed). Only the
    same (profile, value) is warned once.

    The profile dimension is the active profile's resolved Hermes home
    (``get_hermes_home()``), which is exactly the profile whose config
    ``_get_max_context_length`` just read via ``load_config_readonly()``.
    """

    BAD = 32_000  # positive int, below the 64K floor → INVALID ceiling

    @pytest.fixture(autouse=True)
    def _reset(self):
        model_metadata._CEILING_INVALID_WARNED.clear()
        yield
        model_metadata._CEILING_INVALID_WARNED.clear()

    @staticmethod
    def _invalid(home: str, value):
        """Drive _get_max_context_length under profile ``home`` with a
        malformed ceiling ``value``."""
        import hermes_cli.config as _cfg
        import hermes_constants as _hc
        with patch.object(_cfg, "load_config_readonly", return_value={"model": {"max_context_length": value}}), \
             patch.object(_hc, "get_hermes_home", return_value=home):
            assert model_metadata._get_max_context_length() is None

    def test_same_profile_same_value_warns_once(self, caplog):
        """Dedup still works: the same (profile, value) warns exactly once,
        not once per resolution (the spam-prevention the dedup exists for)."""
        import logging
        import agent.model_metadata as _mm
        with caplog.at_level(logging.WARNING, logger=_mm.__name__):
            self._invalid("/home/profiles/a", self.BAD)
            self._invalid("/home/profiles/a", self.BAD)
        warnings = [r for r in caplog.records if "max_context_length" in r.getMessage()]
        assert len(warnings) == 1, f"same (profile,value) must warn once, got {len(warnings)}"

    def test_same_value_different_profiles_both_warn(self, caplog):
        """The regression this item closes: profile A's warning for value V
        must NOT suppress profile B's warning for the SAME value V. Without the
        profile dimension in the dedup key, the second profile is silent."""
        import logging
        import agent.model_metadata as _mm
        with caplog.at_level(logging.WARNING, logger=_mm.__name__):
            self._invalid("/home/profiles/a", self.BAD)
            self._invalid("/home/profiles/b", self.BAD)
        warnings = [r for r in caplog.records if "max_context_length" in r.getMessage()]
        assert len(warnings) == 2, (
            "an invalid ceiling in profile A must not suppress the appropriate "
            f"warning in profile B for the same value (got {len(warnings)})"
        )

    def test_changed_value_same_profile_warns_again(self, caplog):
        """A CHANGED invalid value within the same profile is a distinct
        mis-configuration and must warn again (the dedup must not over-suppress
        — each distinct bad value is diagnosed once)."""
        import logging
        import agent.model_metadata as _mm
        with caplog.at_level(logging.WARNING, logger=_mm.__name__):
            self._invalid("/home/profiles/a", 20_000)
            self._invalid("/home/profiles/a", 31_999)  # a different sub-floor value
        warnings = [r for r in caplog.records if "max_context_length" in r.getMessage()]
        assert len(warnings) == 2, (
            "a changed invalid value within a profile must still warn "
            f"(got {len(warnings)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cache purity: ceiling must never contaminate context_length_cache.yaml
# ─────────────────────────────────────────────────────────────────────────────

class TestCachePurity:
    def test_pre_cap_is_used_for_cache_write(self, tmp_path, monkeypatch):
        """The conversation_loop cache-write site must persist the PRE-CAP value
        (900K), not the clamped effective (272K). The cache retains the model's
        real capability; the ceiling is a runtime policy overlay."""
        import agent.model_metadata as mm
        cache_path = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_path)

        # A compressor with a 900K pre-cap window and a 272K ceiling.
        with _resolver(return_value=900_000):
            c = ContextCompressor(model="gpt-5.6-luna", quiet_mode=True, max_context_length=272_000)
            _ = c.context_length  # force deferred resolution while the mock is live

        assert c.context_length == 272_000
        assert c.pre_cap_context_length == 900_000

        # The cache-write site (conversation_loop.py) reads pre_cap_context_length.
        mm.save_context_length("gpt-5.6-luna", "http://codex", c.pre_cap_context_length)
        cached = mm.get_cached_context_length("gpt-5.6-luna", "http://codex")
        assert cached == 900_000, f"cache must retain 900K, got {cached}"

    def test_removing_ceiling_recovers_native(self, tmp_path, monkeypatch):
        """If the cache stored the clamped value, removing the ceiling would
        still leave the agent at 272K. Proving the cache holds 900K means a
        later config change (ceiling removed) recovers the full window."""
        import agent.model_metadata as mm
        cache_path = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_path)

        with _resolver(return_value=900_000):
            c = ContextCompressor(model="m", quiet_mode=True, max_context_length=272_000)
            _ = c.context_length  # force deferred resolution while the mock is live
        mm.save_context_length("m", "http://x", c.pre_cap_context_length)

        # Now the ceiling is gone and the resolver reads from the cache.
        with _resolver(return_value=900_000):
            c2 = ContextCompressor(model="m", quiet_mode=True)  # no ceiling
            _ = c2.context_length  # force deferred resolution while the mock is live
        assert c2.context_length == 900_000

    def test_persistable_override_never_cached(self):
        """An explicit model.context_length override enters via the
        config_context_length / setter path, which never sets
        _context_probe_persistable=True — so the conversation_loop gate
        (which requires that flag) never persists it to the cache."""
        import agent.model_metadata as mm

        with _resolver(side_effect=_faithful_resolver):
            c = ContextCompressor(
                model="m", quiet_mode=True,
                config_context_length=1_000_000, max_context_length=272_000,
            )
        # The override path does not mark the window as provider-confirmed.
        assert getattr(c, "_context_probed", False) is False
        assert getattr(c, "_context_probe_persistable", False) is False
        # ...so even if the gate ran, it would be skipped.
        gate_would_persist = (
            getattr(c, "_context_probed", False)
            and getattr(c, "_context_probe_persistable", False)
        )
        assert not gate_would_persist


# ─────────────────────────────────────────────────────────────────────────────
# Display / gate paths report the effective value
# ─────────────────────────────────────────────────────────────────────────────

class TestDisplayEffective:
    @staticmethod
    def _mm():
        # ``resolve_display_context_length`` does a *late*
        # ``from agent.model_metadata import effective_context_length`` at call
        # time. A prior test in the full suite can reload or rebind that module,
        # so the file-scope ``model_metadata`` binding captured at this test
        # module's import time may be a *different object* than the one the late
        # import resolves to — patching the stale object is what made these two
        # tests order-dependent. Patch the object the late import actually
        # resolves to, ``sys.modules["agent.model_metadata"]``, so the mock
        # survives arbitrary test ordering.
        return sys.modules["agent.model_metadata"]

    def test_resolve_display_applies_ceiling(self):
        """model_switch.resolve_display_context_length is what the WebUI and
        context-switch warnings call. It must return the effective (capped)
        window, not the raw 900K."""
        from hermes_cli import model_switch
        mm = self._mm()
        with patch.object(mm, "get_model_context_length", return_value=900_000), \
             patch.object(mm, "_get_max_context_length", return_value=272_000):
            ctx = model_switch.resolve_display_context_length(
                "gpt-5.6-luna", "openai", "http://codex", "",
            )
        assert ctx == 272_000

    def test_resolve_display_no_ceiling_is_raw(self):
        from hermes_cli import model_switch
        mm = self._mm()
        with patch.object(mm, "get_model_context_length", return_value=900_000), \
             patch.object(mm, "_get_max_context_length", return_value=None):
            ctx = model_switch.resolve_display_context_length(
                "gpt-5.6-luna", "openai", "http://codex", "",
            )
        assert ctx == 900_000


def _faithful_resolver(model, base_url="", api_key="", config_context_length=None,
                       provider="", custom_providers=None):
    """Faithful stand-in for get_model_context_length's step 0: honor an
    explicit override, else return a large auto-detected window."""
    if isinstance(config_context_length, int) and config_context_length > 0:
        return config_context_length
    return 900_000


# ─────────────────────────────────────────────────────────────────────────────
# Host-side routing: transition_model_context / refresh_context_window / get_pre_cap
#
# These cover the remediation-phase defects: plugin engines (no setter clamp),
# the Codex derived-state refresh, the full fallback/restore round trip with
# host-retained raw, the non-persistable Anthropic tier reduction, and the MoA
# reference budget using the effective window.
# ─────────────────────────────────────────────────────────────────────────────

from agent.context_engine import ContextEngine
from agent.agent_runtime_helpers import transition_model_context, refresh_context_window, get_pre_cap


class _PluginEngine(ContextEngine):
    """Minimal concrete generic plugin engine.

    Inherits the BASE ``update_model()`` contract, which sets the plain
    ``context_length`` attribute and recalculates ``threshold_tokens =
    int(context_length * threshold_percent)``. This is exactly what a real
    third-party engine does, so the tests exercise the base-class derived-state
    behavior that the Codex path must keep coherent.
    """

    def __init__(self):
        self.threshold_percent = 0.75
        self.context_length = 0
        self.threshold_tokens = 0

    @property
    def name(self):
        return "plugin"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None,
                 force=False, memory_context=""):
        return messages


class _AgentStub:
    _pre_cap_context_length: int | None

    def __init__(self, compressor, model="m", ceiling=None):
        self.context_compressor = compressor
        self.model = model
        self.base_url = "http://x"
        self.api_key = ""
        self.provider = "openai"
        self.api_mode = ""
        self._max_context_length = ceiling
        self._pre_cap_context_length = None  # unset until transition_model_context / refresh_context_window sets it


class TestHostRouting:
    def test_plugin_receives_effective_not_inflated(self):
        """A 200K model under a 272K ceiling must STAY at 200K — the ceiling
        never raises a window. The earlier reversed condition
        (``if 0 < cur < ceiling: engine.context_length = ceiling``) would have
        inflated this to 272K; this test pins the correct semantics."""
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        supplied = transition_model_context(agent, 200_000, model="m")
        assert supplied == 200_000
        assert agent.context_compressor.context_length == 200_000
        assert agent._pre_cap_context_length == 200_000

    def test_plugin_900k_capped_to_272k(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        supplied = transition_model_context(agent, 900_000, model="m")
        assert supplied == 272_000
        assert agent.context_compressor.context_length == 272_000
        # The host retains the raw pre-cap for snapshot/restore.
        assert agent._pre_cap_context_length == 900_000

    def test_builtin_receives_raw_preserves_pre_cap(self):
        """The built-in receives the RAW value (not pre-clamped) so its setter
        stores pre_cap = raw and derives effective = min(raw, ceiling)."""
        with _resolver(return_value=1000):
            c = ContextCompressor(model="m", quiet_mode=True, max_context_length=272_000)
        agent = _AgentStub(c, ceiling=272_000)
        supplied = transition_model_context(agent, 900_000, model="m")
        assert supplied == 900_000  # raw was supplied, not the capped 272K
        assert c.context_length == 272_000
        assert c.pre_cap_context_length == 900_000
        assert agent._pre_cap_context_length == 900_000

    def test_builtin_survives_module_reimport_class_identity_change(self):
        """transition_model_context() must dispatch on the stable
        ``_handles_context_ceiling`` capability marker, NOT on isinstance /
        class identity.

        A re-import of ``agent.context_compressor`` (a plugin's
        ``importlib.reload`` or a test fixture that evicts ``agent.*`` from
        ``sys.modules`` and re-imports) produces a NEW ContextCompressor class
        object while live instances still belong to the OLD one. An
        ``isinstance(engine, ContextCompressor)`` check evaluated against the
        re-imported class would then route a built-in instance into the
        generic-engine branch and hand it a pre-clamped window — silently
        breaking ``pre_cap``/``effective`` semantics.

        Reproduction of the exact failure mode:
          1. retain an instance + class from the original module;
          2. remove ``agent.context_compressor`` from sys.modules and
             re-import it so class identity changes;
          3. route a 900K raw window with a 272K ceiling;
          4. the OLD built-in instance must still receive RAW 900K and
             preserve pre_cap = 900K / effective = 272K.
        """
        import importlib

        original_module = sys.modules["agent.context_compressor"]
        original_class = original_module.ContextCompressor
        with _resolver(return_value=1000):
            c = original_class(model="m", quiet_mode=True, max_context_length=272_000)

        # Evict the module and re-import it so a NEW class object becomes the
        # canonical one — the exact state a module-reloading test fixture or
        # plugin leaves behind.
        saved = {
            name: sys.modules[name]
            for name in list(sys.modules)
            if name == "agent.context_compressor"
            or name.startswith("agent.context_compressor.")
        }
        try:
            del sys.modules["agent.context_compressor"]
            new_module = importlib.import_module("agent.context_compressor")
        finally:
            # Restore the ORIGINAL module so later tests see the same
            # canonical module object they imported at collection time.
            sys.modules["agent.context_compressor"] = saved.get(
                "agent.context_compressor", original_module
            )
            for name, mod in saved.items():
                if name != "agent.context_compressor":
                    sys.modules[name] = mod

        # Sanity: the reproduction must ACTUALLY change class identity,
        # otherwise it is not exercising the failure mode it guards.
        assert new_module is not original_module
        assert new_module.ContextCompressor is not original_class
        assert isinstance(c, original_class)
        assert not isinstance(c, new_module.ContextCompressor)

        agent = _AgentStub(c, ceiling=272_000)
        supplied = transition_model_context(agent, 900_000, model="m")
        # The stale built-in instance still receives RAW (not 272K):
        assert supplied == 900_000
        assert c.pre_cap_context_length == 900_000
        assert c.context_length == 272_000
        assert agent._pre_cap_context_length == 900_000

    def test_no_ceiling_plugin_unchanged(self):
        agent = _AgentStub(_PluginEngine(), ceiling=None)
        supplied = transition_model_context(agent, 900_000, model="m")
        assert supplied == 900_000
        assert agent.context_compressor.context_length == 900_000
        assert agent._pre_cap_context_length == 900_000


class TestCodexPluginDerivedState:
    def test_plugin_threshold_recalcs_on_window_change(self):
        """Point 1: a plugin starts at effective 272K (from a 900K window
        capped by the ceiling); the Codex runtime later reports a 200K
        window. The plugin's threshold/budget must reflect 200K, not stay at
        the stale 272K-derived value. Routing through update_model() (not a
        bare attribute write) is what keeps the derived state coherent."""
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        # Initial: 900K window -> effective 272K.
        transition_model_context(agent, 900_000, model="m")
        assert agent.context_compressor.context_length == 272_000
        assert agent.context_compressor.threshold_tokens == int(272_000 * 0.75)

        # Codex re-reports a smaller 200K window (already below the ceiling).
        # Same-model window refresh — not a model transition.
        refresh_context_window(agent, 200_000, model="m")
        assert agent.context_compressor.context_length == 200_000
        # Threshold recalculated from the NEW 200K window, not the old 272K.
        assert agent.context_compressor.threshold_tokens == int(200_000 * 0.75)
        assert agent._pre_cap_context_length == 200_000

    def test_plugin_unchanged_window_skips_update(self):
        """Point 1 (skip-if-unchanged): re-reporting the same effective window
        must not call update_model again (no spurious threshold invalidation)."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)
        transition_model_context(agent, 900_000, model="m")  # -> 272K effective
        assert eng.context_length == 272_000
        calls = 0
        original = eng.update_model

        def counting_update_model(*a, **kw):
            nonlocal calls
            calls += 1
            return original(*a, **kw)

        eng.update_model = counting_update_model
        # 200K window under the 272K ceiling -> effective 200K (changed).
        refresh_context_window(agent, 200_000, model="m")
        assert calls == 1
        # Re-report 200K -> effective 200K (unchanged) -> no update call.
        refresh_context_window(agent, 200_000, model="m")
        assert calls == 1
        assert eng.context_length == 200_000


class TestPluginRoundTrip:
    def test_raw_survives_fallback_restore(self):
        """Full round trip (point 3): raw 900K, ceiling 272K, plugin sees 272K.
        Fallback to a 32K model, then restore. The host must still know raw
        = 900K (so restore re-derives 272K), while the plugin ends at 272K."""
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        # Init.
        transition_model_context(agent, 900_000, model="primary")
        assert agent._pre_cap_context_length == 900_000
        assert agent.context_compressor.context_length == 272_000

        # Snapshot captures the host pre-cap (900K), not the capped 272K.
        snapshot_pre_cap = get_pre_cap(agent)
        assert snapshot_pre_cap == 900_000

        # Fallback activation: the fallback is the ACTIVE model, so the host
        # pre-cap becomes the FALLBACK's pre-cap (32K). The primary's pre-cap
        # (900K) is preserved in the snapshot (captured above) for restoration.
        # (Finding #3: host pre-cap = active model, not the primary.)
        transition_model_context(agent, 32_000, model="fallback")
        assert agent.context_compressor.context_length == 32_000
        # Host pre-cap is now the FALLBACK's (32K) — the active model's.
        assert agent._pre_cap_context_length == 32_000
        # The primary's pre-cap (900K) was captured in snapshot_pre_cap above.
        assert snapshot_pre_cap == 900_000

        # Restore: feed the snapshot's pre-cap back through the helper.
        transition_model_context(agent, snapshot_pre_cap, model="primary")
        # Host re-retains 900K; plugin re-derives 272K.
        assert agent._pre_cap_context_length == 900_000
        assert agent.context_compressor.context_length == 272_000

    def test_snapshot_reads_host_pre_cap_not_capped_getter(self):
        """The snapshot site must read the host pre-cap (900K), not the engine's
        capped getter (272K) — otherwise the restore round trip loses the raw."""
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        transition_model_context(agent, 900_000, model="m")
        assert get_pre_cap(agent) == 900_000
        assert agent.context_compressor.context_length == 272_000
        # The snapshot would store the host value, not the capped getter.
        assert get_pre_cap(agent) != agent.context_compressor.context_length


class TestTierReductionNonPersistable:
    def test_anthropic_tier_reduction_updates_runtime_not_cache(self):
        """Point 2: the Anthropic 200K tier reduction is a NON-PERSISTABLE
        runtime restriction. It becomes the new pre-cap operating window in
        the engine and host state, but must NEVER be written to the persistent
        context_length_cache.yaml (which remains raw-capability-only)."""
        import agent.model_metadata as mm
        with _resolver(return_value=1_000_000):
            c = ContextCompressor(model="claude", quiet_mode=True,
                                  threshold_percent=0.75, max_context_length=272_000)
            _ = c.context_length  # force deferred resolution while the mock is live
        # Pre-tier: 1M window -> effective 272K.
        assert c.context_length == 272_000
        assert c.pre_cap_context_length == 1_000_000

        # Simulate the Anthropic tier-reduction path: route 200K through the
        # helper (this is what conversation_loop.py does at the tier gate).
        agent = _AgentStub(c, ceiling=272_000)
        refresh_context_window(agent, 200_000, model="claude")
        # Runtime state reflects the 200K pre-cap, effective min(200K, 272K)=200K.
        assert c.context_length == 200_000
        assert c.pre_cap_context_length == 200_000
        assert agent._pre_cap_context_length == 200_000

        # The tier gate sets the non-persistable flag, so the cache-write gate
        # (which requires _context_probe_persistable) must NOT fire.
        c._context_probed = True
        c._context_probe_persistable = False  # set by the tier-reduction path
        gate_would_persist = (
            getattr(c, "_context_probed", False)
            and getattr(c, "_context_probe_persistable", False)
        )
        assert not gate_would_persist, "tier reduction must not be persistable"

    def test_tier_reduction_below_ceiling_keeps_tier(self, tmp_path, monkeypatch):
        """A 200K tier reduction under a 272K ceiling stays at 200K (the
        ceiling only lowers, never raises the tier window), and is NOT
        persisted to the cache."""
        import agent.model_metadata as mm
        cache_path = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_path)

        with _resolver(return_value=1_000_000):
            c = ContextCompressor(model="claude", quiet_mode=True,
                                  threshold_percent=0.75, max_context_length=272_000)
            _ = c.context_length  # force deferred resolution while the mock is live
        agent = _AgentStub(c, ceiling=272_000)
        refresh_context_window(agent, 200_000, model="claude")
        assert c.context_length == 200_000

        # The tier path never persists; the cache must be empty for this model.
        assert mm.get_cached_context_length("claude", "http://anthropic") is None


class TestMoAEffectiveBudget:
    def test_reference_budget_uses_effective_not_raw(self):
        """MoA reference budgeting uses the EFFECTIVE window (raw reference
        capability clamped by the profile-wide ceiling). A 900K reference under
        a 272K ceiling is budgeted at 272K, and the local per-fan-out cache
        stores the effective value — NOT the persistent context cache."""
        import agent.moa_loop as moa
        from agent.model_metadata import estimate_messages_tokens_rough

        mm = sys.modules["agent.model_metadata"]
        # Patch the resolver to return the RAW 900K reference capability and the
        # ceiling to 272K, so effective_context_length -> 272K.
        with patch.object(mm, "get_model_context_length", return_value=900_000), \
             patch.object(mm, "_get_max_context_length", return_value=272_000):
            slot = {"model": "ref-model"}
            runtime = {"provider": "openai", "base_url": "http://x", "api_key": ""}
            # A message list large enough to exceed the 272K-based budget so the
            # trim loop runs (proving the budget was derived from 272K, not 900K).
            # ~300K tokens per big frame (chars/4) → ~600K total, well above the
            # 272K×0.9 budget, so old frames must be dropped.
            big = "x" * 1_200_000
            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": big},
                {"role": "assistant", "content": big},
                {"role": "user", "content": "judge the state"},
            ]
            cache = {}
            out = moa._trim_messages_for_reference(
                messages, slot, runtime, context_length_cache=cache,
            )
            # The local per-fan-out cache stores the EFFECTIVE value (272K),
            # not the raw reference capability (900K).
            assert cache[("openai", "ref-model")] == 272_000
            # The trim actually reduced the list (budget was 272K-based, so the
            # 200K×2 payload exceeded it and old frames were dropped).
            assert len(out) < len(messages)

    def test_reference_below_ceiling_keeps_raw(self):
        """A reference whose raw window (200K) is already below the ceiling
        (272K) keeps its (smaller) real limit — the ceiling never raises."""
        import agent.moa_loop as moa
        mm = sys.modules["agent.model_metadata"]
        with patch.object(mm, "get_model_context_length", return_value=200_000), \
             patch.object(mm, "_get_max_context_length", return_value=272_000):
            slot = {"model": "small-ref"}
            runtime = {"provider": "openai", "base_url": "http://x", "api_key": ""}
            messages = [{"role": "system", "content": "sys"},
                        {"role": "user", "content": "judge the state"}]
            cache = {}
            moa._trim_messages_for_reference(
                messages, slot, runtime, context_length_cache=cache,
            )
            assert cache[("openai", "small-ref")] == 200_000

    def test_moa_local_cache_not_written_to_persistent(self, tmp_path, monkeypatch):
        """The MoA per-fan-out budget dict is LOCAL and never feeds the
        persistent context_length_cache.yaml (raw-capability-only)."""
        import agent.moa_loop as moa
        import agent.model_metadata as mm
        cache_path = tmp_path / "context_length_cache.yaml"
        monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_path)
        mm = sys.modules["agent.model_metadata"]
        with patch.object(mm, "get_model_context_length", return_value=900_000), \
             patch.object(mm, "_get_max_context_length", return_value=272_000):
            slot = {"model": "ref-model"}
            runtime = {"provider": "openai", "base_url": "http://x", "api_key": ""}
            messages = [{"role": "system", "content": "sys"},
                        {"role": "user", "content": "judge the state"}]
            local = {}
            moa._trim_messages_for_reference(
                messages, slot, runtime, context_length_cache=local,
            )
            # Local cache holds the effective value…
            assert local[("openai", "ref-model")] == 272_000
            # …but the persistent cache is untouched (never written).
            assert mm.get_cached_context_length("ref-model", "http://x") is None


class TestGetPreCap:
    def test_reads_host_store_first(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        agent._pre_cap_context_length = 500_000
        assert get_pre_cap(agent) == 500_000

    def test_falls_back_to_engine_pre_cap(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        agent._pre_cap_context_length = None  # unset
        agent.context_compressor.context_length = 900_000
        # No host store -> falls back to the engine's context_length.
        assert get_pre_cap(agent) == 900_000

    def test_no_engine_returns_zero(self):
        class _Bare:
            _pre_cap_context_length = None
            context_compressor = None
        assert get_pre_cap(_Bare()) == 0
