"""Tests for the plugin redaction-pattern registry.

Covers ``agent.redact.register_redaction_patterns`` (validation,
dedupe, additive semantics, matcher rebuild), the
``PluginContext.register_redaction_patterns`` wiring, and the bundled
``nvapi-redaction`` reference plugin.

All tests call ``redact_sensitive_text(..., force=True)`` so results
don't depend on the HERMES_REDACT_SECRETS environment of the test run,
and reset the plugin registry around each test so module-global state
never leaks between tests.
"""

import importlib.util

import pytest

import agent.redact as redact_mod
from agent.redact import (
    _reset_plugin_redaction_patterns,
    redact_sensitive_text,
    register_redaction_patterns,
)


NVAPI_KEY = "nvapi-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-abcdEFGH"
NVAPI_PATTERN = r"nvapi-[A-Za-z0-9_-]{20,}"


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_plugin_redaction_patterns()
    yield
    _reset_plugin_redaction_patterns()


# ── Baseline ────────────────────────────────────────────────────────────


def test_unregistered_format_passes_through():
    # Documents the gap the registry closes: an nvapi- key is not a
    # built-in prefix, so without a plugin it survives redaction.
    out = redact_sensitive_text(f"connect failed: {NVAPI_KEY}", force=True)
    assert NVAPI_KEY in out


# ── Core registry semantics ─────────────────────────────────────────────


def test_registered_pattern_masks_token():
    assert register_redaction_patterns([NVAPI_PATTERN], source="test") == 1
    out = redact_sensitive_text(f"connect failed: {NVAPI_KEY}", force=True)
    assert NVAPI_KEY not in out
    # Head/tail mask preserved for debuggability (same rule as built-ins).
    assert "nvapi-" in out and "..." in out


def test_builtins_unaffected_by_registration():
    register_redaction_patterns([NVAPI_PATTERN], source="test")
    sk = "sk-proj-AbCdEf1234567890GhIjKl"
    out = redact_sensitive_text(f"key={sk}", force=True)
    assert sk not in out


def test_invalid_regex_rejected():
    assert register_redaction_patterns([r"nvapi-[unclosed"], source="test") == 0
    out = redact_sensitive_text(f"x {NVAPI_KEY}", force=True)
    assert NVAPI_KEY in out  # nothing registered


def test_pattern_without_literal_prefix_rejected():
    # No literal anchor -> would defeat the pre-screen gate and could
    # match everything. Must be rejected.
    assert register_redaction_patterns([r".*secret.*"], source="test") == 0
    assert register_redaction_patterns([r"[A-Za-z0-9]{30,}"], source="test") == 0
    # One literal char is still too short.
    assert register_redaction_patterns([r"x[A-Za-z0-9]{30,}"], source="test") == 0


def test_duplicate_and_builtin_patterns_deduped():
    assert register_redaction_patterns([NVAPI_PATTERN], source="test") == 1
    assert register_redaction_patterns([NVAPI_PATTERN], source="test") == 0
    # A pattern already shipped in core is skipped too.
    builtin = redact_mod._PREFIX_PATTERNS[0]
    assert register_redaction_patterns([builtin], source="test") == 0
    # Same pattern twice in one call counts once.
    _reset_plugin_redaction_patterns()
    assert register_redaction_patterns([NVAPI_PATTERN, NVAPI_PATTERN], source="test") == 1


def test_non_string_and_empty_entries_skipped():
    assert register_redaction_patterns([None, "", "   ", 42], source="test") == 0
    assert register_redaction_patterns(None, source="test") == 0


def test_file_read_sentinel_uses_plugin_prefix_label():
    register_redaction_patterns([NVAPI_PATTERN], source="test")
    out = redact_sensitive_text(
        f"api_base_key: {NVAPI_KEY}", force=True, file_read=True,
    )
    assert NVAPI_KEY not in out
    # Non-reusable sentinel carries the vendor label, no secret bytes.
    assert "«redacted:nvapi-…»" in out


def test_reset_restores_baseline():
    register_redaction_patterns([NVAPI_PATTERN], source="test")
    _reset_plugin_redaction_patterns()
    out = redact_sensitive_text(f"x {NVAPI_KEY}", force=True)
    assert NVAPI_KEY in out
    # Built-ins still intact after reset.
    sk = "sk-proj-AbCdEf1234567890GhIjKl"
    assert sk not in redact_sensitive_text(sk, force=True)


# ── PluginContext wiring ────────────────────────────────────────────────


def test_plugin_context_method_registers():
    import hermes_cli.plugins as plugins_mod

    manager = plugins_mod.PluginManager()
    manifest = plugins_mod.PluginManifest(name="test-redactor")
    ctx = plugins_mod.PluginContext(manifest, manager)

    assert ctx.register_redaction_patterns([NVAPI_PATTERN]) == 1
    out = redact_sensitive_text(f"boom {NVAPI_KEY}", force=True)
    assert NVAPI_KEY not in out


def test_plugin_context_method_never_raises(monkeypatch):
    import hermes_cli.plugins as plugins_mod

    def _boom(patterns, source=""):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr("agent.redact.register_redaction_patterns", _boom)

    manager = plugins_mod.PluginManager()
    manifest = plugins_mod.PluginManifest(name="test-redactor")
    ctx = plugins_mod.PluginContext(manifest, manager)
    assert ctx.register_redaction_patterns([NVAPI_PATTERN]) == 0


# ── Top-level alternation guard ─────────────────────────────────────────


def test_top_level_alternation_rejected():
    # 'ab|.*' compiles and has the accepted 'ab' literal prefix, but the
    # '.*' branch is unprefixed — accepting it would redact everything.
    assert register_redaction_patterns([r"ab|.*"], source="test") == 0
    assert register_redaction_patterns([r"ab|cd"], source="test") == 0
    clean = "nothing here resembles a credential"
    assert redact_sensitive_text(clean, force=True) == clean


def test_grouped_alternation_and_literal_pipe_accepted():
    # Alternation inside a group after the prefix keeps the guarantee.
    assert register_redaction_patterns(
        [r"zq(?:tok|key)-[A-Za-z0-9]{20,}"], source="test"
    ) == 1
    # Escaped pipes and character-class pipes are literals, not branches.
    assert register_redaction_patterns([r"xy\|[A-Za-z0-9]{20,}"], source="test") == 1
    assert register_redaction_patterns([r"wv[|][A-Za-z0-9]{20,}"], source="test") == 1


# ── Plugin register() end-to-end (synthetic, written at test time) ──────


_SYNTHETIC_PLUGIN = f'''
NVAPI_PATTERN = r"{NVAPI_PATTERN}"


def register(ctx):
    ctx.register_redaction_patterns([NVAPI_PATTERN])
'''


def _load_synthetic_plugin(tmp_path):
    plugin_init = tmp_path / "synthetic_redactor.py"
    plugin_init.write_text(_SYNTHETIC_PLUGIN, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("synthetic_redactor", plugin_init)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_register_end_to_end(tmp_path):
    import hermes_cli.plugins as plugins_mod

    demo = _load_synthetic_plugin(tmp_path)
    manager = plugins_mod.PluginManager()
    manifest = plugins_mod.PluginManifest(name="synthetic-redactor")
    demo.register(plugins_mod.PluginContext(manifest, manager))

    out = redact_sensitive_text(
        f"NIM request failed: 401 for key {NVAPI_KEY}", force=True,
    )
    assert NVAPI_KEY not in out
    assert "nvapi-" in out  # label survives for debuggability


def test_registered_pattern_no_prose_false_positive(tmp_path):
    demo = _load_synthetic_plugin(tmp_path)
    register_redaction_patterns([demo.NVAPI_PATTERN], source="test")
    prose = "the nvapi-endpoint docs describe rate limits"
    assert redact_sensitive_text(prose, force=True) == prose
