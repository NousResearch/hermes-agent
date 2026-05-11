"""Tests for the bundled observability/langfuse plugin."""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "langfuse"


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------

class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "langfuse"
        assert data["version"]
        # All six hooks the plugin implements.
        assert set(data["hooks"]) == {
            "pre_api_request", "post_api_request",
            "pre_llm_call", "post_llm_call",
            "pre_tool_call", "post_tool_call",
        }
        # Required env vars are the user-facing HERMES_ prefixed keys.
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in data["requires_env"]
        assert "HERMES_LANGFUSE_SECRET_KEY" in data["requires_env"]


# ---------------------------------------------------------------------------
# Plugin discovery: langfuse is opt-in (not loaded unless explicitly enabled).
# This guards against someone accidentally re-introducing a per-hook
# load_config() gate or making the plugin auto-load.
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        """Scanner should find the plugin but NOT load it by default."""
        from hermes_cli import plugins as plugins_mod

        # Isolated HERMES_HOME so we don't read the developer's config.yaml.
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        # observability/langfuse appears in the plugin registry …
        loaded = manager._plugins.get("observability/langfuse")
        assert loaded is not None, "plugin not discovered"
        # … but is not loaded (opt-in default → no config.yaml means nothing enabled)
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


# ---------------------------------------------------------------------------
# Runtime gate: _get_langfuse() returns None and caches _INIT_FAILED when
# credentials are missing. Guards against regressing toward the rejected
# per-hook load_config() design.
# ---------------------------------------------------------------------------

class TestRuntimeGate:
    def _fresh_plugin(self):
        """Import the plugin module fresh (clears any cached client)."""
        mod_name = "plugins.observability.langfuse"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    def test_get_langfuse_returns_none_without_credentials(self, monkeypatch):
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        langfuse_plugin = self._fresh_plugin()
        assert langfuse_plugin._get_langfuse() is None

    def test_get_langfuse_caches_failure_no_config_load(self, monkeypatch):
        """A miss must be cached — no per-hook config.yaml reads, no env re-reads."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        langfuse_plugin = self._fresh_plugin()

        # Prime the cache with one call.
        assert langfuse_plugin._get_langfuse() is None

        # Now block os.environ.get — a correctly-cached plugin must not
        # touch env again.
        import os
        called = {"n": 0}
        real_get = os.environ.get

        def tracking_get(key, default=None):
            if key.startswith(("HERMES_LANGFUSE_", "LANGFUSE_")):
                called["n"] += 1
            return real_get(key, default)

        monkeypatch.setattr(os.environ, "get", tracking_get)

        for _ in range(20):
            assert langfuse_plugin._get_langfuse() is None

        assert called["n"] == 0, (
            f"_get_langfuse() re-read env {called['n']} times after cache miss — "
            "it should short-circuit via _INIT_FAILED"
        )

    def test_get_langfuse_does_not_import_hermes_config(self, monkeypatch):
        """The plugin must not re-read config.yaml per hook."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        # Drop any cached import of hermes_cli.config.
        sys.modules.pop("hermes_cli.config", None)

        langfuse_plugin = self._fresh_plugin()
        for _ in range(20):
            langfuse_plugin._get_langfuse()

        assert "hermes_cli.config" not in sys.modules, (
            "langfuse plugin imported hermes_cli.config — regression toward "
            "the rejected per-hook load_config() design"
        )


# ---------------------------------------------------------------------------
# Hooks are inert when the client is unavailable.
# ---------------------------------------------------------------------------

class TestHooksInert:
    def test_hooks_noop_without_client(self, monkeypatch):
        """All 6 hooks must return without raising when _get_langfuse() is None."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        sys.modules.pop("plugins.observability.langfuse", None)
        import importlib
        mod = importlib.import_module("plugins.observability.langfuse")

        # Each hook should just return; no exceptions.
        mod.on_pre_llm_call(task_id="t", session_id="s", messages=[{"role": "user", "content": "hi"}])
        mod.on_pre_llm_request(task_id="t", session_id="s", api_call_count=1, messages=[])
        mod.on_post_llm_call(task_id="t", session_id="s", api_call_count=1)
        mod.on_pre_tool_call(tool_name="read_file", args={}, task_id="t", session_id="s")
        mod.on_post_tool_call(tool_name="read_file", args={}, result="ok", task_id="t", session_id="s")


# ---------------------------------------------------------------------------
# Placeholder-credential guard (#23823).
#
# Regression coverage for the silent-failure bug: when an operator leaves
# HERMES_LANGFUSE_PUBLIC_KEY / SECRET_KEY at a template value like
# "placeholder", "test-key", or "your-langfuse-key", the SDK accepts the
# credentials at construction time (it does no server-side validation
# eagerly) but drops every trace at flush time, with no signal in the
# Hermes logs.  The fix in `_get_langfuse()` validates the documented
# `pk-lf-` / `sk-lf-` prefix Langfuse always issues, surfaces a one-shot
# warning naming the offending env var(s), and short-circuits via the
# same `_INIT_FAILED` path used for missing credentials so subsequent
# hook invocations don't re-log.
# ---------------------------------------------------------------------------


class _FakeLangfuse:
    """Stand-in for the real :class:`langfuse.Langfuse` so tests don't
    need the optional ``langfuse`` SDK installed.  The plugin's runtime
    gate refuses to proceed past ``if Langfuse is None`` when the SDK
    is missing, which would short-circuit before the placeholder check
    can fire.  Patching ``plugin.Langfuse`` with this class lets the
    placeholder validator exercise its full code path."""

    instances: list["_FakeLangfuse"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeLangfuse.instances.append(self)


class TestPlaceholderKeyDetection:
    LOGGER_NAME = "plugins.observability.langfuse"

    def _fresh_plugin(self, monkeypatch=None):
        mod_name = "plugins.observability.langfuse"
        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)
        if monkeypatch is not None:
            # Pretend the SDK is installed so `_get_langfuse()` actually
            # reaches the placeholder check.  Real SDK calls are never
            # made because the placeholder/missing-credentials paths
            # return before constructing a client.
            _FakeLangfuse.instances.clear()
            monkeypatch.setattr(mod, "Langfuse", _FakeLangfuse, raising=False)
        return mod

    @staticmethod
    def _clear_env(monkeypatch):
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

    # -- helper unit tests (no SDK stub needed: these don't go through
    #    _get_langfuse, they exercise the pure-Python helpers directly) ------

    def test_redact_key_preview_empty(self, monkeypatch):
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._redact_key_preview("") == "<empty>"

    def test_redact_key_preview_short_value_echoed(self, monkeypatch):
        """Short placeholder strings are echoed in full so the operator
        can see exactly which template they forgot to replace."""
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._redact_key_preview("placeholder") == "'placeholder'"
        assert plugin._redact_key_preview("test-key") == "'test-key'"

    def test_redact_key_preview_long_value_truncated(self, monkeypatch):
        """If an operator pasted a real secret into the wrong env var the
        preview must NOT echo it in full — only the leading 6 chars."""
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        result = plugin._redact_key_preview("sk-lf-abcdefghijklmnop")
        assert "abcdefghij" not in result
        assert result.startswith("'sk-lf-")
        assert result.endswith("...'")

    def test_validate_langfuse_key_accepts_documented_prefix(self, monkeypatch):
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._validate_langfuse_key(
            "HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-real-public-xyz"
        ) is None
        assert plugin._validate_langfuse_key(
            "HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz"
        ) is None

    def test_validate_langfuse_key_rejects_wrong_prefix(self, monkeypatch):
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        msg = plugin._validate_langfuse_key(
            "HERMES_LANGFUSE_PUBLIC_KEY", "placeholder"
        )
        assert msg is not None
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in msg
        assert "pk-lf-" in msg

    def test_validate_langfuse_key_unknown_name_passes(self, monkeypatch):
        """Defensive: an env var with no registered prefix is trusted."""
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._validate_langfuse_key("HERMES_LANGFUSE_BASE_URL", "anything") is None

    # -- end-to-end _get_langfuse() behaviour --------------------------------
    # These tests pass `monkeypatch` to _fresh_plugin() so the helper can
    # stub out `Langfuse` (the optional SDK).  Without that, every call
    # short-circuits at `if Langfuse is None` before reaching the
    # placeholder validator — masking the very behaviour we're testing.

    def test_placeholder_public_key_warns_and_skips(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        text = caplog.text
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in text
        assert "'placeholder'" in text
        assert "pk-lf-" in text
        # The valid secret value must NOT appear (the var NAME does, in
        # the "or unset ..." hint, but the value preview shouldn't).
        assert "'sk-lf-" not in text
        # Never constructed the SDK client — short-circuited before that.
        assert _FakeLangfuse.instances == []

    def test_placeholder_secret_key_warns_and_skips(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-real-public-xyz")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "test-key")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        text = caplog.text
        assert "HERMES_LANGFUSE_SECRET_KEY" in text
        assert "'test-key'" in text
        assert "sk-lf-" in text
        # The valid public value must NOT appear.
        assert "'pk-lf-" not in text
        assert _FakeLangfuse.instances == []

    def test_both_placeholders_one_warning_with_both_keys(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "placeholder")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert len(warnings) == 1, (
            f"Expected a single combined warning; got {len(warnings)}:\n"
            + "\n".join(r.getMessage() for r in warnings)
        )
        text = warnings[0].getMessage()
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in text
        assert "HERMES_LANGFUSE_SECRET_KEY" in text

    def test_repeated_calls_do_not_re_warn(self, monkeypatch, caplog):
        """The cached ``_INIT_FAILED`` sentinel must short-circuit
        subsequent calls so each hook invocation isn't a fresh log
        line — otherwise a busy gateway will spam the operator's
        terminal."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "placeholder")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            for _ in range(15):
                assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert len(warnings) == 1, (
            f"Warning fired {len(warnings)} times across 15 calls; "
            "expected 1 (cached via _INIT_FAILED)"
        )

    @pytest.mark.parametrize("placeholder", [
        "placeholder",
        "test-key",
        "your-langfuse-key",
        "change-me",
        "xxx",
        "dummy-key-here",
        "<your-key>",
        "REPLACE_ME",
    ])
    def test_common_placeholders_detected(self, monkeypatch, caplog, placeholder):
        """A grab-bag of values that real-world ``.env.example`` templates
        use as stand-ins.  Any of them in either key must trip the guard."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", placeholder)
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in caplog.text

    def test_legacy_LANGFUSE_PUBLIC_KEY_also_validated(self, monkeypatch, caplog):
        """The plugin reads both the canonical HERMES_-prefixed env var and
        the legacy bare ``LANGFUSE_PUBLIC_KEY``.  The validator must run on
        whichever value ``_get_langfuse()`` actually consumed."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        # Warning names the canonical user-facing env var (the bare
        # LANGFUSE_PUBLIC_KEY is a backwards-compat alias for the
        # HERMES_-prefixed one — operators set the HERMES_-prefixed one).
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in caplog.text
        assert "'placeholder'" in caplog.text

    def test_missing_credentials_still_skip_silently(self, monkeypatch, caplog):
        """Missing-creds is the documented opt-out path (operator hasn't
        configured the plugin yet) — it must remain SILENT.  Regression
        guard against the placeholder validator accidentally running on
        empty values and re-introducing log noise for unconfigured
        installs."""
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert warnings == []

    def test_sdk_not_installed_still_skips_silently(self, monkeypatch, caplog):
        """If the langfuse SDK isn't installed at all, the placeholder
        check should never run — there's nothing the operator can do
        about a credential mismatch when the package is missing, and
        re-warning here would dilute the actually-actionable SDK-missing
        signal upstream.  The ``Langfuse is None`` guard at the top of
        ``_get_langfuse`` already handles this; this test pins that
        behaviour."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "placeholder")
        # NO monkeypatch on Langfuse here — falls back to whatever the
        # plugin imported at module load (None if SDK absent).
        plugin = self._fresh_plugin()
        monkeypatch.setattr(plugin, "Langfuse", None, raising=False)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert warnings == []

    def test_valid_prefixes_do_not_trigger_placeholder_warning(self, monkeypatch, caplog):
        """Real Langfuse keys (``pk-lf-…`` / ``sk-lf-…``) must pass the
        guard and proceed to SDK init.  We stub the SDK constructor with
        a recording fake so the assertion can confirm BOTH that the
        placeholder warning didn't fire AND that the client was actually
        constructed — the latter is the success signal the bug report
        wanted."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-real-public-xyz")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            client = plugin._get_langfuse()
        assert isinstance(client, _FakeLangfuse)
        assert client.kwargs["public_key"] == "pk-lf-real-public-xyz"
        assert client.kwargs["secret_key"] == "sk-lf-real-secret-xyz"
        assert "placeholders" not in caplog.text.lower(), (
            f"Valid Langfuse keys tripped the placeholder guard: {caplog.text!r}"
        )
