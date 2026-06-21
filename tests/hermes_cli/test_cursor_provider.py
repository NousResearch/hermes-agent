"""Tests for the Cursor provider registration across all four registries.

Covers:
- ``providers/`` plugin discovery (``ProviderProfile``)
- ``hermes_cli.providers`` HERMES_OVERLAYS + ALIASES + labels
- ``hermes_cli.auth`` PROVIDER_REGISTRY + alias resolver
- ``hermes_cli.models`` ProviderEntry + ``_PROVIDER_MODELS`` snapshot + aliases
- ``resolve_external_process_provider_credentials`` for cursor
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class CursorComposerContextWindowTests(unittest.TestCase):
    """Composer family must report cursor's actual 200K cap, not 256K.

    Cursor docs pin Composer 2 and Composer 2.5 (both fast and standard)
    to a 200K context window (the base model supports 256K but cursor
    truncates). Status-bar % and compression thresholds depend on this
    being accurate — otherwise users see 67K/256K (26%) when in reality
    they're at 67K/200K (33%) and 12K closer to the actual ceiling.

    Regression: prior to 186bf25c the composer ids fell through to the
    256K DEFAULT_FALLBACK_CONTEXT and inflated the usable window by 56K.
    """

    def test_composer_25_fast_is_200k(self) -> None:
        from agent.model_metadata import get_model_context_length

        ctx = get_model_context_length(
            "composer-2.5-fast", "cursor://agent", "", None, "cursor"
        )
        self.assertEqual(ctx, 200_000)

    def test_composer_25_is_200k(self) -> None:
        from agent.model_metadata import get_model_context_length

        ctx = get_model_context_length(
            "composer-2.5", "cursor://agent", "", None, "cursor"
        )
        self.assertEqual(ctx, 200_000)

    def test_composer_2_family_is_200k(self) -> None:
        from agent.model_metadata import get_model_context_length

        for model in ("composer-2", "composer-2-fast"):
            with self.subTest(model=model):
                ctx = get_model_context_length(
                    model, "cursor://agent", "", None, "cursor"
                )
                self.assertEqual(ctx, 200_000)


class CursorProviderRegistryTests(unittest.TestCase):
    # ---- providers/ plugin profile ----

    def test_plugin_profile_registered(self) -> None:
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile, "cursor plugin profile not discovered")
        self.assertEqual(profile.name, "cursor")
        self.assertEqual(profile.api_mode, "chat_completions")
        self.assertEqual(profile.auth_type, "external_process")
        self.assertEqual(profile.base_url, "cursor://agent")
        self.assertIn("CURSOR_API_KEY", profile.env_vars)
        self.assertEqual(profile.supports_health_check, False)
        # Fallback model list must include the new 2.5 family.
        self.assertIn("composer-2.5", profile.fallback_models)
        self.assertIn("composer-2.5-fast", profile.fallback_models)

    def test_plugin_profile_aliases_resolve(self) -> None:
        from providers import get_provider_profile

        for alias in ("cursor-agent", "cursor-cli", "cursor-sub", "cursor-subscription"):
            with self.subTest(alias=alias):
                self.assertEqual(get_provider_profile(alias).name, "cursor")

    # ---- hermes_cli.providers HERMES_OVERLAYS ----

    def test_hermes_overlay_present(self) -> None:
        from hermes_cli.providers import get_provider, normalize_provider, get_label

        pdef = get_provider("cursor")
        self.assertIsNotNone(pdef)
        self.assertEqual(pdef.id, "cursor")
        self.assertEqual(pdef.auth_type, "external_process")
        self.assertEqual(pdef.base_url, "cursor://agent")
        self.assertIn("CURSOR_API_KEY", pdef.api_key_env_vars)

        # Aliases
        for alias in ("cursor-agent", "cursor-cli", "cursor-sub", "anysphere"):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_provider(alias), "cursor")

        # Label override
        self.assertEqual(get_label("cursor"), "Cursor")

    # ---- hermes_cli.auth PROVIDER_REGISTRY ----

    def test_auth_registry_present(self) -> None:
        from hermes_cli.auth import PROVIDER_REGISTRY

        self.assertIn("cursor", PROVIDER_REGISTRY)
        entry = PROVIDER_REGISTRY["cursor"]
        self.assertEqual(entry.auth_type, "external_process")
        self.assertEqual(entry.inference_base_url, "cursor://agent")
        self.assertIn("CURSOR_API_KEY", entry.api_key_env_vars)

    # ---- hermes_cli.models picker + catalog ----

    def test_picker_entry_present(self) -> None:
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        self.assertIn("cursor", slugs)
        entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "cursor")
        self.assertEqual(entry.label, "Cursor")
        # Picker description: short, no em-dash, mentions "100+ models",
        # mirrors OpenRouter's ``OpenRouter (100+ models, pay-per-use)``.
        self.assertEqual(entry.tui_desc, "Cursor (100+ models, subscription)")
        self.assertNotIn("—", entry.tui_desc)

    def test_model_catalog_snapshot(self) -> None:
        from hermes_cli.models import _PROVIDER_MODELS

        self.assertIn("cursor", _PROVIDER_MODELS)
        models = _PROVIDER_MODELS["cursor"]
        # Must include composer-2.5 family and frontier models.
        self.assertIn("auto", models)
        self.assertIn("composer-2.5", models)
        self.assertIn("composer-2.5-fast", models)
        self.assertIn("gpt-5.5-medium", models)
        self.assertIn("claude-opus-4-7-high", models)
        self.assertIn("gemini-3.1-pro", models)
        # No stale junk
        self.assertNotIn("", models)
        self.assertEqual(len(models), len(set(models)))

    def test_model_alias_map(self) -> None:
        # Use private alias map directly — it's the same one the picker uses.
        from hermes_cli.models import _PROVIDER_ALIASES

        for alias in ("cursor-agent", "cursor-cli", "cursor-sub", "anysphere"):
            with self.subTest(alias=alias):
                self.assertEqual(_PROVIDER_ALIASES.get(alias), "cursor")

    def test_provider_model_ids_lists_cursor(self) -> None:
        from hermes_cli.models import provider_model_ids

        # Patch shutil.which to None so the live-fetch path returns nothing
        # and the function falls back to the static snapshot — keeps the test
        # independent of the user's actual cursor-agent install.
        with patch("shutil.which", return_value=None):
            models = provider_model_ids("cursor")
        self.assertIn("composer-2.5", models)
        self.assertIn("auto", models)
        self.assertEqual(models[:3], ["auto", "composer-2.5", "composer-2.5-fast"])

    def test_plugin_fetch_models_honors_custom_cursor_command(self) -> None:
        """Live model discovery must work with wrappers, not only PATH installs."""
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile)
        assert profile is not None

        def which_side_effect(command: str, *args, **kwargs):
            if command == "/opt/hermes/bin/cursor-agent-wrapper":
                return command
            return None

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "/opt/hermes/bin/cursor-agent-wrapper"}, clear=False), \
             patch("shutil.which", side_effect=which_side_effect), \
             patch("subprocess.check_output", return_value="composer-2.5 - Composer 2.5\n") as check_output:
            models = profile.fetch_models()

        self.assertEqual(models, ["composer-2.5"])
        check_output.assert_called_once_with(
            ["/opt/hermes/bin/cursor-agent-wrapper", "--list-models"],
            text=True,
            timeout=8.0,
        )

    def test_plugin_fetch_models_honors_cursor_agent_path_fallback(self) -> None:
        """Legacy CURSOR_AGENT_PATH must work when HERMES_CURSOR_COMMAND is unset."""
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile)
        assert profile is not None

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "", "CURSOR_AGENT_PATH": "/opt/cursor/bin/cursor-agent"}, clear=False), \
             patch("shutil.which", return_value="/opt/cursor/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="composer-2.5-fast - Composer 2.5 Fast\n") as check_output:
            models = profile.fetch_models()

        self.assertEqual(models, ["composer-2.5-fast"])
        check_output.assert_called_once_with(
            ["/opt/cursor/bin/cursor-agent", "--list-models"],
            text=True,
            timeout=8.0,
        )

    def test_plugin_fetch_models_cursor_command_wins_over_cursor_agent_path(self) -> None:
        """HERMES_CURSOR_COMMAND is the canonical override when both env vars exist."""
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile)
        assert profile is not None

        def which_side_effect(command: str, *args, **kwargs):
            return {
                "/opt/hermes/bin/cursor-agent-wrapper": "/resolved/hermes-wrapper",
                "/legacy/cursor-agent": "/resolved/legacy-agent",
            }.get(command)

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "/opt/hermes/bin/cursor-agent-wrapper", "CURSOR_AGENT_PATH": "/legacy/cursor-agent"}, clear=False), \
             patch("shutil.which", side_effect=which_side_effect), \
             patch("subprocess.check_output", return_value="auto - Auto\n") as check_output:
            models = profile.fetch_models()

        self.assertEqual(models, ["auto"])
        check_output.assert_called_once_with(
            ["/resolved/hermes-wrapper", "--list-models"],
            text=True,
            timeout=8.0,
        )

    def test_plugin_fetch_models_ignores_blank_env_overrides(self) -> None:
        """Whitespace-only env vars should fall back to cursor-agent on PATH."""
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile)
        assert profile is not None

        def which_side_effect(command: str, *args, **kwargs):
            if command == "cursor-agent":
                return "/usr/local/bin/cursor-agent"
            return None

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "   ", "CURSOR_AGENT_PATH": "\t"}, clear=False), \
             patch("shutil.which", side_effect=which_side_effect), \
             patch("subprocess.check_output", return_value="composer-2 - Composer 2\n") as check_output:
            models = profile.fetch_models()

        self.assertEqual(models, ["composer-2"])
        check_output.assert_called_once_with(
            ["/usr/local/bin/cursor-agent", "--list-models"],
            text=True,
            timeout=8.0,
        )

    def test_plugin_fetch_models_uses_raw_command_when_which_cannot_resolve_it(self) -> None:
        """Absolute or relative wrapper paths can still be invoked directly."""
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile)
        assert profile is not None

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "./bin/cursor-wrapper"}, clear=False), \
             patch("shutil.which", return_value=None), \
             patch("subprocess.check_output", return_value="gpt-5.5-medium - GPT 5.5 Medium\n") as check_output:
            models = profile.fetch_models()

        self.assertEqual(models, ["gpt-5.5-medium"])
        check_output.assert_called_once_with(
            ["./bin/cursor-wrapper", "--list-models"],
            text=True,
            timeout=8.0,
        )

    def test_plugin_fetch_models_threads_cursor_args_into_probe(self) -> None:
        """Profile-specific CLI args should apply to live model discovery."""
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile)
        assert profile is not None

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "./bin/cursor-wrapper", "HERMES_CURSOR_ARGS": "--profile mazi"}, clear=False), \
             patch("shutil.which", return_value=None), \
             patch("subprocess.check_output", return_value="composer-2.5 - Composer 2.5\n") as check_output:
            models = profile.fetch_models()

        self.assertEqual(models, ["composer-2.5"])
        check_output.assert_called_once_with(
            ["./bin/cursor-wrapper", "--profile", "mazi", "--list-models"],
            text=True,
            timeout=8.0,
        )

    def test_plugin_fetch_models_returns_none_when_cursor_cli_fails(self) -> None:
        """Discovery failure must fail closed so callers use the static fallback list."""
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile)
        assert profile is not None

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "/missing/cursor-agent"}, clear=False), \
             patch("shutil.which", return_value=None), \
             patch("subprocess.check_output", side_effect=FileNotFoundError):
            models = profile.fetch_models()

        self.assertIsNone(models)

    def test_plugin_fetch_models_prioritizes_important_cursor_models(self) -> None:
        """Large live Cursor catalogs should open with Composer/agentic picks first."""
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile)
        assert profile is not None

        live_catalog = "\n".join([
            "z-random - Random Model",
            "gpt-5.5-medium - GPT 5.5 Medium",
            "composer-2.5-fast - Composer 2.5 Fast",
            "auto - Auto",
            "composer-2.5 - Composer 2.5",
            "claude-opus-4-7-high - Claude Opus 4.7 High",
            "z-random - Random Model Duplicate",
        ])
        with patch("shutil.which", return_value="/usr/local/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value=live_catalog):
            models = profile.fetch_models()

        self.assertEqual(
            models,
            [
                "auto",
                "composer-2.5",
                "composer-2.5-fast",
                "gpt-5.5-medium",
                "claude-opus-4-7-high",
                "z-random",
            ],
        )


class CursorCommandResolverTests(unittest.TestCase):
    """Shared Cursor command resolution contract."""

    def test_resolve_cursor_command_env_precedence(self) -> None:
        from providers.cursor_utils import resolve_cursor_command

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "/new/wrapper", "CURSOR_AGENT_PATH": "/legacy/wrapper"}, clear=False):
            self.assertEqual(resolve_cursor_command(), "/new/wrapper")

    def test_resolve_cursor_command_ignores_blank_env(self) -> None:
        from providers.cursor_utils import resolve_cursor_command

        with patch.dict(os.environ, {"HERMES_CURSOR_COMMAND": "  ", "CURSOR_AGENT_PATH": "\t"}, clear=False):
            self.assertEqual(resolve_cursor_command(), "cursor-agent")

    def test_resolve_cursor_command_path_uses_path_lookup(self) -> None:
        from providers.cursor_utils import resolve_cursor_command_path

        with patch("shutil.which", return_value="/usr/bin/cursor-agent") as which:
            self.assertEqual(resolve_cursor_command_path("cursor-agent"), "/usr/bin/cursor-agent")
        which.assert_called_once_with("cursor-agent")

    def test_resolve_cursor_extra_args_splits_shell_style_env(self) -> None:
        from providers.cursor_utils import resolve_cursor_extra_args

        with patch.dict(os.environ, {"HERMES_CURSOR_ARGS": "--profile mazi --flag 'two words'"}, clear=False):
            self.assertEqual(resolve_cursor_extra_args(), ["--profile", "mazi", "--flag", "two words"])

    def test_resolve_cursor_command_path_preserves_explicit_unresolved_path(self) -> None:
        from providers.cursor_utils import resolve_cursor_command_path

        with patch("shutil.which", return_value=None):
            self.assertEqual(resolve_cursor_command_path("./bin/cursor-wrapper"), "./bin/cursor-wrapper")
            self.assertEqual(resolve_cursor_command_path(r"C:\\Cursor\\cursor-agent.exe"), r"C:\\Cursor\\cursor-agent.exe")

    def test_resolve_cursor_command_path_rejects_unresolved_bare_name(self) -> None:
        from providers.cursor_utils import resolve_cursor_command_path

        with patch("shutil.which", return_value=None):
            self.assertIsNone(resolve_cursor_command_path("cursor-agent"))
            self.assertIsNone(resolve_cursor_command_path(""))

    def test_prioritize_cursor_models_keeps_remainder_after_preferred(self) -> None:
        from providers.cursor_utils import prioritize_cursor_models

        models = prioritize_cursor_models([
            "z-last",
            "composer-2.5-fast",
            "auto",
            "composer-2.5",
            "z-last",
            "other-model",
        ])
        self.assertEqual(models, ["auto", "composer-2.5", "composer-2.5-fast", "z-last", "other-model"])


class CursorStatusTests(unittest.TestCase):
    """``get_external_process_provider_status`` cursor branch."""

    def setUp(self) -> None:
        keys = [
            "HERMES_CURSOR_COMMAND",
            "HERMES_CURSOR_ARGS",
            "CURSOR_AGENT_PATH",
            "CURSOR_API_KEY",
            "HERMES_CURSOR_BASE_URL",
        ]
        self._saved = {k: os.environ.pop(k, None) for k in keys}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_status_reports_configured_when_cli_present(self) -> None:
        from hermes_cli.auth import get_external_process_provider_status

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="✓ Logged in as alice@example.com\n"):
            status = get_external_process_provider_status("cursor")
        self.assertTrue(status["configured"])
        self.assertEqual(status["provider"], "cursor")
        self.assertEqual(status["resolved_command"], "/fake/bin/cursor-agent")
        self.assertEqual(status["base_url"], "cursor://agent")
        self.assertTrue(status["logged_in"])
        self.assertEqual(status["email"], "alice@example.com")

    def test_status_accepts_raw_cursor_command_when_not_on_path(self) -> None:
        """Status should match fetch/runtime behavior for wrapper paths."""
        from hermes_cli.auth import get_external_process_provider_status

        os.environ["HERMES_CURSOR_COMMAND"] = "./bin/cursor-wrapper"
        os.environ["HERMES_CURSOR_ARGS"] = "--profile mazi"
        with patch("shutil.which", return_value=None), \
             patch("subprocess.check_output", return_value="✓ Logged in as alice@example.com\n") as check_output:
            status = get_external_process_provider_status("cursor")

        self.assertTrue(status["configured"])
        self.assertEqual(status["command"], "./bin/cursor-wrapper")
        self.assertEqual(status["resolved_command"], "./bin/cursor-wrapper")
        self.assertEqual(status["args"], ["--profile", "mazi"])
        self.assertTrue(status["logged_in"])
        check_output.assert_called_once_with(
            ["./bin/cursor-wrapper", "--profile", "mazi", "status"],
            text=True,
            timeout=4,
        )

    def test_status_unconfigured_when_default_cursor_agent_missing(self) -> None:
        """Only explicit wrapper paths get raw fallback; bare defaults still require PATH."""
        from hermes_cli.auth import get_external_process_provider_status

        with patch("shutil.which", return_value=None):
            status = get_external_process_provider_status("cursor")

        self.assertFalse(status["configured"])
        self.assertEqual(status["command"], "cursor-agent")
        self.assertIsNone(status["resolved_command"])

    def test_status_reports_logged_out_when_status_lacks_marker(self) -> None:
        from hermes_cli.auth import get_external_process_provider_status

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="Some other text\n"):
            status = get_external_process_provider_status("cursor")
        self.assertTrue(status["configured"])
        self.assertFalse(status["logged_in"])
        self.assertEqual(status["email"], "")

    def test_status_treats_api_key_env_as_authenticated(self) -> None:
        from hermes_cli.auth import get_external_process_provider_status

        os.environ["CURSOR_API_KEY"] = "crsr_token"
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="not logged in\n"):
            status = get_external_process_provider_status("cursor")
        self.assertTrue(status["logged_in"])

    def test_status_unconfigured_when_cli_missing(self) -> None:
        from hermes_cli.auth import get_external_process_provider_status

        with patch("shutil.which", return_value=None):
            status = get_external_process_provider_status("cursor")
        self.assertFalse(status["configured"])
        self.assertIsNone(status["resolved_command"])

    def test_get_auth_status_dispatches_cursor(self) -> None:
        from hermes_cli.auth import get_auth_status

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="✓ Logged in as bob@example.com\n"):
            status = get_auth_status("cursor")
        self.assertTrue(status["logged_in"])
        self.assertEqual(status["email"], "bob@example.com")


class CursorRuntimeProviderTests(unittest.TestCase):
    """``resolve_runtime_provider`` must build a cursor dict, not fall through.

    Without this branch the chat REPL falls through to the openrouter default
    and dies with "Provider resolver returned an empty API key. Set
    OPENROUTER_API_KEY..." — the bug we hit in interactive testing.
    """

    def setUp(self) -> None:
        keys = [
            "HERMES_CURSOR_COMMAND",
            "CURSOR_AGENT_PATH",
            "CURSOR_API_KEY",
            "HERMES_CURSOR_BASE_URL",
        ]
        self._saved = {k: os.environ.pop(k, None) for k in keys}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_runtime_provider_returns_cursor_dict(self) -> None:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            rt = resolve_runtime_provider(requested="cursor")

        self.assertEqual(rt["provider"], "cursor")
        self.assertEqual(rt["api_mode"], "chat_completions")
        self.assertEqual(rt["base_url"], "cursor://agent")
        self.assertEqual(rt["command"], "/fake/bin/cursor-agent")
        # api_key must be either a real key or the sentinel — never empty.
        self.assertTrue(rt["api_key"])
        self.assertEqual(rt["source"], "process")

    def test_runtime_provider_threads_real_api_key(self) -> None:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        os.environ["CURSOR_API_KEY"] = "crsr_real_runtime_test"
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            rt = resolve_runtime_provider(requested="cursor")
        self.assertEqual(rt["api_key"], "crsr_real_runtime_test")

    def test_runtime_provider_exposes_replayable_cursor_trace(self) -> None:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        os.environ["CURSOR_API_KEY"] = "crsr_real_runtime_test"
        os.environ["HERMES_CURSOR_COMMAND"] = "./bin/cursor-wrapper"
        os.environ["HERMES_CURSOR_ARGS"] = "--profile mazi"
        with patch("shutil.which", return_value=None):
            rt = resolve_runtime_provider(requested="cursor")

        trace = rt["trace"]
        self.assertEqual(trace["provider"], "cursor")
        self.assertEqual(trace["route"], "cursor-agent")
        self.assertEqual(trace["auth_source"], "CURSOR_API_KEY")
        self.assertEqual(trace["config_source"], "HERMES_CURSOR_COMMAND")
        self.assertEqual(trace["command"], "./bin/cursor-wrapper")
        self.assertEqual(trace["args"], ["--profile", "mazi"])
        self.assertEqual(trace["base_url"], "cursor://agent")
        self.assertNotIn("crsr_real_runtime_test", str(trace))


class CursorCredsResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        # Stash relevant env so we don't leak state across tests.
        keys = [
            "HERMES_CURSOR_COMMAND",
            "HERMES_CURSOR_ARGS",
            "CURSOR_AGENT_PATH",
            "CURSOR_API_KEY",
            "HERMES_CURSOR_BASE_URL",
        ]
        self._saved = {k: os.environ.pop(k, None) for k in keys}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_creds_resolver_finds_cli_and_returns_dict(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        # Pretend cursor-agent lives at a known fake path.
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            creds = resolve_external_process_provider_credentials("cursor")

        self.assertEqual(creds["provider"], "cursor")
        self.assertEqual(creds["command"], "/fake/bin/cursor-agent")
        self.assertEqual(creds["base_url"], "cursor://agent")
        self.assertEqual(creds["source"], "process")
        # No api_key in env → sentinel value
        self.assertEqual(creds["api_key"], "cursor-agent-login")
        self.assertEqual(creds["args"], [])

    def test_creds_resolver_threads_api_key_when_set(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        os.environ["CURSOR_API_KEY"] = "crsr_real_value"
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            creds = resolve_external_process_provider_credentials("cursor")
        self.assertEqual(creds["api_key"], "crsr_real_value")

    def test_creds_resolver_raises_when_cli_missing(self) -> None:
        from hermes_cli.auth import AuthError, resolve_external_process_provider_credentials

        with patch("shutil.which", return_value=None):
            with self.assertRaises(AuthError) as ctx:
                resolve_external_process_provider_credentials("cursor")
        self.assertEqual(ctx.exception.code, "missing_cursor_cli")
        self.assertIn("Cursor CLI", str(ctx.exception))

    def test_creds_resolver_accepts_raw_cursor_command_when_not_on_path(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        os.environ["HERMES_CURSOR_COMMAND"] = "./bin/cursor-wrapper"
        with patch("shutil.which", return_value=None):
            creds = resolve_external_process_provider_credentials("cursor")

        self.assertEqual(creds["command"], "./bin/cursor-wrapper")

    def test_creds_resolver_honors_extra_args_env(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        os.environ["HERMES_CURSOR_ARGS"] = "--header X-Foo:1 --header X-Bar:2"
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            creds = resolve_external_process_provider_credentials("cursor")
        self.assertEqual(creds["args"], ["--header", "X-Foo:1", "--header", "X-Bar:2"])

    def test_creds_resolver_does_not_disturb_copilot_acp(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        with patch("shutil.which", return_value="/fake/bin/copilot"):
            creds = resolve_external_process_provider_credentials("copilot-acp")
        self.assertEqual(creds["provider"], "copilot-acp")
        self.assertEqual(creds["api_key"], "copilot-acp")
        self.assertEqual(creds["args"], ["--acp", "--stdio"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
