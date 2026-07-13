"""Closed-set provider discovery contracts for isolated runtimes."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_fresh_python(source: str, *, env: dict[str, str] | None = None) -> None:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        env=process_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"fresh interpreter failed with exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_model_metadata_import_does_not_discover_provider_plugins():
    _run_fresh_python(
        """
        import sys
        import providers

        assert providers._discovered is False
        import agent.model_metadata  # noqa: F401
        assert providers._discovered is False
        assert not {
            name
            for name in sys.modules
            if name.startswith("plugins.model_providers.")
        }
        """
    )


def test_url_inference_only_discovers_profiles_for_unknown_hosts():
    _run_fresh_python(
        """
        import providers
        from agent.model_metadata import _infer_provider_from_url
        from providers.base import ProviderProfile

        calls = []

        def fake_list_providers():
            calls.append(True)
            return [
                ProviderProfile(
                    name="extension-provider",
                    base_url="https://models.extension.example/v1",
                )
            ]

        providers.list_providers = fake_list_providers

        assert _infer_provider_from_url("https://chatgpt.com/backend-api/codex") == "openai"
        assert calls == []
        assert _infer_provider_from_url("https://models.extension.example/v1") == "extension-provider"
        assert calls == [True]
        """
    )


def test_isolated_discovery_loads_only_bundled_openai_codex(tmp_path):
    hermes_home = tmp_path / "hermes-home"
    user_plugin = hermes_home / "plugins" / "model-providers" / "openai-codex"
    user_plugin.mkdir(parents=True)
    marker = tmp_path / "user-provider-imported"
    (user_plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(\n"
        "    name='openai-codex',\n"
        "    base_url='https://malicious.example/v1',\n"
        "))\n"
    )

    _run_fresh_python(
        f"""
        import sys
        from pathlib import Path

        import providers
        from providers import (
            ProviderDiscoveryIsolationError,
            configure_isolated_provider_discovery,
            get_provider_profile,
            list_providers,
            register_provider,
        )
        from providers.base import ProviderProfile

        configure_isolated_provider_discovery(frozenset({{"openai-codex"}}))
        assert providers._isolated_discovery_validated is True
        profiles = list_providers()
        assert [profile.name for profile in profiles] == ["openai-codex"]

        profile = get_provider_profile("openai-codex")
        assert profile is not None
        assert profile.base_url == "https://chatgpt.com/backend-api/codex"
        assert not Path({str(marker)!r}).exists()
        assert not {{
            name for name in sys.modules
            if name.startswith("_hermes_user_provider_")
        }}

        loaded_provider_modules = {{
            name for name in sys.modules
            if name.startswith("plugins.model_providers.")
        }}
        assert loaded_provider_modules == {{"plugins.model_providers.openai_codex"}}
        loaded_origin = Path(
            sys.modules["plugins.model_providers.openai_codex"].__file__
        ).resolve()
        expected_origin = (
            Path({str(REPO_ROOT)!r})
            / "plugins"
            / "model-providers"
            / "openai-codex"
            / "__init__.py"
        ).resolve()
        assert loaded_origin == expected_origin

        try:
            register_provider(
                ProviderProfile(
                    name="openai-codex",
                    base_url="https://late-override.example/v1",
                )
            )
        except ProviderDiscoveryIsolationError:
            pass
        else:
            raise AssertionError("late provider override was accepted")
        assert get_provider_profile("openai-codex").base_url == profile.base_url

        # The exact same immutable pin is idempotent; broadening it is not.
        configure_isolated_provider_discovery(frozenset({{"openai-codex"}}))
        try:
            configure_isolated_provider_discovery(
                frozenset({{"openai-codex", "gmi"}})
            )
        except ProviderDiscoveryIsolationError:
            pass
        else:
            raise AssertionError("isolated provider pin was broadened")
        """,
        env={"HERMES_HOME": str(hermes_home)},
    )


def test_isolated_discovery_failure_is_sticky_and_closed():
    _run_fresh_python(
        """
        import providers
        from providers import (
            ProviderDiscoveryIsolationError,
            configure_isolated_provider_discovery,
            list_providers,
        )
        from agent.model_metadata import _infer_provider_from_url

        try:
            configure_isolated_provider_discovery(
                frozenset({"provider-that-is-not-packaged"})
            )
        except ProviderDiscoveryIsolationError:
            pass
        else:
            raise AssertionError("missing provider passed synchronous preflight")

        # Both registry reads and repetition of the same failed pin stay
        # closed after the synchronous preflight failure.
        for action in (
            list_providers,
            lambda: configure_isolated_provider_discovery(
                frozenset({"provider-that-is-not-packaged"})
            ),
        ):
            try:
                action()
            except ProviderDiscoveryIsolationError:
                pass
            else:
                raise AssertionError("missing isolated provider failed open")
        assert providers._REGISTRY == {}
        assert providers._ALIASES == {}

        try:
            _infer_provider_from_url("https://unknown-provider.example/v1")
        except ProviderDiscoveryIsolationError:
            pass
        else:
            raise AssertionError("URL fallback swallowed the isolation failure")
        """
    )


def test_isolated_pin_must_precede_registry_use():
    _run_fresh_python(
        """
        from providers import (
            ProviderDiscoveryIsolationError,
            ProviderProfile,
            configure_isolated_provider_discovery,
            register_provider,
        )

        register_provider(ProviderProfile(name="already-registered"))
        try:
            configure_isolated_provider_discovery(frozenset({"openai-codex"}))
        except ProviderDiscoveryIsolationError:
            pass
        else:
            raise AssertionError("late isolated pin was accepted")
        """
    )
