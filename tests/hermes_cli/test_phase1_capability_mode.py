"""Jarvis Phase 1 inherited-environment capability boundary tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated(
    code: str,
    *,
    mode: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_PHASE1_CAPABILITY_MODE"] = mode
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_phase1_uses_only_inherited_capabilities_and_opens_no_persistent_sources(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    managed = tmp_path / "managed"
    for directory in (home, project, managed):
        directory.mkdir()

    (home / ".env").write_text(
        "OPENROUTER_API_KEY=jarvis-file-provider-sentinel\n"
        "JARVIS_FILE_ONLY_SENTINEL=jarvis-file-only-sentinel\n",
        encoding="utf-8",
    )
    (project / ".env").write_text(
        "JARVIS_PROJECT_SENTINEL=jarvis-project-sentinel\n",
        encoding="utf-8",
    )
    (home / ".op.env").write_text(
        "OP_SERVICE_ACCOUNT_TOKEN=jarvis-op-sentinel\n",
        encoding="utf-8",
    )
    (managed / ".env").write_text(
        "JARVIS_MANAGED_SENTINEL=jarvis-managed-sentinel\n",
        encoding="utf-8",
    )
    (home / "auth.json").write_text(
        '{"credential_pool":{"openrouter":[{"access_token":"jarvis-pool-sentinel"}]}}',
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: openrouter\n"
        "providers:\n"
        "  jarvis-sentinel-custom:\n"
        "    name: jarvis-sentinel-custom\n"
        "    base_url: https://jarvis-sentinel.invalid/v1\n"
        "    key_env: JARVIS_CUSTOM_PROVIDER_KEY\n"
        "    api_key: jarvis-config-provider-sentinel\n"
        "    extra_headers:\n"
        "      Authorization: jarvis-config-header-sentinel\n"
        "secrets:\n"
        "  sentinel_source:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    result = _run_isolated(
        """
        import builtins
        import os
        from pathlib import Path
        import subprocess
        import sys

        home = Path(os.environ["JARVIS_TEST_HOME"])
        project_env = Path(os.environ["JARVIS_TEST_PROJECT_ENV"])
        managed_env = Path(os.environ["JARVIS_TEST_MANAGED_ENV"])
        blocked = {
            (home / ".env").resolve(),
            project_env.resolve(),
            (home / ".op.env").resolve(),
            managed_env.resolve(),
            (home / "auth.json").resolve(),
        }

        original_open = builtins.open
        original_path_open = Path.open

        def _blocked_path(value):
            try:
                return Path(value).resolve() in blocked
            except (OSError, TypeError, ValueError):
                return False

        def guarded_open(file, *args, **kwargs):
            if _blocked_path(file):
                raise AssertionError(f"persistent credential path opened: {Path(file).name}")
            return original_open(file, *args, **kwargs)

        def guarded_path_open(self, *args, **kwargs):
            if _blocked_path(self):
                raise AssertionError(f"persistent credential path opened: {self.name}")
            return original_path_open(self, *args, **kwargs)

        builtins.open = guarded_open
        Path.open = guarded_path_open

        from hermes_cli.phase1_capability import (
            Phase1CapabilityModeError,
            phase1_capability_mode_enabled,
        )
        assert phase1_capability_mode_enabled()

        import hermes_cli.env_loader as env_loader
        env_loader._apply_external_secret_sources = lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("external source applied")
        )
        env_loader._apply_managed_env = lambda: (_ for _ in ()).throw(
            AssertionError("managed environment applied")
        )

        assert env_loader.load_hermes_dotenv(
            hermes_home=home,
            project_env=project_env,
        ) == []
        assert env_loader.hydrate_profile_secret_sources(home) == {}

        from agent.secret_scope import build_profile_secret_scope, get_secret
        scope = build_profile_secret_scope(home)
        assert scope["OPENROUTER_API_KEY"] == os.environ["OPENROUTER_API_KEY"]
        assert scope["JARVIS_APPROVED_TOOL_TOKEN"] == os.environ["JARVIS_APPROVED_TOOL_TOKEN"]
        assert "JARVIS_FILE_ONLY_SENTINEL" not in scope
        assert get_secret("OPENROUTER_API_KEY") == os.environ["OPENROUTER_API_KEY"]

        from hermes_cli.config import reload_env
        assert reload_env() == 0
        assert env_loader.load_hermes_dotenv(
            hermes_home=home,
            project_env=project_env,
        ) == []

        import hermes_cli.runtime_provider as runtime_provider
        runtime_provider.load_pool = lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("credential pool consulted")
        )
        runtime = runtime_provider.resolve_runtime_provider(requested="openrouter")
        assert runtime["api_key"] == os.environ["OPENROUTER_API_KEY"]
        custom_runtime = runtime_provider.resolve_runtime_provider(
            requested="jarvis-sentinel-custom"
        )
        assert custom_runtime["api_key"] == os.environ["JARVIS_CUSTOM_PROVIDER_KEY"]
        assert "extra_headers" not in custom_runtime

        import hermes_cli.main as cli_main
        assert cli_main._has_any_provider_configured()

        import gateway.run as gateway_run
        gateway_run._reload_runtime_env_preserving_config_authority()

        from agent.credential_pool import load_pool
        try:
            load_pool("openrouter")
        except Phase1CapabilityModeError:
            pass
        else:
            raise AssertionError("direct credential-pool expansion did not fail closed")

        from hermes_cli.auth import _load_auth_store
        try:
            _load_auth_store()
        except Phase1CapabilityModeError:
            pass
        else:
            raise AssertionError("direct auth-store expansion did not fail closed")

        from agent.secret_sources.registry import apply_all
        try:
            apply_all({}, home)
        except Phase1CapabilityModeError:
            pass
        else:
            raise AssertionError("direct external-source expansion did not fail closed")

        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "from hermes_cli.phase1_capability import "
                "phase1_capability_mode_enabled; "
                "raise SystemExit(0 if phase1_capability_mode_enabled() else 1)",
            ],
            cwd=os.getcwd(),
            env=os.environ.copy(),
            check=False,
        )
        assert child.returncode == 0
        """,
        mode="1",
        extra_env={
            "HERMES_HOME": str(home),
            "HERMES_MANAGED_DIR": str(managed),
            "JARVIS_TEST_HOME": str(home),
            "JARVIS_TEST_PROJECT_ENV": str(project / ".env"),
            "JARVIS_TEST_MANAGED_ENV": str(managed / ".env"),
            "OPENROUTER_API_KEY": "jarvis-inherited-provider-sentinel",
            "JARVIS_CUSTOM_PROVIDER_KEY": "jarvis-inherited-custom-sentinel",
            "JARVIS_APPROVED_TOOL_TOKEN": "jarvis-inherited-tool-sentinel",
        },
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("invalid_value", ["yes", "2"])
def test_invalid_truthy_mode_fails_before_dotenv_import(invalid_value: str) -> None:
    result = _run_isolated(
        """
        import importlib.abc
        import sys

        class RejectDotenv(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "dotenv" or fullname.startswith("dotenv."):
                    raise AssertionError("dotenv imported before Phase 1 validation")
                return None

        sys.meta_path.insert(0, RejectDotenv())
        try:
            import hermes_cli.env_loader  # noqa: F401
        except Exception as exc:
            if (
                type(exc).__name__ != "Phase1CapabilityModeError"
                or type(exc).__module__ != "hermes_cli.phase1_capability"
            ):
                raise
        else:
            raise AssertionError("invalid truthy Phase 1 value was accepted")
        """,
        mode=invalid_value,
    )
    assert result.returncode == 0, result.stderr


def test_disabled_mode_preserves_normal_dotenv_behavior(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(
        "JARVIS_NORMAL_MODE_SENTINEL=jarvis-normal-mode-sentinel\n",
        encoding="utf-8",
    )

    result = _run_isolated(
        """
        import os
        from pathlib import Path
        from hermes_cli.env_loader import load_hermes_dotenv
        from hermes_cli.phase1_capability import phase1_capability_mode_enabled

        home = Path(os.environ["JARVIS_TEST_HOME"])
        assert not phase1_capability_mode_enabled()
        assert load_hermes_dotenv(hermes_home=home) == [home / ".env"]
        assert os.environ["JARVIS_NORMAL_MODE_SENTINEL"] == "jarvis-normal-mode-sentinel"
        """,
        mode="0",
        extra_env={"HERMES_HOME": str(home), "JARVIS_TEST_HOME": str(home)},
    )

    assert result.returncode == 0, result.stderr
