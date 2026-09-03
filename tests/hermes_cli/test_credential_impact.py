"""Tests for the read-only credential dependency map (``hermes credentials impact``).

Drives real config/plugin-manifest parsing against isolated fixtures — no
mocking of the cross-reference logic itself, only of HERMES_HOME and the
bundled-plugins root so results are deterministic and independent of the
current repo's plugin roster.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli.credential_impact import compute_impact


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    home = tmp_path / "cred_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def bundled_plugins_dir(monkeypatch, tmp_path):
    root = tmp_path / "bundled_plugins"
    root.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(root))
    return root


def _write_yaml(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_config(home, text):
    home.joinpath("config.yaml").write_text(text, encoding="utf-8")


def test_provider_env_var_maps_to_provider_id(hermes_home, bundled_plugins_dir):
    impact = compute_impact("OPENAI_API_KEY")
    assert "openai-api" in impact.providers


def test_unknown_var_is_empty(hermes_home, bundled_plugins_dir):
    impact = compute_impact("TOTALLY_UNKNOWN_VAR_XYZ")
    assert impact.is_empty


def test_declared_in_env_reflects_dotenv_file(hermes_home, bundled_plugins_dir):
    hermes_home.joinpath(".env").write_text("MY_TEST_VAR=placeholder\n", encoding="utf-8")
    impact = compute_impact("MY_TEST_VAR")
    assert impact.declared_in_env is True

    impact_missing = compute_impact("NEVER_DECLARED_VAR")
    assert impact_missing.declared_in_env is False


def test_auxiliary_task_maps_through_resolved_provider(hermes_home, bundled_plugins_dir):
    _write_config(
        hermes_home,
        """
model:
  provider: openai-api
auxiliary:
  summarizer:
    provider: openai-api
  fallback_only:
    provider: some-other-provider
""",
    )
    impact = compute_impact("OPENAI_API_KEY")
    assert impact.auxiliary_tasks == ["summarizer"]


def test_auxiliary_task_falls_back_to_main_model_provider(hermes_home, bundled_plugins_dir):
    _write_config(
        hermes_home,
        """
model:
  provider: openai-api
auxiliary:
  summarizer: {}
""",
    )
    impact = compute_impact("OPENAI_API_KEY")
    assert impact.auxiliary_tasks == ["summarizer"]


def test_mcp_server_env_block_is_detected(hermes_home, bundled_plugins_dir):
    _write_config(
        hermes_home,
        """
mcp:
  servers:
    github:
      command: npx
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: placeholder
    filesystem:
      command: npx
      env: {}
""",
    )
    impact = compute_impact("GITHUB_PERSONAL_ACCESS_TOKEN")
    assert impact.mcp_servers == ["github"]

    impact_other = compute_impact("SOME_UNRELATED_VAR")
    assert impact_other.mcp_servers == []


def test_platform_plugin_manifest_requires_env_is_detected(hermes_home, bundled_plugins_dir):
    _write_yaml(
        bundled_plugins_dir / "platforms" / "acme" / "plugin.yaml",
        """
name: acme-platform
kind: platform
requires_env:
  - name: ACME_BOT_TOKEN
    description: token
""",
    )
    impact = compute_impact("ACME_BOT_TOKEN")
    assert impact.platforms == ["acme-platform"]
    assert impact.plugins == []


def test_standalone_plugin_optional_env_is_detected_as_plugin_not_platform(
    hermes_home, bundled_plugins_dir
):
    _write_yaml(
        bundled_plugins_dir / "disk-cleanup" / "plugin.yaml",
        """
name: disk-cleanup
kind: standalone
optional_env:
  - name: DISK_CLEANUP_THRESHOLD
""",
    )
    impact = compute_impact("DISK_CLEANUP_THRESHOLD")
    assert impact.plugins == ["disk-cleanup"]
    assert impact.platforms == []


def test_category_plugin_env_var_is_detected(hermes_home, bundled_plugins_dir):
    _write_yaml(
        bundled_plugins_dir / "image_gen" / "acme" / "plugin.yaml",
        """
name: acme-image-gen
kind: backend
requires_env:
  - name: ACME_IMAGE_API_KEY
""",
    )
    impact = compute_impact("ACME_IMAGE_API_KEY")
    assert impact.plugins == ["image_gen/acme"]


def test_own_discovery_categories_are_skipped_at_top_level(hermes_home, bundled_plugins_dir):
    # memory/ has its own discovery system — the top-level general scan
    # must not descend into it (mirrors PluginManager's skip_names).
    _write_yaml(
        bundled_plugins_dir / "memory" / "acme" / "plugin.yaml",
        """
name: acme-memory
kind: exclusive
requires_env:
  - name: ACME_MEMORY_TOKEN
""",
    )
    impact = compute_impact("ACME_MEMORY_TOKEN")
    assert impact.plugins == []
    assert impact.platforms == []


def test_impact_never_exposes_credential_values(hermes_home, bundled_plugins_dir):
    hermes_home.joinpath(".env").write_text(
        "SECRET_SHAPED_VAR=super-secret-value-should-never-appear\n", encoding="utf-8"
    )
    impact = compute_impact("SECRET_SHAPED_VAR")
    dumped = json.dumps(impact.__dict__)
    assert "super-secret-value-should-never-appear" not in dumped


def test_credentials_command_cli_json_output(hermes_home, bundled_plugins_dir, capsys):
    import argparse

    from hermes_cli.credential_impact import credentials_command

    args = argparse.Namespace(
        credentials_action="impact", var="OPENAI_API_KEY", json=True
    )
    rc = credentials_command(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["var"] == "OPENAI_API_KEY"
    assert "openai-api" in out["providers"]


def test_credentials_command_requires_var(hermes_home, bundled_plugins_dir, capsys):
    import argparse

    from hermes_cli.credential_impact import credentials_command

    args = argparse.Namespace(credentials_action="impact", var="", json=False)
    rc = credentials_command(args)
    assert rc == 1


def test_credentials_command_unknown_action(hermes_home, bundled_plugins_dir, capsys):
    import argparse

    from hermes_cli.credential_impact import credentials_command

    args = argparse.Namespace(credentials_action=None)
    rc = credentials_command(args)
    assert rc == 1
