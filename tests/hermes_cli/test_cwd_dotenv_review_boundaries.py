"""Real-file contracts for CWD dotenv endpoint and directory trust boundaries."""

import json
import os

import pytest

from hermes_cli.env_loader import load_hermes_dotenv


def _write_config(home, trusted_cwds):
    home.mkdir()
    (home / "config.yaml").write_text(
        json.dumps({"dotenv": {"trusted_cwds": trusted_cwds}}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "AZURE_OPENAI_ENDPOINT",
        "AWS_ENDPOINT_URL",
        "OPENAI_API_BASE",
        "PROVIDER_API_URL",
        "PROVIDER_API_HOST",
        "openai_base_url",
    ],
)
@pytest.mark.parametrize("inherited", [None, "https://user-controlled.invalid"])
def test_project_cannot_install_or_replace_provider_endpoint(
    tmp_path, monkeypatch, name, inherited
):
    home, project = tmp_path / "home", tmp_path / "project"
    project.mkdir()
    _write_config(home, [str(project)])
    (project / ".env").write_text(
        f"{name}=https://project-controlled.invalid\nPROJECT_API_KEY=local\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)
    if inherited is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, inherited)

    load_hermes_dotenv(hermes_home=home, cwd_env=True, cwd_path=project)

    assert os.getenv(name) == inherited
    assert os.getenv("PROJECT_API_KEY") == "local"


def test_symlinked_cwd_uses_canonical_trust_and_rejects_retargeting(tmp_path, monkeypatch):
    home, project, other = (tmp_path / name for name in ("home", "project", "other"))
    project.mkdir()
    other.mkdir()
    _write_config(home, [str(project)])
    (project / ".env").write_text("PROJECT_API_KEY=trusted\n", encoding="utf-8")
    (other / ".env").write_text("PROJECT_API_KEY=untrusted\n", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(project, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    monkeypatch.chdir(alias)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    assert load_hermes_dotenv(hermes_home=home, cwd_env=True) == [project / ".env"]
    assert os.getenv("PROJECT_API_KEY") == "trusted"

    # The configured canonical directory, not the mutable alias, owns trust.
    monkeypatch.chdir(tmp_path)
    alias.unlink()
    alias.symlink_to(other, target_is_directory=True)
    monkeypatch.chdir(alias)
    monkeypatch.delenv("PROJECT_API_KEY")

    assert load_hermes_dotenv(hermes_home=home, cwd_env=True) == []
    assert os.getenv("PROJECT_API_KEY") is None


@pytest.mark.parametrize(
    "trusted_cwds",
    [None, True, 23, "/not-a-list", {}, [None, True, 17, {}, []], ["/bad\x00path"]],
)
def test_malformed_trust_entries_fail_closed(tmp_path, monkeypatch, trusted_cwds):
    home, project = tmp_path / "home", tmp_path / "project"
    project.mkdir()
    _write_config(home, trusted_cwds)
    (project / ".env").write_text("PROJECT_API_KEY=untrusted\n", encoding="utf-8")
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    assert load_hermes_dotenv(hermes_home=home, cwd_env=True, cwd_path=project) == []
    assert os.getenv("PROJECT_API_KEY") is None
