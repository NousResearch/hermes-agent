"""Trusted project dotenv follows the selected config backend, not a local-file bypass."""

import pytest

from hermes_cli import config_backend, env_loader


def test_trust_can_be_supplied_without_a_local_config_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("BACKEND_PROJECT_API_KEY=project-value\n")
    monkeypatch.delenv("BACKEND_PROJECT_API_KEY", raising=False)

    class Backend(config_backend.FileBackend):
        def read_user_doc_readonly(self, home):
            assert home == tmp_path / "home"
            return {"dotenv": {"trusted_cwds": [str(project)]}}

    monkeypatch.setattr(config_backend, "get_config_backend", lambda: Backend())
    loaded = env_loader.load_hermes_dotenv(
        hermes_home=home, cwd_env=True, cwd_path=project, load_external_secrets=False,
    )

    import os

    assert loaded == [project / ".env"]
    assert os.environ["BACKEND_PROJECT_API_KEY"] == "project-value"
    assert not (home / "config.yaml").exists()


def test_backend_refusal_cannot_authorize_project_credentials(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("REFUSED_PROJECT_API_KEY=must-not-load\n")
    monkeypatch.delenv("REFUSED_PROJECT_API_KEY", raising=False)

    class Backend(config_backend.FileBackend):
        def boot(self, home):
            raise config_backend.ConfigBackendUnavailable("test backend refuses configuration")

    monkeypatch.setattr(config_backend, "get_config_backend", lambda: Backend())
    with pytest.raises(config_backend.ConfigBackendUnavailable):
        env_loader.load_hermes_dotenv(
            hermes_home=tmp_path / "home", cwd_env=True, cwd_path=project,
            load_external_secrets=False,
        )

    import os

    assert "REFUSED_PROJECT_API_KEY" not in os.environ
