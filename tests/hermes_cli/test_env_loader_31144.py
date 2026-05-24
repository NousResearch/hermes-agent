"""Regression tests for issue #31144.

When ``HERMES_HOME`` is redirected outside ``~/.hermes`` (Windows installer
layout, profile mode, Docker volume), edits to the doc-canonical
``~/.hermes/.env`` were silently ignored because ``load_hermes_dotenv``
resolved its single ``user_env`` slot via ``HERMES_HOME`` only. The fix
loads the doc-canonical file too — at lower precedence so the
HERMES_HOME-resolved file still wins on key collisions but doc-mandated
edits actually take effect.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli.env_loader import load_hermes_dotenv


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect ``Path.home()`` so we don't touch the real user's ~."""
    fake = tmp_path / "userhome"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake))
    return fake


def test_doc_canonical_env_loaded_when_hermes_home_redirected(fake_home, tmp_path, monkeypatch):
    """``~/.hermes/.env`` must load even when HERMES_HOME points elsewhere.

    This reproduces the Windows-installer layout from #31144: HERMES_HOME
    is set to ``%LOCALAPPDATA%\\hermes`` but the user follows the docs and
    edits ``~/.hermes/.env`` to enable the API server.
    """
    install_tree = tmp_path / "appdata" / "hermes"
    install_tree.mkdir(parents=True)
    install_env = install_tree / ".env"
    install_env.write_text("HERMES_WEB_DIST=/install/web_dist\n", encoding="utf-8")

    doc_dir = fake_home / ".hermes"
    doc_dir.mkdir()
    doc_env = doc_dir / ".env"
    doc_env.write_text("API_SERVER_ENABLED=true\nAPI_SERVER_KEY=user-key\n", encoding="utf-8")

    monkeypatch.delenv("API_SERVER_ENABLED", raising=False)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    monkeypatch.delenv("HERMES_WEB_DIST", raising=False)

    loaded = load_hermes_dotenv(hermes_home=install_tree)

    assert doc_env in loaded
    assert install_env in loaded
    assert loaded.index(doc_env) < loaded.index(install_env)
    assert os.getenv("API_SERVER_ENABLED") == "true"
    assert os.getenv("API_SERVER_KEY") == "user-key"
    assert os.getenv("HERMES_WEB_DIST") == "/install/web_dist"


def test_hermes_home_env_wins_on_key_collision(fake_home, tmp_path, monkeypatch):
    """HERMES_HOME-resolved ``.env`` must override the doc-canonical file.

    Profile-mode users keep per-profile overrides in ``${HERMES_HOME}/.env``
    while sharing common keys via ``~/.hermes/.env``. Profile-specific
    values must win.
    """
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    (profile_root / ".env").write_text(
        "OPENROUTER_API_KEY=profile-key\n", encoding="utf-8"
    )

    doc_dir = fake_home / ".hermes"
    doc_dir.mkdir()
    (doc_dir / ".env").write_text(
        "OPENROUTER_API_KEY=shared-key\nGITHUB_TOKEN=ghp_shared\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    load_hermes_dotenv(hermes_home=profile_root)

    assert os.getenv("OPENROUTER_API_KEY") == "profile-key"
    assert os.getenv("GITHUB_TOKEN") == "ghp_shared"


def test_doc_canonical_skipped_when_same_file_as_hermes_home(fake_home, monkeypatch):
    """The fallback must NOT double-load when HERMES_HOME == ~/.hermes.

    Standard installs (HERMES_HOME unset, or pointing at the canonical
    ``~/.hermes``) used to have a single ``.env`` load — keep it that way
    so we don't append a duplicate path to the loaded list.
    """
    canonical = fake_home / ".hermes"
    canonical.mkdir()
    env_file = canonical / ".env"
    env_file.write_text("FOO=bar\n", encoding="utf-8")

    monkeypatch.delenv("FOO", raising=False)

    loaded = load_hermes_dotenv(hermes_home=canonical)

    assert loaded == [env_file]
    assert os.getenv("FOO") == "bar"


def test_doc_canonical_absent_no_op(fake_home, tmp_path, monkeypatch):
    """No ``~/.hermes/.env`` on disk → behavior unchanged from pre-fix."""
    install_tree = tmp_path / "appdata" / "hermes"
    install_tree.mkdir(parents=True)
    (install_tree / ".env").write_text("FOO=installer\n", encoding="utf-8")

    monkeypatch.delenv("FOO", raising=False)
    assert not (fake_home / ".hermes" / ".env").exists()

    loaded = load_hermes_dotenv(hermes_home=install_tree)

    assert loaded == [install_tree / ".env"]
    assert os.getenv("FOO") == "installer"


def test_project_env_only_overrides_when_no_user_envs(fake_home, tmp_path, monkeypatch):
    """project_env keeps its existing fallback semantics.

    When neither the doc-canonical nor the HERMES_HOME ``.env`` exists,
    project ``.env`` must still override stale shell vars (preserves
    pre-#31144 behavior verified by ``test_env_loader.py``).
    """
    install_tree = tmp_path / "appdata" / "hermes"
    install_tree.mkdir(parents=True)
    project_env = tmp_path / "project" / ".env"
    project_env.parent.mkdir()
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://stale.example/v1")
    assert not (fake_home / ".hermes" / ".env").exists()

    loaded = load_hermes_dotenv(hermes_home=install_tree, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://project.example/v1"
