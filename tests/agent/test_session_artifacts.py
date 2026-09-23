"""Session-scoped artifacts directory resolution."""

from agent.session_artifacts import resolve_session_artifacts_dir


def test_configured_root_resolves_to_session_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"sessions": {"artifacts_dir": str(tmp_path)}},
    )

    resolved = resolve_session_artifacts_dir("session-123")

    assert resolved == str(tmp_path / "session-123" / "artifacts")
    assert not (tmp_path / "session-123" / "artifacts").exists()


def test_parent_artifacts_dir_is_inherited_before_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"sessions": {"artifacts_dir": str(tmp_path / "new-root")}},
    )
    monkeypatch.setenv("HERMES_SESSION_ARTIFACTS_DIR", str(tmp_path / "parent-artifacts"))

    assert resolve_session_artifacts_dir("child-123", parent_session_id="parent-123") == str(tmp_path / "parent-artifacts")


def test_gateway_context_artifacts_dir_is_inherited_before_config(monkeypatch, tmp_path):
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"sessions": {"artifacts_dir": str(tmp_path / "new-root")}},
    )
    tokens = set_session_vars(artifacts_dir=str(tmp_path / "parent-artifacts"))
    try:
        assert resolve_session_artifacts_dir("child-123", parent_session_id="parent-123") == str(tmp_path / "parent-artifacts")
    finally:
        clear_session_vars(tokens)
