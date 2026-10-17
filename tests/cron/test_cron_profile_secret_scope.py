"""Cron ticks bind the complete owning profile scope, including credentials."""

from agent.secret_scope import current_secret_scope
from cron.scheduler_provider import _profile_cron_scope


def test_profile_cron_scope_preserves_launch_env_and_isolates_satellite(tmp_path, monkeypatch):
    root = tmp_path / "root"
    alpha = root / "profiles" / "alpha"
    alpha.mkdir(parents=True)
    (root / ".env").write_text("FILE_ONLY=root-file\n", encoding="utf-8")
    (alpha / ".env").write_text(
        "ENV_ONLY=alpha-token\nFILE_ONLY=alpha-file\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("ENV_ONLY", "launch-token")
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: root)

    seen = []
    for home in (root, alpha, root):
        with _profile_cron_scope(home):
            scope = current_secret_scope()
            assert scope is not None
            seen.append((scope.get("ENV_ONLY"), scope.get("FILE_ONLY")))

    assert seen == [
        ("launch-token", "root-file"),
        ("alpha-token", "alpha-file"),
        ("launch-token", "root-file"),
    ]
    assert current_secret_scope() is None
