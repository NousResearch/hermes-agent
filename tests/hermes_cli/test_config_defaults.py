from agent.conversation_worktree_policy import resolve_conversation_worktree_policy
from hermes_cli.config import DEFAULT_CONFIG, load_config


def test_conversation_worktree_defaults_are_platform_neutral_and_disabled():
    worktree = DEFAULT_CONFIG["conversation_worktree"]

    assert worktree is None


def test_normal_load_config_preserves_legacy_conversation_worktree_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "desktop:\n"
        "  conversation_worktree:\n"
        "    enabled: true\n"
        f"    source_worktree: {tmp_path / 'stable'}\n"
        f"    worktree_root: {tmp_path / 'worktrees'}\n",
        encoding="utf-8",
    )

    policy = resolve_conversation_worktree_policy(load_config())

    assert policy.enabled is True
    assert policy.source_worktree == (tmp_path / "stable").resolve()
    assert policy.legacy_location is True


def test_normal_load_config_keeps_explicit_top_level_policy_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "conversation_worktree:\n"
        "  enabled: false\n"
        "desktop:\n"
        "  conversation_worktree:\n"
        "    enabled: true\n"
        f"    source_worktree: {tmp_path / 'stable'}\n"
        f"    worktree_root: {tmp_path / 'worktrees'}\n",
        encoding="utf-8",
    )

    policy = resolve_conversation_worktree_policy(load_config())

    assert policy.enabled is False
    assert policy.legacy_location is False
