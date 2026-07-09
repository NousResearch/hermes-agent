"""Regression tests for IRC setup recommendations."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_irc_user_facing_examples_do_not_recommend_libera():
    """Hermes should not steer agentic IRC users toward Libera.Chat."""
    user_facing_paths = [
        "plugins/platforms/irc/adapter.py",
        "plugins/platforms/irc/plugin.yaml",
        "hermes_cli/config.py",
        "website/docs/user-guide/messaging/irc.md",
        "website/docs/reference/environment-variables.md",
    ]

    offenders = []
    for rel_path in user_facing_paths:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        if "irc.libera.chat" in text or "Libera.Chat" in text:
            offenders.append(rel_path)

    assert offenders == []
