"""Whole-document config.yaml writes must preserve hand-written comments (#92554).

`save_config` and `atomic_config_write` re-serialise the entire document. With the PyYAML
dumper that silently destroys the user's rationale for non-default settings, so existing
files go through ruamel round-trip mode instead.

`hermes config set` is covered separately by #72581 and is not exercised here.
"""

import textwrap

import pytest
import yaml

CONFIG_WITH_COMMENTS = textwrap.dedent("""\
    # TOP COMMENT — must survive
    model:
      provider: test
      # why this ceiling: a page must leave write_file in ONE turn
      max_tokens: 16384
    agent:
      verify_on_stop: true      # deliberate: unattended work gets checked
    _config_version: 40
    """)

COMMENTS = ("TOP COMMENT", "must leave write_file in ONE turn", "deliberate: unattended work")


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    from hermes_cli import config as config_module

    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_WITH_COMMENTS, encoding="utf-8")
    monkeypatch.setattr(config_module, "get_config_path", lambda: path)
    monkeypatch.setattr(config_module, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config_module, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(config_module, "is_managed", lambda: False)
    config_module._RAW_CONFIG_CACHE.pop(str(path), None)
    return path


def _assert_comments_survive(path):
    text = path.read_text(encoding="utf-8")
    for comment in COMMENTS:
        assert comment in text, f"comment lost: {comment!r}"


def test_atomic_config_write_preserves_comments(config_path):
    """The shared fail-closed writer (plugins enable, gateway slash commands, doctor)."""
    from hermes_cli.config import atomic_config_write

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["_config_version"] = 41
    atomic_config_write(config_path, data)

    _assert_comments_survive(config_path)
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["_config_version"] == 41


def test_save_config_preserves_comments_across_migration(config_path):
    """A migration bumping _config_version must not erase the file's rationale."""
    from hermes_cli import config as config_module

    raw = config_module.read_raw_config()
    raw["_config_version"] = 41
    config_module.save_config(raw)

    _assert_comments_survive(config_path)
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["_config_version"] == 41
    assert saved["agent"]["verify_on_stop"] is True
    assert saved["model"]["max_tokens"] == 16384


def test_new_config_still_gets_example_blocks(tmp_path):
    """A file that does not exist yet keeps the PyYAML path so extra_content lands."""
    from hermes_cli.config import atomic_config_write

    path = tmp_path / "fresh.yaml"
    atomic_config_write(path, {"model": {"provider": "test"}}, extra_content="# ── Security ──\n")

    text = path.read_text(encoding="utf-8")
    assert "── Security ──" in text
    assert yaml.safe_load(text)["model"]["provider"] == "test"
