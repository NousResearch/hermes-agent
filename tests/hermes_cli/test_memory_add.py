"""Tests for `hermes memory add` — no-LLM built-in memory writes."""

from hermes_cli.memory_write import add_builtin_memory


def test_memory_add_writes_memory_md(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"memory_char_limit": 2200, "user_char_limit": 1375}},
    )

    result = add_builtin_memory("Prefer dark themes", target="memory")
    assert result.get("success") is True
    text = (memories / "MEMORY.md").read_text(encoding="utf-8")
    assert "Prefer dark themes" in text


def test_memory_add_user_target(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "memories").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"memory_char_limit": 2200, "user_char_limit": 1375}},
    )

    result = add_builtin_memory("Timezone: US Pacific", target="user")
    assert result.get("success") is True
    text = (hermes_home / "memories" / "USER.md").read_text(encoding="utf-8")
    assert "Timezone: US Pacific" in text


def test_memory_add_rejects_empty(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "memories").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})

    result = add_builtin_memory("   ", target="memory")
    assert result.get("success") is False
    assert "empty" in str(result.get("error", "")).lower()
