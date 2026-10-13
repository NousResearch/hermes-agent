"""Keep multilingual Telegram menus within a conservative aggregate text budget."""

import pytest

from hermes_cli import commands_platforms as platforms

# Local safety policy, not a claimed public Telegram API limit.
_TEXT_BUDGET = 7000


def _text_bytes(rows):
    return sum(len(name.encode("utf-8")) + len(desc.encode("utf-8")) for name, desc in rows)


@pytest.mark.parametrize("text", ["Описание команды " * 14, "命令说明" * 60, "🔧" * 64, "A" * 256], ids=["ru", "cjk", "emoji", "ascii"])
@pytest.mark.parametrize("with_skills", [False, True])
@pytest.mark.parametrize("cap", [0, 1, 60, 100, 200])
def test_large_menu_shortens_descriptions_without_dropping_names(text, with_skills, cap, monkeypatch):
    native = [("x" * 29 + f"{i:03}", text) for i in range(40 if with_skills else 120)]
    dynamic = [(f"skill_{i:02}", text, f"/skill_{i:02}", f"skill_{i:02}") for i in range(80)] if with_skills else []
    monkeypatch.setattr(platforms, "telegram_bot_commands", lambda **kwargs: native)
    monkeypatch.setattr(platforms, "_collect_gateway_skill_entries", lambda **kwargs: (dynamic, 0))
    monkeypatch.setattr(platforms, "_prioritize_telegram_menu_candidates", lambda rows: rows)
    source = native + [(name, desc) for name, desc, *_ in dynamic]
    menu, hidden = platforms.telegram_menu_commands(max_commands=cap)
    expected = source[:min(cap, 100)]
    assert [name for name, _ in menu] == [name for name, _ in expected]
    assert hidden == len(source) - len(menu)
    assert _text_bytes(menu) <= _TEXT_BUDGET
    assert all(1 <= len(desc) <= 256 and "\ufffd" not in desc for _, desc in menu)
    assert platforms.telegram_bot_commands() == native  # the full description source is not modified
    assert native == [("x" * 29 + f"{i:03}", text) for i in range(len(native))]


def test_menu_within_budget_keeps_descriptions_exactly(monkeypatch):
    native = [(f"cmd_{i}", "Короткое описание") for i in range(60)]
    monkeypatch.setattr(platforms, "telegram_bot_commands", lambda **kwargs: native)
    monkeypatch.setattr(platforms, "_collect_gateway_skill_entries", lambda **kwargs: ([], 0))
    monkeypatch.setattr(platforms, "_prioritize_telegram_menu_candidates", lambda rows: rows)
    assert _text_bytes(native) < _TEXT_BUDGET
    assert platforms.telegram_menu_commands(max_commands=60) == (native, 0)
