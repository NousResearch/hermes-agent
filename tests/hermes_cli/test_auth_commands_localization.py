from types import SimpleNamespace

import pytest

from hermes_cli.auth_commands import (
    _interactive_add,
    _interactive_auth,
    _interactive_remove,
    _interactive_strategy,
    auth_reset_command,
)


class _Entry:
    def __init__(self, label, auth_type="api_key", source="manual", entry_id="cred-1"):
        self.label = label
        self.auth_type = auth_type
        self.source = source
        self.id = entry_id


class _Pool:
    def __init__(self, entries=None):
        self._entries = entries or []

    def has_credentials(self):
        return bool(self._entries)

    def entries(self):
        return self._entries

    def reset_statuses(self):
        return len(self._entries)


def test_interactive_auth_menu_is_localized(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.auth_commands.auth_list_command", lambda _args: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    _interactive_auth()

    out = capsys.readouterr().out
    assert "자격 증명 풀 상태" in out
    assert "무엇을 할까요?" in out
    assert "자격 증명 추가" in out
    assert "provider 회전 전략 설정" in out


def test_interactive_add_prompts_are_localized(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.auth_commands.PROVIDER_REGISTRY", {"anthropic": object()})
    monkeypatch.setattr("hermes_cli.auth_commands._get_custom_provider_names", lambda: [])
    recorded = {}

    def fake_auth_add(args):
        recorded["provider"] = args.provider
        recorded["auth_type"] = args.auth_type
        recorded["label"] = args.label

    responses = iter(["anthropic", "2", "업무 계정"])

    def fake_input(prompt):
        print(prompt, end="")
        return next(responses)

    monkeypatch.setattr("hermes_cli.auth_commands.auth_add_command", fake_auth_add)
    monkeypatch.setattr("builtins.input", fake_input)

    _interactive_add()

    out = capsys.readouterr().out
    assert "알려진 provider:" in out
    assert "API key와 OAuth 로그인을 모두 지원해요." in out
    assert "유형 [1/2]:" in out
    assert "라벨 / 계정 이름(선택):" in out
    assert recorded == {"provider": "anthropic", "auth_type": "oauth", "label": "업무 계정"}


def test_interactive_remove_empty_pool_is_localized(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.auth_commands.PROVIDER_REGISTRY", {"openrouter": object()})
    monkeypatch.setattr("hermes_cli.auth_commands._get_custom_provider_names", lambda: [])
    monkeypatch.setattr("hermes_cli.auth_commands.load_pool", lambda _provider: _Pool())

    def fake_input(prompt):
        print(prompt, end="")
        return "openrouter"

    monkeypatch.setattr("builtins.input", fake_input)

    _interactive_remove()

    out = capsys.readouterr().out
    assert "자격 증명을 제거할 provider" in out
    assert "openrouter 용 자격 증명이 없어요." in out


def test_auth_reset_command_is_localized(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.auth_commands.load_pool", lambda _provider: _Pool([_Entry("a"), _Entry("b")]))

    auth_reset_command(SimpleNamespace(provider="anthropic"))

    out = capsys.readouterr().out
    assert "anthropic 자격 증명 2개의 상태를 초기화했어요" in out


def test_interactive_strategy_invalid_choice_is_localized(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.auth_commands.PROVIDER_REGISTRY", {"openrouter": object()})
    monkeypatch.setattr("hermes_cli.auth_commands._get_custom_provider_names", lambda: [])
    monkeypatch.setattr("hermes_cli.auth_commands.get_pool_strategy", lambda _provider: "fill_first")
    responses = iter(["openrouter", "9"])

    def fake_input(prompt):
        print(prompt, end="")
        return next(responses)

    monkeypatch.setattr("builtins.input", fake_input)

    _interactive_strategy()

    out = capsys.readouterr().out
    assert "openrouter 의 현재 전략" in out
    assert "전략 [1-4]:" in out
    assert "올바르지 않은 선택이에요." in out
