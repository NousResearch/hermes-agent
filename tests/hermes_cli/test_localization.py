from types import SimpleNamespace

from agent.i18n import pluralize, t
from hermes_cli.commands import gateway_help_lines
from hermes_cli.status import show_status
from hermes_cli.vercel_auth import describe_vercel_auth


def test_translation_catalog_uses_russian(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "ru")

    assert t("commands.help.title", default="Available Commands") == "Доступные команды"
    assert t("cli.fresh_start", default="Fresh start!") == "Новый сеанс! Экран очищен, беседа сброшена."


def test_pluralize_prefers_russian_forms():
    assert pluralize(1, "инструмент", "инструмента", "инструментов", language="ru") == "инструмент"
    assert pluralize(2, "инструмент", "инструмента", "инструментов", language="ru") == "инструмента"
    assert pluralize(5, "инструмент", "инструмента", "инструментов", language="ru") == "инструментов"
    assert pluralize(21, "инструмент", "инструмента", "инструментов", language="ru") == "инструмент"


def test_gateway_help_lines_are_localized(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "ru")

    lines = gateway_help_lines()

    assert any("Показать доступные команды" in line for line in lines)
    assert any("альтернативные имена" in line for line in lines)


def test_show_status_uses_russian_locale(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("HERMES_LANGUAGE", "ru")
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4", "terminal": {"backend": "local"}}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Статус Hermes Agent" in output
    assert "Провайдеры API-ключей" in output
    assert "Платформы сообщений" in output


def test_vercel_auth_is_localized(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "ru")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")

    auth_status = describe_vercel_auth()

    assert "OIDC-token" in auth_status.label
    assert any("режим: OIDC" in line for line in auth_status.detail_lines)
