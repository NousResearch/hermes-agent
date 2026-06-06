import json
from types import SimpleNamespace

import pytest

import hermes_cli.main as hermes_main


def test_model_list_json_calls_provider_model_ids_without_tty(monkeypatch, capsys):
    calls = []

    def fake_provider_model_ids(provider, *, force_refresh=False):
        calls.append((provider, force_refresh))
        return ["gpt-test-a", "gpt-test-b"]

    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids",
        fake_provider_model_ids,
    )
    monkeypatch.setattr(
        hermes_main,
        "_require_tty",
        lambda command_name: (_ for _ in ()).throw(AssertionError("TTY required")),
    )

    hermes_main.cmd_model(
        SimpleNamespace(
            model_action="list",
            provider="openai-codex",
            json_output=True,
            force_refresh=True,
        )
    )

    assert json.loads(capsys.readouterr().out) == ["gpt-test-a", "gpt-test-b"]
    assert calls == [("openai-codex", True)]


def test_model_list_text_prints_one_model_per_line(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids",
        lambda provider, *, force_refresh=False: ["model-a", "model-b"],
    )

    hermes_main.cmd_model(
        SimpleNamespace(
            model_action="list",
            provider="nous",
            json_output=False,
            force_refresh=False,
        )
    )

    assert capsys.readouterr().out.splitlines() == ["model-a", "model-b"]


def test_model_providers_json_calls_list_available_providers_without_tty(
    monkeypatch,
    capsys,
):
    providers = [
        {
            "id": "openai-codex",
            "label": "OpenAI Codex",
            "aliases": ["codex"],
            "authenticated": True,
        }
    ]
    monkeypatch.setattr("hermes_cli.models.list_available_providers", lambda: providers)
    monkeypatch.setattr(
        hermes_main,
        "_require_tty",
        lambda command_name: (_ for _ in ()).throw(AssertionError("TTY required")),
    )

    hermes_main.cmd_model(SimpleNamespace(model_action="providers", json_output=True))

    assert json.loads(capsys.readouterr().out) == providers


def test_list_available_providers_returns_unique_provider_ids():
    from hermes_cli.models import list_available_providers

    provider_ids = [row["id"] for row in list_available_providers()]

    assert len(provider_ids) == len(set(provider_ids))


def test_bare_model_command_still_requires_tty(monkeypatch):
    class TtyGate(Exception):
        pass

    calls = []
    monkeypatch.setattr(
        hermes_main,
        "_require_tty",
        lambda command_name: (
            calls.append(command_name),
            (_ for _ in ()).throw(TtyGate()),
        ),
    )
    monkeypatch.setattr(
        hermes_main,
        "select_provider_and_model",
        lambda args=None: (_ for _ in ()).throw(
            AssertionError("interactive model picker should not run")
        ),
    )

    with pytest.raises(TtyGate):
        hermes_main.cmd_model(SimpleNamespace())

    assert calls == ["model"]
