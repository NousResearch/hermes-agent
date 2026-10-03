"""An explicit auxiliary.<task>.provider beats a healthy main model.

``_resolve_auto_route`` used to return the main client whenever it could be
built, so ``auxiliary.compression.provider: minimax-oauth`` still summarized
with the chat model. See #125707.
"""

from agent import auxiliary_client as aux


def _main_runtime():
    return {"provider": "custom", "model": "Qwen3"}


def test_explicit_compression_provider_beats_healthy_main(monkeypatch):
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {
        "provider": "minimax-oauth",
        "model": "MiniMax-M3",
        "base_url": "https://api.minimax.io/anthropic",
    } if task == "compression" else {})
    seen = {}

    def fake_resolve(provider, model=None, **_kwargs):
        seen["provider"] = provider
        seen["model"] = model
        client = object()
        return client, model

    monkeypatch.setattr(aux, "resolve_provider_client", fake_resolve)
    monkeypatch.setattr(
        aux, "_try_main_provider_route",
        lambda *_args, **_kwargs: ("MAIN", "qwen", "custom"),
    )

    client, model, label = aux._resolve_auto_route(_main_runtime(), task="compression")

    assert seen == {"provider": "minimax-oauth", "model": "MiniMax-M3"}
    assert model == "MiniMax-M3"
    assert label == "minimax-oauth"
    assert client != "MAIN"


def test_auto_compression_provider_still_uses_main(monkeypatch):
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: {"provider": "auto"})

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("explicit resolve should not run")

    monkeypatch.setattr(aux, "resolve_provider_client", fail_resolve)
    monkeypatch.setattr(
        aux, "_try_main_provider_route",
        lambda *_args, **_kwargs: ("MAIN", "qwen", "custom"),
    )

    client, model, label = aux._resolve_auto_route(_main_runtime(), task="compression")

    assert (client, model, label) == ("MAIN", "qwen", "custom")


def test_unusable_explicit_provider_falls_through_to_main(monkeypatch):
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: {
        "provider": "minimax-oauth",
        "model": "MiniMax-M3",
    })
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(
        aux, "_try_main_provider_route",
        lambda *_args, **_kwargs: ("MAIN", "qwen", "custom"),
    )

    client, model, label = aux._resolve_auto_route(_main_runtime(), task="compression")

    assert (client, model, label) == ("MAIN", "qwen", "custom")
