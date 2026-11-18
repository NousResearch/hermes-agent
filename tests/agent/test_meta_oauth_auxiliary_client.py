"""Explicit meta-oauth auxiliary routes must build a Responses client.

Regression: meta-oauth fell through to the generic oauth_external plugin
arm, whose profile-supplied client is None by design, so every explicitly
configured auxiliary task silently returned (None, None) and fell back.
"""

import pytest


def test_meta_oauth_aux_client_uses_subscription_credentials(monkeypatch):
    from agent import auxiliary_client as aux

    monkeypatch.setattr(
        aux, "_resolve_meta_oauth_for_aux",
        lambda *_args: ("fake-key", "https://api.meta.ai/v1"),
    )
    client, model = aux.resolve_provider_client("meta-oauth", "muse-ember-1.0")
    assert model == "muse-ember-1.0"
    assert isinstance(client, aux.CodexAuxiliaryClient)


def test_meta_oauth_aux_client_without_model_fails_closed(monkeypatch):
    from agent import auxiliary_client as aux

    called = False

    def _unexpected():
        nonlocal called
        called = True
        return ("fake-key", "https://api.meta.ai/v1")

    monkeypatch.setattr(aux, "_resolve_meta_oauth_for_aux", _unexpected)
    client, model = aux._build_meta_oauth_aux_client("")
    assert (client, model) == (None, None)
    assert called is False
