"""Regression for #126768: unscoped pool.select blocks Codex aux calls.

When a Codex credential carries a per-model entitlement cooldown (e.g. for
gpt-5.4-mini after a 400), an unscoped ``pool.select()`` treats ANY active
model cooldown as blocking (see ``model_cooldown_until(entry, None)``), so
auxiliary tasks for entitled models (e.g. gpt-5.4) fail with
'no credentials found'. The fix threads the resolved model through
``pool.select(model=...)``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import agent.auxiliary_client as aux
import agent.auxiliary_model_scope as aux_model_scope
from hermes_cli.auth import write_credential_pool


COOLED_MODEL = "gpt-5.4-mini"
ENTITLED_MODEL = "gpt-5.4"


@pytest.fixture
def codex_pool_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    # No singleton fallback: pool is the only authority.
    monkeypatch.setattr(aux, "_read_codex_singleton_token", lambda: None)
    monkeypatch.setattr(aux, "_codex_base_url_override", lambda: "")
    # Avoid real SDK construction; CodexAuxiliaryClient only stores the client.
    monkeypatch.setattr(
        aux,
        "_create_openai_client",
        lambda api_key, base_url, **kwargs: SimpleNamespace(
            api_key=api_key, base_url=base_url
        ),
    )
    write_credential_pool(
        "openai-codex",
        [
            {
                "id": "cred-1",
                "label": "acct-1",
                "auth_type": "oauth",
                "priority": 0,
                "source": "manual",
                "access_token": "codex-token-1",
                "refresh_token": "rt-1",
                "expires_at": time.time() + 3600,
                "model_cooldowns": {COOLED_MODEL: time.time() + 600},
            }
        ],
    )
    return home


def test_entitled_model_selects_despite_sibling_cooldown(codex_pool_home):
    pool_present, entry = aux_model_scope._select_pool_entry("openai-codex", model=ENTITLED_MODEL)
    assert pool_present is True
    assert entry is not None
    assert aux._pool_runtime_api_key(entry) == "codex-token-1"

    token, base_url = aux._resolve_codex_credential_and_base(model=ENTITLED_MODEL)
    assert token == "codex-token-1"
    assert base_url

    client, model = aux._build_codex_client(ENTITLED_MODEL)
    assert client is not None
    assert model == ENTITLED_MODEL


def test_cooled_down_model_does_not_select(codex_pool_home):
    pool_present, entry = aux_model_scope._select_pool_entry("openai-codex", model=COOLED_MODEL)
    assert pool_present is True
    assert entry is None

    token, _base = aux._resolve_codex_credential_and_base(model=COOLED_MODEL)
    assert not token

    # Old unscoped behaviour: pool.select() with no model stays conservative
    # and treats any active model cooldown as blocking.
    pool_present, entry = aux_model_scope._select_pool_entry("openai-codex")
    assert pool_present is True
    assert entry is None


def test_raw_codex_branch_passes_model_scoping(codex_pool_home):
    client, model = aux.resolve_provider_client(
        "openai-codex", model=ENTITLED_MODEL, raw_codex=True
    )
    assert client is not None
    assert model == ENTITLED_MODEL

    client, _model = aux.resolve_provider_client(
        "openai-codex", model=COOLED_MODEL, raw_codex=True
    )
    assert client is None


def test_backward_compat_when_model_is_none(codex_pool_home, monkeypatch):
    # _select_pool_entry without a model stays unscoped (conservative).
    pool_present, entry = aux_model_scope._select_pool_entry("openai-codex", model=None)
    assert pool_present is True
    assert entry is None

    # A 1-arg monkeypatched _select_pool_entry still works via fallback.
    real_select = aux_model_scope._select_pool_entry
    monkeypatch.setattr(aux_model_scope, "_select_pool_entry", lambda provider: (False, None))
    token, _base = aux._resolve_codex_credential_and_base(model=ENTITLED_MODEL)
    assert token is None

    # A mock pool whose select() takes no kwargs falls back cleanly.
    monkeypatch.setattr(aux_model_scope, "_select_pool_entry", real_select)

    class _NoKwargPool:
        def has_credentials(self):
            return True

        def select(self):
            return SimpleNamespace(
                runtime_api_key="tok", runtime_base_url="https://example.test"
            )

    monkeypatch.setattr(
        aux, "_load_pool_with_credentials", lambda provider, note="": _NoKwargPool()
    )
    pool_present, entry = aux_model_scope._select_pool_entry("openai-codex", model=ENTITLED_MODEL)
    assert pool_present is True
    assert entry is not None


def test_scoped_select_internal_typeerror_does_not_silently_downgrade(
    codex_pool_home, monkeypatch
):
    """A TypeError from *inside* the scoped select is a real bug: it must not trigger
    a silent unscoped retry that re-leases the refused credential (#130053)."""
    real_select = aux_model_scope._select_pool_entry
    monkeypatch.setattr(aux_model_scope, "_select_pool_entry", real_select)

    class _ExplodingPool:
        def has_credentials(self):
            return True

        def select(self, model=None):
            raise TypeError("simulated bug inside scoped select")

    monkeypatch.setattr(
        aux, "_load_pool_with_credentials", lambda provider, note="": _ExplodingPool()
    )
    pool_present, entry = aux_model_scope._select_pool_entry("openai-codex", model=ENTITLED_MODEL)
    assert pool_present is True
    assert entry is None


def test_legacy_pool_without_model_kwarg_falls_back_with_warning(
    codex_pool_home, monkeypatch, caplog
):
    """A legacy pool whose select() takes no model kwarg still works, but loudly (#130053)."""
    real_select = aux_model_scope._select_pool_entry
    monkeypatch.setattr(aux_model_scope, "_select_pool_entry", real_select)

    class _NoKwargPool:
        def has_credentials(self):
            return True

        def select(self):
            return SimpleNamespace(
                runtime_api_key="tok", runtime_base_url="https://example.test"
            )

    monkeypatch.setattr(
        aux, "_load_pool_with_credentials", lambda provider, note="": _NoKwargPool()
    )
    with caplog.at_level("WARNING", logger=aux.logger.name):
        pool_present, entry = aux_model_scope._select_pool_entry("openai-codex", model=ENTITLED_MODEL)
    assert pool_present is True
    assert entry is not None
    assert any("does not accept a model scope" in r.message for r in caplog.records)


def test_compression_model_selects_despite_sibling_cooldown(codex_pool_home, monkeypatch):
    """Regression for #125695 on the auxiliary.compression path (#126768).

    The compression task builds its Codex client from
    ``auxiliary.compression.model``: ``get_text_auxiliary_client("compression")``
    resolves that model, then funnels it through ``_build_codex_client(model)``
    -> ``_resolve_codex_credential_and_base(model=...)`` ->
    ``_select_pool_entry("openai-codex", model=...)``. With an active cooldown
    on an unrelated model (COOLED_MODEL), the scoped compression-model select
    must still return the healthy credential, while unscoped selection stays
    conservative and blocks (documenting the mechanism).
    """
    monkeypatch.setattr(
        aux,
        "_get_auxiliary_task_config",
        lambda task: (
            {"provider": "openai-codex", "model": ENTITLED_MODEL}
            if task == "compression"
            else {}
        ),
    )
    client, model = aux.get_text_auxiliary_client("compression")
    assert client is not None
    assert model == ENTITLED_MODEL

    pool_present, entry = aux_model_scope._select_pool_entry("openai-codex")
    assert pool_present is True
    assert entry is None
