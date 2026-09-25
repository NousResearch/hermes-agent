"""A transient auxiliary billing/quota error must not take main chat down with it. (#119533)

The auxiliary path (compression / vision / titles) and the main turn's client init share one
credential pool. When the pool holds a single OpenRouter entry, auxiliary recovery had nothing
to rotate to yet still benched that entry, so every later main turn resolved no client and the
gateway raised ``No LLM provider configured`` until it was restarted.

Invariant tests: they drive the real ``_recover_provider_pool`` and the real
``resolve_provider_client`` router against a temp ``HERMES_HOME`` and assert the relation
"auxiliary recovery may not leave the owning main route without a usable credential" — not any
particular log line or cooldown constant. The sibling suites cover the narrower units; this one
covers the aux-burn -> main-init hand-off that the issue actually reports.
"""

from __future__ import annotations

import json

import pytest

_SOLE_KEY = "sk-or-test-sole"


class _AuxError(Exception):
    def __init__(self, message: str, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _write_pool(tmp_path, monkeypatch, keys) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"openrouter": [
            {
                "id": f"cred-{index}",
                "label": f"cred-{index}",
                "auth_type": "api_key",
                "priority": index,
                "source": "manual",
                "access_token": key,
                "base_url": "https://openrouter.ai/api/v1",
            }
            for index, key in enumerate(keys)
        ]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.mark.parametrize(
    "message, status_code",
    [
        ("quota exceeded: too many tokens per day, retry later", None),
        ("402 Payment Required: insufficient credits", 402),
        ("rate limit exceeded, please slow down", 429),
    ],
)
def test_aux_quota_failure_leaves_main_turn_a_usable_client(
    tmp_path, monkeypatch, message, status_code
):
    from agent.auxiliary_client import _recover_provider_pool, resolve_provider_client
    from agent.credential_pool import load_pool

    _write_pool(tmp_path, monkeypatch, [_SOLE_KEY])

    # The auxiliary task fails and tries same-provider recovery. With a sole credential there
    # is nothing to rotate to, so recovery reports failure (the caller then falls back to
    # another provider for the side task) WITHOUT burning the shared entry.
    assert _recover_provider_pool(
        "openrouter", _AuxError(message, status_code), failed_api_key=_SOLE_KEY,
    ) is False

    # The invariant: main-turn routing still resolves a client from the same pool.
    assert load_pool("openrouter").select() is not None
    client, _model = resolve_provider_client("openrouter", "z-ai/glm-5.3-flash")
    assert client is not None, "main turn lost its client to an auxiliary failure"


def test_aux_quota_failure_still_rotates_when_a_sibling_key_exists(tmp_path, monkeypatch):
    """Real exhaustion keeps working: with a distinct healthy key, benching the failed one
    is useful and must still happen."""
    from agent.auxiliary_client import _recover_provider_pool
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    _write_pool(tmp_path, monkeypatch, ["failed-key", "healthy-key"])

    assert _recover_provider_pool(
        "openrouter", _AuxError("quota exceeded", 429), failed_api_key="failed-key",
    ) is True

    entries = {entry.runtime_api_key: entry for entry in load_pool("openrouter").entries()}
    assert entries["failed-key"].last_status == STATUS_EXHAUSTED
    selected = load_pool("openrouter").select()
    assert selected is not None and selected.runtime_api_key == "healthy-key"


def test_main_loop_exhaustion_of_a_sole_credential_still_works(tmp_path, monkeypatch):
    """The guard is scoped to the auxiliary path: the main loop's own failure handling keeps
    benching a sole credential, so a genuinely spent key is still taken out of rotation."""
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    _write_pool(tmp_path, monkeypatch, [_SOLE_KEY])

    pool = load_pool("openrouter")
    assert pool.mark_exhausted_and_rotate(
        status_code=402, error_context={"message": "insufficient credits"},
        api_key_hint=_SOLE_KEY,
    ) is None
    assert load_pool("openrouter").entries()[0].last_status == STATUS_EXHAUSTED
