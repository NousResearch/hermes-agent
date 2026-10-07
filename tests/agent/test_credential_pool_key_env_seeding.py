"""User-declared ``providers:`` entries seed the pool from ``key_env`` (#125436).

``_seed_from_env`` returns early for providers absent from
``PROVIDER_REGISTRY`` — which every user-declared endpoint is — so their
``key_env`` credential never entered the pool and the main chat path
(pool as sole source) called out key-less → 401 → silent fallback to
``fallback_providers``.
"""

from __future__ import annotations

import json

import pytest

SYN_KEY = "syn-tokenmall-" + "k" * 24
KEY_ENV = "HERMES_CUSTOM_TOKENMALL_API_KEY"
BASE_URL = "https://tokenmall.example.com/v1"


def _write_home(home, *, env_file=None):
    (home / ".env").write_text(
        "".join(f"{k}={v}\n" for k, v in (env_file or {}).items()), encoding="utf-8"
    )
    (home / "config.yaml").write_text(
        f"providers:\n"
        f"  tokenmall:\n"
        f"    base_url: {BASE_URL}\n"
        f"    api_mode: chat_completions\n"
        f"    key_env: {KEY_ENV}\n",
        encoding="utf-8",
    )
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(KEY_ENV, raising=False)
    return home


def test_key_env_provider_seeds_pool(home):
    from agent.credential_pool import load_pool

    _write_home(home, env_file={KEY_ENV: SYN_KEY})

    pool = load_pool("tokenmall")
    available, _ = pool._available_entries()
    assert [e.source for e in available] == [f"env:{KEY_ENV}"]
    assert [e.runtime_api_key for e in available] == [SYN_KEY]
    assert [e.runtime_base_url for e in available] == [BASE_URL]

    # Env-backed rows persist without their secret (same as registry providers).
    rows = json.loads((home / "auth.json").read_text(encoding="utf-8"))["credential_pool"]["tokenmall"]
    assert all("access_token" not in row for row in rows)


def test_unset_key_env_leaves_pool_empty(home):
    from agent.credential_pool import load_pool

    _write_home(home)  # key_env declared, variable absent from .env and environ

    pool = load_pool("tokenmall")
    available, _ = pool._available_entries()
    assert available == []


def test_registry_provider_still_seeds_normally(home, monkeypatch):
    """The registry branch keeps its behaviour: env var present → seeded."""
    from agent.credential_pool import load_pool

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (home / ".env").write_text(f"DEEPSEEK_API_KEY={SYN_KEY}\n", encoding="utf-8")
    (home / "config.yaml").write_text("", encoding="utf-8")
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()

    pool = load_pool("deepseek")
    available, _ = pool._available_entries()
    assert [e.source for e in available] == ["env:DEEPSEEK_API_KEY"]
