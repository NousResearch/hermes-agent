"""Env-backed pool rows outside the registry tuple are hydrated on load.

``env:VAR`` rows persist without their secret and are re-hydrated by
``_seed_from_env`` on every ``load_pool``.  Before this fix only the
registry's declared ``api_key_env_vars`` were consulted, so a user's second
key (``source: env:PROVIDER_API_KEY_2``) stayed empty and was silently
dropped from round-robin by ``_available_entries`` (salvage of PR #103067).
"""

from __future__ import annotations

import json

import pytest

SYN_PRIMARY = "syn-primary-" + "a" * 24
SYN_SECONDARY = "syn-secondary-" + "b" * 24


def _write_home(home, provider, env_file: dict, extra_var: str):
    (home / ".env").write_text("".join(f"{k}={v}\n" for k, v in env_file.items()), encoding="utf-8")
    (home / "config.yaml").write_text(f"credential_pool_strategies:\n  {provider}: round_robin\n", encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {provider: [
        {"id": "x2", "label": extra_var, "auth_type": "api_key", "priority": 1, "source": f"env:{extra_var}"},
    ]}}), encoding="utf-8")
    from hermes_cli.config import invalidate_env_cache
    invalidate_env_cache()


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_2", "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"):
        monkeypatch.delenv(var, raising=False)
    return home


# openrouter has its own early-return branch in _seed_from_env; deepseek is a
# plain registry provider. Both must honour the on-disk env row.
@pytest.mark.parametrize("provider,primary", [("deepseek", "DEEPSEEK_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")])
def test_on_disk_env_row_outside_registry_tuple_joins_rotation(home, provider, primary):
    from agent.credential_pool import load_pool

    extra = f"{primary}_2"
    _write_home(home, provider, {primary: SYN_PRIMARY, extra: SYN_SECONDARY}, extra)

    pool = load_pool(provider)
    available, _ = pool._available_entries()
    assert sorted(e.runtime_api_key for e in available) == sorted([SYN_PRIMARY, SYN_SECONDARY])
    assert {pool.select().source for _ in range(4)} == {f"env:{primary}", f"env:{extra}"}

    # Hydration is runtime-only: the secret never lands in auth.json.
    rows = json.loads((home / "auth.json").read_text(encoding="utf-8"))["credential_pool"][provider]
    assert all("access_token" not in row for row in rows)


def test_unset_env_row_stays_out_of_rotation_and_on_disk(home):
    from agent.credential_pool import load_pool

    _write_home(home, "deepseek", {"DEEPSEEK_API_KEY": SYN_PRIMARY}, "DEEPSEEK_API_KEY_2")

    pool = load_pool("deepseek")
    available, _ = pool._available_entries()
    assert [e.source for e in available] == ["env:DEEPSEEK_API_KEY"]
    # load_pool() is a non-destructive read for env rows (#9331): the
    # reference survives for the process that does have the var.
    assert "env:DEEPSEEK_API_KEY_2" in {e.source for e in pool._entries}


# Numbered siblings need no auth.json row at all: setting the variable is the
# whole opt-in (#76593). Discovery stops at the first gap so a leftover _5 does
# not silently enter rotation.
@pytest.mark.parametrize("provider,primary", [("deepseek", "DEEPSEEK_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")])
def test_numbered_env_siblings_seed_rotation_without_config(home, provider, primary, monkeypatch):
    from agent.credential_pool import load_pool

    for n in (3, 5):
        monkeypatch.delenv(f"{primary}_{n}", raising=False)
    (home / ".env").write_text(
        f"{primary}={SYN_PRIMARY}\n{primary}_2={SYN_SECONDARY}\n{primary}_3=syn-third-{'c' * 24}\n{primary}_5=syn-fifth-{'e' * 24}\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(f"credential_pool_strategies:\n  {provider}: round_robin\n", encoding="utf-8")
    from hermes_cli.config import invalidate_env_cache
    invalidate_env_cache()

    pool = load_pool(provider)
    available, _ = pool._available_entries()
    assert {e.source for e in available} == {f"env:{primary}", f"env:{primary}_2", f"env:{primary}_3"}
    assert len({pool.select().source for _ in range(6)}) == 3
    rows = json.loads((home / "auth.json").read_text(encoding="utf-8"))["credential_pool"][provider]
    assert len(rows) == 3 and all("access_token" not in row for row in rows)
