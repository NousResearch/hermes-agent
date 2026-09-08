"""Port of ba71e00 env-source hydration onto current main.

When an existing pool entry on disk has source ``env:VAR`` and VAR resolves
through Hermes normal environment/secret-scope resolver, it must be hydrated
exactly as the registry's single declared env source is. This keeps
multi-key round_robin working for OpenCode Go-style pools without
per-provider registry code.

See task t_c46cb1ec — narrow forward-port of historical fix
ba71e00db07c6263ee8d44b27dfbce2a92e6b39c (source d83e6fe3e4).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# Synthetic values — never real secrets, never key-shaped literals that trip scanners.
SYN_PRIMARY = "syn-primary-" + "a" * 24
SYN_SECONDARY = "syn-secondary-" + "b" * 24
SYN_TERTIARY = "syn-tertiary-" + "c" * 24
SYN_MANUAL = "syn-manual-" + "d" * 24
SYN_EXTRA = "syn-extra-" + "e" * 24


def _write_env_file(home: Path, **env_vars):
    """Write a .env file under HERMES_HOME and invalidate the memo."""
    home.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in env_vars.items()]
    (home / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from hermes_cli.config import invalidate_env_cache
    invalidate_env_cache()


def _make_pconfig(provider_id: str, env_vars: list[str]):
    """Minimal ProviderConfig for testing."""
    from hermes_cli.auth import ProviderConfig
    return ProviderConfig(
        id=provider_id,
        name=provider_id.title(),
        auth_type="api_key",
        api_key_env_vars=tuple(env_vars),
    )


def _write_auth(home: Path, pool: dict):
    """Write auth.json credential_pool payload."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({"credential_pool": pool}), encoding="utf-8")


def _read_auth(home: Path) -> dict:
    return json.loads((home / "auth.json").read_text(encoding="utf-8"))


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with .env cache cleared and credential envs blanked."""
    home = tmp_path / ".hermes"
    home.mkdir()
    # Override the auto hermetic home — our fixture owns HERMES_HOME for these tests
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import invalidate_env_cache
    invalidate_env_cache()
    # Ensure the credential-shaped envs from conftest are blank (they are by default)
    # Clear generic placeholders explicitly; also clear any remaining
    # credential-shaped variables via bounded suffix/pattern without naming providers.
    for key in [
        "PROVIDER_API_KEY",
        "PROVIDER_API_KEY_2",
        "PROVIDER_API_KEY_3",
        "PROVIDER_API_KEY_4",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key in list(os.environ.keys()):
        if "_API_KEY" in key or key.endswith("_TOKEN") or key.endswith("_BASE_URL"):
            monkeypatch.delenv(key, raising=False)
    # Guarantee no stray secret scope from prior test
    try:
        from agent.secret_scope import set_secret_scope, set_multiplex_active
        # Clear scope, disable multiplex so get_secret fallback is predictable
        set_multiplex_active(False)
        # ensure no scope installed
        from agent.secret_scope import _SECRET_SCOPE
        if _SECRET_SCOPE.get() is not None:
            tok = set_secret_scope(None)
            # reset immediately — leave clean
            from agent.secret_scope import reset_secret_scope
            reset_secret_scope(tok)
    except Exception:
        pass
    return home


class TestSeedFromEnvRespectsExistingPoolEntries:
    """Regression: env-source entries already in auth.json must be seeded."""

    def test_second_env_source_entry_seeded_from_env(self, isolated_hermes_home):
        """An env:PROVIDER_API_KEY_2 entry already in the pool gets
        populated when the env var is set, even though the registry only
        declares the primary PROVIDER_API_KEY."""
        from agent.credential_pool import PooledCredential, _seed_from_env

        _write_env_file(
            isolated_hermes_home,
            PROVIDER_API_KEY=SYN_PRIMARY,
            PROVIDER_API_KEY_2=SYN_SECONDARY,
        )

        pconfig = _make_pconfig("opencode-go", ["PROVIDER_API_KEY"])

        secondary = PooledCredential(
            provider="opencode-go",
            id="sec1234",
            label="secondary",
            auth_type="api_key",
            priority=1,
            source="env:PROVIDER_API_KEY_2",
            access_token="",
        )
        entries = [secondary]

        with patch(
            "agent.credential_pool.PROVIDER_REGISTRY",
            {"opencode-go": pconfig},
        ):
            changed, active_sources = _seed_from_env("opencode-go", entries)

        assert changed is True
        assert "env:PROVIDER_API_KEY" in active_sources
        assert "env:PROVIDER_API_KEY_2" in active_sources

        populated = next((e for e in entries if e.source == "env:PROVIDER_API_KEY_2"), None)
        assert populated is not None, "secondary entry was dropped from pool"
        assert populated.access_token == SYN_SECONDARY

    def test_three_env_source_rows_hydrate_when_only_primary_in_registry(self, isolated_hermes_home):
        """All three OpenCode Go-style env:VAR rows hydrate when only the
        primary variable is in the registry tuple."""
        from agent.credential_pool import PooledCredential, _seed_from_env

        _write_env_file(
            isolated_hermes_home,
            PROVIDER_API_KEY=SYN_PRIMARY,
            PROVIDER_API_KEY_2=SYN_SECONDARY,
            PROVIDER_API_KEY_3=SYN_TERTIARY,
        )
        pconfig = _make_pconfig("opencode-go", ["PROVIDER_API_KEY"])

        # Three pre-existing rows on disk, all empty before seeding
        e1 = PooledCredential(provider="opencode-go", id="pri1", label="primary", auth_type="api_key", priority=0, source="env:PROVIDER_API_KEY", access_token="")
        e2 = PooledCredential(provider="opencode-go", id="sec2", label="secondary", auth_type="api_key", priority=1, source="env:PROVIDER_API_KEY_2", access_token="")
        e3 = PooledCredential(provider="opencode-go", id="ter3", label="tertiary", auth_type="api_key", priority=2, source="env:PROVIDER_API_KEY_3", access_token="")
        entries = [e1, e2, e3]

        with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
            changed, active_sources = _seed_from_env("opencode-go", entries)

        assert changed is True
        assert "env:PROVIDER_API_KEY" in active_sources
        assert "env:PROVIDER_API_KEY_2" in active_sources
        assert "env:PROVIDER_API_KEY_3" in active_sources
        for src, expected in [("env:PROVIDER_API_KEY", SYN_PRIMARY), ("env:PROVIDER_API_KEY_2", SYN_SECONDARY), ("env:PROVIDER_API_KEY_3", SYN_TERTIARY)]:
            ent = next((e for e in entries if e.source == src), None)
            assert ent is not None, f"missing {src}"
            assert ent.access_token == expected
            assert ent.runtime_api_key == expected

    def test_round_robin_cycles_across_hydrated_rows(self, tmp_path, monkeypatch):
        """All hydrated rows are available and round_robin cycles across them."""
        # This test uses load_pool integration + explicit strategy mock,
        # so it exercises the full persist/sanitize/selection pipeline.
        home = tmp_path / "hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        from hermes_cli.config import invalidate_env_cache
        invalidate_env_cache()

        # Write .env with three synthetic keys
        _write_env_file(home, PROVIDER_API_KEY=SYN_PRIMARY, PROVIDER_API_KEY_2=SYN_SECONDARY, PROVIDER_API_KEY_3=SYN_TERTIARY)

        # Pre-create auth.json with three empty env-source rows on disk
        _write_auth(home, {
            "opencode-go": [
                {"id": "pri1", "label": "primary", "auth_type": "api_key", "priority": 0, "source": "env:PROVIDER_API_KEY", "access_token": ""},
                {"id": "sec2", "label": "secondary", "auth_type": "api_key", "priority": 1, "source": "env:PROVIDER_API_KEY_2", "access_token": ""},
                {"id": "ter3", "label": "tertiary", "auth_type": "api_key", "priority": 2, "source": "env:PROVIDER_API_KEY_3", "access_token": ""},
            ]
        })

        # Force round_robin for opencode-go (simpler than config.yaml file)
        monkeypatch.setattr("agent.credential_pool.get_pool_strategy", lambda provider: "round_robin" if provider == "opencode-go" else "fill_first")

        # Use real provider registry for opencode-go (single var) — do not mock,
        # so we prove the fix bridges the gap between registry tuple and disk rows.
        from agent.credential_pool import load_pool
        pool = load_pool("opencode-go")
        entries = pool.entries()
        # All three should be available (runtime_api_key non-empty)
        available = [e for e in entries if e.runtime_api_key]
        assert len(available) == 3, f"expected 3 available, got {[e.source for e in available]!r}"
        sources = {e.source for e in available}
        assert sources == {"env:PROVIDER_API_KEY", "env:PROVIDER_API_KEY_2", "env:PROVIDER_API_KEY_3"}
        tokens = {e.runtime_api_key for e in available}
        assert SYN_PRIMARY in tokens and SYN_SECONDARY in tokens and SYN_TERTIARY in tokens

        # Selection must cycle — with 3 available and round_robin, 6 selects
        # should return each key at least once and not collapse to a single key.
        seen_ids = []
        seen_tokens = set()
        for _ in range(6):
            ent = pool.select()
            assert ent is not None, "select returned None with available entries"
            seen_ids.append(ent.id)
            seen_tokens.add(ent.runtime_api_key)
        # All three keys must have been selected at least once over 6 rounds
        assert SYN_PRIMARY in seen_tokens
        assert SYN_SECONDARY in seen_tokens
        assert SYN_TERTIARY in seen_tokens
        # And the id sequence must not be constant (not degraded to single-key)
        assert len(set(seen_ids)) > 1
        # Round robin with 3 entries cycles; with 6 picks each appears twice in some order.
        # At minimum verify we saw at least 3 distinct ids over the window.
        assert len(set(seen_ids)) == 3

    def test_dotenv_and_secret_scope_resolution_used(self, tmp_path, monkeypatch):
        """Normal .env and active secret-scope resolution is used; no os.environ bypass."""
        from agent.credential_pool import PooledCredential, _seed_from_env
        from agent.secret_scope import set_secret_scope, set_multiplex_active, get_secret
        from hermes_cli.config import invalidate_env_cache

        home = tmp_path / "hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        invalidate_env_cache()

        # Part A: .env file path works (no monkeypatched env needed)
        _write_env_file(home, PROVIDER_API_KEY_2=SYN_SECONDARY)
        pconfig = _make_pconfig("opencode-go", ["PROVIDER_API_KEY"])
        e = PooledCredential(provider="opencode-go", id="sec2", label="secondary", auth_type="api_key", priority=1, source="env:PROVIDER_API_KEY_2", access_token="")
        entries = [e]
        with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
            changed, _ = _seed_from_env("opencode-go", entries)
        assert changed is True
        assert entries[0].access_token == SYN_SECONDARY

        # Part B: secret-scope path — when multiplexing is active, the value
        # comes from the installed scope, not from .env/os.environ.
        # Start clean: remove .env, clear os.environ var
        (home / ".env").write_text("", encoding="utf-8")
        invalidate_env_cache()
        monkeypatch.delenv("PROVIDER_API_KEY_2", raising=False)
        # Ensure .env does not contain the key, and os.environ does not either
        assert get_secret("PROVIDER_API_KEY_2", "") in ("", None)  # no scope yet, no env -> empty (multiplex inactive atm)
        # Activate multiplex and install a scope that DOES contain the key
        set_multiplex_active(True)
        scope = {"PROVIDER_API_KEY_2": SYN_SECONDARY, "PROVIDER_API_KEY": SYN_PRIMARY}
        tok = set_secret_scope(scope)
        try:
            # Now get_env_prefer_dotenv should resolve via scope
            from agent.credential_pool import get_env_prefer_dotenv
            assert get_env_prefer_dotenv("PROVIDER_API_KEY_2") == SYN_SECONDARY
            # Seed again with fresh entry (existing token empty)
            e2 = PooledCredential(provider="opencode-go", id="sec2b", label="secondary", auth_type="api_key", priority=1, source="env:PROVIDER_API_KEY_2", access_token="")
            entries2 = [e2]
            with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
                changed2, _ = _seed_from_env("opencode-go", entries2)
            assert changed2 is True
            assert entries2[0].access_token == SYN_SECONDARY
        finally:
            from agent.secret_scope import reset_secret_scope
            reset_secret_scope(tok)
            set_multiplex_active(False)
            invalidate_env_cache()

        # Part C: no direct os.environ bypass — when multiplexing is active,
        # a value that lives ONLY in os.environ (not in scope/.env) must NOT be used.
        # This proves we route through get_secret, not raw os.environ.get.
        _write_env_file(home, **{})  # empty .env
        # Put the secret ONLY in real os.environ (bypass the scope)
        os.environ["PROVIDER_API_KEY_2"] = SYN_EXTRA
        set_multiplex_active(True)
        tok2 = set_secret_scope({"PROVIDER_API_KEY": SYN_PRIMARY})  # scope lacks KEY2
        try:
            from agent.credential_pool import get_env_prefer_dotenv
            # Should NOT see the os.environ value because multiplex-active + scope miss is fail-closed
            resolved = get_env_prefer_dotenv("PROVIDER_API_KEY_2")
            assert resolved == "" or resolved is None or resolved == "", f"expected empty, got {resolved!r} — direct os.environ bypass!"
            e3 = PooledCredential(provider="opencode-go", id="sec3", label="secondary", auth_type="api_key", priority=1, source="env:PROVIDER_API_KEY_2", access_token="")
            entries3 = [e3]
            with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
                changed3, active3 = _seed_from_env("opencode-go", entries3)
            # Secondary must remain empty — proves no direct os.environ bypass.
            # Primary may be seeded (it is in scope) even when secondary is not.
            assert "env:PROVIDER_API_KEY_2" not in active3, f"secondary should not be in active_sources, got {active3!r}"
            # Find secondary entry — it should still be empty
            sec_entry = next((e for e in entries3 if e.source == "env:PROVIDER_API_KEY_2"), None)
            assert sec_entry is not None
            assert sec_entry.access_token == "", f"secondary leaked via os.environ bypass: {sec_entry.access_token!r}"
        finally:
            reset_secret_scope(tok2)
            set_multiplex_active(False)
            os.environ.pop("PROVIDER_API_KEY_2", None)
            invalidate_env_cache()

    def test_duplicate_registry_source_rows_do_not_multiply(self, isolated_hermes_home):
        """If the registry declares a VAR and the pool also has it, no duplicate is created."""
        from agent.credential_pool import PooledCredential, _seed_from_env
        _write_env_file(isolated_hermes_home, PROVIDER_API_KEY=SYN_PRIMARY, PROVIDER_API_KEY_2=SYN_SECONDARY)
        pconfig = _make_pconfig("opencode-go", ["PROVIDER_API_KEY"])
        existing = PooledCredential(provider="opencode-go", id="prim1234", label="primary", auth_type="api_key", priority=0, source="env:PROVIDER_API_KEY", access_token="")
        e2 = PooledCredential(provider="opencode-go", id="sec1234", label="secondary", auth_type="api_key", priority=1, source="env:PROVIDER_API_KEY_2", access_token="")
        # Also add a duplicate of the secondary source (simulating a corrupted pool with two same-source rows)
        dup = PooledCredential(provider="opencode-go", id="dup999", label="duplicate-secondary", auth_type="api_key", priority=2, source="env:PROVIDER_API_KEY_2", access_token="")
        entries = [existing, e2, dup]
        with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
            changed, active_sources = _seed_from_env("opencode-go", entries)
        assert changed is True
        assert "env:PROVIDER_API_KEY" in active_sources
        assert "env:PROVIDER_API_KEY_2" in active_sources
        # No extra env var entries — env:PROVIDER_API_KEY_2 appears once in active_sources
        # and the pool should have been de-duplicated to one entry per source by _upsert logic.
        # Count entries per source after seeding
        deduped_sources = [e.source for e in entries]
        # _upsert_entry de-duplicates same-source rows, keeping first; second duplicate should have been removed on first upsert that touched that source
        # So we expect at most 2 entries (primary + secondary), not 3
        assert deduped_sources.count("env:PROVIDER_API_KEY_2") == 1, f"duplicate source not deduped: {deduped_sources!r}"
        assert deduped_sources.count("env:PROVIDER_API_KEY") == 1

    def test_manual_source_entry_not_touched_by_seed(self, isolated_hermes_home):
        """Manual entries in auth.json must not be re-seeded from env vars."""
        from agent.credential_pool import PooledCredential, _seed_from_env
        _write_env_file(isolated_hermes_home, PROVIDER_API_KEY=SYN_PRIMARY)
        pconfig = _make_pconfig("opencode-go", ["PROVIDER_API_KEY"])
        manual = PooledCredential(provider="opencode-go", id="man1234", label="manual", auth_type="api_key", priority=0, source="manual", access_token=SYN_MANUAL)
        entries = [manual]
        with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
            _seed_from_env("opencode-go", entries)
        populated = next((e for e in entries if e.source == "manual"), None)
        assert populated is not None
        assert populated.access_token == SYN_MANUAL

    def test_env_source_entry_with_no_env_value_unchanged(self, isolated_hermes_home):
        """An env-source entry whose env var is unset stays empty and unavailable."""
        from agent.credential_pool import PooledCredential, _seed_from_env
        pconfig = _make_pconfig("opencode-go", ["PROVIDER_API_KEY"])
        secondary = PooledCredential(provider="opencode-go", id="sec1234", label="secondary", auth_type="api_key", priority=1, source="env:PROVIDER_API_KEY_2", access_token="")
        entries = [secondary]
        with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
            changed, _ = _seed_from_env("opencode-go", entries)
        assert changed is False
        populated = next((e for e in entries if e.source == "env:PROVIDER_API_KEY_2"), None)
        assert populated is not None
        assert populated.access_token == ""
        # Also verify it is considered unavailable (filtered by _available_entries)
        from agent.credential_pool import CredentialPool
        pool = CredentialPool(provider="opencode-go", entries=entries)
        avail, _ = pool._available_entries()
        assert len(avail) == 0, "empty env entry should be filtered as unavailable"

    def test_raw_secrets_not_written_to_auth_json(self, tmp_path, monkeypatch):
        """Raw synthetic secrets remain runtime-only; auth.json holds only fingerprint metadata."""
        home = tmp_path / "hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        from hermes_cli.config import invalidate_env_cache
        invalidate_env_cache()
        _write_env_file(home, PROVIDER_API_KEY=SYN_PRIMARY, PROVIDER_API_KEY_2=SYN_SECONDARY, PROVIDER_API_KEY_3=SYN_TERTIARY)
        _write_auth(home, {
            "opencode-go": [
                {"id": "pri1", "label": "primary", "auth_type": "api_key", "priority": 0, "source": "env:PROVIDER_API_KEY", "access_token": ""},
                {"id": "sec2", "label": "secondary", "auth_type": "api_key", "priority": 1, "source": "env:PROVIDER_API_KEY_2", "access_token": ""},
                {"id": "ter3", "label": "tertiary", "auth_type": "api_key", "priority": 2, "source": "env:PROVIDER_API_KEY_3", "access_token": ""},
            ]
        })
        # Also test dedup + manual mix in same file — manual should keep its token on disk (it is not borrowed)
        # Add a manual provider entry in same file to ensure isolation
        from agent.credential_pool import load_pool
        # Force strategy so load doesn't change selection expectations; not needed for persist test but harmless
        monkeypatch.setattr("agent.credential_pool.get_pool_strategy", lambda p: "round_robin" if p == "opencode-go" else "fill_first")
        pool = load_pool("opencode-go")
        # In-memory must have hydrated secrets
        for ent in pool.entries():
            if ent.source.startswith("env:"):
                assert ent.runtime_api_key in (SYN_PRIMARY, SYN_SECONDARY, SYN_TERTIARY)
        # On-disk must NOT contain raw synthetic secrets
        raw = (home / "auth.json").read_text(encoding="utf-8")
        assert SYN_PRIMARY not in raw, "raw primary secret leaked to auth.json"
        assert SYN_SECONDARY not in raw, "raw secondary secret leaked to auth.json"
        assert SYN_TERTIARY not in raw, "raw tertiary secret leaked to auth.json"
        # Fingerprints must be present instead, and source refs preserved
        data = _read_auth(home)
        entries = data.get("credential_pool", {}).get("opencode-go", [])
        assert len(entries) == 3
        for ent in entries:
            assert ent.get("source", "").startswith("env:"), f"expected env source, got {ent!r}"
            # Borrowed rows are sanitized: access_token removed, fingerprint kept
            assert "access_token" not in ent or ent.get("access_token") == "", f"raw access_token persisted: {ent!r}"
            assert ent.get("secret_fingerprint", "").startswith("sha256:"), f"missing fingerprint: {ent!r}"
            assert ent.get("secret_source") is None or isinstance(ent.get("secret_source"), str)

    def test_no_pool_entries_unchanged_behavior(self, isolated_hermes_home):
        """If auth.json has no env-source entries, behavior matches pre-fix (only registry tuple)."""
        from agent.credential_pool import _seed_from_env
        _write_env_file(isolated_hermes_home, PROVIDER_API_KEY=SYN_PRIMARY)
        pconfig = _make_pconfig("opencode-go", ["PROVIDER_API_KEY"])
        with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
            changed, active_sources = _seed_from_env("opencode-go", [])
        assert changed is True
        assert "env:PROVIDER_API_KEY" in active_sources
        assert "env:PROVIDER_API_KEY_2" not in active_sources

    def test_env_var_name_extraction_handles_empty_suffix(self, isolated_hermes_home):
        """A malformed source 'env:' with empty var name must not be treated as env var."""
        from agent.credential_pool import PooledCredential, _seed_from_env
        _write_env_file(isolated_hermes_home, PROVIDER_API_KEY=SYN_PRIMARY)
        pconfig = _make_pconfig("opencode-go", ["PROVIDER_API_KEY"])
        malformed = PooledCredential(provider="opencode-go", id="bad1", label="bad", auth_type="api_key", priority=0, source="env:", access_token="")
        entries = [malformed]
        with patch("agent.credential_pool.PROVIDER_REGISTRY", {"opencode-go": pconfig}):
            changed, active_sources = _seed_from_env("opencode-go", entries)
        # Should still seed the registry var, but not crash or add empty env var
        assert "env:PROVIDER_API_KEY" in active_sources
        assert "" not in active_sources
        assert "env:" not in active_sources
        # Malformed entry stays empty
        assert entries[0].access_token == ""
