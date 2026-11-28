"""Regression tests for #150583: a profile's external-secret snapshot is keyed to the ``.env`` it was
built from, so editing that file re-hydrates instead of serving the pre-edit answer for the life of the
process.

The shadow only exists for a source that does NOT override existing values (``override_existing: false``,
the base default for helper/plugin sources); Bitwarden and 1Password default to overriding, so the fixture
pins the flag instead of relying on a source's default.

Observable per case, against a cold process on the same ``.env``:
- a line ADDED to ``.env`` must win over the value the source applied before the edit;
- a line REMOVED must let the source apply the value again (it was skipped as shadowed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import env_loader

PROBE = "PROBE_SECRET_150583"


@pytest.fixture(autouse=True)
def _reset():
    env_loader.reset_secret_source_cache()
    yield
    env_loader.reset_secret_source_cache()
    from agent.secret_sources import registry

    registry._reset_registry_for_tests()


@pytest.fixture
def source(monkeypatch):
    """Stubbed source reporting ``PROBE``, with ``override_existing: false`` so a .env line shadows it."""
    import agent.secret_sources.bitwarden as bw
    from agent.secret_sources import registry

    calls = {"n": 0}

    def fake_fetch(**_kwargs):
        calls["n"] += 1
        return {PROBE: "from-source"}, []

    monkeypatch.setattr(bw, "find_bws", lambda **_kwargs: Path("/fake/bws"))
    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", fake_fetch)
    registry._reset_registry_for_tests()
    return calls


def _home(tmp_path: Path, dotenv: str) -> Path:
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: proj\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n"
        "    override_existing: false\n",
        encoding="utf-8",
    )
    (home / ".env").write_text(dotenv, encoding="utf-8")
    return home


def _scope_value(home: Path) -> str | None:
    from agent.secret_scope import build_profile_secret_scope

    return build_profile_secret_scope(home).get(PROBE)


def test_added_dotenv_line_replaces_the_cached_source_value(tmp_path, source):
    home = _home(tmp_path, "BWS_ACCESS_TOKEN=0.t\n")
    assert env_loader.hydrate_profile_secret_sources(home) == {PROBE: "from-source"}
    assert _scope_value(home) == "from-source"

    (home / ".env").write_text(f"BWS_ACCESS_TOKEN=0.t\n{PROBE}=local-override\n", encoding="utf-8")

    assert env_loader.hydrate_profile_secret_sources(home) == {}
    assert source["n"] == 2
    assert _scope_value(home) == "local-override"


def test_removed_dotenv_line_resolves_the_source_value_again(tmp_path, source):
    home = _home(tmp_path, f"BWS_ACCESS_TOKEN=0.t\n{PROBE}=local-override\n")
    assert env_loader.hydrate_profile_secret_sources(home) == {}
    assert _scope_value(home) == "local-override"

    (home / ".env").write_text("BWS_ACCESS_TOKEN=0.t\n", encoding="utf-8")

    assert env_loader.hydrate_profile_secret_sources(home) == {PROBE: "from-source"}
    assert source["n"] == 2
    assert _scope_value(home) == "from-source"


def test_untouched_dotenv_keeps_the_snapshot_without_refetching(tmp_path, source):
    home = _home(tmp_path, "BWS_ACCESS_TOKEN=0.t\n")

    first = env_loader.hydrate_profile_secret_sources(home)
    second = env_loader.hydrate_profile_secret_sources(home)

    assert first == second == {PROBE: "from-source"}
    assert source["n"] == 1
