"""Regression tests for #100339: cloned / borrowed single-use Anthropic OAuth
grants must never fork across profiles.

Real imports, real temp HERMES_HOME root + named profile, real auth.json I/O.
The Anthropic token endpoint is replaced at the ``urllib.request.urlopen``
boundary with genuine single-use semantics (a refresh token redeems once;
a second POST returns ``invalid_grant``).
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.request

import pytest


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """Root HERMES_HOME with an expired-but-refreshable Anthropic pool row."""
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    # Keep host ~/.claude and host auth.json out of the picture.
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # The pytest seat-belt in the root write-through compares the global path
    # against $HOME/.hermes/auth.json; our root is elsewhere, so writes go.
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]

    expired = int((time.time() - 3600) * 1000)
    store = {
        "version": 1,
        "providers": {},
        "credential_pool": {
            "anthropic": [{
                "id": "abc123", "label": "team-grant", "auth_type": "oauth",
                "priority": 0, "source": "manual:hermes_pkce",
                "access_token": "sk-ant-oat01-AT0", "refresh_token": "sk-ant-ort-RT0",
                "expires_at_ms": expired, "base_url": "https://api.anthropic.com",
            }],
            "openai": [{
                "id": "key001", "label": "static", "auth_type": "api_key",
                "priority": 0, "source": "manual", "access_token": "sk-static-key",
            }],
        },
    }
    (root / "auth.json").write_text(json.dumps(store))

    server = {"valid": {"sk-ant-ort-RT0"}, "spent": set(), "n": 0, "log": []}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        assert "oauth/token" in req.full_url
        body = req.data.decode()
        if req.get_header("Content-type", "").startswith("application/json"):
            rt = json.loads(body)["refresh_token"]
        else:
            from urllib.parse import parse_qsl
            rt = dict(parse_qsl(body))["refresh_token"]
        if rt in server["spent"] or rt not in server["valid"]:
            server["log"].append(("REUSE", rt))
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                io.BytesIO(b'{"error":"invalid_grant","error_description":"refresh_token_reused"}'),
            )
        server["n"] += 1
        server["spent"].add(rt)
        server["valid"].discard(rt)
        new_rt = f"sk-ant-ort-RT{server['n']}"
        server["valid"].add(new_rt)
        server["log"].append(("ROTATE", rt, new_rt))
        return _Resp(json.dumps({
            "access_token": f"sk-ant-oat01-AT{server['n']}",
            "refresh_token": new_rt, "expires_in": 28800, "token_type": "Bearer",
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def use(home):
        """Switch the process to *home* (root or a profile dir)."""
        monkeypatch.setenv("HERMES_HOME", str(home))
        hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
        import hermes_cli.auth as auth_mod
        auth_mod._global_auth_store_cache = None

    def pool_rows(home):
        p = home / "auth.json"
        if not p.exists():
            return None
        return (json.loads(p.read_text()).get("credential_pool") or {}).get("anthropic")

    return {"root": root, "server": server, "use": use, "rows": pool_rows}


def _profile(fleet, name, **kw):
    from hermes_cli.profiles import create_profile
    fleet["use"](fleet["root"])
    return create_profile(name, **kw)


# ── A. cloning never copies single-use OAuth grants ──────────────────────

def test_clone_all_strips_oauth_grant_but_keeps_api_keys(fleet):
    (fleet["root"] / ".anthropic_oauth.json").write_text(
        json.dumps({"accessToken": "sk-ant-oat01-AT0", "refreshToken": "sk-ant-ort-RT0", "expiresAt": 1})
    )
    pdir = _profile(fleet, "forge", clone_all=True)
    store = json.loads((pdir / "auth.json").read_text())
    assert "anthropic" not in store["credential_pool"], "OAuth grant was forked into the clone"
    assert store["credential_pool"]["openai"][0]["access_token"] == "sk-static-key"
    assert not (pdir / ".anthropic_oauth.json").exists()


def test_strip_helper_drops_device_code_blocks_and_reports(tmp_path):
    from hermes_cli.auth import strip_cloned_single_use_oauth_grants
    pdir = tmp_path / "p"
    pdir.mkdir()
    (pdir / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"openai-codex": {"access_token": "a", "refresh_token": "r"}, "nous": {"agent_key": "k"}},
        "credential_pool": {
            "xai-oauth": [{"id": "x", "auth_type": "oauth", "access_token": "t", "refresh_token": "r"}],
            "anthropic": [
                {"id": "legacy", "access_token": "sk-ant-oat01-legacy"},  # no auth_type field
                {"id": "key", "auth_type": "api_key", "access_token": "sk-ant-api03-x"},
            ],
        },
    }))
    summary = strip_cloned_single_use_oauth_grants(pdir)
    store = json.loads((pdir / "auth.json").read_text())
    assert sorted(summary["pool"]) == ["anthropic", "xai-oauth"]
    assert summary["providers"] == ["openai-codex"]
    assert "xai-oauth" not in store["credential_pool"]
    assert [e["id"] for e in store["credential_pool"]["anthropic"]] == ["key"]
    assert "openai-codex" not in store["providers"] and "nous" in store["providers"]


def test_strip_helper_is_a_noop_without_credentials(tmp_path):
    from hermes_cli.auth import strip_cloned_single_use_oauth_grants
    assert strip_cloned_single_use_oauth_grants(tmp_path) == {"pool": [], "providers": [], "files": []}


# ── B. borrowed rotation commits to root, never a profile copy ───────────

def test_first_profile_rotation_does_not_strand_root_or_siblings(fleet):
    from agent.credential_pool import load_pool

    forge = _profile(fleet, "forge")
    atlas = _profile(fleet, "atlas")

    fleet["use"](forge)
    sel = load_pool("anthropic").select()
    assert sel is not None and sel.access_token == "sk-ant-oat01-AT1"
    # The rotated pair landed in ROOT; forge did not grow a local copy.
    assert fleet["rows"](forge) is None
    assert fleet["rows"](fleet["root"])[0]["refresh_token"] == "sk-ant-ort-RT1"

    for home in (atlas, fleet["root"], forge):
        fleet["use"](home)
        sel = load_pool("anthropic").select()
        assert sel is not None and sel.access_token == "sk-ant-oat01-AT1", home
    assert [e[0] for e in fleet["server"]["log"]] == ["ROTATE"], fleet["server"]["log"]
    assert fleet["rows"](atlas) is None and fleet["rows"](forge) is None


def test_agent_init_resolver_sees_sibling_rotation(fleet):
    from agent.anthropic_credentials import resolve_anthropic_token
    from agent.credential_pool import load_pool

    forge = _profile(fleet, "forge")
    atlas = _profile(fleet, "atlas")
    fleet["use"](forge)
    load_pool("anthropic").select()
    fleet["use"](atlas)
    assert resolve_anthropic_token() == "sk-ant-oat01-AT1"


def test_borrowing_profile_load_pool_does_not_materialize_local_copy(fleet):
    from agent.credential_pool import load_pool

    fresh = _profile(fleet, "fresh")
    fleet["use"](fresh)
    pool = load_pool("anthropic")
    assert [e.id for e in pool.entries()] == ["abc123"]
    assert pool._borrowed_root_ids == {"abc123"}
    assert fleet["rows"](fresh) is None


def test_borrower_prune_never_deletes_root_singleton_grant(fleet, tmp_path):
    """Root's hermes_pkce row is seeded from ROOT's .anthropic_oauth.json; a
    profile without that file must not prune (and write-through-delete) it."""
    from agent.credential_pool import load_pool

    root = fleet["root"]
    (root / ".anthropic_oauth.json").write_text(json.dumps({
        "accessToken": "sk-ant-oat01-AT0", "refreshToken": "sk-ant-ort-RT0",
        "expiresAt": int((time.time() - 3600) * 1000),
    }))
    store = json.loads((root / "auth.json").read_text())
    store["active_provider"] = "anthropic"
    del store["credential_pool"]["anthropic"]
    (root / "auth.json").write_text(json.dumps(store))
    fleet["use"](root)
    root_rows = [e for e in load_pool("anthropic").entries()]
    assert [e.source for e in root_rows] == ["hermes_pkce"]

    kid = _profile(fleet, "kid")
    fleet["use"](kid)
    pool = load_pool("anthropic")
    assert [e.source for e in pool.entries()] == ["hermes_pkce"], "borrowed root grant was pruned"
    assert fleet["rows"](root) and fleet["rows"](root)[0]["source"] == "hermes_pkce"
    assert fleet["rows"](kid) is None

    # Rotating from the profile commits BOTH the pool row and the singleton at ROOT.
    sel = pool.select()
    assert sel is not None and sel.access_token == "sk-ant-oat01-AT1"
    assert json.loads((root / ".anthropic_oauth.json").read_text())["refreshToken"] == "sk-ant-ort-RT1"
    assert not (kid / ".anthropic_oauth.json").exists()
    assert fleet["rows"](root)[0]["refresh_token"] == "sk-ant-ort-RT1"


def test_profile_auth_add_owns_only_its_own_rows(fleet):
    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool

    kid = _profile(fleet, "kid")
    fleet["use"](kid)
    pool = load_pool("anthropic")
    pool.add_entry(PooledCredential(
        provider="anthropic", id="own001", label="mine", auth_type=AUTH_TYPE_OAUTH,
        priority=0, source="manual:hermes_pkce", access_token="sk-ant-oat01-MINE",
        refresh_token="rt-mine",
    ))
    assert [e["id"] for e in fleet["rows"](kid)] == ["own001"], "borrowed root row was copied into the profile"
    assert [e["id"] for e in fleet["rows"](fleet["root"])] == ["abc123"]


def test_classic_mode_persist_is_unchanged(fleet):
    from agent.credential_pool import load_pool

    fleet["use"](fleet["root"])
    sel = load_pool("anthropic").select()
    assert sel is not None and sel.access_token == "sk-ant-oat01-AT1"
    assert fleet["rows"](fleet["root"])[0]["refresh_token"] == "sk-ant-ort-RT1"
