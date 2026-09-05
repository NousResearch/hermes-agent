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
        auth_mod._oauth_heal_clean_marks.clear()

    # Process-wide notice buffer: start each test clean.
    import hermes_cli.auth as _auth_mod
    _auth_mod._oauth_heal_notices.clear()
    _auth_mod._oauth_heal_clean_marks.clear()

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


# ── C. one-time heal for installs that ALREADY forked the grant ──────────
#
# Fleets created on pre-fix code hold profile-local copies of the root grant
# (verbatim --clone-all, or the old borrowed-persist). The heal runs inside
# the profile's load_pool(): consolidate to ROOT (freshest rotation wins),
# strip the profile copy, borrow root from then on.

def _fork(fleet, name, *, rotated_to=None):
    """Create *name* with a pre-fix style verbatim copy of root's auth.json.

    ``rotated_to=N`` makes the copy the LIVE pair (RT<N>, spent RT0 server-side)
    to emulate a profile that already refreshed on the old code.
    """
    pdir = _profile(fleet, name)
    pdir.mkdir(parents=True, exist_ok=True)
    store = json.loads((fleet["root"] / "auth.json").read_text())
    if rotated_to is not None:
        row = store["credential_pool"]["anthropic"][0]
        row["access_token"] = f"sk-ant-oat01-AT{rotated_to}"
        row["refresh_token"] = f"sk-ant-ort-RT{rotated_to}"
        row["expires_at_ms"] = int((time.time() - 60) * 1000)  # newer, still expired
        srv = fleet["server"]
        srv["spent"].add("sk-ant-ort-RT0")
        srv["valid"].discard("sk-ant-ort-RT0")
        srv["valid"].add(f"sk-ant-ort-RT{rotated_to}")
        srv["n"] = rotated_to
    (pdir / "auth.json").write_text(json.dumps(store))
    return pdir


def test_heal_consolidates_existing_forks_to_the_live_copy(fleet, caplog):
    """root + atlas hold spent RT0; forge already rotated to RT1 on old code."""
    import logging
    from agent.credential_pool import load_pool

    forge = _fork(fleet, "forge", rotated_to=1)
    atlas = _fork(fleet, "atlas")
    assert fleet["rows"](forge)[0]["refresh_token"] == "sk-ant-ort-RT1"
    assert fleet["rows"](atlas)[0]["refresh_token"] == "sk-ant-ort-RT0"

    with caplog.at_level(logging.INFO, logger="hermes_cli.auth"):
        fleet["use"](forge)
        sel = load_pool("anthropic").select()
    assert sel is not None and sel.access_token == "sk-ant-oat01-AT2"
    # forge's live pair was adopted by ROOT, then rotated there; forge holds nothing.
    assert fleet["rows"](forge) is None
    assert fleet["rows"](fleet["root"])[0]["refresh_token"] == "sk-ant-ort-RT2"
    assert fleet["rows"](fleet["root"])[0]["id"] == "abc123"
    healed = [r.message for r in caplog.records if "consolidated forked anthropic OAuth grant" in r.message]
    assert len(healed) == 1 and "profile forge" in healed[0] and "root updated" in healed[0]

    for home in (atlas, fleet["root"], forge):
        fleet["use"](home)
        sel = load_pool("anthropic").select()
        assert sel is not None and sel.access_token == "sk-ant-oat01-AT2", home
    assert fleet["rows"](atlas) is None and fleet["rows"](forge) is None
    # Exactly one rotation by us (RT1 -> RT2); the spent RT0 was never replayed.
    assert [e[0] for e in fleet["server"]["log"]] == ["ROTATE"], fleet["server"]["log"]
    # API-key rows in the profiles were not touched.
    for home in (forge, atlas):
        store = json.loads((home / "auth.json").read_text())
        assert store["credential_pool"]["openai"][0]["access_token"] == "sk-static-key"


def test_heal_is_idempotent_and_logs_once(fleet, caplog):
    import logging
    from agent.credential_pool import load_pool
    from hermes_cli.auth import consume_oauth_heal_notices, heal_forked_single_use_oauth_grants

    kid = _fork(fleet, "kid")
    fleet["use"](kid)
    with caplog.at_level(logging.INFO, logger="hermes_cli.auth"):
        load_pool("anthropic")
        assert fleet["rows"](kid) is None
        notices = consume_oauth_heal_notices()
        assert len(notices) == 1 and "profile kid" in notices[0]
        root_before = (fleet["root"] / "auth.json").read_text()
        # Second and third loads: nothing to do, nothing written, nothing logged.
        assert heal_forked_single_use_oauth_grants("anthropic") is None
        load_pool("anthropic")
    assert consume_oauth_heal_notices() == []
    assert (fleet["root"] / "auth.json").read_text() == root_before
    assert sum("consolidated forked" in r.message for r in caplog.records) == 1


def test_heal_never_deletes_the_only_surviving_copy(fleet):
    """Root lost its grant (user ran `hermes auth remove` at root); the profile's
    copy is the only one left — and an independent second account stays put."""
    from agent.credential_pool import load_pool

    kid = _fork(fleet, "kid", rotated_to=1)
    store = json.loads((fleet["root"] / "auth.json").read_text())
    del store["credential_pool"]["anthropic"]
    (fleet["root"] / "auth.json").write_text(json.dumps(store))

    fleet["use"](kid)
    sel = load_pool("anthropic").select()
    assert sel is not None and sel.access_token == "sk-ant-oat01-AT2"
    assert fleet["rows"](kid) and fleet["rows"](kid)[0]["refresh_token"] == "sk-ant-ort-RT2"
    assert "anthropic" not in (json.loads((fleet["root"] / "auth.json").read_text())["credential_pool"])


def test_heal_leaves_a_different_account_alone(fleet):
    """A profile row whose JWT identity names ANOTHER account is not root's grant."""
    import base64
    from agent.credential_pool import load_pool

    def jwt(sub):
        payload = base64.urlsafe_b64encode(json.dumps({"sub": sub, "exp": int(time.time()) + 3600}).encode()).rstrip(b"=")
        return "h." + payload.decode() + ".s"

    root_store = json.loads((fleet["root"] / "auth.json").read_text())
    root_store["credential_pool"]["xai-oauth"] = [{
        "id": "rootx", "auth_type": "oauth", "priority": 0, "source": "manual:device_code",
        "access_token": jwt("alice"), "refresh_token": "xr-alice",
    }]
    (fleet["root"] / "auth.json").write_text(json.dumps(root_store))
    kid = _profile(fleet, "kid")
    kid.mkdir(parents=True, exist_ok=True)
    (kid / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"xai-oauth": [
            {"id": "kidx", "auth_type": "oauth", "priority": 0, "source": "manual:device_code",
             "access_token": jwt("bob"), "refresh_token": "xr-bob"},
            {"id": "kidk", "auth_type": "api_key", "priority": 1, "source": "manual",
             "access_token": "xai-static"},
        ]},
    }))
    fleet["use"](kid)
    load_pool("xai-oauth")
    rows = (json.loads((kid / "auth.json").read_text())["credential_pool"])["xai-oauth"]
    assert [r["id"] for r in rows] == ["kidx", "kidk"]
    assert json.loads((fleet["root"] / "auth.json").read_text())["credential_pool"]["xai-oauth"][0]["refresh_token"] == "xr-alice"


def test_heal_pkce_singleton_shape_commits_live_pair_to_root_singleton(fleet):
    """`hermes auth` PKCE shape: root + profile each have .anthropic_oauth.json +
    a hermes_pkce-seeded row; the profile's copy is the rotated (live) one."""
    from agent.credential_pool import load_pool

    root = fleet["root"]
    store = json.loads((root / "auth.json").read_text())
    store["active_provider"] = "anthropic"
    del store["credential_pool"]["anthropic"]
    (root / "auth.json").write_text(json.dumps(store))
    (root / ".anthropic_oauth.json").write_text(json.dumps({
        "accessToken": "sk-ant-oat01-AT0", "refreshToken": "sk-ant-ort-RT0",
        "expiresAt": int((time.time() - 3600) * 1000),
    }))
    fleet["use"](root)
    load_pool("anthropic")  # seeds root's hermes_pkce row from the singleton

    kid = _profile(fleet, "kid")
    kid.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(root / "auth.json", kid / "auth.json")
    (kid / ".anthropic_oauth.json").write_text(json.dumps({
        "accessToken": "sk-ant-oat01-AT1", "refreshToken": "sk-ant-ort-RT1",
        "expiresAt": int((time.time() - 60) * 1000),
    }))
    kstore = json.loads((kid / "auth.json").read_text())
    kstore["credential_pool"]["anthropic"][0].update(
        access_token="sk-ant-oat01-AT1", refresh_token="sk-ant-ort-RT1",
        expires_at_ms=int((time.time() - 60) * 1000),
    )
    (kid / "auth.json").write_text(json.dumps(kstore))
    srv = fleet["server"]
    srv["spent"].add("sk-ant-ort-RT0"); srv["valid"] = {"sk-ant-ort-RT1"}; srv["n"] = 1

    fleet["use"](kid)
    sel = load_pool("anthropic").select()
    assert sel is not None and sel.access_token == "sk-ant-oat01-AT2"
    assert not (kid / ".anthropic_oauth.json").exists()
    assert fleet["rows"](kid) is None
    assert json.loads((root / ".anthropic_oauth.json").read_text())["refreshToken"] == "sk-ant-ort-RT2"
    fleet["use"](root)
    sel = load_pool("anthropic").select()
    assert sel is not None and sel.access_token == "sk-ant-oat01-AT2"
    assert [e[0] for e in srv["log"]] == ["ROTATE"], srv["log"]


def test_heal_is_a_noop_in_classic_mode(fleet):
    from hermes_cli.auth import heal_forked_single_use_oauth_grants
    fleet["use"](fleet["root"])
    before = (fleet["root"] / "auth.json").read_text()
    assert heal_forked_single_use_oauth_grants("anthropic") is None
    assert (fleet["root"] / "auth.json").read_text() == before


# ── C. a SHARED root store is not a fork (#101356) ───────────────────────

def _seed_codex_grant(root):
    """Give the root store an openai-codex pool row AND a providers block."""
    fresh = int((time.time() + 3600) * 1000)
    store = json.loads((root / "auth.json").read_text())
    store["credential_pool"]["openai-codex"] = [{
        "id": "cdx001", "label": "codex", "auth_type": "oauth", "priority": 0,
        "source": "manual:device_code", "access_token": "cdx-AT0",
        "refresh_token": "cdx-RT0", "expires_at_ms": fresh,
    }]
    store["providers"]["openai-codex"] = {
        "tokens": {"access_token": "cdx-AT0", "refresh_token": "cdx-RT0", "expires_at_ms": fresh},
        "last_refresh": fresh / 1000.0,
    }
    (root / "auth.json").write_text(json.dumps(store))


def _shared_profile(fleet, name, *, link):
    """Profile whose auth.json IS the root store (``link`` makes the alias)."""
    pdir = _profile(fleet, name)
    pdir.mkdir(parents=True, exist_ok=True)
    alias = pdir / "auth.json"
    if alias.is_symlink() or alias.exists():
        alias.unlink()
    link(fleet["root"] / "auth.json", alias)
    return pdir


def test_heal_skips_profile_auth_json_symlinked_to_the_root_store(fleet):
    """#101356: `ln -s ~/.hermes/auth.json <profile>/auth.json` shares ONE store.
    Both sides of the consolidation read the same file, so every row looks like
    a fork of itself — healing would strip the shared grant through the link."""
    from hermes_cli.auth import consume_oauth_heal_notices, heal_forked_single_use_oauth_grants

    root = fleet["root"]
    _seed_codex_grant(root)
    before = (root / "auth.json").read_text()

    shared = _shared_profile(fleet, "shared", link=lambda target, alias: alias.symlink_to(target))
    fleet["use"](shared)

    assert heal_forked_single_use_oauth_grants("openai-codex") is None
    assert (root / "auth.json").read_text() == before
    assert (shared / "auth.json").is_symlink()
    assert consume_oauth_heal_notices() == []
    store = json.loads((root / "auth.json").read_text())
    assert [r["id"] for r in store["credential_pool"]["openai-codex"]] == ["cdx001"]
    assert store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "cdx-RT0"


def test_heal_skips_profile_auth_json_hardlinked_to_the_root_store(fleet):
    """Same class as the symlink: a hardlink resolves to a different name but
    is the same inode, so it is still one store, not a forked copy."""
    from hermes_cli.auth import heal_forked_single_use_oauth_grants

    root = fleet["root"]
    _seed_codex_grant(root)
    before = (root / "auth.json").read_text()

    shared = _shared_profile(fleet, "twin", link=lambda target, alias: os.link(target, alias))
    fleet["use"](shared)

    assert heal_forked_single_use_oauth_grants("openai-codex") is None
    assert (root / "auth.json").read_text() == before
    assert (shared / "auth.json").samefile(root / "auth.json")


def test_heal_leaves_an_aliased_anthropic_singleton_alone(fleet):
    """Separate auth.jsons but a profile `.anthropic_oauth.json` symlinked to
    root's: one shared grant, not a fork. The heal must not self-compare it
    or unlink the alias (#101356 sibling site)."""
    from hermes_cli.auth import heal_forked_single_use_oauth_grants

    root = fleet["root"]
    (root / ".anthropic_oauth.json").write_text(json.dumps({
        "accessToken": "AT-shared", "refreshToken": "RT-shared",
        "expiresAt": int((time.time() + 3600) * 1000),
    }))
    kid = _profile(fleet, "kid")
    kid.mkdir(parents=True, exist_ok=True)
    (kid / "auth.json").write_text(json.dumps({"providers": {}, "credential_pool": {}}))
    (kid / ".anthropic_oauth.json").symlink_to(root / ".anthropic_oauth.json")
    before = (root / ".anthropic_oauth.json").read_text()

    fleet["use"](kid)
    assert heal_forked_single_use_oauth_grants("anthropic") is None
    assert (kid / ".anthropic_oauth.json").is_symlink()
    assert (root / ".anthropic_oauth.json").read_text() == before


def test_heal_same_store_skip_is_memoized_off_the_hot_path(fleet, monkeypatch):
    """The shared-store skip must record the clean mark so load_pool()'s
    per-call heal does not re-stat/resolve both paths every model call."""
    from hermes_cli import auth as auth_mod

    root = fleet["root"]
    _seed_codex_grant(root)
    shared = _shared_profile(fleet, "shared", link=lambda target, alias: alias.symlink_to(target))
    fleet["use"](shared)

    assert auth_mod.heal_forked_single_use_oauth_grants("openai-codex") is None
    assert "openai-codex" in auth_mod._oauth_heal_clean_marks
    calls = []
    monkeypatch.setattr(auth_mod, "_is_same_auth_store", lambda *a: calls.append(a) or True)
    assert auth_mod.heal_forked_single_use_oauth_grants("openai-codex") is None
    assert calls == [], "same-store check ran again despite the clean mark"


# ── E. the clean mark outlives the process ──────────────────────────────
#
# The in-memory mark only silences the heal for one process, so every fresh
# `hermes` invocation re-paid its two nested EXCLUSIVE auth-store locks to
# rediscover a store it had already cleared. Persisting the mark removes that,
# but a mark that outlives the process must also invalidate on anything the
# heal reads -- including the ROOT store, which the in-memory fingerprint
# could safely ignore precisely because it died with the process.

def _new_process(auth_mod):
    """Simulate a fresh `hermes` invocation: in-memory state gone, disk kept."""
    auth_mod._oauth_heal_clean_marks.clear()
    auth_mod._global_auth_store_cache = None


def _count_locks(monkeypatch):
    """Count _auth_store_lock acquisitions, still really taking them.

    The heal imports the lock from ``hermes_cli.auth`` inside the function, so
    patching it on that module is what the call site actually resolves.
    """
    import contextlib

    import hermes_cli.auth as auth_mod

    taken = []
    real = auth_mod._auth_store_lock

    @contextlib.contextmanager
    def counting(*a, **k):
        taken.append(k.get("target_path"))
        with real(*a, **k):
            yield

    monkeypatch.setattr(auth_mod, "_auth_store_lock", counting)
    return taken


def _kid_with_api_key_only(fleet, name="kid"):
    """Profile whose store exists and is genuinely fork-free, so the heal has
    to run its locked body to find that out (not the no-files fast path)."""
    pdir = _profile(fleet, name)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"openai": [{
            "id": "k1", "label": "static", "auth_type": "api_key",
            "priority": 0, "source": "manual", "access_token": "sk-local",
        }]},
    }))
    return pdir


def test_clean_mark_persists_so_a_fresh_process_takes_no_auth_lock(fleet, monkeypatch):
    import hermes_cli.auth as auth_mod
    from hermes_cli import auth_oauth_grants as grants

    kid = _kid_with_api_key_only(fleet)
    fleet["use"](kid)
    first = _count_locks(monkeypatch)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None
    assert len(first) == 2, "cold heal should take the profile and root locks"
    assert grants._oauth_heal_clean_mark_path().exists(), "clean mark not written"

    _new_process(auth_mod)
    second = _count_locks(monkeypatch)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None
    assert second == [], "a fresh process re-locked the auth store to redo a clean heal"


def test_persisted_mark_still_re_heals_when_the_root_store_gains_a_grant(fleet):
    """The mark may not outlive the facts. Root acquiring a counterpart turns
    a row the heal deliberately KEPT into a fork it must strip -- with the
    profile's own files untouched, so only root's stamp can catch it."""
    import hermes_cli.auth as auth_mod
    from hermes_cli import auth_oauth_grants as grants

    root = fleet["root"]
    store = json.loads((root / "auth.json").read_text())
    store["credential_pool"].pop("anthropic")
    (root / "auth.json").write_text(json.dumps(store))

    kid = _kid_with_api_key_only(fleet, "kid2")
    fork = {
        "id": "abc123", "label": "team-grant", "auth_type": "oauth",
        "priority": 0, "source": "manual:hermes_pkce",
        "access_token": "sk-ant-oat01-AT0", "refresh_token": "sk-ant-ort-RT0",
        "expires_at_ms": int((time.time() + 3600) * 1000),
        "base_url": "https://api.anthropic.com",
    }
    kid_store = json.loads((kid / "auth.json").read_text())
    kid_store["credential_pool"]["anthropic"] = [dict(fork)]
    (kid / "auth.json").write_text(json.dumps(kid_store))

    fleet["use"](kid)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None
    assert fleet["rows"](kid), "the only surviving copy must not be stripped"
    marked = grants._oauth_heal_clean_mark_path().read_text()

    store["credential_pool"]["anthropic"] = [dict(fork)]
    (root / "auth.json").write_text(json.dumps(store))
    _new_process(auth_mod)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is not None, (
        "the persisted mark skipped a heal that had become necessary")
    assert not fleet["rows"](kid), "the fork survived in the profile store"

    # A heal that actually did work writes no mark, so the one on disk is now
    # stale -- it describes the pre-heal files and can no longer match. The
    # next process re-checks, finds the store clean, and re-stamps.
    _new_process(auth_mod)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None
    assert grants._oauth_heal_clean_mark_path().read_text() != marked


def test_unreadable_clean_mark_falls_back_to_healing(fleet, monkeypatch):
    """The mark is a cache: corrupt, truncated or wrong-shaped content must
    mean 'unknown' and re-run the heal, never raise into a credential path."""
    import hermes_cli.auth as auth_mod
    from hermes_cli import auth_oauth_grants as grants

    kid = _kid_with_api_key_only(fleet, "kid3")
    fleet["use"](kid)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None
    mark = grants._oauth_heal_clean_mark_path()

    taken = _count_locks(monkeypatch)
    for junk in ("{not json", "[]", '{"anthropic": "clean"}', ""):
        mark.write_text(junk)
        _new_process(auth_mod)
        taken.clear()
        assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None
        assert len(taken) == 2, f"{junk!r} did not fall back to the locked heal"


def test_persisted_mark_invalidates_on_a_size_change_that_keeps_the_mtime(fleet, monkeypatch):
    """An mtime-only key is fine for a mark that dies with the process; one
    that outlives it must also survive `rsync -t` / `tar -p` / a restore,
    which put a file's timestamp back after rewriting its contents."""
    import hermes_cli.auth as auth_mod

    kid = _kid_with_api_key_only(fleet, "kid4")
    fleet["use"](kid)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None

    target = kid / "auth.json"
    before = target.stat()
    target.write_text(target.read_text() + " " * 32)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_mtime_ns == before.st_mtime_ns, "mtime was not restored"

    _new_process(auth_mod)
    taken = _count_locks(monkeypatch)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None
    assert len(taken) == 2, "a same-mtime rewrite kept the stale clean mark alive"


def test_clean_mark_records_no_credential_material(fleet):
    """It lands next to the store it describes; it must stay a stat cache."""
    import hermes_cli.auth as auth_mod
    from hermes_cli import auth_oauth_grants as grants

    kid = _kid_with_api_key_only(fleet, "kid5")
    fleet["use"](kid)
    assert auth_mod.heal_forked_single_use_oauth_grants("anthropic") is None
    body = grants._oauth_heal_clean_mark_path().read_text()
    for secret in ("sk-ant-oat01", "sk-ant-ort", "sk-static-key", "sk-local",
                   "access_token", "refresh_token"):
        assert secret not in body, f"clean mark leaked {secret}"
