"""Single-use Anthropic OAuth grants never fork across profiles (#100339) and are never
inherited from the root (#111724).

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


@pytest.mark.parametrize(
    "link",
    [
        lambda target, alias: alias.symlink_to(target),
        lambda target, alias: os.link(target, alias),
    ],
    ids=["symlink", "hardlink"],
)
def test_strip_helper_leaves_shared_root_auth_store_unchanged(fleet, link):
    """A shared auth store is one grant, not a cloned credential copy."""
    from hermes_cli.auth import strip_cloned_single_use_oauth_grants

    root = fleet["root"]
    _seed_codex_grant(root)
    before = (root / "auth.json").read_text()
    shared = _shared_profile(fleet, "shared", link=link)

    assert strip_cloned_single_use_oauth_grants(shared) == {
        "pool": [], "providers": [], "files": [],
    }
    assert (root / "auth.json").read_text() == before


def test_strip_helper_fails_closed_when_root_store_cannot_be_resolved(fleet, monkeypatch):
    """Credential hygiene must not mutate auth when store identity is unknown."""
    from hermes_cli.auth import strip_cloned_single_use_oauth_grants
    import hermes_constants

    root = fleet["root"]
    _seed_codex_grant(root)
    before = (root / "auth.json").read_text()
    shared = _shared_profile(
        fleet, "shared", link=lambda target, alias: alias.symlink_to(target))
    monkeypatch.setattr(
        hermes_constants, "get_default_hermes_root",
        lambda: (_ for _ in ()).throw(OSError("root unavailable")))

    assert strip_cloned_single_use_oauth_grants(shared) == {
        "pool": [], "providers": [], "files": [],
    }
    assert (root / "auth.json").read_text() == before


def test_strip_helper_fails_closed_when_store_identity_check_errors(fleet, monkeypatch):
    """A transient stat failure must not be interpreted as two stores."""
    from hermes_cli.auth import strip_cloned_single_use_oauth_grants

    root = fleet["root"]
    _seed_codex_grant(root)
    copied = _profile(fleet, "copied")
    (copied / "auth.json").write_text((root / "auth.json").read_text())
    before = (copied / "auth.json").read_text()
    monkeypatch.setattr(
        type(copied), "samefile",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("stat unavailable")))

    assert strip_cloned_single_use_oauth_grants(copied) == {
        "pool": [], "providers": [], "files": [],
    }
    assert (copied / "auth.json").read_text() == before


# ── B. a named profile never borrows the root grant ───────────────────────


def test_named_profile_does_not_inherit_or_rotate_the_root_grant(fleet):
    """An empty profile sees no Anthropic credential and the root's single-use pair is untouched;
    the root itself still refreshes and persists its own rotation (control)."""
    from agent.anthropic_credentials import resolve_anthropic_token
    from agent.credential_pool import load_pool

    forge = _profile(fleet, "forge")
    fleet["use"](forge)
    assert load_pool("anthropic").select() is None
    assert resolve_anthropic_token() is None
    assert fleet["rows"](forge) in (None, [])
    assert fleet["server"]["log"] == []
    assert fleet["rows"](fleet["root"])[0]["refresh_token"] == "sk-ant-ort-RT0"

    fleet["use"](fleet["root"])
    sel = load_pool("anthropic").select()
    assert sel is not None and sel.access_token == "sk-ant-oat01-AT1"
    assert fleet["rows"](fleet["root"])[0]["refresh_token"] == "sk-ant-ort-RT1"


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
    assert [e["id"] for e in fleet["rows"](kid)] == ["own001"]
    assert [e["id"] for e in fleet["rows"](fleet["root"])] == ["abc123"]


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


def _link_same_file(target, alias):
    """Make ``alias`` the SAME store as ``target``: a symlink where the host
    permits one, a hardlink where it does not (Windows without SeCreateSymbolicLink
    privilege raises WinError 1314). Both flavors are one file — the heal contract
    (``_is_same_auth_store``) is ``samefile``/resolved-path equality, not the link
    flavor, so the test exercises the same contract on either host."""
    try:
        alias.symlink_to(target)
    except (OSError, NotImplementedError):
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        os.link(target, alias)


def _shared_profile(fleet, name, *, link=_link_same_file):
    """Profile whose auth.json IS the root store (``link`` makes the alias)."""
    pdir = _profile(fleet, name)
    pdir.mkdir(parents=True, exist_ok=True)
    alias = pdir / "auth.json"
    if alias.is_symlink() or alias.exists():
        alias.unlink()
    link(fleet["root"] / "auth.json", alias)
    return pdir
