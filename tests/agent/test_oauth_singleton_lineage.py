"""Real file-I/O regressions for independent Anthropic profile logins.

Only remote model metadata is stubbed; credential resolution and healing is real.
"""
from pathlib import Path
import json
import socket

import pytest


def _grant(tag, *, row_id=None, newer=False, source="hermes_pkce"):
    return {"id": row_id or tag, "auth_type": "oauth", "source": source, "priority": 0,
            "access_token": "sk-ant-" + "oat01-fixture-" + tag,
            "refresh_token": "fixture-refresh-" + tag,
            "expires_at_ms": 4102448400000 if newer else 4102444800000}


def _seed(home, rows, single):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model:\n  provider: anthropic\n  default: claude-sonnet-4\nnous:\n  guest: false\n")
    (home / "auth.json").write_text(json.dumps({"version": 1, "active_provider": "anthropic",
        "providers": {}, "credential_pool": {"anthropic": rows}}))
    if single is not None:
        (home / ".anthropic_oauth.json").write_text(json.dumps({
            "accessToken": single["access_token"], "refreshToken": single["refresh_token"],
            "expiresAt": single["expires_at_ms"]}))


def _snapshot(home):
    return {name: (home / name).read_bytes() if (home / name).exists() else None
            for name in ("auth.json", ".anthropic_oauth.json")}


def _rows(home):
    return json.loads((home / "auth.json").read_text()).get("credential_pool", {}).get("anthropic", [])


def _single(home):
    path = home / ".anthropic_oauth.json"
    return json.loads(path.read_text())["accessToken"] if path.exists() else None


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    fake_home = tmp_path / "os-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(fake_home / "claude"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    for key in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("No network is allowed in this credential fixture")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    import hermes_constants
    from hermes_cli import auth, inventory, model_switch_providers
    from agent import models_dev
    auth._oauth_heal_notices.clear()
    auth._oauth_heal_clean_marks.clear()
    catalog = {"anthropic": {"id": "anthropic", "name": "Anthropic", "env": ["ANTHROPIC_API_KEY"],
                            "models": {"claude-sonnet-4": {"id": "claude-sonnet-4", "name": "Claude Sonnet"}}}}
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda *a, **k: catalog)
    monkeypatch.setattr(model_switch_providers, "_build_curated_lists", lambda *a, **k: {"anthropic": ["claude-sonnet-4"]})
    monkeypatch.setattr(model_switch_providers, "_live_or_curated_ids", lambda slug, curated, *a, **k: curated.get(slug, []))
    for name in ("_apply_pricing", "_apply_capabilities", "_apply_featured", "_prewarm_pricing_async"):
        monkeypatch.setattr(inventory, name, lambda *a, **k: None)

    def use(home):
        monkeypatch.setenv("HERMES_HOME", str(home))
        hermes_constants._default_hermes_root_memo = None
        auth._global_auth_store_cache = None
        # Keep clean marks: A→B→A must work without clearing the real cache.
    use(root)
    return root, use


@pytest.mark.parametrize("shape", ["mixed", "profile-singleton-only", "profile-pool-only", "root-singleton-only", "root-pool-only"])
@pytest.mark.parametrize("root_newer", [False, True])
def test_independent_logins_survive_model_options_and_profile_switches(fleet, shape, root_newer):
    from agent.credential_pool import load_pool
    from hermes_cli.inventory import build_model_options_payload, load_picker_context
    root, use = fleet
    a, b = root / "profiles" / "a", root / "profiles" / "b"
    rg = _grant("root", newer=root_newer, source="manual:hermes_pkce" if shape == "root-pool-only" else "hermes_pkce")
    ag = _grant("a", newer=not root_newer, source="manual:hermes_pkce" if shape == "profile-pool-only" else "hermes_pkce")
    bg = _grant("b", newer=True)
    _seed(root, [] if shape == "root-singleton-only" else [rg], None if shape == "root-pool-only" else rg)
    _seed(a, [] if shape == "profile-singleton-only" else [ag], None if shape == "profile-pool-only" else ag)
    _seed(b, [bg], bg)
    root_before = _snapshot(root)
    singles_before = {home: _snapshot(home)[".anthropic_oauth.json"] for home in (a, b)}
    for home, grant in ((a, ag), (b, bg), (a, ag)):
        use(home)
        payload = build_model_options_payload(load_picker_context(), explicit_only=True)
        assert "anthropic" in {p["slug"] for p in payload["providers"]}
        for _ in range(2):
            entries = load_pool("anthropic").entries()
            assert [(e.access_token, e.refresh_token) for e in entries] == [(grant["access_token"], grant["refresh_token"])]
        assert _snapshot(root) == root_before
        assert _snapshot(home)[".anthropic_oauth.json"] == singles_before[home]
        assert _rows(home)[0]["id"] != rg["id"]


@pytest.mark.parametrize("case", ["rotated", "root-newer", "non-first-match", "independent-profile-singleton", "independent-root-singleton", "singleton-only", "save-failure"])
def test_healing_tracks_the_specific_lineage_and_commits_before_unlink(fleet, monkeypatch, case):
    from hermes_cli import auth
    root, use = fleet
    profile = root / "profiles" / "fork"
    old = _grant("old", row_id="copied", newer=case == "root-newer")
    new = _grant("new", row_id="copied", newer=case != "root-newer")
    unrelated = _grant("independent", source="manual:hermes_pkce")
    root_rows, profile_rows = [old], [new]
    root_single, profile_single = old, new
    if case == "non-first-match":
        root_rows.insert(0, unrelated)
    if case == "independent-profile-singleton":
        profile_single = unrelated
    if case == "independent-root-singleton":
        root_single = unrelated
    if case == "singleton-only":
        root_rows, profile_rows = [], []
        new["refresh_token"] = old["refresh_token"]  # actual shared token, not a provider-wide match
    _seed(root, root_rows, root_single)
    _seed(profile, profile_rows, profile_single)
    before_root, before_profile = _snapshot(root), _snapshot(profile)
    use(profile)
    if case == "save-failure":
        def cannot_save(*args, **kwargs):
            raise OSError("fixture write failure")
        from agent import anthropic_credentials
        monkeypatch.setattr(anthropic_credentials, "_write_hermes_oauth_credentials", cannot_save)
        assert auth.heal_forked_single_use_oauth_grants("anthropic") is None
        assert _snapshot(profile) == before_profile
        assert _snapshot(root) == before_root
        return
    summary = auth.heal_forked_single_use_oauth_grants("anthropic")
    assert summary is not None
    winner = old if case == "root-newer" else new
    assert _single(profile) == (unrelated["access_token"] if case == "independent-profile-singleton" else None)
    assert _single(root) == (unrelated["access_token"] if case == "independent-root-singleton" else winner["access_token"])
    if root_rows:
        assert next(r for r in _rows(root) if r["id"] == "copied")["access_token"] == winner["access_token"]
    if case == "non-first-match":
        assert _rows(root)[0] == unrelated
    settled = _snapshot(root), _snapshot(profile)
    assert auth.heal_forked_single_use_oauth_grants("anthropic") is None
    assert (_snapshot(root), _snapshot(profile)) == settled


def test_pool_adoption_preserves_an_older_singletons_lineage(fleet):
    from hermes_cli import auth
    from agent.credential_pool import load_pool
    root, use = fleet
    profile = root / "profiles" / "fork"
    old = _grant("old", row_id="copied", source="manual:hermes_pkce")
    new = _grant("new", row_id="copied", newer=True, source="manual:hermes_pkce")
    _seed(root, [old], None)
    _seed(profile, [new], old)
    use(profile)
    assert auth.heal_forked_single_use_oauth_grants("anthropic") is not None
    assert _single(profile) is None
    assert [e.refresh_token for e in load_pool("anthropic").entries()] == [new["refresh_token"]]


@pytest.mark.parametrize("boundary", ["singleton", "root-pool", "profile-pool"])
def test_partial_persistence_retries_without_reseeding_spent_tokens(fleet, monkeypatch, boundary):
    from hermes_cli import auth
    from agent import anthropic_credentials as ac
    from agent.credential_pool import load_pool
    root, use = fleet
    profile = root / "profiles" / "fork"
    old, new = _grant("old", row_id="copied"), _grant("new", row_id="copied", newer=True)
    _seed(root, [old], old)
    _seed(profile, [new], new)
    before_root, before_profile = _snapshot(root), _snapshot(profile)
    original = ac._write_hermes_oauth_credentials if boundary == "singleton" else auth._save_auth_store
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        target = kwargs.get("target_path")
        intended = (boundary == "singleton" or
                    target == (root if boundary == "root-pool" else profile) / "auth.json")
        if intended and not failed:
            failed = True
            raise OSError("fixture partial persistence failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(ac if boundary == "singleton" else auth,
                        "_write_hermes_oauth_credentials" if boundary == "singleton" else "_save_auth_store",
                        fail_once)
    use(profile)
    assert auth.heal_forked_single_use_oauth_grants("anthropic") is None
    assert failed
    assert _snapshot(profile) == before_profile
    if boundary == "singleton":
        assert _snapshot(root) == before_root
    else:
        assert _single(root) == new["access_token"]
        assert _rows(root)[0]["access_token"] == (old if boundary == "root-pool" else new)["access_token"]
    assert auth.heal_forked_single_use_oauth_grants("anthropic") is not None
    assert _single(profile) is None
    assert _single(root) == new["access_token"]
    assert _rows(root)[0]["refresh_token"] == new["refresh_token"]
    use(root)
    assert [e.refresh_token for e in load_pool("anthropic").entries()] == [new["refresh_token"]]
