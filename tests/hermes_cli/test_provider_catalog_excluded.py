"""``model_catalog.excluded_providers`` must also hide the desktop Providers tabs.

#67971 added the key and taught every ``/model`` picker surface to honour it, but
the desktop Settings → Providers pane derives its two tabs from
``provider_catalog()`` — ``_catalog_provider_env_metadata`` (API keys) and
``_build_oauth_catalog`` (Accounts) — and nothing between that function and the
endpoint read the exclusion. An install holding one provider credential still got
a key card and a sign-in card for every other rail.

These tests pin the visibility predicate (:mod:`hermes_cli.provider_catalog`) and
both endpoints, plus the two rules that keep the setting from locking anyone out:

* a var/account with a credential already on disk stays visible — a stored secret
  must remain clearable from the GUI;
* an env var declared by several providers is hidden only when EVERY owner of it
  is excluded.
"""

from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_ALIASES
from hermes_cli.provider_catalog import (
    excluded_provider_slugs,
    provider_catalog,
    provider_is_excluded,
    visible_provider_catalog,
)
from hermes_cli.web_server import _SESSION_TOKEN, app
import hermes_cli.web_routers.oauth as _rt_oauth

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def _make_home(root, monkeypatch, *, excluded=None, env_lines=""):
    """An isolated HERMES_HOME whose config.yaml names ``excluded`` (or omits the
    key entirely when None), plus a .env holding ``env_lines``."""
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"model": "old-model"}
    if excluded is not None:
        cfg["model_catalog"] = {"excluded_providers": excluded}
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    (root / ".env").write_text(env_lines)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _first_aliased_canonical():
    """(canonical_slug, alias) for the first canonical provider that has an alias
    in ``_PROVIDER_ALIASES`` — picked at runtime so a rename cannot rot the test."""
    canonical = {p.slug for p in CANONICAL_PROVIDERS}
    for alias, canon in _PROVIDER_ALIASES.items():
        if canon in canonical and alias.lower() != canon.lower():
            return canon, alias
    return None, None


def _env_payload(query=""):
    resp = client.get(f"/api/env{query}", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _oauth_ids(query=""):
    resp = client.get(f"/api/providers/oauth{query}", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    return [p["id"] for p in resp.json()["providers"]]


# --------------------------------------------------------------- the predicate


def test_visible_catalog_equals_full_catalog_when_unset(tmp_path, monkeypatch):
    """No exclusion configured → visibility is the identity on the catalog."""
    _make_home(tmp_path / "home", monkeypatch)
    assert excluded_provider_slugs() == frozenset()
    assert [d.slug for d in visible_provider_catalog()] == [d.slug for d in provider_catalog()]


def test_visible_catalog_hides_excluded_slug(tmp_path, monkeypatch):
    _make_home(tmp_path / "home", monkeypatch, excluded=["openrouter"])
    slugs = [d.slug for d in visible_provider_catalog()]
    assert "openrouter" not in slugs
    assert "nous" in slugs, "only the named provider may disappear"
    assert provider_is_excluded("openrouter")
    assert not provider_is_excluded("nous")


def test_visible_catalog_hides_by_alias(tmp_path, monkeypatch):
    """An alias entry hides the canonical provider, as it does in `hermes model`
    (hermes_cli/main_provider_setup.py) — e.g. ``aws`` hides ``bedrock``."""
    canon, alias = _first_aliased_canonical()
    assert canon, "no aliased canonical provider to test"
    _make_home(tmp_path / "home", monkeypatch, excluded=[alias])
    assert canon not in [d.slug for d in visible_provider_catalog()], (
        f"excluding alias {alias!r} must hide {canon!r}"
    )
    assert provider_is_excluded(canon)


def test_provider_is_excluded_is_case_and_whitespace_insensitive(tmp_path, monkeypatch):
    _make_home(tmp_path / "home", monkeypatch, excluded=["  OpenRouter  ", "", None])
    assert provider_is_excluded("openrouter")
    assert provider_is_excluded("OPENROUTER")
    assert provider_is_excluded(" openrouter ")
    assert "openrouter" not in [d.slug for d in visible_provider_catalog()]


def test_provider_catalog_universe_unchanged_by_exclusion(tmp_path, monkeypatch):
    """``provider_catalog()`` stays the whole universe — the parity contract
    (tests/hermes_cli/test_provider_parity.py) is asserted against it, and
    "hidden" is a per-install view, not a smaller catalog."""
    _make_home(tmp_path / "home", monkeypatch, excluded=["openrouter", "nous"])
    slugs = [d.slug for d in provider_catalog()]
    assert "openrouter" in slugs and "nous" in slugs


# ------------------------------------------------------------- Keys tab (/api/env)


def test_keys_tab_drops_unset_var_of_excluded_provider(tmp_path, monkeypatch):
    _make_home(tmp_path / "home", monkeypatch, excluded=["openrouter"])
    payload = _env_payload()
    assert "OPENROUTER_API_KEY" not in payload
    # GEMINI_API_KEY is another keys-tab provider's var (NOUS_API_KEY is not a
    # Keys-tab card at all: nous is an Accounts provider).
    assert "GEMINI_API_KEY" in payload, "only the excluded provider's card may disappear"


def test_keys_tab_keeps_set_var_of_excluded_provider(tmp_path, monkeypatch):
    """A stored secret stays visible even for an excluded provider: the GUI must
    remain able to clear it."""
    _make_home(
        tmp_path / "home", monkeypatch, excluded=["openrouter"],
        env_lines="OPENROUTER_API_KEY=sk-or-v1-secret\n",
    )
    row = _env_payload().get("OPENROUTER_API_KEY")
    assert row is not None, "a key on disk must stay listed so it can be removed"
    assert row["is_set"] is True
    assert row["category"] == "provider"


def test_keys_tab_keeps_var_shared_with_a_visible_provider(tmp_path, monkeypatch):
    """DASHSCOPE_API_KEY is declared by four alibaba* providers; excluding one of
    them must not hide the card the others still need — and the surviving row must
    be grouped under a provider that is still visible, not under the one the user
    excluded (the desktop groups cards by ``provider_label``)."""
    owners = [d for d in provider_catalog() if "DASHSCOPE_API_KEY" in d.api_key_env_vars]
    assert len(owners) > 1, "expected a multi-owner env var"
    excluded, remaining = owners[0], owners[1:]
    _make_home(tmp_path / "home", monkeypatch, excluded=[excluded.slug])
    row = _env_payload().get("DASHSCOPE_API_KEY")
    assert row is not None, (
        f"{[d.slug for d in remaining]} still declare it — one excluded owner must not hide the var"
    )
    assert row["provider"] != excluded.slug, "row must not be filed under the excluded provider"
    assert row["provider"] in {d.slug for d in remaining}
    assert row["provider_label"] in {d.label for d in remaining}


def test_keys_tab_hides_aws_vars_only_when_bedrock_excluded_and_unset(tmp_path, monkeypatch):
    """AWS_REGION / AWS_PROFILE reach the Keys tab through ``_AUTH_TYPE_ENV_VARS``
    (bedrock's aws_sdk auth), not through ``api_key_env_vars`` — the exclusion has
    to cover that branch too, and a configured region still has to be editable."""
    _make_home(tmp_path / "unset", monkeypatch, excluded=["bedrock"])
    payload = _env_payload()
    assert "AWS_REGION" not in payload
    assert "AWS_PROFILE" not in payload

    _make_home(
        tmp_path / "set", monkeypatch, excluded=["bedrock"],
        env_lines="AWS_REGION=us-west-2\n",
    )
    payload = _env_payload()
    assert payload.get("AWS_REGION", {}).get("is_set") is True
    assert "AWS_PROFILE" not in payload


def test_keys_tab_builds_the_catalog_once_under_exclusion(tmp_path, monkeypatch):
    """With exclusions configured, one ``GET /api/env`` must build ``provider_catalog()``
    once and read the exclusion list once: the visible catalog is a filter over the list
    already in hand, not a second discovery walk plus a second config read."""
    import hermes_cli.provider_catalog as pc

    _make_home(tmp_path / "home", monkeypatch, excluded=["openrouter"])
    real_catalog, real_excluded = pc.provider_catalog, pc.excluded_provider_slugs
    catalog_calls, excluded_calls = [], []

    def _counted_catalog(*a, **k):
        catalog_calls.append(1)
        return real_catalog(*a, **k)

    def _counted_excluded(*a, **k):
        excluded_calls.append(1)
        return real_excluded(*a, **k)

    monkeypatch.setattr(pc, "provider_catalog", _counted_catalog)
    monkeypatch.setattr(pc, "excluded_provider_slugs", _counted_excluded)
    payload = _env_payload()
    assert "OPENROUTER_API_KEY" not in payload, "the filter must still apply"
    assert len(catalog_calls) == 1, f"provider_catalog() built {len(catalog_calls)}x per request"
    assert len(excluded_calls) == 1, f"excluded_provider_slugs() read {len(excluded_calls)}x per request"


def test_keys_tab_exclusion_follows_the_requested_profile(tmp_path, monkeypatch):
    """``?profile=<name>`` must apply THAT profile's exclusions. ``_profile_scope``
    scopes the config read through the HERMES_HOME contextvar
    (hermes_cli/web_server_profiles.py), so a read outside the block would
    silently apply the default profile's list to every profile's Keys tab."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": "old-model"}))
    (tmp_path / ".env").write_text("")
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"model_catalog": {"excluded_providers": ["openrouter"]}})
    )
    (profile_home / ".env").write_text("")

    assert "OPENROUTER_API_KEY" in _env_payload(), "default profile excludes nothing"
    assert "OPENROUTER_API_KEY" not in _env_payload("?profile=coder")


# ------------------------------------------- Accounts tab (/api/providers/oauth)


def test_accounts_tab_drops_logged_out_excluded_provider(tmp_path, monkeypatch):
    """Covers the hand-written ``_OAUTH_PROVIDER_CATALOG`` cards (anthropic) as well
    as the catalog-derived rows (nous). Driven through the app because that list is
    a LateState proxy (hermes_cli/web_routers/oauth.py). Status is forced
    logged-out so a real login on the developer's machine cannot flip the check."""
    _make_home(tmp_path / "home", monkeypatch, excluded=["nous", "anthropic"])
    with patch.object(_rt_oauth, "_resolve_provider_status", return_value={"logged_in": False}):
        ids = _oauth_ids()
    assert "nous" not in ids
    assert "anthropic" not in ids
    # `claude-code` is an alias card for the same Anthropic credential.
    assert "claude-code" not in ids
    assert ids, "unexcluded providers must still be offered"


def test_accounts_tab_keeps_logged_in_excluded_provider(tmp_path, monkeypatch):
    """A connected account stays listed even when excluded — otherwise the only
    way to disconnect it disappears with the card."""
    _make_home(tmp_path / "home", monkeypatch, excluded=["nous", "anthropic"])

    def _status(provider_id, status_fn):
        return {"logged_in": provider_id == "nous"}

    with patch.object(_rt_oauth, "_resolve_provider_status", side_effect=_status):
        ids = _oauth_ids()
    assert "nous" in ids, "a logged-in account must stay disconnectable"
    assert "anthropic" not in ids
