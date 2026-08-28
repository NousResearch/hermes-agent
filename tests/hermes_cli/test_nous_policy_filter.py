"""Narrowing the Nous model lists to an org's policy.

The inference gateway omits policy-blocked rows from an authenticated
``GET /v1/models`` with no marker field, so the keys of the authenticated
catalog read are the reachable set. These helpers turn that into a filter the
pickers can apply without a second round trip, and — just as importantly —
decline to filter when the evidence cannot support it.
"""

from __future__ import annotations

import base64
import json

import pytest

import hermes_cli.models as models_mod
import hermes_cli.nous_account as account_mod
from hermes_cli.models import nous_policy_allowed_ids, restrict_to_nous_policy
from hermes_cli.nous_account import nous_policy_present


def _jwt(claims: dict) -> str:
    def seg(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{seg({'alg': 'RS256'})}.{seg(claims)}.sig"


class TestRestrictToNousPolicy:
    def test_none_leaves_the_list_untouched(self):
        ids = ["a/one", "b/two"]
        assert restrict_to_nous_policy(ids, None) == ids

    def test_empty_set_leaves_the_list_untouched(self):
        """Empty is a failed read, not an org that may reach nothing."""
        ids = ["a/one", "b/two"]
        assert restrict_to_nous_policy(ids, set()) == ids

    def test_drops_ids_outside_the_policy(self):
        assert restrict_to_nous_policy(
            ["a/one", "b/two", "c/three"], {"a/one", "c/three"}
        ) == ["a/one", "c/three"]

    def test_preserves_curated_order(self):
        """The pickers show a curated order deliberately; filtering must not
        reorder it into the catalog's alphabetical order."""
        curated = ["z/last", "a/first", "m/middle"]
        allowed = {"a/first", "m/middle", "z/last"}
        assert restrict_to_nous_policy(curated, allowed) == curated

    def test_keeps_a_free_sibling_when_its_base_is_reachable(self):
        """Portal free recommendations are ``:free`` ids; the gateway admits a
        row when any of its requestable ids passes."""
        assert restrict_to_nous_policy(["vendor/model:free"], {"vendor/model"}) == [
            "vendor/model:free"
        ]

    def test_keeps_a_free_id_listed_in_its_own_right(self):
        assert restrict_to_nous_policy(
            ["vendor/model:free"], {"vendor/model:free"}
        ) == ["vendor/model:free"]

    def test_drops_a_free_sibling_whose_base_is_blocked(self):
        assert restrict_to_nous_policy(["vendor/model:free"], {"other/model"}) == []


class TestNousPolicyAllowedIds:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        models_mod._pricing_cache.clear()
        models_mod._pricing_cache_retry_after.clear()
        yield
        models_mod._pricing_cache.clear()
        models_mod._pricing_cache_retry_after.clear()

    def _patch(self, monkeypatch, *, policy_present, api_key="sk-test", pricing=None):
        calls = []
        monkeypatch.setattr(
            account_mod, "nous_policy_present", lambda: policy_present
        )
        monkeypatch.setattr(
            models_mod,
            "_resolve_nous_pricing_credentials",
            lambda: (api_key, "https://inference.example.com"),
        )

        def _fake_fetch(**kwargs):
            calls.append(kwargs)
            return pricing if pricing is not None else {}

        monkeypatch.setattr(models_mod, "fetch_models_with_pricing", _fake_fetch)
        return calls

    def test_returns_the_authenticated_catalog_keys(self, monkeypatch):
        calls = self._patch(
            monkeypatch,
            policy_present=True,
            pricing={"a/one": {}, "b/two": {}},
        )
        assert nous_policy_allowed_ids() == {"a/one", "b/two"}
        assert len(calls) == 1
        assert calls[0]["api_key"] == "sk-test"

    def test_declines_to_filter_an_unrestricted_org(self, monkeypatch):
        calls = self._patch(monkeypatch, policy_present=False, pricing={"a/one": {}})
        assert nous_policy_allowed_ids() is None
        assert calls == [], "an unrestricted org should not pay for the read"

    def test_declines_to_filter_when_the_claim_is_unknown(self, monkeypatch):
        """Absent is an older mint, not an unrestricted org."""
        calls = self._patch(monkeypatch, policy_present=None, pricing={"a/one": {}})
        assert nous_policy_allowed_ids() is None
        assert calls == []

    def test_declines_to_filter_on_an_anonymous_read(self, monkeypatch):
        """An anonymous read returns the full catalog; treating it as the
        policy-filtered set would silently widen the list to everything."""
        self._patch(monkeypatch, policy_present=True, api_key="", pricing={"a/one": {}})
        assert nous_policy_allowed_ids() is None

    def test_declines_to_filter_on_an_empty_read(self, monkeypatch):
        """A failed fetch must not read as an org that may reach nothing."""
        self._patch(monkeypatch, policy_present=True, pricing={})
        assert nous_policy_allowed_ids() is None


class TestNousPolicyPresent:
    def _patch_token(self, monkeypatch, token):
        import hermes_cli.auth as auth_mod

        monkeypatch.setattr(
            auth_mod,
            "get_provider_auth_state",
            lambda _p: {"access_token": token} if token is not None else {},
        )

    @pytest.mark.parametrize("claim,expected", [(True, True), (False, False)])
    def test_reads_the_claim(self, monkeypatch, claim, expected):
        self._patch_token(monkeypatch, _jwt({"policy_present": claim}))
        assert nous_policy_present() is expected

    def test_absent_claim_is_unknown_not_false(self, monkeypatch):
        self._patch_token(monkeypatch, _jwt({"org_id": "org_1"}))
        assert nous_policy_present() is None

    def test_non_boolean_claim_is_unknown(self, monkeypatch):
        """The gateway refuses to read a corrupt claim as "no policy"."""
        self._patch_token(monkeypatch, _jwt({"policy_present": "yes"}))
        assert nous_policy_present() is None

    def test_no_token_is_unknown(self, monkeypatch):
        self._patch_token(monkeypatch, None)
        assert nous_policy_present() is None

    def test_undecodable_token_is_unknown(self, monkeypatch):
        self._patch_token(monkeypatch, "not-a-jwt")
        assert nous_policy_present() is None


class TestNousPolicyNotice:
    """A governed org is told its choice is restricted, rather than left to
    read an omitted model as one Hermes does not support."""

    def _patch(self, monkeypatch, present):
        monkeypatch.setattr(account_mod, "nous_policy_present", lambda: present)

    def test_shows_a_line_for_a_governed_org(self, monkeypatch):
        self._patch(monkeypatch, True)
        assert "restricts which models" in account_mod.nous_policy_notice()

    @pytest.mark.parametrize("present", [False, None])
    def test_silent_otherwise(self, monkeypatch, present):
        """Absent is an older mint, not an unrestricted org — either way there
        is nothing truthful to say."""
        self._patch(monkeypatch, present)
        assert account_mod.nous_policy_notice() == ""

    def test_names_no_models(self, monkeypatch):
        """Policy is an allowlist, so the blocked set is most of the catalog;
        the notice must not try to enumerate it."""
        self._patch(monkeypatch, True)
        notice = account_mod.nous_policy_notice()
        assert "/" not in notice, f"looks like it names a model: {notice}"
        assert len(notice.splitlines()) == 1


class TestAllowlistOutsideTheCuratedList:
    """An allowlist can name a model the curated manifest has never heard of.

    Intersecting alone leaves the picker empty in that case — strictly worse
    than showing an unfiltered list, because the one model the org may use is
    the one that got dropped.
    """

    def test_surfaces_an_allowed_model_the_curated_list_lacks(self):
        assert restrict_to_nous_policy(
            ["vendor/a", "vendor/b"], {"amazon/nova-2-lite-v1"}, rescue_empty=True
        ) == ["amazon/nova-2-lite-v1"]

    def test_does_not_append_when_the_curated_overlap_is_non_empty(self):
        """A jurisdiction or provider policy narrows the catalog without
        emptying the curated overlap. Appending its remainder would push
        non-curated alphabetical ids into a deliberately curated order."""
        kept = restrict_to_nous_policy(
            ["z/curated", "a/curated"],
            {"z/curated", "a/curated", "new/model"},
            rescue_empty=True,
        )
        assert kept == ["z/curated", "a/curated"]

    def test_jurisdiction_policy_never_grows_the_list(self):
        """Regression: a region filter leaves few enough models to slip under
        the size cap, so a size-only guard let it append."""
        curated = ["vendor/one", "vendor/two", "vendor/three"]
        reachable = {"vendor/one", "vendor/two"} | {f"cn/model-{i}" for i in range(20)}
        assert restrict_to_nous_policy(curated, reachable, rescue_empty=True) == [
            "vendor/one",
            "vendor/two",
        ]

    def test_does_not_append_a_free_sibling_already_covered(self):
        assert restrict_to_nous_policy(
            ["vendor/m:free"], {"vendor/m"}, rescue_empty=True
        ) == ["vendor/m:free"]

    def test_a_provider_only_policy_does_not_bury_the_curated_order(self):
        """Such a policy leaves the whole catalog reachable; appending it would
        drop hundreds of alphabetical ids into the picker."""
        curated = ["vendor/one", "vendor/two"]
        catalog = {f"vendor/model-{i}" for i in range(300)} | set(curated)
        assert restrict_to_nous_policy(curated, catalog, rescue_empty=True) == curated


class TestRescueIsOptIn:
    """The empty-intersection rescue is meaningful only for the list a user
    picks from. Any list whose emptiness carries meaning must not get it."""

    def test_no_rescue_by_default(self):
        assert restrict_to_nous_policy([], {"a/one", "b/two"}) == []

    def test_rescue_only_when_asked(self):
        assert restrict_to_nous_policy(
            [], {"a/one"}, rescue_empty=True
        ) == ["a/one"]

    def test_an_already_empty_unavailable_list_is_never_filled(self):
        """Regression: a paid-tier user has no gated models, so the
        unavailable list is legitimately empty. Rescuing it read that as
        "nothing survived" and pushed the whole reachable set into the picker's
        unavailable block."""
        reachable = {f"cn/model-{i}" for i in range(42)}
        assert restrict_to_nous_policy([], reachable) == []

    def test_rescue_does_not_resurrect_a_fully_blocked_list(self):
        """A list whose every entry was blocked is a real filter result, not a
        signal to show something else — unless the caller asked for the
        rescue, which only the selectable list does."""
        assert restrict_to_nous_policy(["x/blocked"], {"y/allowed"}) == []
