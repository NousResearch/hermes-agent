"""Tests for hermes_cli.security_advisories.

The advisory module is the user-facing detection / remediation surface
for supply-chain attacks (e.g. the Mini Shai-Hulud worm of May 2026 that
poisoned mistralai 2.4.6 on PyPI). These tests exercise the public API in
isolation — no real package metadata, no real config, no real cache.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

import hermes_cli.security_advisories as adv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_advisory() -> adv.Advisory:
    """A self-contained Advisory used across tests."""
    return adv.Advisory(
        id="test-advisory-2026-99",
        title="Test advisory",
        summary="Pretend this package has been compromised.",
        url="https://example.com/advisory",
        compromised=(
            ("fake-malicious-pkg", frozenset({"6.6.6"})),
        ),
        remediation=(
            "pip uninstall -y fake-malicious-pkg",
            "Rotate any credentials that may have been exposed.",
        ),
        published="2026-01-01",
        severity="critical",
    )


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HERMES_HOME so banner cache and config writes are sandboxed."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "cache").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def patched_version(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """Override _installed_version with a controllable lookup table."""
    table: dict[str, str] = {}
    monkeypatch.setattr(adv, "_installed_version", lambda pkg: table.get(pkg))
    yield table


# ---------------------------------------------------------------------------
# detect_compromised
# ---------------------------------------------------------------------------


class TestDetectCompromised:
    def test_no_match_returns_empty_list(self, fake_advisory, patched_version):
        # No matching package installed.
        hits = adv.detect_compromised(advisories=[fake_advisory])
        assert hits == []

    def test_exact_version_match(self, fake_advisory, patched_version):
        patched_version["fake-malicious-pkg"] = "6.6.6"
        hits = adv.detect_compromised(advisories=[fake_advisory])
        assert len(hits) == 1
        assert hits[0].advisory.id == fake_advisory.id
        assert hits[0].package == "fake-malicious-pkg"
        assert hits[0].installed_version == "6.6.6"


    def test_empty_compromised_set_matches_any_version(
        self, patched_version
    ):
        # An advisory with an empty version set is a "any version is suspect"
        # wildcard — used when an entire maintainer namespace is owned.
        wildcard = adv.Advisory(
            id="wildcard",
            title="Whole namespace owned",
            summary="x",
            url="x",
            compromised=(("evil-namespace", frozenset()),),
            remediation=("uninstall it",),
        )
        patched_version["evil-namespace"] = "0.0.1"
        hits = adv.detect_compromised(advisories=[wildcard])
        assert len(hits) == 1
        assert hits[0].installed_version == "0.0.1"


# ---------------------------------------------------------------------------
# Acknowledgement persistence
# ---------------------------------------------------------------------------


class TestAck:
    def test_get_acked_ids_empty_when_no_config(self, monkeypatch):
        # load_config raises → returns empty set, doesn't crash.
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert adv.get_acked_ids() == set()



    def test_ack_advisory_persists_id(self, isolated_home, monkeypatch):
        # Stub the config layer end-to-end with a tiny in-memory store so we
        # don't depend on the full hermes_cli.config bootstrap.
        store: dict = {"security": {}}
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: store
        )
        monkeypatch.setattr(
            "hermes_cli.config.save_config",
            lambda cfg: store.update(cfg) or None,
        )
        assert adv.ack_advisory("test-advisory-2026-99") is True
        assert "test-advisory-2026-99" in store["security"]["acked_advisories"]
        # Idempotent.
        adv.ack_advisory("test-advisory-2026-99")
        assert (
            store["security"]["acked_advisories"].count("test-advisory-2026-99")
            == 1
        )



# ---------------------------------------------------------------------------
# Banner cache rate limiting
# ---------------------------------------------------------------------------


class TestBannerCache:
    def test_first_call_returns_due_hits(
        self, fake_advisory, isolated_home, monkeypatch
    ):
        monkeypatch.setattr(adv, "get_acked_ids", lambda: set())
        hit = adv.AdvisoryHit(
            advisory=fake_advisory,
            package="fake-malicious-pkg",
            installed_version="6.6.6",
        )
        due = adv.hits_due_for_banner([hit])
        assert due == [hit]


    def test_call_after_window_re_banners(
        self, fake_advisory, isolated_home, monkeypatch
    ):
        monkeypatch.setattr(adv, "get_acked_ids", lambda: set())
        hit = adv.AdvisoryHit(
            advisory=fake_advisory,
            package="fake-malicious-pkg",
            installed_version="6.6.6",
        )
        adv.hits_due_for_banner([hit])
        # Backdate the cache so it looks like the banner was shown more
        # than 24h ago — should re-banner.
        cache_path = adv._banner_cache_path()
        assert cache_path is not None
        old_lines = cache_path.read_text(encoding="utf-8").splitlines()
        backdated = []
        for line in old_lines:
            parts = line.split(None, 1)
            if len(parts) == 2:
                backdated.append(f"{parts[0]} {time.time() - 48 * 3600}")
        cache_path.write_text("\n".join(backdated) + "\n", encoding="utf-8")
        again = adv.hits_due_for_banner([hit])
        assert again == [hit]

    def test_acked_hits_never_banner(
        self, fake_advisory, isolated_home, monkeypatch
    ):
        monkeypatch.setattr(adv, "get_acked_ids", lambda: {fake_advisory.id})
        hit = adv.AdvisoryHit(
            advisory=fake_advisory,
            package="fake-malicious-pkg",
            installed_version="6.6.6",
        )
        assert adv.hits_due_for_banner([hit]) == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:

    def test_full_remediation_text_contains_all_steps(self, fake_advisory):
        hit = adv.AdvisoryHit(
            advisory=fake_advisory,
            package="fake-malicious-pkg",
            installed_version="6.6.6",
        )
        body = "\n".join(adv.full_remediation_text(hit))
        # All remediation steps must be present.
        for step in fake_advisory.remediation:
            assert step in body
        assert fake_advisory.url in body
        assert fake_advisory.summary in body





# ---------------------------------------------------------------------------
# Real catalog smoke test
# ---------------------------------------------------------------------------


class TestRealCatalog:
    def test_advisories_well_formed(self):
        """Every shipped advisory must be self-consistent.

        Catches data-entry mistakes (empty IDs, missing remediation, bad
        compromised tuples) before they ship.
        """
        seen_ids: set[str] = set()
        for advisory in adv.ADVISORIES:
            assert advisory.id, "advisory has empty id"
            assert advisory.id not in seen_ids, f"duplicate id {advisory.id}"
            seen_ids.add(advisory.id)
            assert advisory.title, f"{advisory.id}: empty title"
            assert advisory.summary, f"{advisory.id}: empty summary"
            assert advisory.remediation, f"{advisory.id}: empty remediation"
            assert advisory.url.startswith("http"), \
                f"{advisory.id}: bad url {advisory.url!r}"
            assert advisory.compromised, \
                f"{advisory.id}: empty compromised tuple"
            for pkg, versions in advisory.compromised:
                assert pkg, f"{advisory.id}: empty package name"
                assert isinstance(versions, frozenset), \
                    f"{advisory.id}: versions must be frozenset"


def _node_advisory(fixed_version: str) -> adv.Advisory:
    return adv.Advisory(
        id="test-node-cve", title="Test Node CVE", summary="test", url="https://example.invalid",
        vulnerable_below=(("node", fixed_version),),
        remediation=("Upgrade via `hermes doctor --upgrade-node`.",))


def test_installed_below_fixed_version_is_a_hit(monkeypatch):
    monkeypatch.setattr(adv, "_installed_node_version", lambda: "22.9.0")
    hits = adv.detect_compromised(advisories=(_node_advisory("22.14.0"),))
    assert len(hits) == 1
    assert hits[0].package == "node"
    assert hits[0].installed_version == "22.9.0"


def test_installed_at_or_above_fixed_version_is_not_a_hit(monkeypatch):
    monkeypatch.setattr(adv, "_installed_node_version", lambda: "22.14.0")
    assert adv.detect_compromised(advisories=(_node_advisory("22.14.0"),)) == []
    monkeypatch.setattr(adv, "_installed_node_version", lambda: "24.0.0")
    assert adv.detect_compromised(advisories=(_node_advisory("22.14.0"),)) == []


def test_no_resolvable_node_never_raises(monkeypatch):
    monkeypatch.setattr(adv, "_installed_node_version", lambda: None)
    assert adv.detect_compromised(advisories=(_node_advisory("22.14.0"),)) == []


def test_installed_node_version_resolves_whichever_node_hermes_uses(monkeypatch):
    """Must go through find_node_executable (managed-first, PATH fallback), not a bespoke
    resolver — reuse the existing precedence rather than duplicating it."""
    with patch("hermes_constants.find_node_executable", return_value="/opt/hermes/node/bin/node") as find, \
         patch("subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "v22.9.0\n"
        assert adv._installed_node_version() == "22.9.0"
    find.assert_called_once_with("node")


def test_prerelease_suffix_is_not_treated_as_below_the_fixed_version(monkeypatch):
    """Documents the deliberate _semver_tuple behavior: a pre-release of the fixed version itself
    (e.g. `24.0.0-rc.1` for a fix landing at `24.0.0`) must NOT count as vulnerable, since the
    suffix is stripped before comparison and the numeric core is equal, not lower."""
    monkeypatch.setattr(adv, "_installed_node_version", lambda: "24.0.0-rc.1")
    assert adv.detect_compromised(advisories=(_node_advisory("24.0.0"),)) == []


def test_multiple_vulnerable_below_sources_in_one_advisory_are_independent(monkeypatch):
    """A single advisory naming both a Node range and a PyPI-package range must report a hit for
    each source independently, not just the first one checked or a merged/deduped single hit."""
    advisory = adv.Advisory(
        id="multi-source-cve", title="t", summary="s", url="https://example.invalid",
        vulnerable_below=(("node", "22.14.0"), ("some-pkg", "3.0.0")),
        remediation=("upgrade",))
    monkeypatch.setattr(adv, "_installed_node_version", lambda: "22.9.0")
    monkeypatch.setattr(adv, "_installed_version", lambda pkg: "2.9.0" if pkg == "some-pkg" else None)

    hits = adv.detect_compromised(advisories=(advisory,))

    packages_hit = {hit.package for hit in hits}
    assert packages_hit == {"node", "some-pkg"}
    assert len(hits) == 2


def test_advisory_with_both_compromised_and_vulnerable_below_reports_both(monkeypatch):
    """The two mechanisms (exact-set `compromised` and range-based `vulnerable_below`) must
    compose within a single advisory, not be mutually exclusive."""
    advisory = adv.Advisory(
        id="combo-advisory", title="t", summary="s", url="https://example.invalid",
        compromised=(("mistralai", frozenset({"2.4.6"})),),
        vulnerable_below=(("node", "22.14.0"),),
        remediation=("upgrade",))
    monkeypatch.setattr(adv, "_installed_version", lambda pkg: "2.4.6" if pkg == "mistralai" else None)
    monkeypatch.setattr(adv, "_installed_node_version", lambda: "22.9.0")

    hits = adv.detect_compromised(advisories=(advisory,))

    packages_hit = {hit.package for hit in hits}
    assert packages_hit == {"mistralai", "node"}


def test_installed_node_version_resolver_exception_never_raises(monkeypatch):
    """If `find_node_executable` itself raises (not just returns None — e.g. a broken import or
    an unexpected internal error), the whole advisory scan must not crash on it."""
    def raise_it(name):
        raise RuntimeError("boom")
    monkeypatch.setattr("hermes_constants.find_node_executable", raise_it)
    assert adv._installed_node_version() is None


def test_installed_node_version_subprocess_exception_never_raises(monkeypatch):
    """`node --version` failing to even execute (missing binary despite find_node_executable
    returning a path, permissions, etc.) must not crash the scan either."""
    def raise_it(*a, **k):
        raise OSError("no such file")
    monkeypatch.setattr("hermes_constants.find_node_executable", lambda name: "/opt/hermes/node/bin/node")
    with patch("subprocess.run", raise_it):
        assert adv._installed_node_version() is None


def test_installed_node_version_empty_output_is_none_not_empty_string(monkeypatch):
    """An empty/garbage `--version` output must resolve to None (never a hit against any range),
    not an empty string that a naive semver-tuple parse could otherwise choke on or silently
    treat as version 0.0.0."""
    with patch("hermes_constants.find_node_executable", return_value="/opt/hermes/node/bin/node"), \
         patch("subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "v\n"
        assert adv._installed_node_version() is None


def test_existing_exact_version_advisories_are_unaffected(monkeypatch):
    """Regression guard: the pre-existing mistralai-style entries must behave identically —
    vulnerable_below defaults to () and contributes nothing when absent."""
    monkeypatch.setattr(adv, "_installed_version", lambda pkg: "2.4.6" if pkg == "mistralai" else None)
    advisory = adv.Advisory(
        id="regression-check", title="t", summary="s", url="https://example.invalid",
        compromised=(("mistralai", frozenset({"2.4.6"})),))
    hits = adv.detect_compromised(advisories=(advisory,))
    assert len(hits) == 1
    assert hits[0].package == "mistralai"
