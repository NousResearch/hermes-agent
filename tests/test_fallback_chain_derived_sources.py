"""Derived (``{source: ...}``) rungs in the fallback provider chain.

Regression context: on 2026-09-21 every ``claude-apx-N`` rung pinned in ten
profiles' ``fallback_providers`` was simultaneously unusable (seven weekly-capped,
one five-hour-capped, one pointing at a credential that had been disabled in the
registry). Workers that hit a rate limit on the primary walked the entire chain,
were rejected by every rung, and died in about two minutes — while four usable
seats, which appeared in no chain, sat idle. The list was written when those
rungs were healthy and nothing ever re-derived it.

These tests pin the mechanism that makes a chain re-derivable at read time, and
in particular the failure behaviour: the chain is the path you are already on
*because something broke*, so a broken resolver must degrade to "no rungs from
this source", never to an exception.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hermes_cli.fallback_config import (
    get_fallback_chain,
    last_expansion_report,
    register_chain_source,
    registered_chain_sources,
    unregister_chain_source,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fallback_chain_pool_health_20260921.json"


@pytest.fixture
def source():
    """Register a uniquely-named source and always clean it up."""
    registered: list[str] = []

    def _register(name, resolver):
        register_chain_source(name, resolver)
        registered.append(name)
        return name

    yield _register
    for name in registered:
        unregister_chain_source(name)


def _providers(chain):
    return [entry["provider"] for entry in chain]


class TestLiteralChainUnchanged:
    """The derived mechanism must not disturb ordinary hand-written chains."""

    def test_literal_entries_keep_order_and_dedupe(self):
        chain = get_fallback_chain(
            {
                "fallback_providers": [
                    {"provider": "a", "model": "m"},
                    {"provider": "b", "model": "m"},
                    {"provider": "a", "model": "m"},
                ]
            }
        )
        assert _providers(chain) == ["a", "b"]

    def test_legacy_fallback_model_still_appended(self):
        chain = get_fallback_chain(
            {
                "fallback_providers": [{"provider": "a", "model": "m"}],
                "fallback_model": {"provider": "legacy", "model": "m"},
            }
        )
        assert _providers(chain) == ["a", "legacy"]

    def test_malformed_entries_are_skipped_not_fatal(self):
        chain = get_fallback_chain(
            {"fallback_providers": [{"provider": "a"}, None, 7, {"model": "m"}, {"provider": "b", "model": "m"}]}
        )
        assert _providers(chain) == ["b"]


class TestDerivedExpansion:
    def test_source_expands_in_place_between_literal_rungs(self, source):
        source("healthiest", lambda entry: [{"provider": "pool-3", "model": entry["model"]},
                                            {"provider": "pool-1", "model": entry["model"]}])
        chain = get_fallback_chain(
            {
                "fallback_providers": [
                    {"provider": "first", "model": "opus"},
                    {"source": "healthiest", "model": "opus"},
                    {"provider": "terminal", "model": "gpt"},
                ]
            }
        )
        # Position matters: a derived block is not appended to the end, it is
        # spliced where the author put it.
        assert _providers(chain) == ["first", "pool-3", "pool-1", "terminal"]

    def test_resolver_receives_the_entry_so_it_can_read_its_own_options(self, source):
        seen = {}

        def resolver(entry):
            seen.update(entry)
            return [{"provider": "p", "model": entry["model"]}]

        source("s", resolver)
        chain = get_fallback_chain(
            {"fallback_providers": [{"source": "s", "model": "fable", "limit": 4}]}
        )
        assert seen["limit"] == 4 and seen["model"] == "fable"
        assert chain[0]["model"] == "fable"

    def test_derived_rungs_dedupe_against_literal_ones(self, source):
        source("s", lambda e: [{"provider": "dup", "model": "m"}, {"provider": "new", "model": "m"}])
        chain = get_fallback_chain(
            {
                "fallback_providers": [
                    {"provider": "dup", "model": "m"},
                    {"source": "s"},
                ]
            }
        )
        assert _providers(chain) == ["dup", "new"]

    def test_expansion_is_re_evaluated_on_every_read(self, source):
        """The whole point: health changes between calls must be picked up."""
        state = {"healthy": ["a"]}
        source("s", lambda e: [{"provider": p, "model": "m"} for p in state["healthy"]])
        cfg = {"fallback_providers": [{"source": "s"}]}

        assert _providers(get_fallback_chain(cfg)) == ["a"]
        state["healthy"] = ["b", "c"]
        assert _providers(get_fallback_chain(cfg)) == ["b", "c"]

    def test_resolver_output_is_normalised_like_literal_entries(self, source):
        source("s", lambda e: [
            {"provider": "  spaced  ", "model": " m ", "base_url": "http://x/"},
            {"provider": "", "model": "m"},          # dropped: no provider
            "not-a-dict",                             # dropped: wrong type
        ])
        chain = get_fallback_chain({"fallback_providers": [{"source": "s"}]})
        assert _providers(chain) == ["spaced"]
        assert chain[0]["model"] == "m"
        assert chain[0]["base_url"] == "http://x"


class TestDegradation:
    """A chain is the failure path. Nothing here may raise."""

    def test_unregistered_source_is_skipped_and_reported(self, caplog):
        with caplog.at_level(logging.WARNING):
            chain = get_fallback_chain(
                {"fallback_providers": [{"source": "nope"}, {"provider": "term", "model": "m"}]}
            )
        assert _providers(chain) == ["term"]
        assert last_expansion_report()["nope"] == {"count": 0, "error": "no resolver registered"}
        assert "not registered" in caplog.text

    def test_raising_resolver_does_not_propagate(self, source, caplog):
        def boom(entry):
            raise RuntimeError("usage snapshot unreadable")

        source("s", boom)
        with caplog.at_level(logging.WARNING):
            chain = get_fallback_chain(
                {"fallback_providers": [{"source": "s"}, {"provider": "term", "model": "m"}]}
            )
        # The literal terminal rung survives a broken resolver — that is the
        # property that keeps a deployment from losing its whole chain to a bug
        # in the health lookup.
        assert _providers(chain) == ["term"]
        assert last_expansion_report()["s"]["error"].startswith("RuntimeError")

    def test_empty_expansion_is_recorded_so_silence_is_attributable(self, source):
        source("s", lambda e: [])
        chain = get_fallback_chain({"fallback_providers": [{"source": "s"}]})
        assert chain == []
        # An empty chain and a chain whose source produced nothing look identical
        # in a config dump; the report is how an operator tells them apart.
        assert last_expansion_report()["s"] == {"count": 0, "error": None}

    def test_report_is_cleared_between_reads(self, source):
        source("s", lambda e: [{"provider": "a", "model": "m"}])
        get_fallback_chain({"fallback_providers": [{"source": "s"}]})
        get_fallback_chain({"fallback_providers": [{"provider": "a", "model": "m"}]})
        assert last_expansion_report() == {}


class TestRegistration:
    def test_rejects_empty_name_and_non_callable(self):
        with pytest.raises(ValueError):
            register_chain_source("  ", lambda e: [])
        with pytest.raises(TypeError):
            register_chain_source("x", "not-callable")  # type: ignore[arg-type]

    def test_reregistration_replaces_rather_than_duplicates(self, source):
        source("s", lambda e: [{"provider": "old", "model": "m"}])
        register_chain_source("s", lambda e: [{"provider": "new", "model": "m"}])
        chain = get_fallback_chain({"fallback_providers": [{"source": "s"}]})
        assert _providers(chain) == ["new"]
        assert "s" in registered_chain_sources()


class TestAgainstMeasuredPoolHealth:
    """Drive a realistic resolver with the health snapshot from the incident."""

    @staticmethod
    def _load():
        return json.loads(FIXTURE.read_text())

    @staticmethod
    def _healthy(snapshot, registry, now):
        """The policy the fleet actually wants: usable seats, best first.

        Selection is on POSITIVE evidence only — registry-enabled, *observed*
        in the snapshot, weekly window explicitly ``allowed``, and not inside an
        unexpired five-hour cap. Absence of a rejection is deliberately not
        treated as health: sub-vps-10 is enabled in the registry but has no
        usage observation at all, and a "not rejected" rule would have promoted
        an unproven seat into the failure path. ``allowed_warning`` is excluded
        for the same reason — a seat at 96% of its weekly limit will not survive
        the incident you are handing it.

        Ordered by earliest five-hour reset, so the rung most likely to free up
        first is tried first.
        """
        out = []
        for key, snap in snapshot.items():
            if not (registry.get(key) or {}).get("enabled"):
                continue
            five, seven = snap.get("five_hour") or {}, snap.get("seven_day") or {}
            if seven.get("status") != "allowed":
                continue
            if five.get("status") == "rejected" and five.get("resets_at", "") > now:
                continue
            out.append((five.get("resets_at", ""), key))
        return [key for _, key in sorted(out)]

    def test_incident_snapshot_would_have_produced_an_all_dead_chain(self):
        """The pinned rungs really were all unusable — this is the motivation."""
        data = self._load()
        healthy = self._healthy(data["snapshot"], data["registry"], data["evaluate_at"])
        pinned = {p.replace("claude-apx-", "sub-vps-") for p in data["chain_as_pinned"]}
        assert healthy, "fixture must still contain usable seats to contrast against"
        assert pinned.isdisjoint(healthy), "fixture no longer reproduces the incident"

    def test_unobserved_seat_is_not_treated_as_healthy(self):
        """Enabled + never measured != usable (the sub-vps-10 trap)."""
        data = self._load()
        assert data["registry"]["sub-vps-10"]["enabled"] is True
        assert "sub-vps-10" not in data["snapshot"]
        assert "sub-vps-10" not in self._healthy(
            data["snapshot"], data["registry"], data["evaluate_at"]
        )

    def test_derived_source_selects_the_usable_seats_instead(self, source):
        data = self._load()
        now = data["evaluate_at"]
        assert self._healthy(data["snapshot"], data["registry"], now) == data[
            "expected_live_after_resets"
        ]

        source(
            "healthiest-pool-seat",
            lambda entry: [
                {"provider": f"claude-apx-{key.rsplit('-', 1)[-1]}", "model": entry["model"]}
                for key in self._healthy(data["snapshot"], data["registry"], now)
            ],
        )
        chain = get_fallback_chain(
            {
                "fallback_providers": [
                    {"source": "healthiest-pool-seat", "model": "claude-opus-5"},
                    {"provider": "openai-codex", "model": "gpt-6-astra-900k"},
                ]
            }
        )
        # Exactly the four seats measured usable at 2026-09-21T07:41Z, and the
        # non-Claude terminal rung still last.
        assert _providers(chain) == [
            "claude-apx-13",
            "claude-apx-14",
            "claude-apx-16",
            "claude-apx-12",
            "openai-codex",
        ]

    def test_a_terminal_non_pool_rung_survives_a_total_pool_outage(self, source):
        """Every seat capped => the chain is the terminal rung, not empty."""
        data = self._load()
        everything_capped = {
            key: {
                "five_hour": {"status": "rejected", "resets_at": "2099-01-01T00:00:00Z"},
                "seven_day": {"status": "rejected", "resets_at": "2099-01-01T00:00:00Z"},
            }
            for key in data["snapshot"]
        }
        source(
            "healthiest-pool-seat",
            lambda entry: [
                {"provider": f"claude-apx-{k.rsplit('-', 1)[-1]}", "model": entry["model"]}
                for k in self._healthy(everything_capped, data["registry"], data["evaluate_at"])
            ],
        )
        chain = get_fallback_chain(
            {
                "fallback_providers": [
                    {"source": "healthiest-pool-seat", "model": "claude-opus-5"},
                    {"provider": "openai-codex", "model": "gpt-6-astra-900k"},
                ]
            }
        )
        assert _providers(chain) == ["openai-codex"]
