"""Kimi Coding catalog alignment: the static curated list must not advertise
ids the coding endpoint rejects.

The kimi-coding static table carried ids the live coding endpoint does not
serve (``kimi-k3`` is rejected with 401 "model id does not exist … use k3";
the k2-thinking/k2-turbo generations are gone from the roster). Curated-first
merging put ``kimi-k3`` at the top of the picker, so picking the FIRST row
suggested by Hermes failed with HTTP 401 (#105650). The endpoint's real
catalog is ``kimi-for-coding``, ``kimi-for-coding-highspeed``, ``k3`` and
``k3-256k`` (verified live 2026-09-21 and recorded in models.dev's
kimi-code-plan-global entry).

These tests pin: (1) the bare ``k3`` slug is present in the curated list so
the picker offers the id the endpoint actually serves; (2) ``kimi-k3`` no
longer leads the list it serves (moonshot legacy users still match it through
the search alias, and the live fetch filters it off non-coding endpoints);
(3) retired generations stay out.
"""

from __future__ import annotations

import hermes_cli.models_catalog_static as statics


def _kimi_curated() -> list[str]:
    return list(statics._PROVIDER_MODELS["kimi-coding"])


class TestKimiCodingCatalogAlignment:
    def test_bare_k3_is_in_curated_list(self):
        """The coding endpoint only serves the bare ``k3`` — the curated list
        must offer it (the search alias maps k3/kimi-k3 for discovery)."""
        assert "k3" in _kimi_curated()

    def test_rejected_kimi_k3_is_not_first(self):
        """``kimi-k3`` was the picker's first row and the endpoint 401s it —
        it must not lead the curated list anymore."""
        curated = _kimi_curated()
        assert not curated or curated[0] != "kimi-k3"

    def test_retired_generations_removed(self):
        """k2-thinking / k2-turbo generations are no longer on the endpoint."""
        curated = [m.lower() for m in _kimi_curated()]
        for retired in ("kimi-k2-thinking", "kimi-k2-thinking-turbo",
                        "kimi-k2-turbo-preview", "kimi-k2-0905-preview"):
            assert retired not in curated, f"retired id {retired} must not be curated"

    def test_live_served_ids_present(self):
        """Every id the endpoint actually serves (2026-09-21) is either curated
        or reachable via its alias."""
        curated = {m.lower() for m in _kimi_curated()}
        for served in ("kimi-for-coding", "kimi-for-coding-highspeed", "k3", "k3-256k"):
            assert served in curated, f"live id {served} missing from curated list"

    def test_k3_dedup_alias_still_holds(self):
        """The picker dedup folds k3 -> kimi-k3; after this change both forms
        never appear as separate rows."""
        from hermes_cli.model_search import model_alias_canonical

        assert model_alias_canonical("k3") == "kimi-k3"
