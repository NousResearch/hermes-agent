"""v0.2 feature 4 — Constitutional critic (reward-hacking detector).

Five cases:

1. Cheater synthesised by a mocked critic is flagged (risk ≥ 0.7).
2. Honest candidate passes through with risk ≤ 0.3.
3. Malformed JSON response soft-fails to risk=0 (no crash).
4. The raw ``fitness`` rows in SQLite are untouched by the critic hook;
   penalties only affect in-memory ``Individual.fitness``.
5. Overriding ``critic_model`` routes critic calls to a different model
   (we verify by inspecting the request body through a mocked httpx).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "darwinian-evolver" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import critic     # noqa: E402
import llm        # noqa: E402
import storage    # noqa: E402


def _install_judge_post(response_json: str):
    """Replace httpx.AsyncClient.post with one that returns ``response_json``."""
    original = httpx.AsyncClient.post
    seen_bodies: list[dict] = []

    async def fake_post(self, url, json=None, **kw):
        seen_bodies.append(json)
        class _R:
            status_code = 200
            headers: dict = {}
            def raise_for_status(self_): pass
            def json(self_inner):
                return {
                    "choices": [{"message": {"content": response_json}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                }
        return _R()

    httpx.AsyncClient.post = fake_post  # type: ignore[assignment]

    def restore():
        httpx.AsyncClient.post = original  # type: ignore[assignment]

    return restore, seen_bodies


class TestConstitutionalCritic:
    def test_flags_obvious_cheater(self):
        payload = json.dumps({
            "risk": 0.9,
            "evidence": "pattern-matches the literal expected output",
            "signal_tags": ["literal_short_circuit"],
        })
        restore, _ = _install_judge_post(payload)
        try:
            async def _run():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="") as c:
                    cc = critic.ConstitutionalCritic(client=c, threshold=0.5)
                    return await cc.review("return 'expected'", fitness_value=0.99)
            out = asyncio.run(_run())
            assert out.risk >= 0.7
            assert "literal_short_circuit" in out.signal_tags
        finally:
            restore()

    def test_passes_honest_candidate(self):
        payload = json.dumps({"risk": 0.05, "evidence": "no obvious gaming", "signal_tags": []})
        restore, _ = _install_judge_post(payload)
        try:
            async def _run():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="") as c:
                    cc = critic.ConstitutionalCritic(client=c, threshold=0.5)
                    return await cc.review("Summarize the paragraph in one sentence.")
            out = asyncio.run(_run())
            assert out.risk <= 0.3
            assert out.signal_tags == []
        finally:
            restore()

    def test_malformed_json_soft_fails(self):
        restore, _ = _install_judge_post("I cannot decide.")
        try:
            async def _run():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="") as c:
                    cc = critic.ConstitutionalCritic(client=c, threshold=0.5)
                    return await cc.review("candidate text")
            out = asyncio.run(_run())
            assert out.risk == 0.0
            assert out.evidence == ""
        finally:
            restore()

    def test_penalty_only_applies_above_threshold(self):
        # risk=0.3 < threshold=0.5 → penalty=0
        low = critic.CriticReview(risk=0.3)
        high = critic.CriticReview(risk=0.8)
        cc = critic.ConstitutionalCritic(
            client=type("Stub", (), {"model": "m"})(),  # duck-typed
            threshold=0.5,
        )
        assert cc.penalty(low, fitness=1.0) == 0.0
        # Above threshold: penalty = risk * fitness.
        assert cc.penalty(high, fitness=1.0) == pytest.approx(0.8)

    def test_critic_model_override_routes_to_cheaper_model(self):
        restore, seen = _install_judge_post(json.dumps({"risk": 0.1, "evidence": ""}))
        try:
            async def _run():
                async with llm.LLMClient(model="big-model", base_url="http://x", api_key="") as c:
                    cc = critic.ConstitutionalCritic(
                        client=c, threshold=0.5, model_override="small-cheap",
                    )
                    await cc.review("any candidate")
                    # Also confirm that after the critic call the client
                    # returned to the main model.
                    assert c.model == "big-model"
            asyncio.run(_run())
            # The mocked httpx recorded the request — verify model override.
            assert seen, "expected at least one request"
            assert seen[0]["model"] == "small-cheap"
        finally:
            restore()


class TestCriticAuditIntegrity:
    """Raw fitness rows must not change; only the in-memory Individual.fitness does."""

    def test_sqlite_fitness_row_unchanged_after_review(self, tmp_path):
        conn = storage.open_db(tmp_path / "lineage.db")
        cid = storage.insert_candidate(conn, "cheater", 0)
        storage.record_fitness(conn, cid, "fitness", 0.99)

        # Simulate a review that assigns risk=0.9; write to SQLite via
        # the helper used by the runner; fitness row must stay put.
        storage.record_critic_evaluation(
            conn, cid, 0,
            risk=0.9, evidence="cheats", signal_tags=["literal_short_circuit"],
            model="m",
        )
        row = conn.execute(
            "SELECT value FROM fitness WHERE candidate_id = ?",
            (cid,),
        ).fetchone()
        assert row["value"] == 0.99

        crit_row = storage.get_critic_evaluation(conn, cid, 0)
        assert crit_row is not None
        assert crit_row["risk"] == 0.9
        assert crit_row["signal_tags"] == ["literal_short_circuit"]
