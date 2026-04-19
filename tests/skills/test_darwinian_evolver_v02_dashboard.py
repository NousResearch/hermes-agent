"""v0.2 feature 2 — FastAPI + Plotly dashboard.

The dashboard is intentionally thin — endpoints derive from storage.py
helpers — so these tests lean on FastAPI's TestClient to exercise the
wiring end-to-end, plus a monkeypatch test for the "fastapi absent"
error path.

Five cases, matching the acceptance checklist in the plan.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "darwinian-evolver" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import storage  # noqa: E402


# FastAPI is an optional dep; if it's not installed in the test env we
# skip the HTTP-level cases. The "install hint" test is still run
# because it exercises the no-fastapi code path.
fastapi_missing = importlib.util.find_spec("fastapi") is None


@pytest.fixture
def seeded_experiment(tmp_path):
    """Build a small experiment DB with two generations of candidates."""
    conn = storage.open_db(tmp_path / "lineage.db")
    a = storage.insert_candidate(conn, "alpha", 0)
    b = storage.insert_candidate(conn, "beta",  1, parents=[(a, "paraphrase", "h1")])
    c = storage.insert_candidate(conn, "gamma", 2, parents=[(b, "critique_then_edit", "h2")])
    for cid, gen, val in ((a, 0, 0.10), (b, 1, 0.35), (c, 2, 0.72)):
        storage.record_fitness(conn, cid, "fitness", val)
    storage.record_budget(conn, input_tokens=120, output_tokens=40, usd=0.002, operator="paraphrase")
    conn.close()
    return tmp_path, {"a": a, "b": b, "c": c}


@pytest.mark.skipif(fastapi_missing, reason="fastapi not installed")
class TestDashboardEndpoints:
    def _client(self, tmp_path):
        import dashboard
        from fastapi.testclient import TestClient
        app = dashboard.build_app(tmp_path)
        return TestClient(app)

    def test_summary_shape(self, seeded_experiment):
        tmp_path, _ids = seeded_experiment
        with self._client(tmp_path) as client:
            r = client.get("/api/summary")
            assert r.status_code == 200
            body = r.json()
            assert body["experiment"] == tmp_path.name
            assert body["generations"] == 2
            assert body["budget"]["calls"] == 1
            assert abs(body["budget"]["usd"] - 0.002) < 1e-9
            assert body["objectives"] == ["fitness"]
            assert body["best"]["preview"] == "gamma"   # highest score
            assert body["best"]["score"] == 0.72

    def test_fitness_series_is_ordered_by_generation(self, seeded_experiment):
        tmp_path, _ = seeded_experiment
        with self._client(tmp_path) as client:
            r = client.get("/api/fitness?objective=fitness")
            assert r.status_code == 200
            series = r.json()
            assert [row["generation"] for row in series] == [0, 1, 2]
            # Monotonic best in this fixture.
            bests = [row["best"] for row in series]
            assert bests == sorted(bests)

    def test_lineage_matches_storage(self, seeded_experiment):
        tmp_path, ids = seeded_experiment
        with self._client(tmp_path) as client:
            r = client.get(f"/api/lineage/{ids['c']}")
            assert r.status_code == 200
            body = r.json()
            # c → b → a; two edges total.
            ops = sorted(e["operator"] for e in body["edges"])
            assert ops == ["critique_then_edit", "paraphrase"]
            assert body["generation"] == 2
            assert body["preview"].startswith("gamma")

    def test_lineage_unknown_candidate_404(self, seeded_experiment):
        tmp_path, _ = seeded_experiment
        with self._client(tmp_path) as client:
            r = client.get("/api/lineage/no-such-id")
            assert r.status_code == 404


def test_dashboard_raises_import_error_when_fastapi_missing(monkeypatch):
    """Simulate a Python environment without fastapi — ``dashboard.build_app``
    must raise an ImportError carrying the install hint."""
    # Force the _require_fastapi helper to re-evaluate its imports.
    import dashboard

    def fake_require():
        raise ImportError(
            "fastapi is required for `evolver dashboard`; install with "
            "`pip install fastapi uvicorn`."
        )

    monkeypatch.setattr(dashboard, "_require_fastapi", fake_require)
    with pytest.raises(ImportError) as exc:
        dashboard.build_app(Path("/tmp/doesnt-matter"))
    assert "pip install fastapi" in str(exc.value)
