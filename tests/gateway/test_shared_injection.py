"""RED tests for event-driven lead shared-memory injection (R5, F-04).

Locked decision 2: per-task completion for leads (event-driven). This seam
injects distilled shared-surface points into the lead's scope when a kanban
task completes. Contract shared with the nightly batch
(scripts/shared_memory_inject.py): same store, same embedding resolver,
source.event=shared_injection provenance.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Severian is an intentionally external Kensei integration, not a core Hermes
# dependency. Its dedicated integration environment supplies the package and
# SQLAlchemy; base CI must skip this module cleanly when that stack is absent.
_HAS_SEVERIAN_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("severian", "sqlalchemy")
)
pytestmark = pytest.mark.skipif(
    not _HAS_SEVERIAN_STACK,
    reason="shared injection requires the Severian integration stack",
)


@pytest.fixture()
def store_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Scratch store — never touches the live shared store."""
    return (tmp_path / "t.db", tmp_path / "t.fts", tmp_path / "t.vec")


@pytest.fixture()
def registered_team(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a team in the module registry (extension point for stand-up)."""
    import gateway.shared_injection as si

    monkeypatch.setitem(
        si.TEAMS,
        "team-content",
        {"lead": "ceecee", "participants": ["ceecee-writer", "ceecee-social"]},
    )


def _build_bundle(store_paths: tuple[Path, Path, Path], embedding: object | None = None):
    from severian.composition import build_bundle

    db, fts, vec = store_paths
    return build_bundle(
        backend="sqlite", database=db, fts=fts, vectors=vec, embedding=embedding,
    )


class TestInjectOnTaskCompletion:
    def test_completion_injects_one_observation_into_lead_scope(
        self, store_paths: tuple[Path, Path, Path], registered_team: None,
    ) -> None:
        """Exact task-completion event causes one idempotent injection."""
        from gateway.shared_injection import inject_on_task_completion

        b = _build_bundle(store_paths)
        try:
            result = inject_on_task_completion(
                bundle=b,
                task_id="task-1",
                title="Ship memory wiring",
                board="main",
                team="team-content",
            )
            assert result["injected"] == 1
            # Second completion for the same task must NOT duplicate.
            again = inject_on_task_completion(
                bundle=b,
                task_id="task-1",
                title="Ship memory wiring",
                board="main",
                team="team-content",
            )
            assert again["injected"] == 0
            assert again["skipped"] == "already_injected"
        finally:
            b.close()

    def test_different_task_injects_again(
        self, store_paths: tuple[Path, Path, Path], registered_team: None,
    ) -> None:
        """A different task still injects — idempotency is per task_id."""
        from gateway.shared_injection import inject_on_task_completion

        b = _build_bundle(store_paths)
        try:
            inject_on_task_completion(
                bundle=b, task_id="task-1", title="First", board="main", team="team-content",
            )
            r2 = inject_on_task_completion(
                bundle=b, task_id="task-2", title="Second", board="main", team="team-content",
            )
            assert r2["injected"] == 1
        finally:
            b.close()

    def test_unknown_team_never_writes(
        self, store_paths: tuple[Path, Path, Path],
    ) -> None:
        """Wrong team/profile cannot cross scope — unknown team is a no-op."""
        from gateway.shared_injection import inject_on_task_completion

        b = _build_bundle(store_paths)
        try:
            result = inject_on_task_completion(
                bundle=b, task_id="task-9", title="X", board="main", team="team-unknown",
            )
            assert result["injected"] == 0
            assert result["skipped"] == "unknown_team"
            # Nothing landed in any scope.
            from severian.domain.models import Scope

            sc = Scope(tenant_id="kensei", profile_id="team-unknown",
                       collection_id="team", agent_id="team-lead", session_id="shared")
            assert b.repository.list_observations(sc) == []
        finally:
            b.close()

    def test_shared_points_are_distilled_from_team_surface(
        self, store_paths: tuple[Path, Path, Path], registered_team: None,
    ) -> None:
        """Injection content carries source.event=shared_injection provenance."""
        from gateway.shared_injection import inject_on_task_completion
        from severian.domain.models import Scope

        b = _build_bundle(store_paths)
        try:
            inject_on_task_completion(
                bundle=b, task_id="task-3", title="Wiring", board="main", team="team-content",
            )
            sc = Scope(tenant_id="kensei", profile_id="team-content",
                       collection_id="team", agent_id="ceecee", session_id="shared")
            obs = b.repository.list_observations(sc)
            assert len(obs) == 1
            src = dict(obs[0].source)
            assert src.get("event") == "shared_injection"
            assert src.get("task_id") == "task-3"
        finally:
            b.close()
