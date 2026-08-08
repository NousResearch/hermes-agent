"""Tests for the multi-board kanban layer (``hermes kanban boards …``).

Covers the pieces added when boards became a first-class concept:

* Slug validation and normalisation.
* Path resolution for ``default`` (legacy ``<root>/kanban.db``) vs
  named boards (``<root>/kanban/boards/<slug>/kanban.db``).
* Current-board persistence via ``<root>/kanban/current`` and
  ``HERMES_KANBAN_BOARD`` env var.
* ``connect(board=)`` isolation — writes on one board don't leak.
* ``create_board`` / ``list_boards`` / ``remove_board`` round trip.
* CLI surface: ``hermes kanban boards list/create/switch/rm``.
* ``_default_spawn`` injects ``HERMES_KANBAN_BOARD`` into worker env.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state.

    The autouse hermetic conftest already nukes credentials + TZ; this
    fixture layers a per-test HERMES_HOME plus a path-init cache reset
    so each test sees a truly empty board set.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    # Also reset hermes_constants cache so get_default_hermes_root() re-reads.
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    # Kanban module-level init cache must not leak between tests.
    kb._INITIALIZED_PATHS.clear()
    return home


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

class TestSlugValidation:
    @pytest.mark.parametrize("good", [
        "default", "atm10-server", "hermes-agent", "proj_1", "a",
        "very-long-but-still-ok-slug-with-hyphens-and-numbers-1234",
    ])
    def test_accepts_valid(self, good):
        assert kb._normalize_board_slug(good) == good


    def test_empty_returns_none(self):
        assert kb._normalize_board_slug(None) is None
        assert kb._normalize_board_slug("") is None
        assert kb._normalize_board_slug("   ") is None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestPathResolution:
    def test_default_board_legacy_path(self, fresh_home):
        """The default board's DB lives at ``<root>/kanban.db`` for back-compat."""
        assert kb.kanban_db_path() == fresh_home / "kanban.db"
        assert kb.kanban_db_path(board="default") == fresh_home / "kanban.db"

    def test_named_board_under_boards_dir(self, fresh_home):
        p = kb.kanban_db_path(board="atm10-server")
        assert p == fresh_home / "kanban" / "boards" / "atm10-server" / "kanban.db"


    def test_env_var_db_override_still_wins(self, fresh_home, tmp_path, monkeypatch):
        """``HERMES_KANBAN_DB`` pins the file regardless of board= arg."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board="ignored") == forced


# ---------------------------------------------------------------------------
# Current-board resolution
# ---------------------------------------------------------------------------

class TestCurrentBoard:



    def test_stale_file_pointer_falls_back_to_default(self, fresh_home):
        current = fresh_home / "kanban" / "current"
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("missing-board\n", encoding="utf-8")

        assert kb.get_current_board() == "default"
        assert not kb.board_exists("missing-board")
        assert [b["slug"] for b in kb.list_boards()] == ["default"]



    def test_kanban_db_path_reads_current(self, fresh_home):
        """kanban_db_path() with no args respects the on-disk pointer."""
        kb.create_board("my-proj")
        kb.set_current_board("my-proj")
        expected = fresh_home / "kanban" / "boards" / "my-proj" / "kanban.db"
        assert kb.kanban_db_path() == expected


# ---------------------------------------------------------------------------
# Board CRUD
# ---------------------------------------------------------------------------

class TestBoardCRUD:






    @pytest.mark.parametrize("archive", [True, False])
    def test_remove_clears_init_cache_for_recreated_db(self, fresh_home, archive):
        # Regression for #23833: poll loops that call connect(board=slug) right
        # after remove_board() recreate an empty kanban.db at the same path
        # (connect() does mkdir(exist_ok=True)). If _INITIALIZED_PATHS still
        # contains the resolved path, the CREATE TABLE pass is skipped and
        # downstream readers hit `no such table: task_events`.
        kb.create_board("recycle")
        # First connect populates _INITIALIZED_PATHS for this DB.
        with kb.connect(board="recycle") as conn:
            kb.create_task(conn, title="t1", assignee="dev")
        db_path = kb.board_dir("recycle") / "kanban.db"
        assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

        kb.remove_board("recycle", archive=archive)
        # remove_board must drop the cache entry so a re-create through
        # connect() gets a fresh schema-init pass.
        assert str(db_path.resolve()) not in kb._INITIALIZED_PATHS

        # Simulate the event-stream poll: re-open the same slug. connect()
        # recreates the directory + empty .db; the schema must be re-applied.
        with kb.connect(board="recycle") as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "task_events" in tables
        assert "tasks" in tables

    def test_rename_updates_metadata(self, fresh_home):
        kb.create_board("slug-immutable")
        kb.write_board_metadata("slug-immutable", name="New Display Name")
        assert kb.read_board_metadata("slug-immutable")["name"] == "New Display Name"
        # Slug must not change.
        assert kb.board_exists("slug-immutable")


class TestBoardPresentation:
    @staticmethod
    def config():
        return {
            "schema": 1,
            "mode": "projection",
            "columns": [
                {
                    "id": "waiting-view",
                    "label": "Waiting",
                    "helper": "Not currently active.",
                    "read_only": True,
                    "match": {
                        "any": [
                            {"status_in": ["todo", "blocked"]},
                            {
                                "all": [
                                    {"status_in": ["running"]},
                                    {"not": {"evidence": "live_worker"}},
                                ]
                            },
                        ]
                    },
                },
                {
                    "id": "active-view",
                    "label": "Active",
                    "helper": "Verified work is active.",
                    "read_only": True,
                    "match": {
                        "all": [
                            {"status_in": ["running"]},
                            {"evidence": "live_worker"},
                        ]
                    },
                },
            ],
            "unmatched": {
                "column": "waiting-view",
                "diagnostic": "No configured rule matched; lifecycle state is unchanged.",
            },
        }

    def test_validated_round_trip_preserves_unrelated_metadata(self, fresh_home):
        kb.create_board("neutral-board", name="Neutral", description="keep me")
        before = kb.read_board_presentation("neutral-board")
        assert before["presentation"] is None

        written = kb.write_board_presentation(
            "neutral-board", self.config(), expected_revision=before["revision"]
        )

        assert written["presentation"] == self.config()
        assert len(written["digest"]) == 64
        metadata = kb.read_board_metadata("neutral-board")
        assert metadata["name"] == "Neutral"
        assert metadata["description"] == "keep me"

    def test_stale_revision_cannot_overwrite_newer_metadata(self, fresh_home):
        kb.create_board("neutral-board", name="Neutral")
        stale = kb.read_board_presentation("neutral-board")["revision"]
        kb.write_board_metadata("neutral-board", description="new metadata")

        with pytest.raises(kb.BoardPresentationConflict):
            kb.write_board_presentation(
                "neutral-board", self.config(), expected_revision=stale
            )
        assert kb.read_board_metadata("neutral-board")["description"] == "new metadata"
        assert kb.read_board_presentation("neutral-board")["presentation"] is None

    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda c: c.update(schema=2), "schema"),
            (lambda c: c.update(mode="canonical"), "mode"),
            (lambda c: c["columns"][0].update(id="running"), "lifecycle status"),
            (lambda c: c["columns"][0].update(label="  "), "label"),
            (lambda c: c["columns"][0].update(read_only=False), "read_only"),
            (lambda c: c["columns"][0].update(match={"python": "pass"}), "predicate"),
            (lambda c: c["unmatched"].update(diagnostic=""), "unmatched diagnostic"),
        ],
    )
    def test_invalid_or_unsafe_config_is_rejected(self, mutate, message):
        config = self.config()
        mutate(config)
        with pytest.raises(ValueError, match=message):
            kb.validate_board_presentation(config)

    def test_deeply_nested_config_is_rejected_without_recursion_crash(self):
        config = self.config()
        predicate: dict[str, object] = {"status_in": ["ready"]}
        for _ in range(10_000):
            predicate = {"not": predicate}
        config["columns"][0]["match"] = predicate

        with pytest.raises(ValueError, match="nesting limit"):
            kb.validate_board_presentation(config)

    def test_recursive_config_is_rejected_with_controlled_validation_error(self):
        config = self.config()
        recursive: dict[str, object] = {"not": {}}
        recursive["not"] = recursive
        config["columns"][0]["match"] = recursive

        with pytest.raises(ValueError, match="recursive structure"):
            kb.validate_board_presentation(config)

    def test_excessive_config_nodes_are_rejected_before_canonical_encoding(self):
        config = self.config()
        config["unknown"] = [None] * 10_000

        with pytest.raises(ValueError, match="structural size limit"):
            kb.validate_board_presentation(config)

    def test_deeply_nested_ondisk_config_falls_back_with_error(self, fresh_home):
        kb.create_board("neutral-board")
        path = kb.board_metadata_path("neutral-board")
        depth = 10_000
        raw = (
            '{"name":"Neutral","presentation":{"schema":1,"mode":"projection",'
            '"columns":[{"id":"queue-view","label":"Queue","helper":"Queue",'
            '"read_only":true,"match":'
            + '{"not":' * depth
            + '{"status_in":["ready"]}'
            + '}' * depth
            + '}],"unmatched":{"column":"queue-view","diagnostic":"No rule matched."}}}'
        )
        path.write_text(raw, encoding="utf-8")

        state = kb.read_board_presentation("neutral-board")
        assert state["presentation"] is None
        assert state["error"] == "board metadata is not valid JSON"

    @pytest.mark.parametrize(
        "malformed",
        [
            b'{"name":"Keep","description":"must survive","presentation":',
            b'[{"unrelated":"must survive"}]',
            b"\xff\xfeinvalid-json-encoding",
            b'{"presentation":' + b"9" * 5_000 + b"}",
            b'{"unrelated":NaN}',
            b'{"unrelated":1e999}',
            b'{"name":"Keep","custom":"first","custom":"second"}',
        ],
    )
    def test_presentation_mutations_refuse_malformed_metadata_without_overwriting(
        self, fresh_home, malformed
    ):
        kb.create_board("neutral-board", name="Keep")
        kb.write_board_metadata("neutral-board", description="must survive")
        path = kb.board_metadata_path("neutral-board")
        path.write_bytes(malformed)
        revision = kb.read_board_presentation("neutral-board")["revision"]

        mutations = (
            lambda: kb.write_board_presentation(
                "neutral-board", self.config(), expected_revision=revision
            ),
            lambda: kb.clear_board_presentation(
                "neutral-board", expected_revision=revision
            ),
            lambda: kb.write_board_metadata("neutral-board", name="Updated"),
        )
        for mutate in mutations:
            with pytest.raises(ValueError, match="metadata is invalid; refusing"):
                mutate()
            assert path.read_bytes() == malformed

    @pytest.mark.parametrize("read_error", [PermissionError, FileNotFoundError])
    def test_metadata_mutations_refuse_unreadable_or_disappearing_metadata(
        self, fresh_home, monkeypatch, read_error
    ):
        kb.create_board("neutral-board", name="Keep")
        path = kb.board_metadata_path("neutral-board")
        original = path.read_bytes()
        real_read_text = Path.read_text

        def denied_read_text(candidate, *args, **kwargs):
            if candidate == path:
                raise read_error("denied")
            return real_read_text(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied_read_text)
        state = kb.read_board_presentation("neutral-board")
        assert state["error"] == "board metadata could not be read"
        mutations = (
            lambda: kb.write_board_presentation(
                "neutral-board", self.config(), expected_revision=state["revision"]
            ),
            lambda: kb.write_board_metadata("neutral-board", name="Updated"),
        )
        for mutate in mutations:
            with pytest.raises(ValueError, match="metadata is invalid; refusing"):
                mutate()
            assert path.read_bytes() == original

    def test_clear_is_revision_guarded(self, fresh_home):
        kb.create_board("neutral-board")
        state = kb.read_board_presentation("neutral-board")
        written = kb.write_board_presentation(
            "neutral-board", self.config(), expected_revision=state["revision"]
        )
        cleared = kb.clear_board_presentation(
            "neutral-board", expected_revision=written["revision"]
        )
        assert cleared["presentation"] is None

    @staticmethod
    def task(*, status="todo", body=None, result=None, block_kind=None):
        return kb.Task(
            id="t_neutral",
            title="Neutral task",
            body=body,
            assignee="worker",
            status=status,
            priority=0,
            created_by="user",
            created_at=1,
            started_at=None,
            completed_at=None,
            workspace_kind="scratch",
            workspace_path=None,
            claim_lock=None,
            claim_expires=None,
            tenant=None,
            block_kind=block_kind,
            result=result,
        )

    def test_projection_evaluates_allowlisted_rules_without_changing_status(self):
        task = self.task(status="running")
        assignment = kb.project_task_presentation(
            task, self.config(), live_worker=True
        )
        assert assignment["column_id"] == "active-view"
        assert assignment["matched"] is True
        assert assignment["canonical_status"] == "running"
        assert task.status == "running"
        assert assignment["diagnostics"] == []

    def test_stale_running_has_explicit_diagnostic(self):
        assignment = kb.project_task_presentation(
            self.task(status="running"), self.config(), live_worker=False
        )
        assert assignment["column_id"] == "waiting-view"
        assert {d["kind"] for d in assignment["diagnostics"]} == {"stale_running"}

    def test_live_worker_requires_matching_current_run_pid_claim_and_heartbeat(
        self, fresh_home
    ):
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="active", assignee="worker")
            claimed = kb.claim_task(conn, task_id, claimer="host:test")
            assert claimed is not None
            assert claimed.current_run_id is not None
            kb._set_worker_pid(conn, task_id, os.getpid())
            assert kb.heartbeat_worker(
                conn, task_id, expected_run_id=claimed.current_run_id
            )
            current = kb.get_task(conn, task_id)
            assert current is not None
            assert current.last_heartbeat_at is not None
            assert current.current_run_id is not None
            assert kb.presentation_live_worker(conn, current) is True

            stale = int(current.last_heartbeat_at) - 10_000
            conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?",
                (stale, task_id),
            )
            conn.execute(
                "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?",
                (stale, current.current_run_id),
            )
            refreshed = kb.get_task(conn, task_id)
            assert refreshed is not None
            assert kb.presentation_live_worker(
                conn, refreshed, now=int(current.last_heartbeat_at)
            ) is False

    def test_live_worker_rejects_any_future_heartbeat(self, fresh_home):
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="active", assignee="worker")
            claimed = kb.claim_task(conn, task_id, claimer="host:test")
            assert claimed is not None
            assert claimed.current_run_id is not None
            kb._set_worker_pid(conn, task_id, os.getpid())
            assert kb.heartbeat_worker(
                conn, task_id, expected_run_id=claimed.current_run_id
            )
            current = kb.get_task(conn, task_id)
            assert current is not None
            assert current.last_heartbeat_at is not None

            assert kb.presentation_live_worker(
                conn, current, now=int(current.last_heartbeat_at) - 1
            ) is False

    def test_needs_input_requires_exact_nonempty_question_and_resume(self):
        config = self.config()
        config["columns"].insert(
            0,
            {
                "id": "input-view",
                "label": "Input",
                "helper": "A concrete answer is required.",
                "read_only": True,
                "match": {
                    "all": [
                        {"status_in": ["blocked"]},
                        {"block_kind_in": ["needs_input"]},
                        {"markers_present": ["Question", "Resumes"]},
                    ]
                },
            },
        )
        valid = self.task(
            status="blocked",
            block_kind="needs_input",
            body="**Question:** Choose one option\n**Resumes:** Validation continues",
        )
        malformed = self.task(
            status="blocked",
            block_kind="needs_input",
            body="Question: vague prose without exact fields",
        )

        assert kb.project_task_presentation(valid, config)["column_id"] == "input-view"
        result = kb.project_task_presentation(malformed, config)
        assert result["column_id"] == "waiting-view"
        assert {d["kind"] for d in result["diagnostics"]} == {"malformed_needs_input"}

        stale_result = self.task(
            status="blocked",
            block_kind="needs_input",
            body="Current body has no structured input gate.",
            result="**Question:** stale question\n**Resumes:** stale condition",
        )
        result = kb.project_task_presentation(stale_result, config)
        assert result["column_id"] == "waiting-view"
        assert {d["kind"] for d in result["diagnostics"]} == {"malformed_needs_input"}

    def test_health_proof_is_structured_current_evidence_not_free_text(self):
        valid = (
            '**Display:** Live\n**Health Proof**: '
            '{"schema":1,"checked_at":950,"max_age_seconds":120,'
            '"healthy":true,"checks":[{"name":"endpoint","ok":true,'
            '"evidence":"https://status.example.test/check/42"}]}'
        )
        stale = valid.replace('"checked_at":950', '"checked_at":800')
        future = valid.replace('"checked_at":950', '"checked_at":1200')
        free_text = "**Display:** Live\n**Health Proof:** looks fine"
        no_evidence = valid.replace(
            ',"evidence":"https://status.example.test/check/42"', ""
        )

        markers = kb.presentation_markers(valid, now=1000)
        assert markers["Display: Live"] is True
        assert markers["Health Proof"] is True
        assert kb.presentation_markers(stale, now=1000)["Health Proof"] is False
        assert kb.presentation_markers(future, now=1000)["Health Proof"] is False
        assert kb.presentation_markers(
            valid.replace('"checked_at":950', '"checked_at":1001'), now=1000
        )["Health Proof"] is False
        assert kb.presentation_markers(free_text, now=1000)["Health Proof"] is False
        assert kb.presentation_markers(no_evidence, now=1000)["Health Proof"] is False
        duplicated = valid + "\n" + valid.splitlines()[1]
        assert kb.presentation_markers(duplicated, now=1000)["Health Proof"] is False
        duplicate_key = valid.replace(
            '"healthy":true', '"healthy":false,"healthy":true'
        )
        assert kb.presentation_markers(duplicate_key, now=1000)["Health Proof"] is False
        deeply_nested = "**Health Proof:** " + "[" * 10_000 + "0" + "]" * 10_000
        assert kb.presentation_markers(deeply_nested, now=1000)["Health Proof"] is False
        oversized_integer = (
            '**Health Proof:** {"schema":1,"checked_at":'
            + "9" * 5_000
            + ',"max_age_seconds":120,"healthy":true,'
            '"checks":[{"name":"endpoint","ok":true,"evidence":"receipt://check"}]}'
        )
        assert kb.presentation_markers(oversized_integer, now=1000)["Health Proof"] is False

    def test_completed_task_requires_display_intent_and_valid_health_for_live_view(self):
        config = {
            "schema": 1,
            "mode": "projection",
            "columns": [
                {
                    "id": "live-view",
                    "label": "Live",
                    "helper": "Completed and currently healthy.",
                    "read_only": True,
                    "match": {
                        "all": [
                            {"status_in": ["done"]},
                            {"markers_present": ["Display: Live"]},
                            {"evidence": "valid_health_proof"},
                        ]
                    },
                },
                {
                    "id": "release-view",
                    "label": "Release",
                    "helper": "Completed without sufficient live evidence.",
                    "read_only": True,
                    "match": {"status_in": ["done"]},
                },
            ],
            "unmatched": {
                "column": "release-view",
                "diagnostic": "No configured projection matched.",
            },
        }
        valid = (
            '**Display:** Live\n**Health Proof:** '
            '{"schema":1,"checked_at":950,"max_age_seconds":120,'
            '"healthy":true,"checks":[{"name":"endpoint","ok":true,'
            '"evidence":"receipt://check/42"}]}'
        )
        free_text = "**Display:** Live\n**Health Proof:** healthy"

        live = kb.project_task_presentation(
            self.task(status="done", body=valid), config, now=1000
        )
        assert live["column_id"] == "live-view"
        assert live["diagnostics"] == []

        insufficient = kb.project_task_presentation(
            self.task(status="done", body=free_text), config, now=1000
        )
        assert insufficient["column_id"] == "release-view"
        assert {item["kind"] for item in insufficient["diagnostics"]} == {
            "invalid_health_proof"
        }

        no_intent = kb.project_task_presentation(
            self.task(status="done", body=valid.replace("**Display:** Live\n", "")),
            config,
            now=1000,
        )
        assert no_intent["column_id"] == "release-view"

    def test_metadata_revision_compare_and_swap_serializes_concurrent_writers(
        self, fresh_home, monkeypatch
    ):
        kb.create_board("neutral-board")
        revision = kb.read_board_presentation("neutral-board")["revision"]
        barrier = threading.Barrier(2)
        real_read = kb.read_board_metadata

        def synchronized_read(board=None):
            value = real_read(board)
            if threading.current_thread().name.startswith("presentation-writer"):
                try:
                    barrier.wait(timeout=0.2)
                except threading.BrokenBarrierError:
                    pass
            return value

        monkeypatch.setattr(kb, "read_board_metadata", synchronized_read)

        def write(label):
            config = self.config()
            config["columns"][0]["label"] = label
            try:
                kb.write_board_presentation(
                    "neutral-board", config, expected_revision=revision
                )
                return "written"
            except kb.BoardPresentationConflict:
                return "conflict"

        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="presentation-writer"
        ) as pool:
            outcomes = list(pool.map(write, ("First", "Second")))

        assert sorted(outcomes) == ["conflict", "written"]

    def test_successful_writer_returns_its_own_committed_state(
        self, fresh_home, monkeypatch
    ):
        kb.create_board("neutral-board")
        initial = kb.read_board_presentation("neutral-board")["revision"]
        first_released = threading.Event()
        second_done = threading.Event()
        real_lock = kb._board_metadata_lock
        results = {}
        errors = {}

        @contextlib.contextmanager
        def delayed_return_lock(path):
            with real_lock(path):
                yield
            if threading.current_thread().name == "writer-first":
                first_released.set()
                assert second_done.wait(timeout=5)

        monkeypatch.setattr(kb, "_board_metadata_lock", delayed_return_lock)

        def writer_first():
            try:
                config = self.config()
                config["columns"][0]["label"] = "First"
                results["first"] = kb.write_board_presentation(
                    "neutral-board", config, expected_revision=initial
                )
            except Exception as exc:
                errors["first"] = exc

        def writer_second():
            try:
                assert first_released.wait(timeout=5)
                current = kb.read_board_presentation("neutral-board")
                config = self.config()
                config["columns"][0]["label"] = "Second"
                results["second"] = kb.write_board_presentation(
                    "neutral-board", config, expected_revision=current["revision"]
                )
            except Exception as exc:
                errors["second"] = exc
            finally:
                second_done.set()

        first = threading.Thread(target=writer_first, name="writer-first")
        second = threading.Thread(target=writer_second, name="writer-second")
        first.start()
        second.start()
        first.join(timeout=10)
        second.join(timeout=10)

        assert not first.is_alive() and not second.is_alive()
        assert errors == {}
        assert results["first"]["presentation"]["columns"][0]["label"] == "First"
        assert results["second"]["presentation"]["columns"][0]["label"] == "Second"
        final = kb.read_board_presentation("neutral-board")
        assert final["presentation"]["columns"][0]["label"] == "Second"

    def test_unmatched_uses_explicit_fallback_diagnostic(self):
        result = kb.project_task_presentation(
            self.task(status="done"), self.config()
        )
        assert result["column_id"] == "waiting-view"
        assert result["matched"] is False
        assert {d["kind"] for d in result["diagnostics"]} == {
            "unmatched_projection"
        }


# ---------------------------------------------------------------------------
# Connection isolation
# ---------------------------------------------------------------------------

class TestConnectionIsolation:
    def test_tasks_do_not_leak_across_boards(self, fresh_home):
        kb.create_board("alpha")
        kb.create_board("beta")

        with kb.connect(board="alpha") as conn:
            kb.create_task(conn, title="alpha-task-1", assignee="dev")
            kb.create_task(conn, title="alpha-task-2", assignee="dev")

        with kb.connect(board="beta") as conn:
            kb.create_task(conn, title="beta-only", assignee="dev")

        with kb.connect(board="alpha") as conn:
            a = kb.list_tasks(conn)
        with kb.connect(board="beta") as conn:
            b = kb.list_tasks(conn)
        with kb.connect(board="default") as conn:
            d = kb.list_tasks(conn)

        assert {t.title for t in a} == {"alpha-task-1", "alpha-task-2"}
        assert {t.title for t in b} == {"beta-only"}
        assert d == []

    def test_connect_without_args_uses_current(self, fresh_home):
        kb.create_board("curr")
        kb.set_current_board("curr")
        with kb.connect() as conn:
            kb.create_task(conn, title="implicit", assignee="x")
        with kb.connect(board="curr") as conn:
            tasks = kb.list_tasks(conn)
        assert [t.title for t in tasks] == ["implicit"]

    def test_connect_env_var_overrides_current(self, fresh_home, monkeypatch):
        kb.create_board("persist")
        kb.create_board("envwin")
        kb.set_current_board("persist")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "envwin")
        with kb.connect() as conn:
            kb.create_task(conn, title="via-env", assignee="x")
        with kb.connect(board="envwin") as conn:
            assert [t.title for t in kb.list_tasks(conn)] == ["via-env"]
        with kb.connect(board="persist") as conn:
            assert kb.list_tasks(conn) == []


# ---------------------------------------------------------------------------
# Worker spawn env injection
# ---------------------------------------------------------------------------

class TestWorkerSpawnEnv:
    """Ensure the dispatcher pins ``HERMES_KANBAN_BOARD`` / DB / workspaces on spawn.

    We monkey-patch ``subprocess.Popen`` to capture the child env without
    actually spawning anything.
    """

    def test_default_spawn_sets_env_vars(self, fresh_home, monkeypatch):
        captured = {}

        class FakeProc:
            pid = 12345

        def fake_popen(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        kb.create_board("spawntest")

        task = kb.Task(
            id="t_abc",
            title="worker test",
            body=None,
            assignee="teknium",
            status="ready",
            priority=0,
            created_by="user",
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="scratch",
            workspace_path=None,
            claim_lock=None,
            claim_expires=None,
            tenant=None,
        )

        kb._default_spawn(task, str(fresh_home / "ws"), board="spawntest")

        env = captured["env"]
        assert env["HERMES_KANBAN_BOARD"] == "spawntest"
        assert env["HERMES_KANBAN_TASK"] == "t_abc"
        # DB path should match the per-board DB, not the legacy default.
        expected_db = fresh_home / "kanban" / "boards" / "spawntest" / "kanban.db"
        assert env["HERMES_KANBAN_DB"] == str(expected_db)
        expected_ws = fresh_home / "kanban" / "boards" / "spawntest" / "workspaces"
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(expected_ws)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def _cli(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``hermes kanban …`` with PYTHONPATH pinned to the worktree."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_WORKTREE)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban"] + args,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
        timeout=30,
    )


class TestCLI:
    def test_boards_list_default_only(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        res = _cli(["boards", "list", "--json"], env_extra=env)
        assert res.returncode == 0, res.stderr
        data = json.loads(res.stdout)
        slugs = [b["slug"] for b in data]
        assert slugs == ["default"]
        assert data[0]["is_current"] is True


    def test_per_board_task_isolation_via_cli(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "projA"], env_extra=env).returncode == 0
        assert _cli(["boards", "create", "projB"], env_extra=env).returncode == 0

        # Create one task on each via --board.
        r = _cli(["--board", "projA", "create", "Task A", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr
        r = _cli(["--board", "projB", "create", "Task B", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr

        # list on each board only shows its own.
        listA = _cli(["--board", "projA", "list", "--json"], env_extra=env)
        listB = _cli(["--board", "projB", "list", "--json"], env_extra=env)
        listD = _cli(["list", "--json"], env_extra=env)

        titlesA = [t["title"] for t in json.loads(listA.stdout)]
        titlesB = [t["title"] for t in json.loads(listB.stdout)]
        titlesD = [t["title"] for t in json.loads(listD.stdout)]

        assert titlesA == ["Task A"]
        assert titlesB == ["Task B"]
        assert titlesD == []



    def test_boards_rm_archives(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        _cli(["boards", "create", "rmme"], env_extra=env)
        r = _cli(["boards", "rm", "rmme"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "archived" in r.stdout
        # Default board list no longer shows it.
        res = _cli(["boards", "list", "--json"], env_extra=env)
        slugs = [b["slug"] for b in json.loads(res.stdout)]
        assert "rmme" not in slugs

    def test_presentation_show_validate_set_and_clear(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "neutral"], env_extra=env).returncode == 0
        candidate = tmp_path / "candidate.json"
        candidate.write_text(
            json.dumps(TestBoardPresentation.config()), encoding="utf-8"
        )

        validate = _cli(
            ["boards", "presentation", "validate", "neutral", "--file", str(candidate)],
            env_extra=env,
        )
        assert validate.returncode == 0, validate.stderr
        assert "valid" in validate.stdout.lower()

        shown = _cli(
            ["boards", "presentation", "show", "neutral", "--json"],
            env_extra=env,
        )
        initial = json.loads(shown.stdout)
        assert initial["presentation"] is None

        set_result = _cli(
            [
                "boards", "presentation", "set", "neutral", "--file", str(candidate),
                "--if-revision", initial["revision"],
            ],
            env_extra=env,
        )
        assert set_result.returncode == 0, set_result.stderr
        written = json.loads(set_result.stdout)
        assert written["presentation"] == TestBoardPresentation.config()

        cleared = _cli(
            [
                "boards", "presentation", "clear", "neutral",
                "--if-revision", written["revision"],
            ],
            env_extra=env,
        )
        assert cleared.returncode == 0, cleared.stderr
        assert json.loads(cleared.stdout)["presentation"] is None

    def test_presentation_cli_rejects_deep_json_without_traceback(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "neutral"], env_extra=env).returncode == 0
        candidate = tmp_path / "deep.json"
        depth = 10_000
        candidate.write_text(
            '{"schema":1,"mode":"projection","columns":[{"id":"queue-view",'
            '"label":"Queue","helper":"Queue","read_only":true,"match":'
            + "[" * depth
            + "0"
            + "]" * depth
            + '}],"unmatched":{"column":"queue-view","diagnostic":"No match."}}',
            encoding="utf-8",
        )
        assert candidate.stat().st_size < kb.PRESENTATION_MAX_BYTES

        result = _cli(
            ["boards", "presentation", "validate", "neutral", "--file", str(candidate)],
            env_extra=env,
        )
        assert "invalid JSON" in result.stderr
        assert "Traceback" not in result.stderr

    def test_presentation_cli_rejects_duplicate_json_keys(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "neutral"], env_extra=env).returncode == 0
        candidate = tmp_path / "duplicate.json"
        raw = json.dumps(TestBoardPresentation.config())
        candidate.write_text(
            raw.replace('"schema": 1', '"schema": 2, "schema": 1', 1),
            encoding="utf-8",
        )

        result = _cli(
            ["boards", "presentation", "validate", "neutral", "--file", str(candidate)],
            env_extra=env,
        )
        assert result.returncode == 2
        assert "duplicate JSON object key" in result.stderr
        assert "Traceback" not in result.stderr

    @pytest.mark.parametrize(
        "command",
        (
            ["boards", "rename", "neutral", "New name"],
            ["boards", "set-default-workdir", "neutral", "/tmp/work"],
        ),
    )
    def test_legacy_board_mutations_reject_malformed_metadata_cleanly(
        self, tmp_path, command
    ):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "neutral"], env_extra=env).returncode == 0
        path = tmp_path / "kanban" / "boards" / "neutral" / "board.json"
        malformed = b'{"name":"first","name":"second"}\n'
        path.write_bytes(malformed)

        result = _cli(command, env_extra=env)

        assert result.returncode == 2
        assert "board metadata is invalid" in result.stderr
        assert "Traceback" not in result.stderr
        assert path.read_bytes() == malformed

