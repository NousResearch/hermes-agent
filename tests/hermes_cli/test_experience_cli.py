"""`hermes experience` — the human-facing surface over the experience store.

The feature silently adds context to prompts and silently accumulates rows.
These commands are how a person audits that, so the tests care about two
things: the commands report what is actually stored (not a stale or filtered
view), and the destructive ones cannot delete without saying so.

``why`` gets the most attention: it must run the REAL scoring path, because a
diagnostic that approximates the thing it diagnoses is worse than none.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.experience import Experience, normalize_task, task_fingerprint
from hermes_cli import experience as cli
from hermes_state import SessionDB

WS = "/proj"


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A store the CLI commands will open, seeded with three known rows."""
    database = SessionDB(tmp_path / "state.db")

    def _add(task, outcome, **kw):
        norm = normalize_task(task)
        return database.record_experience(Experience(
            task=task, task_norm=norm, task_hash=task_fingerprint(norm),
            outcome=outcome, workspace=kw.pop("workspace", WS), cwd=WS,
            session_id="s1", **kw
        ).to_row())

    ids = {
        "ok": _add("fix the failing build in the payment module", "success",
                   strategy="used read_file → patch", tools=["read_file", "patch"],
                   verification="passed"),
        "bad": _add("deploy the gateway service to staging", "failure",
                    failure_reason="tool errors from terminal", tools=["terminal"],
                    verification="failed"),
        "other": _add("write unit tests for the retry helper", "partial",
                      workspace="/elsewhere", tools=["write_file"],
                      recovery="retried after failure and succeeded: write_file"),
    }
    monkeypatch.setattr(cli, "_open_db", lambda: SessionDB(tmp_path / "state.db"))
    monkeypatch.setattr(cli, "_current_workspace", lambda: WS)
    yield database, ids
    database.close()


def _args(**kw):
    kw.setdefault("json", False)
    return SimpleNamespace(**kw)


# ── stats ───────────────────────────────────────────────────────────────


class TestStats:
    def test_reports_the_real_counts(self, db, capsys):
        assert cli._cmd_stats(_args()) == 0
        out = capsys.readouterr().out
        assert "3 recorded, 3 live" in out
        assert "tests passed 1" in out
        assert "tests failed 1" in out
        assert "recovered    1" in out

    def test_json_matches_the_store(self, db, capsys):
        database, _ = db
        cli._cmd_stats(_args(json=True))
        assert json.loads(capsys.readouterr().out) == database.experience_stats()

    def test_empty_store_says_so_without_crashing(self, db, capsys):
        database, _ = db
        database.clear_experiences()
        assert cli._cmd_stats(_args()) == 0
        assert "nothing recorded yet" in capsys.readouterr().out


# ── list ────────────────────────────────────────────────────────────────


class TestList:
    def test_lists_every_workspace_by_default(self, db, capsys):
        assert cli._cmd_list(_args(workspace=None, outcome=None, limit=20, all=False)) == 0
        out = capsys.readouterr().out
        assert "payment module" in out and "retry helper" in out

    def test_workspace_filter(self, db, capsys):
        cli._cmd_list(_args(workspace=WS, outcome=None, limit=20, all=False))
        out = capsys.readouterr().out
        assert "payment module" in out
        assert "retry helper" not in out, "a row from another project leaked in"

    def test_dot_resolves_the_current_workspace(self, db, capsys):
        cli._cmd_list(_args(workspace=".", outcome=None, limit=20, all=False))
        out = capsys.readouterr().out
        assert "payment module" in out and "retry helper" not in out

    def test_outcome_filter(self, db, capsys):
        cli._cmd_list(_args(workspace=None, outcome="failure", limit=20, all=False))
        out = capsys.readouterr().out
        assert "gateway service" in out and "payment module" not in out

    def test_limit_is_respected(self, db, capsys):
        cli._cmd_list(_args(workspace=None, outcome=None, limit=1, all=False,
                            json=True))
        assert len(json.loads(capsys.readouterr().out)) == 1

    def test_superseded_rows_are_hidden_unless_asked_for(self, db, capsys):
        database, ids = db
        database.record_experience_correction(ids["ok"], "wrong file")

        cli._cmd_list(_args(workspace=None, outcome=None, limit=20, all=False))
        assert "payment module" not in capsys.readouterr().out

        cli._cmd_list(_args(workspace=None, outcome=None, limit=20, all=True))
        out = capsys.readouterr().out
        assert "payment module" in out
        assert "superseded" in out, "the marker needs explaining, not just showing"

    def test_json_is_machine_readable(self, db, capsys):
        cli._cmd_list(_args(workspace=WS, outcome=None, limit=20, all=False, json=True))
        rows = json.loads(capsys.readouterr().out)
        assert {r["workspace"] for r in rows} == {WS}

    def test_no_match_is_not_an_error(self, db, capsys):
        assert cli._cmd_list(_args(workspace="/nowhere", outcome=None,
                                   limit=20, all=False)) == 0
        assert "no rows match" in capsys.readouterr().out


# ── show ────────────────────────────────────────────────────────────────


class TestShow:
    def test_short_prefix_resolves(self, db, capsys):
        _, ids = db
        assert cli._cmd_show(_args(id=ids["bad"][:8])) == 0
        out = capsys.readouterr().out
        assert ids["bad"] in out
        assert "tool errors from terminal" in out
        assert "verification failed" in out

    def test_full_id_resolves(self, db, capsys):
        _, ids = db
        assert cli._cmd_show(_args(id=ids["ok"])) == 0
        assert "payment module" in capsys.readouterr().out

    def test_unknown_id_fails_loudly(self, db, capsys):
        assert cli._cmd_show(_args(id="deadbeef")) == 1
        assert "no row matching" in capsys.readouterr().err

    def test_ambiguous_prefix_refuses(self, db, capsys):
        """An empty prefix matches every id — refuse rather than pick one."""
        assert cli._cmd_show(_args(id="")) == 1
        assert "ambiguous" in capsys.readouterr().err


# ── why ─────────────────────────────────────────────────────────────────


class TestWhy:
    def test_shows_the_block_that_would_be_injected(self, db, capsys):
        assert cli._cmd_why(_args(query="the payment module build is broken",
                                  workspace=WS)) == 0
        out = capsys.readouterr().out
        assert "<experience-context>" in out
        assert "fix the failing build in the payment module" in out
        assert "not instructions" in out, "the data-boundary note must be visible too"

    def test_shows_the_score_and_the_floor(self, db, capsys):
        cli._cmd_why(_args(query="the payment module build is broken", workspace=WS))
        out = capsys.readouterr().out
        assert "floor" in out and "candidates" in out
        assert "score" in out

    def test_a_non_matching_prompt_says_nothing_would_be_injected(self, db, capsys):
        assert cli._cmd_why(_args(query="what is the weather in Paris",
                                  workspace=WS)) == 0
        out = capsys.readouterr().out
        assert "no match" in out
        assert "<experience-context>" not in out

    def test_it_agrees_with_the_real_retrieval_path(self, db, capsys):
        """A diagnostic that disagrees with the code it explains is a liability."""
        from agent.experience import format_experience_block, rank_rows
        from agent.experience_runtime import experience_config

        database, _ = db
        cfg = experience_config()
        expected = format_experience_block(
            rank_rows(
                database.fetch_experience_candidates(workspace=WS),
                "the payment module build is broken",
                limit=int(cfg["max_results"]),
                min_score=float(cfg["min_score"]),
            ),
            max_chars=int(cfg["max_context_chars"]),
        )

        cli._cmd_why(_args(query="the payment module build is broken",
                           workspace=WS, json=True))
        assert json.loads(capsys.readouterr().out)["block"] == expected

    def test_it_reports_when_the_feature_is_off(self, db, capsys, monkeypatch):
        monkeypatch.setenv("HERMES_EXPERIENCE", "0")
        cli._cmd_why(_args(query="the payment module build is broken", workspace=WS))
        assert "DISABLED" in capsys.readouterr().out

    def test_it_reports_when_only_retrieval_is_off(self, db, capsys, monkeypatch):
        monkeypatch.setenv("HERMES_EXPERIENCE_RETRIEVAL", "0")
        cli._cmd_why(_args(query="the payment module build is broken", workspace=WS))
        assert "retrieval off" in capsys.readouterr().out


# ── forget ──────────────────────────────────────────────────────────────


class TestForget:
    def test_deletes_one_row_with_yes(self, db, capsys):
        database, ids = db
        assert cli._cmd_forget(_args(id=ids["bad"][:8], all=False, yes=True)) == 0
        assert database.get_experience(ids["bad"]) is None
        assert database.experience_stats()["total"] == 2

    def test_delete_is_not_the_same_as_supersede(self, db):
        """`forget` must remove the row, not merely hide it from retrieval."""
        database, ids = db
        cli._cmd_forget(_args(id=ids["ok"], all=False, yes=True))
        assert ids["ok"] not in {r["id"] for r in database.export_experiences()}

    def test_all_clears_the_store(self, db, capsys):
        database, _ = db
        assert cli._cmd_forget(_args(id=None, all=True, yes=True)) == 0
        assert database.experience_stats()["total"] == 0

    def test_refuses_without_yes_when_not_a_terminal(self, db, capsys, monkeypatch):
        """A piped or CI invocation must not silently delete."""
        database, ids = db
        monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
        assert cli._cmd_forget(_args(id=ids["ok"], all=False, yes=False)) == 1
        assert database.get_experience(ids["ok"]) is not None
        assert "refusing to delete" in capsys.readouterr().err

    def test_declining_the_prompt_keeps_the_row(self, db, capsys, monkeypatch):
        database, ids = db
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda *_: "n")
        assert cli._cmd_forget(_args(id=ids["ok"], all=False, yes=False)) == 1
        assert database.get_experience(ids["ok"]) is not None

    def test_accepting_the_prompt_deletes(self, db, monkeypatch):
        database, ids = db
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda *_: "y")
        assert cli._cmd_forget(_args(id=ids["ok"], all=False, yes=False)) == 0
        assert database.get_experience(ids["ok"]) is None

    def test_unknown_id_deletes_nothing(self, db, capsys):
        database, _ = db
        assert cli._cmd_forget(_args(id="deadbeef", all=False, yes=True)) == 1
        assert database.experience_stats()["total"] == 3

    def test_no_id_and_no_all_is_a_usage_error(self, db, capsys):
        assert cli._cmd_forget(_args(id=None, all=False, yes=True)) == 1
        assert "give an id" in capsys.readouterr().err

    def test_all_on_an_empty_store_is_a_no_op(self, db, capsys):
        database, _ = db
        database.clear_experiences()
        assert cli._cmd_forget(_args(id=None, all=True, yes=True)) == 0
        assert "nothing to forget" in capsys.readouterr().out


# ── prune ───────────────────────────────────────────────────────────────


class TestPrune:
    def test_enforces_the_row_cap(self, db, capsys):
        database, _ = db
        assert cli._cmd_prune(_args(max_rows=1, max_age_days=365)) == 0
        assert database.experience_stats()["total"] == 1
        assert "pruned 2 of 3" in capsys.readouterr().out

    def test_a_no_op_prune_reports_zero(self, db, capsys):
        cli._cmd_prune(_args(max_rows=2000, max_age_days=365))
        assert "pruned 0 of 3" in capsys.readouterr().out


# ── wiring ──────────────────────────────────────────────────────────────


class TestWiring:
    def test_every_subcommand_is_registered_with_a_handler(self):
        import argparse

        parser = argparse.ArgumentParser(prog="experience")
        cli.register_cli(parser)
        for argv in (
            ["stats"], ["list"], ["ls"], ["show", "abc"], ["why", "a query"],
            ["forget", "abc"], ["forget", "--all"], ["prune"],
        ):
            args = parser.parse_args(argv)
            assert callable(getattr(args, "func", None)), f"{argv} has no handler"

    def test_bare_invocation_prints_help_rather_than_failing(self, capsys):
        import argparse

        parser = argparse.ArgumentParser(prog="experience")
        cli.register_cli(parser)
        assert parser.parse_args([]).func(None) == 0
        assert "usage" in capsys.readouterr().out.lower()

    def test_main_registers_the_subcommand(self):
        """Guard against the parser wiring being dropped from main.py."""
        import inspect

        from hermes_cli import main as main_mod

        src = inspect.getsource(main_mod)
        assert 'subparsers.add_parser(\n        "experience"' in src
        assert "register_cli as _register_experience_cli" in src
        # main.py only routes names it also lists here.
        assert "experience" in main_mod._BUILTIN_SUBCOMMANDS
