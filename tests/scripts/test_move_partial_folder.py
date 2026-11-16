"""scripts/move: a folder moved only when every file under it moved; files that leave a folder that stays move alone."""

import importlib.util
import sys
from pathlib import Path

MOVE = Path(__file__).resolve().parents[2] / "scripts" / "move"

TREE = ["tools/__init__.py", "tools/registry.py", "tools/cronjob_tools.py", "tools/cronjob_job_args.py",
        "tools/blueprints.py", "tools/blueprint_notes.py", "tools/environments/__init__.py",
        "tools/environments/local.py",
        "tests/tools/test_blueprints.py", "tests/tools/test_registry.py", "cron/__init__.py", "cron/jobs.py"]
PARTIAL = [("tools/cronjob_job_args.py", "cron/job_args.py"), ("tools/blueprints.py", "cron/blueprints.py"),
           ("tools/environments/__init__.py", "sandboxes/__init__.py"),
           ("tools/environments/local.py", "sandboxes/local.py"),
           ("tests/tools/test_blueprints.py", "tests/cron/test_blueprints.py")]
WHOLE = [(path, f"hermes_agent/{path}") for path in TREE if path.startswith("tools/")]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, MOVE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_path_strings_name_a_moved_folder_only_when_the_move_emptied_it():
    renames = _load("renames")

    partial = renames.MovedPaths(tuple(PARTIAL), renames.emptied_folders(TREE, PARTIAL))
    # tools/ and tests/tools/ keep files, so a string that names them names nothing that moved.
    assert partial.find_path("tools/") is None
    assert partial.find_path("tests/tools/") is None
    assert partial.find_path("tools/registry.py") is None
    assert partial.find_path("see tools/cronjob_job_args.py") == ("tools/cronjob_job_args.py", "cron/job_args.py")
    assert partial.find_path("tools/environments/") == ("tools/environments/", "sandboxes/")
    # Path arithmetic is flagged by the outermost name the move took away, never by the folder that stays.
    assert partial.folders == {"cronjob_job_args.py", "blueprints.py", "environments", "test_blueprints.py"}
    assert partial.departed_files == {"cronjob_job_args.py", "blueprints.py", "test_blueprints.py"}

    whole = renames.MovedPaths(tuple(WHOLE), renames.emptied_folders(TREE, WHOLE))
    assert whole.find_path("tools/") == ("tools/", "hermes_agent/tools/")
    assert whole.find_path("tools/registry.py") == ("tools/registry.py", "hermes_agent/tools/registry.py")
    assert whole.find_path("tools/environments/") == ("tools/environments/", "hermes_agent/tools/environments/")
    assert whole.folders == {"tools"}
    assert whole.departed_files == frozenset()


def test_docs_glob_follows_a_move_only_when_every_file_it_matches_moved_together():
    rewrite_docs = _load("rewrite_docs")
    text = ("each `tools/*.py` file, `tools/blueprint*.py`, `tools/cronjob_job_*.py`, `tools/environments/*.py` "
            "and `tools/blueprints.py`")

    partial = rewrite_docs.Plan({"tools.cronjob_job_args": "cron.job_args", "tools.blueprints": "cron.blueprints",
                                 "tools.environments": "sandboxes"}, PARTIAL, TREE)
    stats = rewrite_docs.Stats()
    assert rewrite_docs.rewrite_text(text, partial, stats) == (
        "each `tools/*.py` file, `tools/blueprint*.py`, `tools/cronjob_job_*.py`, `sandboxes/*.py` "
        "and `cron/blueprints.py`")
    assert stats.skips["partial-glob"] == 3

    whole = rewrite_docs.Plan({"tools": "hermes_agent.tools"}, WHOLE, TREE)
    assert rewrite_docs.rewrite_text(text, whole) == (
        "each `hermes_agent/tools/*.py` file, `hermes_agent/tools/blueprint*.py`, "
        "`hermes_agent/tools/cronjob_job_*.py`, `hermes_agent/tools/environments/*.py` "
        "and `hermes_agent/tools/blueprints.py`")
