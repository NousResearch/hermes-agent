"""scripts/move/rewrite_docs.py: a glob that takes a whole folder with an ambiguous name stays as written."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "move" / "rewrite_docs.py"


def _load():
    spec = importlib.util.spec_from_file_location("rewrite_docs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_whole_folder_glob_on_ambiguous_root_stays_and_file_glob_moves():
    rewrite_docs = _load()
    tree = ["cron/__init__.py", "cron/jobs.py", "cron/scheduler.py"]
    plan = rewrite_docs.Plan({"cron": "hermes_agent.cron"}, [(p, f"hermes_agent/{p}") for p in tree], tree)

    # A profile home's .gitignore: cron/ there is runtime state, not the moved package.
    assert rewrite_docs.rewrite_text("cron/*\n!cron/jobs.json\n", plan) == "cron/*\n!cron/jobs.json\n"
    assert rewrite_docs.rewrite_text("the `cron/**` tree", plan) == "the `cron/**` tree"
    # A pattern for source files still follows the move.
    assert rewrite_docs.rewrite_text("each `cron/*.py` file", plan) == "each `hermes_agent/cron/*.py` file"
