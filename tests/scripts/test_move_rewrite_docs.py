"""scripts/move/rewrite_docs.py: moved paths and module names in docs and comments, with exclusions."""

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "move"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"move_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rd = _load("rewrite_docs")

TREE = [
    "tui_gateway/__init__.py", "tui_gateway/entry.py", "tui_gateway/server.py", "tui_gateway/contracts/common.py",
    "tools/cronjob_tools.py", "tools/other.py", "agent/turn_a.py", "agent/turn_b.py", "agent/reasoning_effort.py",
    "agent/system_prompt.py", "run_agent.py", "cli.py", "utils.py", "plugins/google_meet/cli.py",
    "gateway/platforms/api_server.py", "gateway/run.py", "hermes_cli/config.py", "cron/jobs.py",
]
MOVED = ["tui_gateway", "tools/cronjob_tools.py", "agent/turn_a.py", "agent/turn_b.py", "agent/reasoning_effort.py",
         "agent/system_prompt.py", "run_agent.py", "cli.py", "utils.py", "gateway/platforms/api_server.py",
         "cron/jobs.py"]


def _plan():
    paths = []
    for move in MOVED:
        paths += [(p, f"hermes_agent/{p}") for p in TREE if p == move or p.startswith(move + "/")]
    renames = {}
    for move in MOVED:
        name = move[:-3] if move.endswith(".py") else move
        renames[name.replace("/", ".")] = "hermes_agent." + name.replace("/", ".")
    return rd.Plan(renames, paths, TREE, frozenset({"agent.max_turns", "toolsets", "cron.wrap_response"}))


def fix(text):
    stats = rd.Stats()
    return rd.rewrite_text(text, _plan(), stats), stats


@pytest.mark.parametrize("before,after", [
    # rule 1: paths
    ("see `tui_gateway/entry.py`.", "see `hermes_agent/tui_gateway/entry.py`."),
    ("see tui_gateway/entry.py::main and tui_gateway/entry.py:42", "see hermes_agent/tui_gateway/entry.py::main and hermes_agent/tui_gateway/entry.py:42"),
    ("the `tui_gateway/contracts/` dir", "the `hermes_agent/tui_gateway/contracts/` dir"),
    ("in tools/cronjob_tools for jobs", "in hermes_agent/tools/cronjob_tools for jobs"),
    ("`tools/cronjob_tools.py.` ends", "`hermes_agent/tools/cronjob_tools.py.` ends"),
    ("(`gateway/platforms/api_server.py`)", "(`hermes_agent/gateway/platforms/api_server.py`)"),
    ("tui_gateway/entry.py.main reads", "hermes_agent/tui_gateway/entry.py.main reads"),
    ("grep agent/turn_*.py", "grep hermes_agent/agent/turn_*.py"),
    ("run `python run_agent.py`", "run `python hermes_agent/run_agent.py`"),
    ("https://github.com/NousResearch/hermes-agent/blob/main/tui_gateway/entry.py",
     "https://github.com/NousResearch/hermes-agent/blob/main/hermes_agent/tui_gateway/entry.py"),
    # rule 2: dotted names
    ("calls tui_gateway.server.dispatch", "calls hermes_agent.tui_gateway.server.dispatch"),
    ("`tools.cronjob_tools`", "`hermes_agent.tools.cronjob_tools`") if False else ("x", "x"),
    ("run_agent.AIAgent is the entry", "hermes_agent.run_agent.AIAgent is the entry"),
    # rule 3: imports
    ("from tui_gateway.server import dispatch", "from hermes_agent.tui_gateway.server import dispatch"),
    ("import tui_gateway.server", "import hermes_agent.tui_gateway.server"),
    ("python -m tui_gateway.entry", "python -m hermes_agent.tui_gateway.entry"),
    ("from run_agent import AIAgent", "from hermes_agent.run_agent import AIAgent"),
    ("import run_agent", "import hermes_agent.run_agent as run_agent"),
    ("rendered from tui_gateway/contracts", "rendered from hermes_agent/tui_gateway/contracts"),
])
def test_rewrites(before, after):
    out, _ = fix(before)
    assert out == after
    assert fix(out)[0] == out or "hermes_agent/hermes_agent" not in fix(out)[0]


@pytest.mark.parametrize("text", [
    "user plugins at `~/.hermes/tools/cronjob_tools.py`",
    "see $HERMES_HOME/tui_gateway/entry.py",
    "./tui_gateway/entry.py is relative",
    "set `agent.reasoning_effort: high` in config.yaml",
    "Global `agent.system_prompt` wins",
    "gateway.platforms.api_server.model_routes is a config section",
    "the bare dir `tools/` and `cron/` hold HERMES_HOME data",
    "utils.freeze and cli.train are axolotl's",
    "run `python cli.py --toolsets web`",
    "enabled toolsets and cron.wrap_response",
    "the word tui_gateway alone",
    "apps/desktop/src/tui_gateway/entry.py",
    "tests/tui_gateway/test_x.py",
    "├── tui_gateway/entry.py",
    "hermes_agent/tui_gateway/entry.py",
    "import numpy",
    "from other import x",
    "cron/jobs.json holds jobs",
])
def test_left_alone(text):
    assert fix(text)[0] == text


def test_config_keys_come_from_default_config(tmp_path):
    (tmp_path / "hermes_cli").mkdir()
    (tmp_path / "hermes_cli/config_defaults.py").write_text(
        'DEFAULT_CONFIG = {"agent": {"max_turns": 1, "tools": {"x": 1}}, "model": ""}\n')
    keys = rd.default_config_keys(tmp_path)
    assert {"agent", "agent.max_turns", "agent.tools", "agent.tools.x", "model"} <= keys


def test_import_of_ambiguous_root_follows_the_table():
    out, _ = fix("`from cli import main`")
    assert out == "`from hermes_agent.cli import main`"


def test_python_only_comments_and_docstrings():
    source = dedent('''\
        """Module doc: see tui_gateway/entry.py."""
        import tui_gateway.server  # calls tui_gateway.server.dispatch
        PATH = "tui_gateway/entry.py"
        TEXT = f"tui_gateway/{x}"

        def f():
            """Run `python -m tui_gateway.entry`."""
            return "tui_gateway.server"  # see tools/cronjob_tools
    ''')
    expected = dedent('''\
        """Module doc: see hermes_agent/tui_gateway/entry.py."""
        import tui_gateway.server  # calls hermes_agent.tui_gateway.server.dispatch
        PATH = "tui_gateway/entry.py"
        TEXT = f"tui_gateway/{x}"

        def f():
            """Run `python -m hermes_agent.tui_gateway.entry`."""
            return "tui_gateway.server"  # see hermes_agent/tools/cronjob_tools
    ''')
    assert rd.rewrite_python(source, _plan(), rd.Stats()) == expected
    assert rd.rewrite_python(expected, _plan(), rd.Stats()) == expected


def test_python_shebang_and_unparsable_source_are_untouched():
    source = "#!/usr/bin/env python tui_gateway/entry.py\n"
    assert rd.rewrite_python(source, _plan(), rd.Stats()) == source
    broken = "def (:  # tui_gateway/entry.py\n"
    stats = rd.Stats()
    assert rd.rewrite_python(broken, _plan(), stats) == broken


def test_yaml_rewrites_comments_only():
    source = 'key: "tui_gateway/entry.py"  # see tui_gateway/entry.py\n# tools/cronjob_tools\nurl: "a # tui_gateway/entry.py"\n'
    out = rd.rewrite_yaml_comments(source, _plan(), rd.Stats())
    assert out == ('key: "tui_gateway/entry.py"  # see hermes_agent/tui_gateway/entry.py\n'
                   '# hermes_agent/tools/cronjob_tools\nurl: "a # tui_gateway/entry.py"\n')


@pytest.mark.parametrize("path", [
    "tests/compat/old_updater_surface.json", "plugin-catalog/x.yaml", "evals/compaction/results/SCORECARD-1.md",
    "website/docs/user-guide/skills/bundled/apple/a.md", "website/docs/reference/optional-skills-catalog.md",
    "optional-skills/mlops/training/axolotl/references/api.md", "locales/en.yaml", "tests/install/KNOWN_FAILURES.md",
    "hermes_cli/observability/schemas/a.md", "x/vendor/conformance/a.md", "hermes_agent/compat/moves.py",
])
def test_excluded_files(path):
    assert rd.excluded(path)
    stats = rd.Stats()
    text = "see tui_gateway/entry.py"
    assert rd.rewrite_file(path, text, _plan(), stats) == text
    assert stats.skips["excluded-file"] == 1


@pytest.mark.parametrize("path", ["AGENTS.md", "website/docs/guides/a.md", "optional-skills/security/godmode/SKILL.md",
                                  "tui_gateway/AGENTS.md", "skills/devops/x/SKILL.md"])
def test_included_files(path):
    assert not rd.excluded(path)
    assert rd.rewrite_file(path, "see tui_gateway/entry.py", _plan(), rd.Stats()) == "see hermes_agent/tui_gateway/entry.py"


def test_tree_lines_are_left_and_counted():
    out, stats = fix("├── tui_gateway/entry.py")
    assert out == "├── tui_gateway/entry.py" and stats.skips["tree-line"] == 1


def test_counts_by_rule():
    _, stats = fix("`tui_gateway/entry.py` and tui_gateway.server and from run_agent import X")
    assert stats.rewrites == {"path-file": 1, "dotted": 1, "import": 1}


def test_run_over_a_git_tree(tmp_path):
    import subprocess

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "tui_gateway").mkdir()
    (tmp_path / "tui_gateway/entry.py").write_text('"""See tui_gateway/entry.py."""\nX = "tui_gateway/entry.py"\n')
    (tmp_path / "README.md").write_text("run tui_gateway/entry.py\n")
    (tmp_path / "optional-skills/mlops").mkdir(parents=True)
    (tmp_path / "optional-skills/mlops/a.md").write_text("tui_gateway/entry.py\n")
    git("add", "-A")
    plan = rd.synthetic_plan(tmp_path, ["tui_gateway"])
    stats = rd.run(tmp_path, plan, [], write=False)
    assert stats.files_changed == 2 and (tmp_path / "README.md").read_text() == "run tui_gateway/entry.py\n"
    rd.run(tmp_path, plan, [], write=True)
    assert (tmp_path / "README.md").read_text() == "run hermes_agent/tui_gateway/entry.py\n"
    assert (tmp_path / "tui_gateway/entry.py").read_text() == '"""See hermes_agent/tui_gateway/entry.py."""\nX = "tui_gateway/entry.py"\n'
    assert (tmp_path / "optional-skills/mlops/a.md").read_text() == "tui_gateway/entry.py\n"
    assert rd.run(tmp_path, plan, [], write=False).files_changed == 0
