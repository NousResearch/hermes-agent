"""scripts/move: the rename table a move commit records, and the rewrite of old module names."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "move"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"move_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renames_mod = _load("renames")
rw = _load("rewrite_imports")

RENAMES = {
    "watchdog": "hermes_agent.watchdog",
    "tui_gateway": "hermes_agent.tui_gateway",
    "pkg.child": "hermes_agent.child",
}
PATHS = [("watchdog.py", "hermes_agent/watchdog.py"),
         ("tui_gateway/entry.py", "hermes_agent/tui_gateway/entry.py")]


NAMES = {"watchdog": {"arm", "kick"}, "tui_gateway": set(), "tui_gateway.entry": {"main"}, "pkg.child": set()}


def _plan():
    return rw.Plan(RENAMES, PATHS, NAMES)


def _rewrite(source, path="x.py"):
    result = rw.rewrite_source(dedent(source), path, _plan())
    return (result.source if result.source is not None else dedent(source)), result


def check_transformed(before, after):
    out, result = _rewrite(before)
    assert out == dedent(after)
    assert rw.rewrite_source(out, "x.py", _plan()).source is None, "second run must change nothing"
    return result


def check_noop(source):
    out, result = _rewrite(source)
    assert out == dedent(source)
    return result


@pytest.mark.parametrize("before, after", [
    ("import watchdog\n", "import hermes_agent.watchdog\n"),
    ("import watchdog as sw\nsw.arm()\n", "import hermes_agent.watchdog as sw\nsw.arm()\n"),
    ("from watchdog import arm, kick\n", "from hermes_agent.watchdog import arm, kick\n"),
    ("from tui_gateway.entry import main\n", "from hermes_agent.tui_gateway.entry import main\n"),
    ("from tui_gateway import entry\n", "from hermes_agent.tui_gateway import entry\n"),
    ("def f():\n    from watchdog import (\n        arm,\n    )\n",
     "def f():\n    from hermes_agent.watchdog import (\n        arm,\n    )\n"),
    ("import tui_gateway.entry\ntui_gateway.entry.main()\n",
     "import hermes_agent.tui_gateway.entry\nhermes_agent.tui_gateway.entry.main()\n"),
    ("import watchdog\ndef f():\n    return watchdog.arm()\n",
     "import hermes_agent.watchdog\ndef f():\n    return hermes_agent.watchdog.arm()\n"),
    ("from pkg import child\n", "from hermes_agent import child\n"),
    ("from pkg import child as c\n", "from hermes_agent import child as c\n"),
])
def test_imports_and_their_attribute_chains_take_the_new_name(before, after):
    check_transformed(before, after)


@pytest.mark.parametrize("before, after, kind", [
    ('cmd = [sys.executable, "-m", "tui_gateway.entry"]\n',
     'cmd = [sys.executable, "-m", "hermes_agent.tui_gateway.entry"]\n', "argv"),
    ('cmd = (python, "-u", "-m", "watchdog")\n', 'cmd = (python, "-u", "-m", "hermes_agent.watchdog")\n', "argv"),
    ('os.system("python -m tui_gateway.entry --x")\n',
     'os.system("python -m hermes_agent.tui_gateway.entry --x")\n', "string-code"),
    ('CODE = "import tui_gateway.entry\\nmain()"\n', 'CODE = "import hermes_agent.tui_gateway.entry\\nmain()"\n',
     "string-code"),
    ("CODE = r\'\'\'\nfrom watchdog import arm\nPopen([sys.executable, \"-m\", \"tui_gateway.entry\"])\n\'\'\'\n",
     "CODE = r\'\'\'\nfrom hermes_agent.watchdog import arm\nPopen([sys.executable, \"-m\", \"hermes_agent.tui_gateway.entry\"])\n\'\'\'\n",
     "string-code"),
    ('sys.modules.pop("watchdog", None)\n', 'sys.modules.pop("hermes_agent.watchdog", None)\n', "sys-modules"),
    ('x = sys.modules["tui_gateway.entry"]\n', 'x = sys.modules["hermes_agent.tui_gateway.entry"]\n',
     "sys-modules"),
    ('if "watchdog" in sys.modules:\n    pass\n', 'if "hermes_agent.watchdog" in sys.modules:\n    pass\n',
     "sys-modules"),
    ('with mock.patch.dict(sys.modules, {"watchdog": fake}):\n    pass\n',
     'with mock.patch.dict(sys.modules, {"hermes_agent.watchdog": fake}):\n    pass\n', "sys-modules"),
    ('monkeypatch.setitem(sys.modules, "watchdog", fake)\n',
     'monkeypatch.setitem(sys.modules, "hermes_agent.watchdog", fake)\n', "sys-modules"),
    ('monkeypatch.delitem(sys.modules, "watchdog", raising=False)\n',
     'monkeypatch.delitem(sys.modules, "hermes_agent.watchdog", raising=False)\n', "sys-modules"),
    ('log = logging.getLogger("tui_gateway.server")\n',
     'log = logging.getLogger("hermes_agent.tui_gateway.server")\n', "logger"),
    ('caplog.set_level(logging.INFO, logger="watchdog")\n',
     'caplog.set_level(logging.INFO, logger="hermes_agent.watchdog")\n', "logger"),
])
def test_strings_the_strict_finder_cannot_see_are_rewritten(before, after, kind):
    assert check_transformed(before, after).counts == {kind: 1}


@pytest.mark.parametrize("source", [
    "def run(agent):\n    agent.watchdog.arm()\n",
    "watchdog = make()\nwatchdog.arm()\n",
    'msg = t("watchdog.restart.title")\n',
    'msg = "tui_gateway.restarting"\n',
    "from . import watchdog\nfrom .tui_gateway import entry\n",
    "import watchdogs\nimport hermes_agent.watchdog\n",
    'cmd = ["-m", "watchdogs"]\n',
    'help = "use --mode watchdog"\n',
    'msg = "importing watchdog failed"\n',
    'msg = "pick from watchdogs import list"\n',
])
def test_names_that_are_not_the_moved_modules_stay(source):
    assert not check_noop(source).flags


@pytest.mark.parametrize("source, kind, name", [
    ('mock.patch("watchdog.arm")\n', rw.STRING_REF, "watchdog.arm"),
    ('monkeypatch.setattr("tui_gateway.entry.main", f)\n', rw.STRING_REF, "tui_gateway.entry.main"),
    ('importlib.import_module("watchdog")\n', rw.STRING_REF, "watchdog"),
    ('importlib.import_module(f"tui_gateway.{name}")\n', rw.DYNAMIC, 'f"tui_gateway.{name}"'),
    ('sys.modules.pop("tui_gateway." + name)\n', rw.DYNAMIC, '"tui_gateway." + name'),
    ('TARGETS = ["watchdog:arm"]\n', rw.OTHER, "watchdog:arm"),
    ('SKIP = ("tui_gateway/entry.py",)\n', rw.BY_PATH, "tui_gateway/entry.py"),
    ('SKIP = ["tui_gateway/"]\n', rw.BY_PATH, "tui_gateway/"),
    ('p = ROOT / "watchdog.py"\n', rw.BY_PATH, "watchdog.py"),
])
def test_sites_left_alone_are_flagged_with_class_and_name(source, kind, name):
    result = check_noop(source)
    assert len(result.flags) == 1
    line, flag_kind, flag_name, _reason = result.flags[0].split("  ", 3)
    assert (line, flag_kind, flag_name) == ("x.py:1", kind, name)


def test_a_docstring_naming_a_moved_path_is_flagged_as_prose():
    [flag] = check_noop('"""Drives tui_gateway/entry.py over stdio."""\n').flags
    assert flag.split("  ")[1:3] == [rw.BY_PATH, "tui_gateway/entry.py"] and "docstring prose" in flag


def test_a_string_gets_the_same_flags_before_and_after_its_dash_m_rewrite():
    before = '"""Run python -m tui_gateway.entry; see tui_gateway/ws.py."""\n'
    out, first = _rewrite(before)
    assert [f.split("  ")[1:3] for f in first.flags] == [[rw.BY_PATH, "tui_gateway/"]]
    assert rw.rewrite_source(out, "x.py", _plan()).flags == first.flags


def test_file_arithmetic_is_flagged_only_in_moved_files():
    source = "ROOT = Path(__file__).parent.parent\n"
    assert not _rewrite(source)[1].flags
    flags = _rewrite(source, path="hermes_agent/watchdog.py")[1].flags
    assert [f.split("  ")[1] for f in flags] == [rw.FILE_MATH]


def test_a_root_name_shared_with_an_unmoved_import_keeps_the_unmoved_uses():
    out, result = _rewrite("import pkg.child\nimport pkg.keep\npkg.child.go()\npkg.keep.stay()\n")
    assert out == "import hermes_agent.child\nimport pkg.keep\nhermes_agent.child.go()\npkg.keep.stay()\n"
    assert not result.flags


def test_a_bare_package_use_after_a_submodule_import_follows_it():
    check_transformed("import tui_gateway.entry\ntui_gateway.server.run()\n",
                      "import hermes_agent.tui_gateway.entry\nhermes_agent.tui_gateway.server.run()\n")


@pytest.mark.parametrize("source", [
    "import watchdog\nfor watchdog in xs:\n    watchdog.run()\n",
    "import watchdog\ntry:\n    import watchdog\nexcept ImportError:\n    watchdog = None\nif watchdog:\n    pass\n",
    "import watchdog\nmatch v:\n    case Foo() as watchdog:\n        pass\n",
    "import watchdog\ndel watchdog\n",
    "import pkg\npkg.child.f()\n",
])
def test_a_name_with_another_binding_or_reached_through_an_unmoved_import_is_flagged(source):
    out, result = _rewrite(source)
    assert result.flags and all(f.split("  ")[1] == rw.OTHER for f in result.flags)
    compile(out, "x.py", "exec")


def test_a_from_import_mixing_a_moved_apart_child_with_a_kept_one_is_flagged():
    assert check_noop("from pkg import child, other\n").flags[0].split("  ")[1:3] == [rw.OTHER, "pkg"]


@pytest.mark.parametrize("source", ['x = sys.modules["wat\\x63hdog"]\n', 'a = "python -m tui\\x5fgateway.entry\\n"\n'])
def test_a_literal_spelling_the_name_with_escapes_is_flagged_not_counted(source):
    result = check_noop(source)
    assert not result.counts and result.flags[0].split("  ")[1] == rw.OTHER


def test_an_escape_elsewhere_in_the_literal_still_rewrites():
    check_transformed('a = "ExecStart=python -m tui_gateway.entry run\\n"\n',
                      'a = "ExecStart=python -m hermes_agent.tui_gateway.entry run\\n"\n')


def test_logger_by_keyword_and_names_inside_top_level_blocks():
    check_transformed('log = logging.getLogger(name="watchdog")\n',
                      'log = logging.getLogger(name="hermes_agent.watchdog")\n')
    assert rw.top_level_names("try:\n    import y\nexcept E:\n    def g(): pass\nif X:\n    Z = 1\n") >= {"y", "g", "Z"}


def test_a_parse_failure_is_reported_not_raised():
    result = rw.rewrite_source("def broken(:\n", "bad.py", _plan())
    assert result.source is None and result.error.startswith("bad.py: parse error")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=60)


def _write(repo, files):
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _write(tmp_path, {
        "watchdog.py": "def arm():\n    pass\n",
        "tui_gateway/__init__.py": "",
        "tui_gateway/entry.py": "def main():\n    pass\n",
        "tui_gateway/server.py": "from tui_gateway.entry import main\n",
        "hermes_agent/__init__.py": "",
        "hermes_agent/compat/__init__.py": "",
        "hermes_agent/compat/moves.py": '"""Doc."""\n\nMOVES: dict[str, str] = {\n    "old": "watchdog",\n}\n',
        "user.py": "import watchdog\nfrom tui_gateway import server\nwatchdog.arm()\n",
    })
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    _git(tmp_path, "mv", "watchdog.py", "hermes_agent/watchdog.py")
    _git(tmp_path, "mv", "tui_gateway", "hermes_agent/tui_gateway")
    _git(tmp_path, "commit", "-qm", "move")
    _git(tmp_path, "tag", "move")
    return tmp_path


def test_a_moved_package_maps_once_by_its_prefix_and_its_empty_init_is_paired(repo):
    assert renames_mod.module_renames(repo, "move") == {
        "watchdog": "hermes_agent.watchdog",
        "tui_gateway": "hermes_agent.tui_gateway",
    }
    assert ("tui_gateway/__init__.py", "hermes_agent/tui_gateway/__init__.py") in \
        renames_mod.renamed_paths(repo, "move")


def test_identical_empty_inits_pair_by_directory_not_by_content(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _write(tmp_path, {"a/__init__.py": "", "a/x.py": "X = 1\n", "b/__init__.py": "", "b/y.py": "Y = 2\n",
                      "c/__init__.py": "", "c/d/__init__.py": "", "c/d/z.py": "Z = 3\n"})
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "h").mkdir()
    for name in ("a", "c"):
        _git(tmp_path, "mv", name, f"h/{name}")
    _write(tmp_path, {"h/__init__.py": ""})
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "move")
    assert renames_mod.module_renames(tmp_path, "HEAD") == {"a": "h.a", "c": "h.c"}


def _cli(repo, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / "rewrite_imports.py"), "--from-commit", "move", *args],
                          cwd=repo, capture_output=True, text=True, check=False, timeout=120)


def test_the_cli_rewrites_records_moves_and_a_second_run_changes_nothing(repo):
    first = _cli(repo)
    assert first.returncode == 0, first.stdout + first.stderr
    assert first.stdout.splitlines()[-1] == "rewritten=4 flagged=0 failed=0"
    assert (repo / "user.py").read_text() == (
        "import hermes_agent.watchdog\nfrom hermes_agent.tui_gateway import server\nhermes_agent.watchdog.arm()\n")
    moves = (repo / "hermes_agent/compat/moves.py").read_text()
    assert '"old": "hermes_agent.watchdog"' in moves and '"tui_gateway": "hermes_agent.tui_gateway"' in moves
    _git(repo, "commit", "-qam", "rewrite")
    second = _cli(repo, "--check")
    assert (second.returncode, second.stdout.splitlines()[-1]) == (0, "rewritten=0 flagged=0 failed=0")


def test_the_cli_fails_on_a_parse_error_and_on_check_with_pending_changes(repo):
    assert _cli(repo, "--check").returncode == 1
    (repo / "broken.py").write_text("import watchdog\ndef broken(:\n")
    _git(repo, "add", "broken.py")
    result = _cli(repo)
    assert result.returncode == 1 and "failed=1" in result.stdout


def test_the_cli_rewrites_extensionless_files_with_a_python_shebang_only(repo):
    _write(repo, {"hermes": "#!/usr/bin/env python3\nimport watchdog\nwatchdog.arm()\n",
                  "launcher": "#!/bin/sh\nexec python -m tui_gateway.entry import watchdog\n"})
    _git(repo, "add", "-A")
    result = _cli(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (repo / "hermes").read_text() == (
        "#!/usr/bin/env python3\nimport hermes_agent.watchdog\nhermes_agent.watchdog.arm()\n")
    assert (repo / "launcher").read_text().startswith("#!/bin/sh\nexec python -m tui_gateway.entry")
