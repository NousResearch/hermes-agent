"""``scripts/check_secret_file_access.py``: every direct ``.env`` / ``auth.json`` site outside the
router is listed per line with a reason, a path handed to a reader imported from the router is not
a site, and the per-file count of listed lines only falls against the base."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("check_secret_file_access", REPO / "scripts" / "check_secret_file_access.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def check(tmp_path, monkeypatch):
    module = _load()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "hermes_cli").mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "ALLOWLIST", tmp_path / "scripts" / "secret_file_access_allowlist.json")
    monkeypatch.setattr(module, "base_allowlist", lambda base, head=None: None)
    return module, tmp_path


def _write(root, rel, src):
    (root / rel).write_text(src, encoding="utf-8")


def _allow(root, entries):
    (root / "scripts" / "secret_file_access_allowlist.json").write_text(json.dumps(entries), encoding="utf-8")


def _lines(module, rel):
    return [text for _lineno, text in module.counts().get(rel, [])]


def test_the_tree_passes_its_own_check():
    assert _load().main([]) == 0


def test_a_new_direct_read_fails(check, capsys):
    module, root = check
    _write(root, "hermes_cli/new_reader.py",
           "import json\ndef read(home):\n    return json.loads((home / 'auth.json').read_text())\n")
    assert module.main([]) == 1
    assert "hermes_cli/new_reader.py:3: a direct .env/auth.json site" in capsys.readouterr().out


@pytest.mark.parametrize("src", [
    "def f(home):\n    return Path(home, '.env')\n",
    "def f(home):\n    return os.path.join(home, 'auth.json')\n",
    "def f():\n    return get_env_path()\n",
    "def f():\n    return _auth_file_path()\n",
    "def f(home):\n    name = 'auth.json'; return (home / name).read_text()\n",
    "def f(home):\n    return open(home + '/.env').read()\n",
    "def f():\n    return Path('~/.hermes/auth.json').expanduser().read_text()\n",
    "def f(home):\n    return [p for p in ('.env',)]\n",
])
def test_every_secret_file_literal_and_path_helper_is_a_site(check, src):
    module, root = check
    _write(root, "hermes_cli/shape.py", src)
    assert [lineno for lineno, _text in module.counts()["hermes_cli/shape.py"]] == [2]


def test_a_path_handed_to_an_imported_router_is_not_a_site(check):
    module, root = check
    _write(root, "hermes_cli/ok.py",
           "from hermes_secret_routing import read_profile_env, read_auth_json as raj\n"
           "import hermes_secret_routing as routing\n"
           "from hermes_cli import auth\n"
           "def f(home):\n"
           "    '''Reads <home>/.env through the store.'''\n"
           "    a = read_profile_env(home / '.env')\n"
           "    b = raj(home / 'auth.json')\n"
           "    c = routing.env_file_exists(home / '.env')\n"
           "    d = auth._load_auth_store(home / 'auth.json')\n"
           "    return a, b, c, d\n")
    assert module.counts() == {}
    assert module.main([]) == 0


@pytest.mark.parametrize("src", [
    # never imported: a same-named function from anywhere else
    "def f(home):\n    return read_profile_env(home / '.env')\n",
    # defined locally: a helper of the same name is not the router
    "from hermes_secret_routing import read_profile_env\n"
    "def read_profile_env(path):\n    return open(path).read()\n"
    "def f(home):\n    return read_profile_env(home / '.env')\n",
    # a local object that happens to have the routed attribute name
    "def f(home, store):\n    return store.read_auth_json(home / 'auth.json')\n",
    # the module name rebound locally
    "import hermes_secret_routing as routing\nrouting = object()\n"
    "def f(home):\n    return routing.read_profile_env(home / '.env')\n",
])
def test_a_reader_that_is_not_the_imported_router_is_not_trusted(check, src):
    module, root = check
    _write(root, "hermes_cli/lookalike.py", src)
    assert len(module.counts()["hermes_cli/lookalike.py"]) == 1


def test_an_allowlisted_file_cannot_add_new_access(check, capsys):
    module, root = check
    _write(root, "hermes_cli/kept.py", "def f(home):\n    return home / '.env'\n")
    _allow(root, {"hermes_cli/kept.py": [{"line": "return home / '.env'", "why": "a path shown to the user"}]})
    assert module.main([]) == 0
    _write(root, "hermes_cli/kept.py",
           "def f(home):\n    return home / '.env'\ndef g(home):\n    return (home / 'auth.json').read_text()\n")
    assert module.main([]) == 1
    assert "hermes_cli/kept.py:4" in capsys.readouterr().out
    # Swapping the allowed site for a new one fails too: the new line is not listed, the old one is gone.
    _write(root, "hermes_cli/kept.py", "def g(home):\n    return (home / 'auth.json').read_text()\n")
    assert module.main([]) == 1
    out = capsys.readouterr().out
    assert "hermes_cli/kept.py:2" in out and "no longer present" in out


def test_every_listed_line_needs_a_reason_and_the_old_count_shape_is_refused(check, capsys):
    module, root = check
    _write(root, "hermes_cli/kept.py", "def f(home):\n    return home / '.env'\n")
    _allow(root, {"hermes_cli/kept.py": [{"line": "return home / '.env'", "why": ""}]})
    assert module.main([]) == 1 and "needs a reason" in capsys.readouterr().out
    _allow(root, {"hermes_cli/kept.py": {"count": 1, "why": "a path shown to the user"}})
    assert module.main([]) == 1 and "must list each allowed line" in capsys.readouterr().out


def test_explicit_allowlist_additions_with_reasons_can_raise_the_file_limit(check, monkeypatch, capsys):
    module, root = check
    _write(root, "hermes_cli/kept.py", "def f(home):\n    return home / '.env'\ndef g(home):\n    return home / 'auth.json'\n")
    two = [{"line": "return home / '.env'", "why": "shown"}, {"line": "return home / 'auth.json'", "why": "shown"}]
    _allow(root, {"hermes_cli/kept.py": two})
    monkeypatch.setattr(module, "base_allowlist", lambda base, head=None: {"hermes_cli/kept.py": two[:1]})
    assert module.main([]) == 0
    monkeypatch.setattr(module, "base_allowlist", lambda base, head=None: {})
    assert module.main([]) == 0
    monkeypatch.setattr(module, "base_allowlist", lambda base, head=None: {"hermes_cli/kept.py": {"count": 2}})
    assert module.main([]) == 0


def test_the_base_allowlist_is_read_from_git(tmp_path, monkeypatch):
    module = _load()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, env=env,
                       check=True, capture_output=True)
    git("init", "-q", "-b", "main")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "secret_file_access_allowlist.json").write_text(
        json.dumps({"a.py": [{"line": "x", "why": "y"}]}), encoding="utf-8")
    git("add", "scripts")
    git("commit", "-q", "-m", "base")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_git", lambda *args: subprocess.run(
        ["git", *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=False).stdout or None)
    assert module.base_allowlist(None) == {"a.py": [{"line": "x", "why": "y"}]}
    assert module.base_allowlist("main") == {"a.py": [{"line": "x", "why": "y"}]}
