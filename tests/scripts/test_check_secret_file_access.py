"""``scripts/check_secret_file_access.py``: new direct ``.env`` / ``auth.json`` access is rejected,
a path handed to a routed reader is not counted, and the allowlist only ratchets down."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def check(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("check_secret_file_access", REPO / "scripts" / "check_secret_file_access.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "hermes_cli").mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "ALLOWLIST", tmp_path / "scripts" / "secret_file_access_allowlist.json")
    return module, tmp_path


def _write(root, rel, src):
    (root / rel).write_text(src, encoding="utf-8")


def test_the_tree_passes_its_own_check():
    spec = importlib.util.spec_from_file_location("check_secret_file_access", REPO / "scripts" / "check_secret_file_access.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([]) == 0


def test_a_new_direct_read_fails(check, capsys):
    module, root = check
    _write(root, "hermes_cli/new_reader.py",
           "import json\ndef read(home):\n    return json.loads((home / 'auth.json').read_text())\n")
    assert module.main([]) == 1
    assert "hermes_cli/new_reader.py: 1 direct" in capsys.readouterr().out


@pytest.mark.parametrize("src", [
    "def f(home):\n    return Path(home, '.env')\n",
    "def f(home):\n    return os.path.join(home, 'auth.json')\n",
    "def f():\n    return get_env_path()\n",
    "def f():\n    return _auth_file_path()\n",
])
def test_every_path_shape_is_counted(check, src):
    module, root = check
    _write(root, "hermes_cli/shape.py", src)
    assert module.counts() == {"hermes_cli/shape.py": [2]}


def test_routed_readers_markers_and_name_lists_are_not_counted(check):
    module, root = check
    _write(root, "hermes_cli/ok.py",
           "NAMES = ('.env', 'auth.json')\n"
           "def f(home):\n"
           "    a = read_profile_env(home / '.env')\n"
           "    b = read_auth_json(home / 'auth.json')\n"
           "    return home / '.env'  # secret-file: ok — a path shown to the user\n")
    assert module.counts() == {}
    assert module.main([]) == 0


def test_the_allowlist_only_ratchets_down_and_needs_a_reason(check, capsys):
    module, root = check
    _write(root, "hermes_cli/kept.py", "def f(home):\n    return home / '.env'\n")
    allow = root / "scripts" / "secret_file_access_allowlist.json"
    allow.write_text(json.dumps({"hermes_cli/kept.py": {"count": 1, "why": "a path shown to the user"}}), encoding="utf-8")
    assert module.main([]) == 0
    allow.write_text(json.dumps({"hermes_cli/kept.py": {"count": 2, "why": "a path shown to the user"}}), encoding="utf-8")
    assert module.main([]) == 1 and "lower its count" in capsys.readouterr().out
    allow.write_text(json.dumps({"hermes_cli/kept.py": {"count": 1, "why": ""}}), encoding="utf-8")
    assert module.main([]) == 1 and "needs a reason" in capsys.readouterr().out
