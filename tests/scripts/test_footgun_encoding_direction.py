"""Encoding-direction diagnostics through the real file scanner, not a cloned loop."""
import importlib.util
from pathlib import Path
import sys

import pytest

READ = "read with encoding='utf-8' (BOM-intolerant — use 'utf-8-sig')"
WRITE = "write with encoding='utf-8-sig' (emits a BOM)"


@pytest.fixture(scope="module")
def linter():
    path = Path(__file__).resolve().parents[2] / "scripts/check-windows-footguns.py"
    spec = importlib.util.spec_from_file_location("encoding_footguns", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source,expected", [
    ('data = path.read_text(encoding="utf-8")', [(1, READ)]),
    ("with open(path, 'r', encoding='utf-8') as f:", [(1, READ)]),
    ("with open(path, encoding='utf-8') as f:", [(1, READ)]),
    ("f = os.fdopen(fd, 'r', encoding='utf-8')", [(1, READ)]),
    ('data = path.read_text(encoding="utf_8")', [(1, READ)]),
    ("with open(path, mode='r', encoding='utf-8') as f:", [(1, READ)]),
    ('path.read_text(encoding="utf-8-sig")', []),
    ('path.write_text(data, encoding="utf-8")', []),
    ("open(path, 'w', encoding='utf-8')", []),
    ("open(path, 'a', encoding='utf-8')", []),
    ("open(path, 'r+', encoding='utf-8')", []),
    ("subprocess.run(cmd, text=True, encoding='utf-8')", []),
    ("path.read_text(encoding='utf-8')  # windows-footgun: ok — owned file", []),
    ("# use path.read_text(encoding='utf-8') here", []),
    ('path.write_text(data, encoding="utf-8-sig")', [(1, WRITE)]),
    ("open(path, 'w', encoding='utf-8-sig')", [(1, WRITE)]),
    ("open(path, 'a', encoding='utf-8-sig')", [(1, WRITE)]),
    ("os.fdopen(fd, 'w', encoding='utf-8-sig')", [(1, WRITE)]),
    ('path.open("a", encoding="utf-8-sig")', [(1, WRITE)]),
    ('path.open("a+", encoding="utf-8-sig")', [(1, WRITE)]),
    ('path.open("r+", encoding="utf-8-sig")', [(1, WRITE)]),
    ('path.write_text(data, encoding="utf_8_sig")', [(1, WRITE)]),
    ("open(path, 'r', encoding='utf-8-sig')", []),
    ("open(path, encoding='utf-8-sig')", []),
    ("codec = 'utf-8-sig'", []),
    ('open(os.path.join(root, "install-stamp.json"), encoding="utf-8-sig")', []),
    ('path.open("r", encoding="utf-8-sig")', []),
    ('path.open("a", encoding="utf-8")', []),
    ('something.open(encoding="utf-8")', []),
    ("path.write_text(data, encoding='utf-8-sig')  # windows-footgun: ok — BOM required", []),
    ("open(p, 'x', encoding='utf-8-sig')", [(1, WRITE)]),
    ("enc = 'utf-8'", []),
    ("path.read_text(encoding='utf-8') # merely mentions windows-footgun", [(1, READ)]),
    ("path.read_text(encoding='utf-8') if hasattr(path, 'read_text') else ''", [(1, READ)]),
    ("path.read_text(encoding='utf-8') if sys.platform != 'win32' else ''", []),
    ('"""Example:\npath.read_text(encoding="utf-8")\n"""\npath.read_text(encoding="utf-8")', [(4, READ)]),
])
def test_encoding_diagnostics(linter, tmp_path, source, expected):
    path = tmp_path / "input.py"
    path.write_text(source + "\n", encoding="utf-8")
    rules = [rule for rule in linter.FOOTGUNS if rule.name in {READ, WRITE}]
    assert {rule.name for rule in rules} == {READ, WRITE}
    assert [(line, rule.name) for line, _, rule in linter.scan_file(path, rules)] == expected
