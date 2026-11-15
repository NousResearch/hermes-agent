"""PLC0415 in the code-health ratchet: under tools/, an import inside a function that main does
not already have fails; the ones main has stay, and a waived lazy import passes."""

from __future__ import annotations

import pytest

from tests.scripts.test_code_health import _commit, _repo, _verdict

_BASE = "def existing():\n    import json\n    return json\n"
_ADDED = "\n\ndef added():\n    import shlex\n    return shlex\n"
_WAIVED = "\n\ndef added():\n    import shlex  # health: allow PLC0415 -- optional extra\n    return shlex\n"
_EDITED = "def existing():\n    import json\n    return json.dumps({})\n"


@pytest.mark.parametrize("path, text, blocks", [
    ("tools/a.py", _BASE + _ADDED, True),
    ("tools/pkg/a.py", _BASE + _ADDED, True),
    ("tools/a.py", _BASE + _WAIVED, False),
    ("tools/a.py", _EDITED, False),
    ("pkg/a.py", _BASE + _ADDED, False),
])
def test_new_function_level_imports_under_tools_block(tmp_path, capsys, path, text, blocks):
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {path: _BASE})
    code, out = _verdict(repo, base, {path: text}, capsys)
    assert code == (1 if blocks else 0), out
    if blocks:
        assert f"{path}:7  PLC0415" in out, out
