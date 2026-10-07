"""Regression coverage for Solstice discovery in the stripped PM runtime."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = ROOT / "plugins" / "model-providers"


class SolsticePMImportTests(unittest.TestCase):
    def test_discovery_does_not_require_httpx_at_import_time(self) -> None:
        script = f"""
import json
import sys

class BlockHttpx:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'httpx' or fullname.startswith('httpx.'):
            raise ModuleNotFoundError(\"No module named 'httpx'\", name='httpx')
        return None


import builtins
_REAL_IMPORT = builtins.__import__
def _blocked(name, *args, **kwargs):
    if name == "httpx":
        raise ModuleNotFoundError("No module named 'httpx'", name="httpx")
    return _REAL_IMPORT(name, *args, **kwargs)
builtins.__import__ = _blocked
sys.meta_path.insert(0, BlockHttpx())
sys.path[:0] = [{str(PLUGIN_ROOT)!r}, {str(ROOT)!r}]
import solstice
print(json.dumps({{'loaded': True, 'registered': 'solstice' in sys.modules}}))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"loaded": True, "registered": True})


if __name__ == "__main__":
    unittest.main()
