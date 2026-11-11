"""The ImportError that stubs the Telegram SDK must be logged with its cause (#87455)."""

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_PROBE = """
import importlib.abc, logging, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == {missing!r} or name.startswith({missing!r} + "."):
            raise ModuleNotFoundError("No module named " + repr(name), name=name)
sys.meta_path.insert(0, Block())
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s|%(message)s", stream=sys.stdout)
import plugins.platforms.telegram.adapter as a
print("AVAILABLE", a.TELEGRAM_AVAILABLE)
"""


@pytest.mark.parametrize(("missing", "level"), [("telegram", "DEBUG"), ("telegram.ext", "WARNING")])
def test_sdk_import_failure_is_logged_with_cause(missing, level):
    if missing != "telegram":
        pytest.importorskip("telegram")  # the partial-install leg needs the top-level package present
    out = subprocess.run([sys.executable, "-c", _PROBE.format(missing=missing)], cwd=_ROOT, capture_output=True,
                         text=True, timeout=120, stdin=subprocess.DEVNULL, check=False).stdout
    assert "AVAILABLE False" in out
    line = next(ln for ln in out.splitlines() if "python-telegram-bot import failed" in ln)
    assert line.startswith(level + "|") and missing in line
    assert "Traceback" in out  # the swallowed cause is printed, not just a summary
