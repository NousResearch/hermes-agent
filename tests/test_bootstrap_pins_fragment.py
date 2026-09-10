"""Generated bootstrap pins must match the shared pin and mirror authorities."""
import subprocess
import sys
from pathlib import Path


def test_fragments_match_the_pin_table():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/gen-bootstrap-pins.py"), "--check"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
