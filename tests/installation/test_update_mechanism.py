"""The stamp's required ``updateMechanism`` field — the writer's contract.

Writer: scripts/write_install_stamp.py refuses to build a stamp without a
valid mechanism. Stamp readers HARD-FAIL a stamp missing the field: a
mechanism-less stamp is a build-lane bug, and guessing would misroute
updates for every install of that artifact. (The reader-side tests land
with the reader port; the build-lane pin tests land when the packager
lanes are wired to this writer.)
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from write_install_stamp import UPDATE_MECHANISMS, build_stamp  # noqa: E402


class TestWriter:
    @pytest.mark.parametrize("mechanism", UPDATE_MECHANISMS)
    def test_valid_mechanisms_are_emitted(self, mechanism):
        stamp = build_stamp(
            commit="a" * 40, source="ci", update_mechanism=mechanism
        )
        assert stamp["updateMechanism"] == mechanism

    def test_invalid_mechanism_refused(self):
        with pytest.raises(SystemExit):
            build_stamp(commit="a" * 40, update_mechanism="carrier-pigeon")

    def test_mechanism_is_required_by_the_cli(self, tmp_path):
        """argparse enforces --update-mechanism; a lane that forgets it dies."""
        out = tmp_path / "install-stamp.json"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "write_install_stamp.py"),
                "--output", str(out),
                "--commit", "b" * 40,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "--update-mechanism" in result.stderr
        assert not out.exists()

    def test_cli_emits_the_field(self, tmp_path):
        out = tmp_path / "install-stamp.json"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "write_install_stamp.py"),
                "--output", str(out),
                "--commit", "c" * 40,
                "--distribution", "docker",
                "--update-mechanism", "external",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(out.read_text(encoding="utf-8-sig"))["updateMechanism"] == "external"
