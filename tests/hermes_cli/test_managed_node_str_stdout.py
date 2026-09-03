"""Ad-hoc regression for hermes_constants._managed_node_tree_outdated:
subprocess.run mocking (str stdout) must not crash the outdated-probe.
RED on origin/main 21b2095d0 (AttributeError: 'str' object has no
attribute 'decode'); GREEN with text=True/str-guard fix.

Windows-sensitive: the crash path is reached through the managed-Node
resolution chain used by `hermes update` on every checkout.
"""
import subprocess
from unittest.mock import patch

import pytest

import hermes_constants as hc


@pytest.fixture
def fake_node_tree(tmp_path):
    """A tmp HERMES_HOME-like dir with a node candidate file present."""
    names = hc._candidate_node_command_names("node")
    assert names, "expected at least one candidate node command name"
    made = [tmp_path / name for name in names]
    for p in made:
        p.write_text("", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "stdout_payload",
    [
        b"v24.11.1\n",   # real subprocess (bytes)
        "",              # test-mock class: str stdout, empty
        "v24.11.1\n",    # test-mock class: str stdout, version
    ],
    ids=["bytes", "str-empty", "str-version"],
)
def test_outdated_probe_survives_str_and_bytes_stdout(fake_node_tree, stdout_payload):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout_payload, stderr="")

    with patch.object(hc, "iter_hermes_node_dirs", lambda home=None: [fake_node_tree]), \
         patch("subprocess.run", side_effect=fake_run):
        result = hc._managed_node_tree_outdated(fake_node_tree)

    # Must return a bool, never raise, whatever stdout class arrives.
    assert isinstance(result, bool)


def test_outdated_probe_reads_version_from_str_stdout(fake_node_tree):
    """A str 'v22.0.0' below target major must be seen as outdated (True),
    not crash and not silently report False."""
    target = hc._HERMES_NODE_TARGET_MAJOR

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=f"v{target - 1}.0.0\n", stderr="")

    with patch.object(hc, "iter_hermes_node_dirs", lambda home=None: [fake_node_tree]), \
         patch("subprocess.run", side_effect=fake_run):
        assert hc._managed_node_tree_outdated(fake_node_tree) is True
