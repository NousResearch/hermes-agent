"""Unit tests for hermes_constants.node_cli_launch_for_shim.

Regression for the Windows NBSP-in-profile-path bug: cmd.exe truncates batch
shim paths containing Unicode whitespace (U+00A0), so npm/npx launches must
be rewritten to node.exe + the CLI script, which CreateProcess runs natively
(Unicode-safe). The ``windows`` flag keeps the derivation testable on POSIX.
"""

from hermes_constants import node_cli_launch_for_shim


def _fake_node_tree(tmp_path):
    node = tmp_path / "node"
    (node / "node_modules" / "npm" / "bin").mkdir(parents=True)
    (node / "node_modules" / "npx" / "bin").mkdir(parents=True)
    (node / "node_modules" / "npm" / "bin" / "npm-cli.js").write_text("", encoding="utf-8")
    (node / "node_modules" / "npx" / "bin" / "npx-cli.js").write_text("", encoding="utf-8")
    (node / "node.exe").write_text("", encoding="utf-8")
    (node / "npm.cmd").write_text("", encoding="utf-8")
    (node / "npx.cmd").write_text("", encoding="utf-8")
    return node


def test_posix_path_never_rewritten():
    assert node_cli_launch_for_shim("/usr/bin/npm", windows=False) == ["/usr/bin/npm"]


def test_windows_npm_shim_resolves_to_node_launch(tmp_path):
    node = _fake_node_tree(tmp_path)
    shim = str(node / "npm.cmd")
    assert node_cli_launch_for_shim(shim, windows=True) == [
        str(node / "node.exe"),
        str(node / "node_modules" / "npm" / "bin" / "npm-cli.js"),
    ]


def test_windows_npx_shim_resolves_to_node_launch(tmp_path):
    node = _fake_node_tree(tmp_path)
    shim = str(node / "npx.cmd")
    assert node_cli_launch_for_shim(shim, windows=True) == [
        str(node / "node.exe"),
        str(node / "node_modules" / "npx" / "bin" / "npx-cli.js"),
    ]


def test_windows_native_exe_unchanged(tmp_path):
    node = _fake_node_tree(tmp_path)
    node_exe = str(node / "node.exe")
    assert node_cli_launch_for_shim(node_exe, windows=True) == [node_exe]


def test_windows_missing_targets_fall_back_to_shim(tmp_path):
    shim = tmp_path / "npm.cmd"
    shim.write_text("", encoding="utf-8")
    assert node_cli_launch_for_shim(str(shim), windows=True) == [str(shim)]


def test_windows_other_batch_shim_unchanged(tmp_path):
    other = tmp_path / "tool.bat"
    other.write_text("", encoding="utf-8")
    assert node_cli_launch_for_shim(str(other), windows=True) == [str(other)]
