"""The launchd refresh must never repoint a booting service onto a launcher that cannot boot.

``refresh_launchd_plist_if_needed()`` regenerates the definition from the invoking process's
code tree. A PM-env CLI whose code tree is a workspace root without a committed dependency
environment regenerates a launcher that refuses at exec ("no dependency environment is
committed for this install"), leaving launchd to crash-loop the job instead of serving
(observed on a real macOS host, Sep 2026). ``launchd_plist_is_bootable`` is the guard's
predicate; it must read both command shapes ``installation_command`` can emit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import gateway as gw

CHECKOUT = "/opt/hermes-checkout"
WORKSPACE = "/opt/hermes-installs/abcd/environments/ef01/workspace"


def _committed_for(monkeypatch, committed_roots):
    monkeypatch.setattr(
        "pm.environments.committed_venv",
        lambda root: Path("/envs/current/venv") if str(root) in committed_roots else None,
    )


def _launcher_plist(root: str) -> str:
    # Emitted when a store Python is recorded for the root: the ``.hermes/bin/hermes`` shim.
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<plist version=\"1.0\">\n<dict>\n"
        "    <key>ProgramArguments</key>\n    <array>\n"
        "        <string>/usr/bin/osascript</string>\n        <string>-e</string>\n"
        "        <string>do shell script &quot;exec {root}/.hermes/bin/hermes --run-module "
        "hermes_cli.stderr_timestamp --error-log /tmp/gateway.error.log -- {root}/.hermes/bin/hermes "
        "gateway run --external-supervisor &gt;&gt; /tmp/gateway.log 2&gt;&gt; /tmp/gateway.error.log&quot;</string>\n"
        "    </array>\n</dict>\n</plist>\n"
    ).format(root=root)


def _bootstrap_plist(root: str) -> str:
    import json
    import plistlib
    import shlex

    code = f"import sys; sys.path.insert(0, {root!r}); import hermes_bootstrap"
    shell = "exec " + shlex.join(["/store/python/bin/python3", "-I", "-c", code])
    return plistlib.dumps({
        "ProgramArguments": ["/usr/bin/osascript", "-e", "do shell script " + json.dumps(shell)]
    }).decode("utf-8")


def test_predicate_reads_both_command_shapes(monkeypatch):
    from hermes_cli import gateway_launchd as gl

    _committed_for(monkeypatch, {CHECKOUT})
    assert gl.launchd_plist_is_bootable(_launcher_plist(CHECKOUT)) is True
    assert gl.launchd_plist_is_bootable(_launcher_plist(WORKSPACE)) is False
    assert gl.launchd_plist_is_bootable(_bootstrap_plist(CHECKOUT)) is True
    assert gl.launchd_plist_is_bootable(_bootstrap_plist(WORKSPACE)) is False
    # Unparsed shapes must never block a refresh.
    assert gl.launchd_plist_is_bootable("<plist><dict/></plist>") is True


def test_refresh_keeps_the_installed_definition_when_regenerated_cannot_boot(tmp_path, monkeypatch):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    installed = _launcher_plist(CHECKOUT)
    plist_path.write_text(installed, encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gw, "generate_launchd_plist", lambda: _launcher_plist(WORKSPACE))
    monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
    _committed_for(monkeypatch, {CHECKOUT})

    assert gw.refresh_launchd_plist_if_needed() is False
    assert plist_path.read_text(encoding="utf-8") == installed


def test_predicate_reads_a_generated_store_python_bootstrap(monkeypatch):
    """The bootstrap shape exactly as the real generator emits it -- escape noise included."""
    from hermes_cli import gateway_launchd as gl

    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda root=None: None)
    _committed_for(monkeypatch, {CHECKOUT})

    monkeypatch.setattr(gw, "PROJECT_ROOT", Path(WORKSPACE))
    generated_workspace = gl.generate_launchd_plist()
    assert "sys.path.insert(0, " in generated_workspace  # the bootstrap shape was produced
    assert gl.launchd_plist_is_bootable(generated_workspace) is False

    monkeypatch.setattr(gw, "PROJECT_ROOT", Path(CHECKOUT))
    assert gl.launchd_plist_is_bootable(gl.generate_launchd_plist()) is True


def test_install_refuses_to_replace_a_booting_definition_with_a_broken_one(tmp_path, monkeypatch):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    installed = _launcher_plist(CHECKOUT)
    plist_path.write_text(installed, encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gw, "generate_launchd_plist", lambda: _launcher_plist(WORKSPACE))
    monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
    _committed_for(monkeypatch, {CHECKOUT})

    gw.launchd_install(force=True)

    assert plist_path.read_text(encoding="utf-8") == installed


@pytest.mark.parametrize("root", ["/opt/with space/hermes", "/opt/user's/hermes", '/opt/with"quote/hermes', "/opt/中文/hermes"])
@pytest.mark.parametrize("store_python", [None, Path("/store/python/bin/python3")])
def test_generated_launchers_preserve_quoted_roots(root, store_python, monkeypatch):
    from hermes_cli import gateway_launchd as gl

    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda root=None: store_python)
    monkeypatch.setattr(gw, "PROJECT_ROOT", Path(root))
    generated = gl.generate_launchd_plist()
    _committed_for(monkeypatch, set())
    assert gl.launchd_plist_is_bootable(generated) is False
    _committed_for(monkeypatch, {root})
    assert gl.launchd_plist_is_bootable(generated) is True
