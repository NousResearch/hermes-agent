"""Regression tests for issue #35200 (installer installed the gateway twice).

When a messaging platform is configured, the ``hermes setup`` wizard installs
the gateway and writes a ``<hermes_home>/.gateway_setup_done`` marker. The shell
installers (``scripts/install.sh`` and ``scripts/install.ps1``) then ran their
own post-setup gateway step, which re-detected the messaging token and installed
/ started the gateway a second time (the ``Service already installed ... Use
--force`` path the reporter saw).

These tests pin the installer-scoped marker contract:
  * the installers read the marker and bail out of their gateway step before
    they probe the ``.env`` for messaging tokens, and
  * the installers discard stale markers and request a fresh marker only for
    their own setup invocation.
"""

from pathlib import Path

from hermes_cli import setup

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
MARKER_NAME = ".gateway_setup_done"


# ---------------------------------------------------------------------------
# Static assertions: the shell installers guard on the marker file, before
# they inspect the .env for messaging tokens (which is what triggers the
# duplicate install/prompt).
# ---------------------------------------------------------------------------
def test_install_sh_checks_marker_before_messaging_detection():
    text = INSTALL_SH.read_text(encoding="utf-8")

    fn_idx = text.index("maybe_start_gateway() {")
    marker_idx = text.index(f'-f "$HERMES_HOME/{MARKER_NAME}"', fn_idx)
    # Messaging-token detection / the gateway prompt come after the guard.
    detect_idx = text.index("HAS_MESSAGING=false", fn_idx)

    assert fn_idx < marker_idx < detect_idx
    consume_idx = text.index(f'rm -f "$HERMES_HOME/{MARKER_NAME}"', marker_idx)

    assert marker_idx < consume_idx < detect_idx


def test_install_ps1_checks_marker_before_messaging_detection():
    text = INSTALL_PS1.read_text(encoding="utf-8")

    fn_idx = text.index("function Start-GatewayIfConfigured")
    marker_idx = text.index(MARKER_NAME, fn_idx)
    detect_idx = text.index("$hasMessaging = $false", fn_idx)

    assert fn_idx < marker_idx < detect_idx
    consume_idx = text.index("Remove-Item (Join-Path $HermesHome \".gateway_setup_done\")", marker_idx)

    assert marker_idx < consume_idx < detect_idx


def test_installers_clear_stale_marker_before_wizard_and_request_a_fresh_one():
    shell_text = INSTALL_SH.read_text(encoding="utf-8")
    shell_fn = shell_text.index("run_setup_wizard() {")
    shell_clear = shell_text.index(f'rm -f "$HERMES_HOME/{MARKER_NAME}"', shell_fn)
    shell_launch = shell_text.index("HERMES_INSTALLER_GATEWAY_MARKER=1", shell_fn)

    assert shell_clear < shell_launch
    assert 'HERMES_HOME="$HERMES_HOME" HERMES_INSTALLER_GATEWAY_MARKER=1' in shell_text

    ps_text = INSTALL_PS1.read_text(encoding="utf-8")
    ps_fn = ps_text.index("function Invoke-SetupWizard")
    ps_clear = ps_text.index("Remove-Item (Join-Path $HermesHome \".gateway_setup_done\")", ps_fn)
    ps_launch = ps_text.index('$env:HERMES_INSTALLER_GATEWAY_MARKER = "1"', ps_fn)

    assert ps_clear < ps_launch


# ---------------------------------------------------------------------------
# Behavioral: the wizard writes an installer-requested marker under HERMES_HOME.
# ---------------------------------------------------------------------------
def test_marker_path_honors_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert setup._gateway_setup_marker_path() == tmp_path / MARKER_NAME


def test_marker_path_defaults_to_home_hermes(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(setup.Path, "home", classmethod(lambda cls: tmp_path))

    assert setup._gateway_setup_marker_path() == tmp_path / ".hermes" / MARKER_NAME


def test_write_marker_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_INSTALLER_GATEWAY_MARKER", "1")

    setup._write_gateway_setup_marker()

    assert (tmp_path / MARKER_NAME).is_file()


def test_write_marker_requires_installer_request(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_INSTALLER_GATEWAY_MARKER", raising=False)

    setup._write_gateway_setup_marker()

    assert not (tmp_path / MARKER_NAME).exists()


def test_write_marker_swallows_oserror(tmp_path, monkeypatch):
    # Point HERMES_HOME at a path whose parent is a file, so write_text raises
    # OSError. The helper must log and not propagate.
    not_a_dir = tmp_path / "regular_file"
    not_a_dir.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(not_a_dir / "sub"))
    monkeypatch.setenv("HERMES_INSTALLER_GATEWAY_MARKER", "1")

    setup._write_gateway_setup_marker()  # must not raise

    assert not (not_a_dir / "sub" / MARKER_NAME).exists()
