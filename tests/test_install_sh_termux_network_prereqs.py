"""Regression tests for Termux network prerequisite handling in install.sh."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_termux_pkg_list_includes_network_basics() -> None:
    text = INSTALL_SH.read_text()
    assert "local termux_pkgs=(clang rust make pkg-config libffi openssl ca-certificates curl)" in text


def test_install_script_has_connectivity_probe_and_termux_guidance() -> None:
    text = INSTALL_SH.read_text()
    assert "check_network_prerequisites()" in text
    assert "https://pypi.org/simple/" in text
    assert "https://duckduckgo.com/" in text
    assert "termux-change-repo" in text
    assert "pkg install -y ca-certificates curl && pkg update" in text
    assert "check_network_prerequisites" in text
