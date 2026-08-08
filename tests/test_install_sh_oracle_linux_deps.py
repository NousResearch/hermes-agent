"""Regression tests for Oracle Linux package-manager coverage in install.sh."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_treats_oracle_linux_as_rpm_family() -> None:
    text = INSTALL_SH.read_text()

    assert "fedora|rhel|centos|rocky|alma|ol|oraclelinux" in text
    assert 'echo "dnf install -y"' in text


def test_install_script_auto_installs_git_via_linux_package_manager() -> None:
    text = INSTALL_SH.read_text()

    assert 'pkg_install="$(get_linux_pkg_install_cmd "$DISTRO" || true)"' in text
    assert '$sudo_cmd $pkg_install git >/dev/null 2>&1 || true' in text
    assert 'log_info "Installing Git via dnf..."' in text
