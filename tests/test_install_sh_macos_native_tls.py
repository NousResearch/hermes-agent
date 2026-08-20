"""Regression for #45183: macOS installer should default uv to native TLS."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_defaults_uv_native_tls_on_macos_only_when_unset() -> None:
    text = INSTALL_SH.read_text()

    assert 'Darwin*)' in text
    assert 'if [ -z "${UV_NATIVE_TLS:-}" ] && [ "${UV_NATIVE_TLS_MACOS_DEFAULTED:-}" != "1" ]; then' in text
    assert 'export UV_NATIVE_TLS=1' in text
    assert 'Using macOS native TLS roots for uv downloads (UV_NATIVE_TLS=1)' in text
