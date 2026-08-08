"""Regression coverage for the Termux Rust-toolchain pre-flight in install.sh.

Issue [#26891](https://github.com/NousResearch/hermes-agent/issues/26891)
reported that the Termux installer would spend minutes downloading
packages and then surface a confusing ``Cannot import 'maturin'`` /
``Target triple not supported by rustup: aarch64-unknown-linux-android``
error from deep inside pip when a kernel-too-old / mirror-broken
device couldn't compile the Rust-based ``jiter`` (a transitive of
``openai==2.24.0``).

The fix adds a ``verify_termux_rust_toolchain`` pre-flight that runs
before pip and fails fast with an actionable error message pointing
at #26891 and the dedicated docs section.  These tests grep the
install script and docs to keep that surface stable.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
TERMUX_DOCS = REPO_ROOT / "website" / "docs" / "getting-started" / "termux.md"


def test_install_script_defines_termux_rust_preflight() -> None:
    text = INSTALL_SH.read_text()
    assert "verify_termux_rust_toolchain()" in text, (
        "The pre-flight helper must exist as a named function so future "
        "edits don't accidentally inline-and-drop it."
    )


def test_install_script_calls_preflight_before_pip_on_termux() -> None:
    """The check must run BEFORE the main pip install — otherwise pip
    wastes minutes before the Rust error surfaces (#26891)."""
    text = INSTALL_SH.read_text()
    preflight_idx = text.find("verify_termux_rust_toolchain\n")
    pip_install_idx = text.find("pip install -e '.[termux-all]'")
    assert preflight_idx > 0, "Pre-flight call site must be invoked, not just defined"
    assert pip_install_idx > 0
    assert preflight_idx < pip_install_idx, (
        "Pre-flight must run BEFORE the .[termux-all] pip install"
    )


def test_install_script_preflight_checks_cargo_and_rustc() -> None:
    """Without both cargo AND rustc the jiter build will fail later."""
    text = INSTALL_SH.read_text()
    assert "command -v cargo" in text
    assert "command -v rustc" in text


def test_install_script_preflight_retries_pkg_install_rust() -> None:
    """Termux mirrors flake; the pre-flight should give them one more
    chance before bailing out hard."""
    text = INSTALL_SH.read_text()
    assert "pkg install -y rust" in text


def test_install_script_preflight_error_message_is_actionable() -> None:
    """The failure message must mention the actual cause (jiter +
    rustup target) and link the user at the issue / docs — otherwise
    we're just printing a different confusing error."""
    text = INSTALL_SH.read_text()
    assert "Hermes Agent cannot install on this Termux environment." in text
    assert "openai==2.24.0" in text
    assert "jiter" in text
    assert "Rust" in text or "rust" in text
    assert "kernel" in text.lower()
    assert "https://github.com/NousResearch/hermes-agent/issues/26891" in text
    assert "Workarounds" in text


def test_termux_docs_have_old_kernel_troubleshooting_section() -> None:
    """The pre-flight error points at the docs; the docs must actually
    explain the workaround.  Lock in the section title + link so a
    later docs refactor doesn't quietly break that contract."""
    text = TERMUX_DOCS.read_text()
    assert "### Old Android kernel (< 5.4) — Rust can't compile `jiter`" in text
    assert "#26891" in text
    assert "aarch64-unknown-linux-android" in text
    assert "pkg install -y rust" in text
