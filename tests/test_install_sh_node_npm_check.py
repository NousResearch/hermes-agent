"""Regression tests for install.sh Node/npm checks (#77003).

A stray `node` symlink without a sibling `npm` (leftover from a node
version manager) made the installer report "✓ Node.js found" and then fail
opaquely at the desktop stage. Node must only count as found when npm
resolves on the same PATH, and npm install stages must not report success
when the install actually failed.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_check_node_requires_npm_alongside_node() -> None:
    """check_node must not report success when only `node` resolves.

    Before the fix, `command -v node` succeeding was enough — a stray node
    symlink (no sibling npm) passed the check, every later `npm install`
    failed silently, and the desktop build died with an opaque
    "Node.js / npm unavailable" (#77003).
    """
    text = INSTALL_SH.read_text()

    # The system-toolchain branch now gates on BOTH node and npm.
    assert (
        "if command -v node &> /dev/null && command -v npm &> /dev/null \\" in text
    )
    # The "node found but npm missing" case has its own explicit branch that
    # falls through to installing the Hermes-managed Node (which bundles npm).
    assert "node found but npm is not on PATH (stray node symlink?)" in text


def test_check_node_managed_requires_npm() -> None:
    """The Hermes-managed Node fallback also requires its npm to exist."""
    text = INSTALL_SH.read_text()
    assert (
        '[ -x "$HERMES_HOME/node/bin/node" ] && [ -x "$HERMES_HOME/node/bin/npm" ] \\'
        in text
    )


def test_node_deps_success_log_is_conditional() -> None:
    """install_node_deps must not print ✓ when the npm install failed.

    Before the fix, `log_success "Node.js dependencies installed"` ran
    unconditionally after a `||` warn, so a failed npm install still read as
    success — hiding the browser-tool degradation from the user (#77003).
    """
    text = INSTALL_SH.read_text()

    # The success log now sits inside the `if run_with_timeout ...; then`
    # branch, so it cannot fire when the install failed.
    node_deps_block = text.split("Installing Node.js dependencies (browser tools)...", 1)[1]
    node_deps_block = node_deps_block.split("Installing browser engine", 1)[0]
    assert 'if run_with_timeout "$NODE_DEPS_TIMEOUT" npm install --silent; then' in node_deps_block
    assert 'log_success "Node.js dependencies installed"' in node_deps_block
    assert 'log_warn "npm install failed or timed out (browser tools may not work)"' in node_deps_block
    # Success and failure are mutually exclusive branches.
    success_pos = node_deps_block.find('log_success "Node.js dependencies installed"')
    warn_pos = node_deps_block.find('log_warn "npm install failed or timed out')
    assert success_pos != -1 and warn_pos != -1
    assert 'else' in node_deps_block[success_pos:warn_pos] or 'else' in node_deps_block[:warn_pos]


def test_tui_deps_success_log_is_conditional() -> None:
    """The TUI npm install follows the same success-only-on-success rule."""
    text = INSTALL_SH.read_text()
    tui_block = text.split("Installing TUI dependencies...", 1)[1]
    tui_block = tui_block.split("restore_dirty_lockfiles", 1)[0]
    assert 'if run_with_timeout "$NODE_DEPS_TIMEOUT" npm install --silent; then' in tui_block
    assert 'log_success "TUI dependencies installed"' in tui_block
    assert 'log_warn "TUI npm install failed or timed out (hermes --tui may not work)"' in tui_block
