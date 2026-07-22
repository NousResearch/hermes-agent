"""Regression for #45279: installer must not shadow system node/npm/npx.

On macOS user installs, ``install_node()`` used to unconditionally symlink
node/npm/npx into ``~/.local/bin``.  When that directory precedes
``/opt/homebrew/bin`` (or nvm) in ``$PATH``, the symlinks silently replace
the user's real Node toolchain with Hermes' bundled copy.

The fix: only create a symlink when ``command -v <cmd>`` finds no existing
binary on ``$PATH``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_install_node_function() -> str:
    """Return the body of the ``install_node()`` shell function."""
    text = INSTALL_SH.read_text()
    start = text.index("install_node()")
    # Find the closing brace at column 0 that ends the function.
    rest = text[start:]
    depth = 0
    for i, ch in enumerate(rest):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return rest[: i + 1]
    raise AssertionError("Could not find end of install_node()")


def test_install_node_does_not_unconditionally_symlink() -> None:
    """The old ``ln -sf`` block must be replaced with a ``command -v`` guard."""
    body = _extract_install_node_function()
    # The unconditional symlinks must be gone.
    assert 'ln -sf "$HERMES_HOME/node/bin/node" "$node_link_dir/node"' not in body
    assert 'ln -sf "$HERMES_HOME/node/bin/npm"  "$node_link_dir/npm"' not in body
    assert 'ln -sf "$HERMES_HOME/node/bin/npx"  "$node_link_dir/npx"' not in body


def test_install_node_checks_command_before_symlinking() -> None:
    """Symlinks must only be created when the command is absent from $PATH."""
    body = _extract_install_node_function()
    assert 'command -v "$_cmd"' in body or "command -v '$_cmd'" in body
    assert "ln -sf" in body  # symlinks still created, just conditionally
