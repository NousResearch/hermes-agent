"""Regression tests for the Hermes-managed Node's npm global prefix.

When the installer falls back to a bundled Node under ``$HERMES_HOME/node``,
npm's default global prefix is that Node dir, so ``npm install -g <pkg>``
drops the package binary in ``$HERMES_HOME/node/bin`` — which is NOT on PATH
(only the command link dir is) and is wiped on every Node upgrade. Users then
report "I can ``npm i -g`` but the package isn't usable on the command line".

The fix redirects the bundled Node's global prefix to the command link dir's
parent (so global bins land in the already-on-PATH link dir alongside
node/npm/npx), scoped to the bundled Node via its prefix-local global npmrc.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"


def test_install_sh_redirects_bundled_npm_global_prefix_to_link_dir() -> None:
    text = INSTALL_SH.read_text()

    # The redirect must target the link dir's PARENT so global bins resolve to
    # <parent>/bin == the command link dir (node/npm/npx live there and it is
    # guaranteed on PATH by the installer's PATH setup).
    assert "configure_managed_node_npm_prefix()" in text
    assert 'printf \'prefix=%s\\n\' "$(dirname "$link_dir")" > "$HERMES_HOME/node/etc/npmrc"' in text


def test_install_sh_repairs_existing_managed_node_on_rerun() -> None:
    """The redirect must run on every install (not just fresh Node installs),
    so re-running the installer repairs pre-existing managed installs whose
    Node is already up to date and would otherwise skip install_node."""
    text = INSTALL_SH.read_text()

    check_node_body = text.split("check_node()", 1)[1].split("\ninstall_node()", 1)[0]
    assert "configure_managed_node_npm_prefix" in check_node_body

    # No-op guard so it's safe to call when there is no managed Node.
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text


def test_node_bootstrap_redirects_bundled_npm_global_prefix_to_link_dir() -> None:
    text = NODE_BOOTSTRAP.read_text()

    assert "_nb_configure_npm_prefix()" in text
    assert 'printf \'prefix=%s\\n\' "$(dirname "$_link_dir")" > "$HERMES_HOME/node/etc/npmrc"' in text

    # Runs at the top of ensure_node so existing managed installs are repaired
    # even when a modern Node is already present (early return path).
    ensure_node_body = text.split("ensure_node()", 1)[1]
    assert "_nb_configure_npm_prefix" in ensure_node_body
    assert '[ -x "$HERMES_HOME/node/bin/npm" ] || return 0' in text
