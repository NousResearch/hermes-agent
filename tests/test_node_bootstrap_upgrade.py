"""Regression tests for the documented managed-Node upgrade path.

``hermes update`` refreshes npm dependencies but never re-provisions the
Hermes-managed Node runtime under ``$HERMES_HOME/node``. A healthy tree keeps
the patch release it was installed with until its major falls out of the
accepted range and the automatic heal starts firing, so upgrading the runtime
(to pick up security patches or a newer LTS line) needs a supported command
instead of reverse-engineering the installer's internals.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"
UPDATING_DOC = REPO_ROOT / "website" / "docs" / "getting-started" / "updating.md"


def test_node_bootstrap_exposes_public_upgrade_entry_point() -> None:
    text = NODE_BOOTSTRAP.read_text()

    assert "upgrade_managed_node()" in text

    # The upgrade must not downgrade a newer managed tree to the library's
    # default line: the target major comes from the live tree, and an explicit
    # HERMES_NODE_TARGET_MAJOR still wins (documented major-jump escape hatch).
    upgrade_body = text.split("upgrade_managed_node()", 1)[1].split("\n}", 1)[0]
    assert '"$HERMES_HOME/node/bin/node"' in upgrade_body
    assert "HERMES_NODE_TARGET_MAJOR=\"$major\" _nb_install_bundled_node" in upgrade_body

    # No managed tree — refuse instead of provisioning a fresh one: this is an
    # upgrade helper, the installer's ensure_node() provisions.
    assert '[ -d "$HERMES_HOME/node" ] || return 1' in upgrade_body


def test_updating_doc_documents_managed_node_upgrade() -> None:
    text = UPDATING_DOC.read_text()

    assert "### Upgrading the self-managed Node.js runtime" in text
    assert "source scripts/lib/node-bootstrap.sh" in text
    assert "upgrade_managed_node" in text
    # The documented path must work from any checkout without copy-paste drift.
    assert "HERMES_NODE_TARGET_MAJOR=26 upgrade_managed_node" in text
