"""Regression tests for Node tarball ownership on root installs.

``tar`` restores the uid/gid recorded inside an archive when it runs as root,
and the official nodejs.org tarballs are built as ``iojs``, uid/gid 1001. A root
install therefore left ``$HERMES_HOME/node`` owned by 1001 rather than root.

That matters because uid 1001 is the first id most distributions hand out after
the initial human account: the next account created on the host silently took
ownership of the Node runtime root executes for browser tools, at mode 0755.

Both Node install paths extract a downloaded tarball, so both must suppress
owner restoration. Either spelling satisfies that: ``-o`` is the extract-mode
synonym of ``--no-same-owner`` in GNU tar, busybox and bsdtar.

See https://github.com/NousResearch/hermes-agent/issues/81525.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"

# A tar extraction call site, in either the `tar xf` or the `tar -xf` form.
# Matched by shape rather than by exact line so a future third call site cannot
# be added without the flag.
EXTRACTION = re.compile(r"^\s*tar\b.*\s-?x[a-z]*f\b")

# A folded `-oxzf` suppresses ownership too; requiring the standalone form
# keeps the flag visible at the call site.
NO_SAME_OWNER = re.compile(r"\s(?:-o|--no-same-owner)(?=\s)")


def _extraction_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if EXTRACTION.match(line)]


def test_install_sh_extracts_node_without_adopting_archive_ownership() -> None:
    lines = _extraction_lines(INSTALL_SH)
    assert lines, "no tar extraction found in install.sh"
    for line in lines:
        assert NO_SAME_OWNER.search(line), line


def test_node_bootstrap_extracts_node_without_adopting_archive_ownership() -> None:
    """The lazy runtime path installs Node too, and runs as whoever runs hermes."""
    lines = _extraction_lines(NODE_BOOTSTRAP)
    assert lines, "no tar extraction found in node-bootstrap.sh"
    for line in lines:
        assert NO_SAME_OWNER.search(line), line
