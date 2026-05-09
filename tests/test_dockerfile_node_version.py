"""Static guards: Dockerfile installs Node from NodeSource at the same major
version pinned by scripts/install.sh.

Background — issue #21656.  Debian 13's apt-shipped nodejs is 20.x.  Some
transitive web/ deps (e.g. language-tags@2.1.0) require Node >= 22.  npm
only WARNs on engine mismatch, so the install completes but the resolver
chain breaks silently and the Vite build dies with a misleading
"can't resolve clsx" error.  scripts/install.sh already pins
NODE_VERSION="22"; the Dockerfile must match.

These are pure text inspections — no `docker build`, no subprocess.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _installer_node_major() -> str:
    """Extract the NODE_VERSION="<major>" pin from scripts/install.sh."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(r'^NODE_VERSION="(\d+)"', text, re.MULTILINE)
    assert match, "scripts/install.sh must declare NODE_VERSION=\"<major>\""
    return match.group(1)


def test_dockerfile_installs_node_from_nodesource_not_debian_apt() -> None:
    """The Dockerfile must not rely on Debian apt's bundled nodejs/npm.

    Debian 13 ships nodejs 20.x; we need >= 22 (see #21656).  A line that
    apt-installs both `nodejs` and `npm` without a NodeSource setup script
    is the regression we are guarding against.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    lower = text.lower()

    assert "deb.nodesource.com/setup_" in lower, (
        "Dockerfile must install Node from NodeSource (deb.nodesource.com/setup_<major>.x), "
        "not from Debian's apt nodejs package, which is too old (#21656)."
    )

    for raw_line in text.splitlines():
        line = raw_line.lower()
        if "apt-get install" not in line:
            continue
        if "nodesource" in line:
            continue
        if " nodejs" in f" {line}" and " npm" in f" {line}":
            raise AssertionError(
                "Dockerfile installs both `nodejs` and `npm` from Debian apt "
                f"(line: {raw_line!r}). This pulls Node 20.x on Debian 13 and "
                "breaks the web/ build (#21656). Use NodeSource setup_<major>.x instead."
            )


def test_dockerfile_node_major_matches_installer() -> None:
    """Dockerfile's NodeSource major must equal scripts/install.sh's NODE_VERSION.

    Drift between the two is exactly the failure mode #21656 reported:
    install.sh upgraded to 22, the Dockerfile kept shipping 20, and the
    container build started failing in CI on PRs that touched web/ deps.
    """
    major = _installer_node_major()
    text = DOCKERFILE.read_text(encoding="utf-8").lower()

    expected = f"deb.nodesource.com/setup_{major}.x"
    assert expected in text, (
        f"scripts/install.sh pins NODE_VERSION=\"{major}\" but the Dockerfile "
        f"does not contain {expected!r}. Bump the NodeSource setup script in "
        "the Dockerfile to match (#21656)."
    )
