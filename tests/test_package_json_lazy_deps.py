"""Invariants for what is eager vs lazy in the root ``package.json``.

The root ``package.json`` is installed by ``hermes update`` on every user,
including users who never opted into a given browser backend. Anything
listed in ``dependencies`` therefore runs its npm postinstall script for
everyone — including binary-fetching backends, on every update.

The contract:

* ``agent-browser`` IS eager. It is the default Chromium-driving backend
  used whenever the agent makes a browser call without a cloud provider
  configured, so it must already be installed before any session starts.
  Its postinstall is also small.

* ``@askjo/camofox-browser`` is NOT eager. It is an explicit opt-in
  alternative browser backend, selected by the user via
  ``hermes tools`` → Browser Automation → Camofox, and only used at
  runtime when ``CAMOFOX_URL`` is set. Its postinstall fetches a ~300MB
  Firefox-fork binary, which silently blocked ``hermes update`` for
  multi-minute stretches on slow / network-restricted connections
  (notably users in China running through a VPN). The package is
  installed on demand by ``tools_config.py`` ``post_setup_key ==
  "camofox"`` when the user actually selects Camofox.

If a future PR re-adds Camofox (or any other binary-postinstall package)
to root ``dependencies``, this test fails — read the lazy-install
guidance in the ``hermes-agent-dev`` skill before changing the
expectations.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _root_package_json() -> dict:
    with (REPO_ROOT / "package.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_camofox_is_not_in_root_dependencies() -> None:
    """Camofox must be opt-in, installed lazily by its post_setup handler."""
    deps = _root_package_json().get("dependencies", {})
    assert "@askjo/camofox-browser" not in deps, (
        "Camofox is a ~300MB binary-postinstall backend that must stay "
        "out of root package.json dependencies. It belongs in the "
        "Camofox post_setup handler in hermes_cli/tools_config.py so it "
        "only installs when the user explicitly selects Camofox via "
        "`hermes tools` → Browser Automation → Camofox."
    )


def test_agent_browser_stays_eager() -> None:
    """agent-browser is the default backend; it must remain eager."""
    deps = _root_package_json().get("dependencies", {})
    assert "agent-browser" in deps, (
        "agent-browser is the default browser-tool backend used by every "
        "session that doesn't have a cloud browser provider configured. "
        "It must stay in root package.json dependencies so it is present "
        "after `hermes setup` / `hermes update` without an explicit "
        "post_setup step."
    )


def test_root_lockfile_has_no_camofox_entries() -> None:
    """Regenerated lockfiles should not contain Camofox tree entries."""
    lock_path = REPO_ROOT / "package-lock.json"
    if not lock_path.exists():
        # Some CI matrix shards skip lockfile materialization.
        return
    text = lock_path.read_text(encoding="utf-8")
    assert "@askjo/camofox-browser" not in text, (
        "package-lock.json still references @askjo/camofox-browser. "
        "Regenerate the lockfile after removing the dep: "
        "`rm package-lock.json && npm install --package-lock-only "
        "--ignore-scripts --no-fund --no-audit`."
    )
    assert "camoufox-js" not in text, (
        "package-lock.json still references camoufox-js (transitive of "
        "@askjo/camofox-browser). Regenerate the lockfile."
    )
