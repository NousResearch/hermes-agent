"""Seam-test + behavior test for apex_overlay.cn_mirror_env (hc-476).

The installer's CN mirror env (HERMES_NODE_DIST_BASE / PLAYWRIGHT_DOWNLOAD_HOST)
only ever lived inside the install-script process; runtime lazy-download paths
(Playwright Chromium autoinstall, managed-Node reinstall/heal) ran without it.
This seam self-applies those vars into the running process on CN deployments.

Pinned here:
* behavior — CN mode sets exactly the runtime vars (setdefault semantics:
  operator overrides win); global mode is a strict no-op;
* registration — plugins/apex-overlay/__init__.py loads the seam;
* keep-in-step — the values match the install-time single source of truth in
  scripts/lib/apexnodes-region-detect.sh and .ps1, and the bash runtime belt
  (node-bootstrap.sh self-sourcing the region lib) stays present.

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from apex_overlay import cn_mirror_env

_REPO = Path(__file__).resolve().parents[2]


def _reset():
    cn_mirror_env._APPLIED = False


def test_cn_mode_sets_runtime_mirror_env(monkeypatch):
    _reset()
    for key in cn_mirror_env.RUNTIME_CN_MIRROR_ENV:
        monkeypatch.delenv(key, raising=False)
    with patch("apex_overlay.region.is_cn_mode", return_value=True):
        assert cn_mirror_env.apply() is True
    assert os.environ["HERMES_NODE_DIST_BASE"] == "https://registry.npmmirror.com/-/binary/node"
    assert os.environ["PLAYWRIGHT_DOWNLOAD_HOST"] == "https://cdn.npmmirror.com/binaries/playwright"


def test_operator_override_wins(monkeypatch):
    _reset()
    monkeypatch.setenv("HERMES_NODE_DIST_BASE", "https://mirror.corp.example/node")
    monkeypatch.delenv("PLAYWRIGHT_DOWNLOAD_HOST", raising=False)
    with patch("apex_overlay.region.is_cn_mode", return_value=True):
        assert cn_mirror_env.apply() is True
    assert os.environ["HERMES_NODE_DIST_BASE"] == "https://mirror.corp.example/node"
    assert os.environ["PLAYWRIGHT_DOWNLOAD_HOST"] == "https://cdn.npmmirror.com/binaries/playwright"


def test_global_mode_is_a_noop(monkeypatch):
    _reset()
    for key in cn_mirror_env.RUNTIME_CN_MIRROR_ENV:
        monkeypatch.delenv(key, raising=False)
    with patch("apex_overlay.region.is_cn_mode", return_value=False):
        assert cn_mirror_env.apply() is True
    for key in cn_mirror_env.RUNTIME_CN_MIRROR_ENV:
        assert key not in os.environ


def test_apply_is_once_per_process(monkeypatch):
    _reset()
    for key in cn_mirror_env.RUNTIME_CN_MIRROR_ENV:
        monkeypatch.delenv(key, raising=False)
    with patch("apex_overlay.region.is_cn_mode", return_value=False) as probe:
        assert cn_mirror_env.apply() is True
        assert cn_mirror_env.apply() is True
        assert probe.call_count == 1


def test_seam_registered_in_apex_overlay_plugin():
    plugin = (_REPO / "plugins" / "apex-overlay" / "__init__.py").read_text(encoding="utf-8")
    assert "cn_mirror_env" in plugin, (
        "apex_overlay.cn_mirror_env is not loaded by plugins/apex-overlay — the "
        "runtime CN mirror env never gets applied and CN lazy downloads regress "
        "to foreign hosts."
    )


def test_values_in_step_with_install_time_source_of_truth():
    """The seam's values must match apexnodes-region-detect.{sh,ps1} verbatim."""
    sh = (_REPO / "scripts" / "lib" / "apexnodes-region-detect.sh").read_text(encoding="utf-8")
    ps1 = (_REPO / "scripts" / "lib" / "apexnodes-region-detect.ps1").read_text(encoding="utf-8")
    for value in cn_mirror_env.RUNTIME_CN_MIRROR_ENV.values():
        assert value in sh, f"{value} missing from apexnodes-region-detect.sh"
        assert value in ps1, f"{value} missing from apexnodes-region-detect.ps1"


def test_node_bootstrap_bash_belt_present():
    """node-bootstrap.sh must self-derive the mirror env when it is absent.

    _ensure_tui_node / heal_hermes_managed_node spawn bash from processes that
    do not load plugins, so the bash side needs its own wiring — sourcing the
    sibling region lib. Losing that line silently regresses CN runtime Node
    reinstalls to nodejs.org.
    """
    nb = (_REPO / "scripts" / "lib" / "node-bootstrap.sh").read_text(encoding="utf-8")
    assert "apexnodes-region-detect.sh" in nb
    assert "apexnodes_apply_cn_mirror_env" in nb
    assert "HERMES_NODE_DIST_BASE" in nb
