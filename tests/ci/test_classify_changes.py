"""Tests for scripts/ci/classify_changes.py.

Check some common patterns of file modifications and the CI lanes they should run.
We should always fail open. We may run a lane we didn't need, never skip one a
change could have broken.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "classify_changes.py"
_spec = importlib.util.spec_from_file_location("classify_changes", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load classify_changes.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
classify = _mod.classify

DEFAULT = {
    "python": True,
    "frontend": True,
    "docker_meta": True,
    "site": True,
    "scan": True,
    "deps": True,
    "mcp_catalog": False,
}


def _lanes(python=False, frontend=False, site=False, scan=False, deps=False, mcp_catalog=False, docker_meta=False) -> dict[str, bool]:
    return {
        "python": python,
        "frontend": frontend,
        "docker_meta": docker_meta,
        "site": site,
        "scan": scan,
        "deps": deps,
        "mcp_catalog": mcp_catalog,
    }


CASES = {
    "docs-only → nothing heavy": (["README.md", "docs/guide.md"], _lanes()),
    "python source → python": (["run_agent.py"], _lanes(python=True, scan=True)),
    "dep manifest → python": (["pyproject.toml"], _lanes(python=True, scan=True, deps=True)),
    "uv.lock → python": (["uv.lock"], _lanes(python=True)),
    "ts package → frontend": (["apps/desktop/src/app.tsx"], _lanes(frontend=True)),
    "ui-tui → frontend": (["ui-tui/src/entry.ts"], _lanes(frontend=True)),
    # Lockfile bump shifts every TS package's tree, but not the Python suite.
    "root lockfile → frontend, not python": (["package-lock.json"], _lanes(frontend=True)),
    "website → site": (["website/docs/intro.md"], _lanes(site=True)),
    # SKILL.md reads like docs, but the skill-doc tests read skills/, so a
    # skill edit must still run Python.
    "skill md → python + site": (["skills/github/SKILL.md"], _lanes(python=True, site=True)),
    # Built dashboard browser bundles are served verbatim — they can't touch
    # Python, so a compiled asset edit must NOT trigger the pytest matrix.
    "dashboard dist js → nothing heavy": (
        ["plugins/kanban/dashboard/dist/index.js"], _lanes()),
    "dashboard dist css → nothing heavy": (
        ["plugins/kanban/dashboard/dist/style.css"], _lanes()),
    # Vite/webpack also emit index.html + asset-manifest.json into dist/ (Greptile P2).
    "dashboard dist html → nothing heavy": (
        ["plugins/kanban/dashboard/dist/index.html"], _lanes()),
    "dashboard dist manifest json → nothing heavy": (
        ["plugins/kanban/dashboard/dist/asset-manifest.json"], _lanes()),
    # But the plugin's .py backend is NOT under dist/ → still python-relevant.
    "dashboard plugin_api.py → python": (
        ["plugins/kanban/dashboard/plugin_api.py"], _lanes(python=True, scan=True)),
    # A plugin .py OUTSIDE dashboard/dist still runs Python (fail-open).
    "plugin source py → python": (
        ["plugins/memory/mem0/__init__.py"], _lanes(python=True, scan=True)),
    # Mixed: a dashboard bundle + a real python file still runs Python.
    "mixed dashboard-js + python → python": (
        ["plugins/kanban/dashboard/dist/index.js", "agent/x.py"],
        _lanes(python=True, scan=True)),
    "dockerfile → docker meta": (["Dockerfile"], _lanes(docker_meta=True)),
    # Media / binary assets can't be imported or executed by Python → nothing heavy.
    "logo png → nothing heavy": (["assets/logo.png"], _lanes()),
    "nested icon svg → nothing heavy": (["hermes_cli/web_dist/assets/x.svg"], _lanes()),
    "sound wav → nothing heavy": (["assets/notify.wav"], _lanes()),
    "font woff2 → nothing heavy": (["web/fonts/inter.woff2"], _lanes(frontend=True)),
    # Git/repo metadata never affects the Python suite.
    "gitignore → nothing heavy": ([".gitignore"], _lanes()),
    "mailmap → nothing heavy": ([".mailmap"], _lanes()),
    "nested gitignore → nothing heavy": (["subdir/.gitignore"], _lanes()),
    # A media file MIXED with real python still runs Python (fail-open).
    "mixed png + python → python": (["docs/logo.png", "agent/x.py"], _lanes(python=True, scan=True)),
    # A .json fixture is NOT media — data a test reads could change behavior → python stays on.
    "json fixture → python (fail-open)": (["tests/fixtures/data.json"], _lanes(python=True)),
    # Unknown top-level file keeps Python on rather than risk a silent skip.
    "unknown toplevel → python": (["Makefile"], _lanes(python=True)),
    "mixed docs+python → python": (["README.md", "agent/x.py"], _lanes(python=True, scan=True)),
    "mixed docs+frontend → frontend": (["README.md", "apps/x.tsx"], _lanes(frontend=True)),
    # Supply-chain lanes
    ".pth file → scan": (["evil.pth"], _lanes(python=True, scan=True)),
    "setup.py → scan": (["setup.py"], _lanes(python=True, scan=True)),
    "mcp catalog manifest → mcp_catalog": (
        ["optional-mcps/foo/manifest.yaml"],
        _lanes(python=True, mcp_catalog=True),
    ),
    "mcp_catalog.py → mcp_catalog": (
        ["hermes_cli/mcp_catalog.py"],
        _lanes(python=True, scan=True, mcp_catalog=True),
    ),
    # Fail open: CI-config / empty / blank diffs run everything.
    ".github change → all": ([".github/workflows/tests.yml"], DEFAULT),
    "action change → all": ([".github/actions/detect-changes/action.yml"], DEFAULT),
    "empty diff → all": ([], DEFAULT),
    "blank lines → all": (["", "  "], DEFAULT),
}


@pytest.mark.parametrize("files,expected", CASES.values(), ids=CASES.keys())
def test_classify(files, expected):
    assert classify(files) == expected
