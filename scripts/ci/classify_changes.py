#!/usr/bin/env python3
"""Classify a PR's changed files into CI work lanes.

Reads newline-separated changed paths on stdin and writes ``key=value``
booleans (one per lane) to ``$GITHUB_OUTPUT`` and stdout. The
``detect-changes`` composite action consumes them so steps gate on
``if: steps.changes.outputs.<lane> == 'true'``.

Lanes:

* ``python``      — pytest / ruff / ty / footguns.
* ``docker_meta`` — Dockerfiles etc.
* ``frontend``    — TS typecheck matrix + desktop build.
* ``site``        — Docusaurus + generated skill docs.
* ``scan``        — supply-chain scan (Python files, .pth, setup hooks).
* ``deps``        — pyproject.toml dependency bounds check.
* ``mcp_catalog`` — bundled MCP catalog / installer review.

Docker is not a lane — it builds on push-to-main and release only,
never per-PR.

Contract — *fail open, never closed*. We may run a lane we didn't need, but
must never skip one a change could break:

* An empty diff, or any ``.github/`` change, runs everything.
* ``python`` is a denylist: skipped only when *every* file is provably prose
  or a frontend-only package; an unrecognized path keeps it on.
* ``skills/`` (incl. ``SKILL.md``) is python-relevant — the skill-doc tests
  read that tree, so a doc-looking edit can still break Python.
"""

from __future__ import annotations

import os
import sys

_FRONTEND = ("ui-tui/", "web/", "apps/")  # TS typecheck-matrix packages
_ROOT_NPM = {"package.json", "package-lock.json"}  # shifts every package's tree
_DOCKER_META = ("docker/", ".hadolint.yml", "Dockerfile") # docker setup
_SITE = ("website/", "skills/", "optional-skills/")  # docs site + skill pages
# Prose/frontend trees that can't touch Python. skills/ is excluded on purpose.
_PY_SKIP = ("docs/", "website/") + _FRONTEND

# Built browser assets served verbatim to the dashboard SPA. A compiled bundle
# under a plugin's ``dashboard/dist/`` (or an installed dashboard's asset dir)
# cannot import or execute Python — it's served as-is by ``serve_plugin_asset``.
# Editing one must NOT trigger the whole pytest matrix. The plugin's ``.py``
# backend (``plugin_api.py``) is NOT under ``dist/`` and stays python-relevant.
# Covers the full set of Vite/webpack ``dist/`` outputs: scripts, styles, source
# maps, fonts, images, plus ``index.html`` / ``asset-manifest.json`` / ``stats.json``.
_BROWSER_ASSET_EXTS = (".js", ".mjs", ".css", ".map", ".woff", ".woff2", ".svg",
                       ".png", ".jpg", ".jpeg", ".gif", ".ico", ".html", ".json",
                       ".txt", ".ttf", ".eot", ".webp", ".avif", ".wasm")


def _is_built_dashboard_asset(p: str) -> bool:
    # e.g. plugins/kanban/dashboard/dist/index.js, plugins/*/dashboard/dist/style.css
    return (
        p.startswith("plugins/")
        and "/dashboard/dist/" in p
        and p.endswith(_BROWSER_ASSET_EXTS)
    )


# Media / binary assets that cannot be imported or executed by Python anywhere in
# the tree — a logo swap, a screenshot, an icon, a sound. These force the whole
# pytest matrix today for zero reason. (Data-ish text like .json/.csv/.txt is NOT
# here: a fixture the tests read could legitimately change behavior → fail-open.)
_MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".avif", ".svg",
               ".bmp", ".tiff", ".wav", ".mp3", ".mp4", ".mov", ".webm", ".ogg",
               ".woff", ".woff2", ".ttf", ".eot", ".otf")

# Git/repo metadata files that never affect the Python test suite.
_REPO_META_FILES = {".gitignore", ".gitattributes", ".mailmap", ".editorconfig",
                    ".dockerignore", ".prettierignore", ".npmignore",
                    "CODEOWNERS", ".gitmodules"}


def _is_media(p: str) -> bool:
    return p.endswith(_MEDIA_EXTS)


def _is_repo_meta(p: str) -> bool:
    # match by basename so nested locations (e.g. subdir/.gitignore) also qualify
    return p in _REPO_META_FILES or p.rsplit("/", 1)[-1] in _REPO_META_FILES

# Supply-chain scan: files that can execute code at install/import time.
_SCAN_EXTS = (".py", ".pth")
_SCAN_FILES = {"setup.cfg", "pyproject.toml"}

# MCP catalog files that require explicit security review.
_MCP_CATALOG_PATHS = ("optional-mcps/",)
_MCP_CATALOG_FILES = {"hermes_cli/mcp_catalog.py"}

def _is_docs(p: str) -> bool:
    if p.startswith(("skills/", "optional-skills/")):
        return False
    return p.endswith((".md", ".mdx")) or p.startswith("docs/") or p.startswith("LICENSE")


def _py_irrelevant(p: str) -> bool:
    return (
        _is_docs(p)
        or _is_built_dashboard_asset(p)
        or _is_media(p)
        or _is_repo_meta(p)
        or p in _ROOT_NPM
        or p.startswith(_PY_SKIP)
        or p.startswith(_DOCKER_META)
    )


def _is_scan(p: str) -> bool:
    return p.endswith(_SCAN_EXTS) or p in _SCAN_FILES


def _is_mcp_catalog(p: str) -> bool:
    return p.startswith(_MCP_CATALOG_PATHS) or p in _MCP_CATALOG_FILES


def classify(files: list[str]) -> dict[str, bool]:
    """Map changed paths to ``{lane: should_run}``."""
    files = [f.strip() for f in files if f.strip()]
    ret = {
        "python": any(not _py_irrelevant(f) for f in files),
        "docker_meta":  any(f.startswith(_DOCKER_META) for f in files),
        "frontend": any(f.startswith(_FRONTEND) or f in _ROOT_NPM for f in files),
        "site": any(f.startswith(_SITE) for f in files),
        "scan": any(_is_scan(f) for f in files),
        "deps": any(f == "pyproject.toml" for f in files),
        "mcp_catalog": any(_is_mcp_catalog(f) for f in files),
    }
    if not files or any(f.startswith(".github/") for f in files):
        ret["python"] = True
        ret["docker_meta"] = True
        ret["frontend"] = True
        ret["site"] = True
        ret["scan"] = True
        ret["deps"] = True

        # explicitly skip mcp catalog here. it's not needed unless those files are modified.
    return ret



def main() -> int:
    lanes = classify(sys.stdin.read().splitlines())
    out = "\n".join(f"{k}={str(v).lower()}" for k, v in lanes.items())
    if dest := os.environ.get("GITHUB_OUTPUT"):
        with open(dest, "a", encoding="utf-8") as fh:
            fh.write(out + "\n")
    print(out)  # echo for local runs + CI step logs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
