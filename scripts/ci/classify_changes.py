#!/usr/bin/env python3
"""Classify a PR's changed files into CI work lanes.

Reads newline-separated changed paths on stdin and writes ``key=value``
booleans (one per lane) to ``$GITHUB_OUTPUT`` and stdout. The
``detect-changes`` composite action consumes them so steps gate on
``if: steps.changes.outputs.<lane> == 'true'``.

Lanes:

* ``python``      — pytest / ruff / ty / footguns.
* ``python_prod`` — Python changes OUTSIDE tests/ — gates jobs that ship or
  run the product (Desktop E2E backend, Docker image) but never import the
  test suite. A tests-only PR keeps ``python`` (pytest must run) while
  skipping those product jobs.
* ``docker_meta`` — Dockerfiles etc.
* ``docker`` — any product change + docker meta
* ``nix``         — ``nix flake check``: the flake inputs and any product change.
* ``frontend``    — TS typecheck matrix + desktop build.
* ``site``        — Docusaurus + generated skill docs.
* ``scan``        — supply-chain scan (Python files, .pth, setup hooks).
* ``deps``        — pyproject.toml dependency bounds check.
* ``uv_lock``     — ``uv lock --check``. Re-resolves the whole graph against
  PyPI, so a diff that touches neither ``pyproject.toml`` nor ``uv.lock``
  must not run it.
* ``npm_lock``    — semantic package-lock.json diff PR comment.
* ``installer``   — PowerShell installer tests (Windows runner).
* ``rust``        — ``cargo test`` for the Tauri bootstrap installer. ``.rs``
  lives under ``apps/``, so without this lane a Rust change matched ``frontend``
  and only the TypeScript matrix ran.
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
* ``nix/``, ``flake.nix`` and ``flake.lock`` are the exception the other way:
  only the flake reads them, so they skip the Python lanes and run ``nix``
  alone. ``pyproject.toml`` and ``uv.lock`` are flake inputs too, but the
  packaging tests read them, so they keep every Python lane.
* ``website/static/oauth/`` is python-relevant too: it publishes the OAuth
  Client ID Metadata Document that ``tests/tools/test_mcp_cimd.py`` checks
  against the pinned callback ports in ``tools/mcp_oauth.py``.
* ``website/docs/`` and ``website/scripts/`` are python-relevant for the same
  reason: the docs tree generates ``llms.txt``, and
  ``tests/website/test_generate_llms_txt.py`` asserts every page reaches it.
"""

from __future__ import annotations

import json
import os
import sys

_FRONTEND = ("ui-tui/", "web/", "apps/")  # TS typecheck-matrix packages
_ROOT_NPM = {"package.json", "package-lock.json"}  # shifts every package's tree
_DOCKER_META = ("docker/", ".hadolint.yml", "Dockerfile") # docker setup
_NIX_PATHS = ("nix/",) # nix files
_NIX_FILES = {"flake.nix", "flake.lock"} # base nix files
_SITE = ("website/", "skills/", "optional-skills/")  # docs site + skill pages
# Prose/frontend trees that can't touch Python. skills/ is excluded on purpose.
_PY_SKIP = ("docs/", "website/") + _FRONTEND
# Published artifacts that live under website/ but that Python asserts about.
# The OAuth Client ID Metadata Document is cross-checked against the pinned
# callback ports in tools/mcp_oauth.py, so editing it alone must still run the
# Python lane — otherwise dropping a redirect URI goes green here and breaks
# every CIMD login on main.
# website/docs/ and website/scripts/ are asserted about the same way. The docs
# tree generates llms.txt — the index every LLM (Hermes included, via the
# hermes-agent skill) reads to learn what Hermes can do — and
# tests/website/test_generate_llms_txt.py holds every page to appearing in it.
# Skipping Python on a docs-only PR is how the index drifted to 53% coverage.
_PY_RELEVANT_SITE = (
    "website/static/oauth/",
    "website/docs/",
    "website/scripts/",
)

# CI-sensitive files: eslint config, workflow files, composite actions.
# Changes here can influence what code the autofix job executes and pushes to
# main, so they require explicit maintainer review (ci-reviewed label).
#
# package.json is deliberately NOT listed here: npm scripts only execute on the
# unprivileged generate-patch runner (contents: read), never on the privileged
# apply-patch job. The two-job split means a malicious package.json script
# can't get push access — it runs on an ephemeral runner with zero write perms.
_CI_REVIEW_FILES = {
    ".prettierrc",
}
_CI_REVIEW_PATHS = (".github/workflows/", ".github/actions/")

# Supply-chain scan: files that can execute code at install/import time.
_SCAN_EXTS = (".py", ".pth")
_SCAN_FILES = {"setup.cfg", "pyproject.toml"}

# MCP catalog files that require explicit security review.
_MCP_CATALOG_PATHS = ("optional-mcps/",)
_MCP_CATALOG_FILES = {"hermes_cli/mcp_catalog.py"}

# Windows installer + its PowerShell tests. These only run on a Windows runner,
# so they get their own lane rather than riding along with ``python``.
_INSTALLER_PATHS = ("scripts/tests/",)
_INSTALLER_FILES = {"scripts/install.ps1", "scripts/install.cmd"}

# Rust crates — currently just the Tauri bootstrap installer (Hermes-Setup).
# These live under ``apps/``, so before this lane existed a ``.rs`` edit matched
# ``frontend`` and nothing more: the TypeScript matrix built, cargo never ran,
# and the crate's unit tests had never executed in CI at all.
_RUST_PATHS = ("apps/bootstrap-installer/src-tauri/",)
_RUST_FILENAMES = {"Cargo.toml", "Cargo.lock"}

def _is_docs(p: str) -> bool:
    if p.startswith(("skills/", "optional-skills/")):
        return False
    return p.endswith((".md", ".mdx")) or p.startswith("docs/") or p.startswith("LICENSE")


def _is_nix(p: str) -> bool:
    return p.startswith(_NIX_PATHS) or p in _NIX_FILES


def _py_irrelevant(p: str) -> bool:
    if p.startswith(_PY_RELEVANT_SITE):
        return False
    return (
        _is_docs(p)
        or p in _ROOT_NPM
        or p.startswith(_PY_SKIP)
        or p.startswith(_DOCKER_META)
        or _is_nix(p)
    )


def _py_test_only(p: str) -> bool:
    """Is ``p`` inside the test suite (never shipped / imported by the product)?

    Product jobs (Desktop E2E's ``hermes serve`` backend, the Docker image)
    run installed code — nothing under ``tests/`` is packaged or importable
    there. scripts/run_tests.sh and run_tests_parallel.py are deliberately
    NOT test-only: they are runner infrastructure, and a bad edit there can
    mask real failures, so they stay conservative (python_prod=true).
    """
    return p.startswith("tests/")


def _is_scan(p: str) -> bool:
    return p.endswith(_SCAN_EXTS) or p in _SCAN_FILES


def _is_mcp_catalog(p: str) -> bool:
    return p.startswith(_MCP_CATALOG_PATHS) or p in _MCP_CATALOG_FILES


def _is_installer(p: str) -> bool:
    return p.startswith(_INSTALLER_PATHS) or p in _INSTALLER_FILES


def _is_rust(p: str) -> bool:
    return (
        p.endswith(".rs")
        or p.startswith(_RUST_PATHS)
        or os.path.basename(p) in _RUST_FILENAMES
    )


def _is_ci_review(p: str) -> bool:
    if p in _CI_REVIEW_FILES or p.startswith(_CI_REVIEW_PATHS):
        return True
    # Any eslint config file at any path — eslint configs can define custom
    # fix functions that execute arbitrary code, so they all require review.
    return os.path.basename(p).startswith("eslint.config.")


def ci_review_files(files: list[str]) -> list[str]:
    """Return the CI-sensitive paths that need maintainer review."""
    return sorted({f.strip() for f in files if f.strip() and _is_ci_review(f.strip())})


def classify(files: list[str]) -> dict[str, bool]:
    """Map changed paths to ``{lane: should_run}``."""
    files = [f.strip() for f in files if f.strip()]
    python = any(not _py_irrelevant(f) for f in files)
    python_prod = any(not _py_irrelevant(f) and not _py_test_only(f) for f in files)
    frontend = any(f.startswith(_FRONTEND) or f in _ROOT_NPM for f in files)
    deps = any(f == "pyproject.toml" for f in files)
    npm_lock = any(f.split("/")[-1] == "package-lock.json" for f in files)
    docker_meta = any(f.startswith(_DOCKER_META) for f in files)
    
    ret = {
        "python": python,
        "python_prod": python_prod,
        "docker": docker_meta or python_prod or frontend,
        "docker_meta": docker_meta,
        "frontend": frontend,
        "site": any(f.startswith(_SITE) for f in files),
        "scan": any(_is_scan(f) for f in files),
        "deps": deps,
        "uv_lock": any(f in ("pyproject.toml", "uv.lock") for f in files),
        "npm_lock": npm_lock,
        "installer": any(_is_installer(f) for f in files),
        "rust": any(_is_rust(f) for f in files),
        "mcp_catalog": any(_is_mcp_catalog(f) for f in files),
        "ci_review": any(_is_ci_review(f) for f in files),
        "nix": python_prod or frontend or any(_is_nix(f) for f in files)
    }
    if not files or any(f.startswith(".github/") for f in files):
        ret["python"] = True
        ret["python_prod"] = True
        ret["docker"] = True
        ret["docker_meta"] = True
        ret["frontend"] = True
        ret["site"] = True
        ret["scan"] = True
        ret["deps"] = True
        ret["uv_lock"] = True
        ret["npm_lock"] = True
        ret["installer"] = True
        ret["rust"] = True
        ret["nix"] = True
        ret["ci_review"] = True

        # explicitly skip mcp catalog here. it's not needed unless those files are modified.
    return ret



def main() -> int:
    files = sys.stdin.read().splitlines()
    lanes = classify(files)
    out = "\n".join([
        *(f"{key}={str(value).lower()}" for key, value in lanes.items()),
        f"ci_review_files={json.dumps(ci_review_files(files))}",
    ])
    if dest := os.environ.get("GITHUB_OUTPUT"):
        with open(dest, "a", encoding="utf-8") as fh:
            fh.write(out + "\n")
    print(out)  # echo for local runs + CI step logs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
