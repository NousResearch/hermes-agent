"""Regression tests for ``_update_node_dependencies()`` (issue #64354).

The legacy implementation ran two passes through
``_run_npm_install_deterministic``:

1. ``npm ci --workspaces=false`` → root-only deps
   (``agent-browser``, ``@streamdown/math``).
2. ``npm ci --workspace ui-tui --workspace web`` → scoped workspaces.

``npm ci`` wipes ``node_modules`` before reifying, so pass 2 removed what pass
1 had installed and never restored the root-only deps — both passes exited 0,
the ``✓`` branch fired, and browser tools were silently broken until the user
ran ``npm install agent-browser`` by hand. Verified with the minimal repro
shipped in #64354.

These tests cover (no real npm/network required):

* The single-pass shape: exactly one helper call, with
  ``--include-workspace-root`` plus both workspace flags, and the legacy
  ``--workspaces=false`` root-only pass gone.
* A failing install (non-zero exit) prints ``⚠`` and refrains from ``✓``.
* ``npm ci`` exits 0 but the resulting ``node_modules/`` is missing every
  root-only dep — the function must print ``⚠`` listing the missing names
  instead of claiming success (this is exactly the silent-failure shape that
  #64354 describes).
* Happy path: when every declared root dep is present after install, ``✓``
  still prints (no regression for users with an intact tree).
* Legacy flat-layout marker (``node_modules/<name>/package.json`` with no
  nested ``node_modules``) still counts as installed.
* An empty ``node_modules/<name>/`` directory is the partial / interrupted
  shape — must be flagged as missing, not mistaken for success.
* Scoped-only manifest (no top-level ``dependencies`` key) does not crash the
  check.
"""

import json
import sys
import types
from pathlib import Path

import pytest


def _fake_hermes_constants_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stub ``hermes_constants`` so ``_update_node_dependencies``
    can ``from hermes_constants import find_node_executable, with_hermes_node_path``
    without requiring the real package or a Node install.
    """
    stub = types.ModuleType("hermes_constants")
    stub.find_node_executable = lambda name: f"/usr/bin/{name}"

    def fake_with_hermes_node_path(env):
        return dict(env)

    stub.with_hermes_node_path = fake_with_hermes_node_path
    monkeypatch.setitem(sys.modules, "hermes_constants", stub)


def _write_pkg(
    root: Path, deps: dict | None = None, *, name: str = "x"
) -> None:
    pkg = {
        "name": name,
        "private": True,
        "scripts": {
            "install:root": "npm install --workspaces=false",
            "install:web": "npm install --workspace web",
            "install:tui": "npm install --workspace ui-tui",
        },
    }
    if deps is not None:
        pkg["dependencies"] = deps
    (root / "package.json").write_text(json.dumps(pkg))


def _fake_installed(root: Path, dep_name: str, *, nested_node_modules: bool = True) -> None:
    """Mark a dep as installed in a hoisted-layout shape (the post-npm-7 default)."""
    dep_dir = root / "node_modules" / dep_name
    dep_dir.mkdir(parents=True, exist_ok=True)
    (dep_dir / "package.json").write_text("{}")
    if nested_node_modules:
        # Hoisted deps have a nested node_modules sub-dir.
        (dep_dir / "node_modules").mkdir(exist_ok=True)


def _capture_helper(monkeypatch, pkg_main, *, returncode: int, stderr: str = ""):
    captured: dict = {"calls": 0, "extra_args": None, "capture_output": None}

    def fake(*_args, extra_args=(), capture_output=True, **_kwargs):
        captured["calls"] += 1
        captured["extra_args"] = tuple(extra_args)
        captured["capture_output"] = capture_output
        return types.SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")

    monkeypatch.setattr(pkg_main, "_run_npm_install_deterministic", fake)
    return captured


@pytest.fixture
def main_pkg(monkeypatch, tmp_path):
    """Import the CLI module against our tmp PROJECT_ROOT and stubbed
    ``hermes_constants``. Returns the loaded module.
    """
    _fake_hermes_constants_module(monkeypatch)
    import hermes_cli.main as pkg_main

    monkeypatch.setattr(pkg_main, "PROJECT_ROOT", tmp_path)
    return pkg_main


# ─── single-pass shape ──────────────────────────────────────────────────────


def test_single_pass_uses_include_workspace_root(main_pkg, monkeypatch) -> None:
    """Exactly one helper call, with --include-workspace-root + both workspaces."""
    _write_pkg(main_pkg.PROJECT_ROOT, {"agent-browser": "0.0.1"})
    _fake_installed(main_pkg.PROJECT_ROOT, "agent-browser")

    captured = _capture_helper(monkeypatch, main_pkg, returncode=0)
    main_pkg._update_node_dependencies()

    assert captured["calls"] == 1, (
        f"expected single-pass install, got {captured['calls']} calls"
    )
    args = captured["extra_args"]
    # Required flags in any order between --no-* and --workspace
    for required in (
        "--include-workspace-root",
        "--workspace",
        "ui-tui",
        "web",
    ):
        assert required in args, f"{required!r} missing from {args!r}"
    assert "--workspaces=false" not in args, (
        "legacy root-only pass must be gone — its interaction with the scoped "
        "pass is exactly what introduced #64354"
    )


def test_legacy_root_pass_eliminated_no_double_call(main_pkg, monkeypatch, capsys) -> None:
    """Belt-and-braces: even when root deps stay around, the helper is hit once, not twice."""
    _write_pkg(main_pkg.PROJECT_ROOT, {"agent-browser": "0.0.1"})
    _fake_installed(main_pkg.PROJECT_ROOT, "agent-browser")

    captured = _capture_helper(monkeypatch, main_pkg, returncode=0)
    main_pkg._update_node_dependencies()

    assert captured["calls"] == 1
    out = capsys.readouterr().out
    assert "✓" in out
    assert "⚠" not in out


# ─── install failure path ──────────────────────────────────────────────────


def test_install_failure_surfaces_warning(main_pkg, monkeypatch, capsys) -> None:
    """Non-zero exit → ⚠, last stderr line surfaced, no ✓."""
    _write_pkg(main_pkg.PROJECT_ROOT, {"agent-browser": "0.0.1"})
    _capture_helper(
        monkeypatch, main_pkg, returncode=1, stderr="npm ERR! ERESOLVE\n"
    )

    main_pkg._update_node_dependencies()

    out = capsys.readouterr().out
    assert "✓" not in out
    assert "⚠" in out
    assert "ERESOLVE" in out


# ─── silent-drop hardening (#64354 violation) ──────────────────────────────


def test_silent_root_drop_emits_warning_listing_missing(
    main_pkg, monkeypatch, capsys
) -> None:
    """npm ci returns 0 but root-only deps are missing → ⚠, list names, no ✓."""
    _write_pkg(
        main_pkg.PROJECT_ROOT,
        {"agent-browser": "0.0.1", "@streamdown/math": "0.0.2"},
    )
    _capture_helper(monkeypatch, main_pkg, returncode=0)
    # node_modules intentionally empty — exactly the #64354 violation.

    main_pkg._update_node_dependencies()

    out = capsys.readouterr().out
    assert "✓" not in out, f"must NOT print ✓ when root deps are missing, got: {out!r}"
    assert "⚠" in out
    assert "agent-browser" in out
    assert "@streamdown/math" in out
    assert "Browser" in out  # tells the user who is impacted


def test_partial_drop_warning_lists_only_missing(main_pkg, monkeypatch, capsys) -> None:
    """If only one of multiple root deps is missing, the warning names just that one."""
    _write_pkg(
        main_pkg.PROJECT_ROOT,
        {"agent-browser": "0.0.1", "@streamdown/math": "0.0.2"},
    )
    _fake_installed(main_pkg.PROJECT_ROOT, "@streamdown/math")  # one present
    _capture_helper(monkeypatch, main_pkg, returncode=0)

    main_pkg._update_node_dependencies()

    out = capsys.readouterr().out
    assert "✓" not in out
    assert "⚠" in out
    assert "agent-browser" in out
    assert "@streamdown/math" not in out


# ─── happy path — don't regress ────────────────────────────────────────────


def test_happy_path_when_tree_intact(main_pkg, monkeypatch, capsys) -> None:
    """When every declared root dep is present after install, ✓ still prints."""
    _write_pkg(main_pkg.PROJECT_ROOT, {"agent-browser": "0.0.1"})
    _fake_installed(main_pkg.PROJECT_ROOT, "agent-browser")
    _capture_helper(monkeypatch, main_pkg, returncode=0)

    main_pkg._update_node_dependencies()

    out = capsys.readouterr().out
    assert "✓" in out
    assert "⚠" not in out


def test_legacy_flat_layout_still_counts_as_installed(
    main_pkg, monkeypatch, capsys
) -> None:
    """npm <7 flat layout: node_modules/<name>/package.json with no nested dir."""
    _write_pkg(main_pkg.PROJECT_ROOT, {"agent-browser": "0.0.1"})
    _fake_installed(main_pkg.PROJECT_ROOT, "agent-browser", nested_node_modules=False)
    _capture_helper(monkeypatch, main_pkg, returncode=0)

    main_pkg._update_node_dependencies()

    out = capsys.readouterr().out
    assert "✓" in out, f"legacy flat layout must count as success, got: {out!r}"
    assert "⚠" not in out


def test_empty_dir_treated_as_missing(main_pkg, monkeypatch, capsys) -> None:
    """Empty ``node_modules/<name>/`` directory = partial / interrupted install."""
    _write_pkg(main_pkg.PROJECT_ROOT, {"agent-browser": "0.0.1"})
    (main_pkg.PROJECT_ROOT / "node_modules" / "agent-browser").mkdir(
        parents=True, exist_ok=True
    )
    # No package.json, no nested node_modules — neither marker present.
    _capture_helper(monkeypatch, main_pkg, returncode=0)

    main_pkg._update_node_dependencies()

    out = capsys.readouterr().out
    assert "✓" not in out
    assert "⚠" in out
    assert "agent-browser" in out


def test_no_top_level_dependencies_key_does_not_crash(
    main_pkg, monkeypatch, capsys
) -> None:
    """A package.json without ``dependencies`` short-circuits the check (no ✓ check, no warning)."""
    _write_pkg(main_pkg.PROJECT_ROOT, deps=None)  # no dependencies key
    _capture_helper(monkeypatch, main_pkg, returncode=0)

    main_pkg._update_node_dependencies()

    out = capsys.readouterr().out
    # No declared deps → no tree to verify → ✓ prints cleanly.
    assert "✓" in out
    assert "⚠" not in out


def test_unreadable_manifest_falls_back_to_success(
    main_pkg, monkeypatch, capsys
) -> None:
    """If package.json can't be read, we don't crash — we fall back to ✓ per
    spec guidance: tree verification is best-effort hardening, not a hard
    gate that breaks installs on transient FS errors."""
    _capture_helper(monkeypatch, main_pkg, returncode=0)

    # Make package.json unreadable / invalid JSON.
    (main_pkg.PROJECT_ROOT / "package.json").write_text("{ not-json")

    main_pkg._update_node_dependencies()

    out = capsys.readouterr().out
    assert "✓" in out
    assert "⚠" not in out
