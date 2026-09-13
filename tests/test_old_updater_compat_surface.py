"""An old `hermes update` must still find every name it loads mid-swap.

`hermes update` replaces the checkout underneath a RUNNING process. The
old process then lazy-imports from the new tree: whatever it asks for
must still exist, or the user's update dies half-applied. The names it
can ask for are the contract of updaters shipped BEFORE the PM migration:
EVERY commit reachable from an explicit pre-PM cutoff, back to the first
cmd_update. The checked-in tests/compat/old_updater_surface.json already
contains a history-plus-tree superset; keep enforcing those names, but do
not continually expand it with new PM updater imports. Moving the
implementation to pm does not retire already-running old updaters.
Historical entrypoints, extractions, renamed and deleted helpers remain
part of the contract.

Regeneration requires a full clone. These tests deliberately consume the
checked-in names and resolve them against the current tree, so shallow CI can
enforce the contract without reconstructing (or silently truncating) it.

`managed_uv._reload_hermes_constants` is the scar proving the failure
mode is real: a live updater hit ``cannot import name 'venv_python_path'
from 'hermes_constants'`` while the NEW file on disk plainly held it.

If this test fails you have two honest options:
* restore the name (a stub with the old signature is fine), or
* prove no updater in the pre-PM history loads it, then review a full
  history audit at the same cutoff (the existing JSON records it under
  stats.history.history_ref):
      python scripts/audit-old-updater-imports.py --ref PRE_PM_COMMIT --freeze \
          tests/compat/old_updater_surface.json
New updater imports are not grounds for regeneration or for advancing the
cutoff to a later origin/main. Hand-trimming the JSON is how someone's
install bricks mid-update.

The test resolves names statically (AST over the files) rather than
importing: importing would execute module side effects and — worse —
resolve against the test venv, not the tree under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SURFACE_FILE = REPO_ROOT / "tests" / "compat" / "old_updater_surface.json"

# One authority for resolution: the audit script itself. The test must
# agree with --check byte-for-byte or the two drift apart.
import importlib.util as _ilu
import sys as _sys

_spec = _ilu.spec_from_file_location(
    "audit_old_updater_imports",
    REPO_ROOT / "scripts" / "audit-old-updater-imports.py",
)
assert _spec is not None and _spec.loader is not None
_audit = _ilu.module_from_spec(_spec)
# dataclass decorators resolve their defining module through sys.modules
# at class-creation time; exec without registration dies on it.
_sys.modules["audit_old_updater_imports"] = _audit
_spec.loader.exec_module(_audit)


def _load_surface() -> dict:
    return json.loads(SURFACE_FILE.read_text())


def _pairs(entries: list[str]) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for entry in entries:
        module, _, symbol = entry.partition("::")
        out.append((module, symbol or None))
    return out


class TestTheFrozenSurfaceStillResolves:
    """Every bare name in the retained frozen superset must exist in THIS tree."""

    @pytest.mark.parametrize(
        "module,symbol",
        _pairs(_load_surface()["bare"]),
        ids=lambda v: v or "<module>",
    )
    def test_bare_name_exists(self, module: str, symbol: str | None):
        ok, why = _audit.resolve_in_tree(module, symbol, REPO_ROOT)
        assert ok, (
            f"{why} — but the retained old-updater surface requires it AFTER "
            f"the checkout swap. Restore the name (a compat stub is fine) or "
            f"regenerate the frozen surface on a full clone; see this "
            f"file's docstring."
        )


class TestTheFrozenFileIsSane:
    """Catch a corrupted or hand-trimmed freeze before it lies to us."""

    def test_has_historical_load_bearing_names(self):
        # Spot-check names loaded by pre-PM updaters. If any of
        # these fall out of the frozen file, the freeze itself went wrong
        # (shallow clone, wrong branch) — the resolver test above would
        # pass vacuously.
        bare = set(_load_surface()["bare"])
        for anchor in (
            "hermes_constants::with_hermes_node_path",
            "hermes_constants::venv_python_path",
            "hermes_cli.gitlock::clear_stale_git_locks",
            "hermes_cli.managed_uv::ensure_uv",
            "hermes_cli.managed_uv::rebuild_venv",
            "hermes_cli._subprocess_compat::run",
        ):
            assert anchor in bare, (
                f"{anchor} missing from the frozen surface — the freeze "
                f"went wrong (wrong branch, or the audit lost its "
                f"entrypoints); regenerate and eyeball the diff."
            )

    def test_frozen_history_is_complete_including_the_retained_union(self):
        frozen = _load_surface()
        stats = frozen["stats"]
        assert stats.get("mode") in {"history", "union"}, (
            "frozen surface must include the complete pre-PM history; "
            "a current-tree-only audit cannot establish that contract."
        )
        # The existing union remains a safe superset, not a requirement to
        # add today's updater imports or regenerate its tree half.
        history = stats["history"] if stats["mode"] == "union" else stats
        assert history.get("complete_history") is True
        assert history["commits"] > 0
        assert history["roots"] and history["entrypoint_paths"]
        assert "hermes_cli/main.py" in history["entrypoint_paths"], (
            "the freeze omitted the original inline cmd_update history"
        )
        assert history["history_ref"]
        analyzed = set(history["files_analyzed"])
        for must_see in ("hermes_cli/update_cmd.py", "hermes_cli/managed_uv.py"):
            assert must_see in analyzed, (
                f"{must_see} was not analyzed for the freeze — the audit "
                f"lost part of the update flow; a vacuously small surface "
                f"cannot guard anything."
            )

        assert not set(frozen["bare"]) & set(frozen["guarded_only"])
