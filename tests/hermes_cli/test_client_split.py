"""Tests for the registry-driven prod/dev volume split (hermes_cli.client_split).

The critical invariants under test: the source volume is never mutated, only
registry-listed profiles survive in each target, and the missing-profile guard
blocks an apply that would silently drop a client.
"""

from pathlib import Path

import pytest

from hermes_cli.clients import parse_registry
from hermes_cli.client_split import (
    compute_plan,
    discover_profiles,
    run_split,
)


def _registry():
    return parse_registry({"clients": [
        {"name": "default", "env": "dev"},
        {"name": "alpha", "env": "prod", "telegram_token_ref": "A"},
        {"name": "bravo", "env": "prod", "telegram_token_ref": "B"},
        {"name": "sandbox", "env": "dev", "telegram_token_ref": "S"},
    ]})


def _make_source(tmp_path: Path, profiles) -> Path:
    src = tmp_path / "data"
    (src / "profiles").mkdir(parents=True)
    for name in profiles:
        d = src / "profiles" / name
        d.mkdir()
        (d / "gateway_state.json").write_text("{}", encoding="utf-8")
    # a root-level file representing the 'default' profile's config
    (src / "config.yaml").write_text("model: x\n", encoding="utf-8")
    return src


class TestComputePlan:
    def test_prod_keeps_only_prod_profiles(self):
        present = {"alpha", "bravo", "sandbox"}
        plan = compute_plan(_registry(), "prod", "/t", present)
        assert plan.keep == ("alpha", "bravo")
        assert plan.prune == ("sandbox",)
        assert plan.missing == ()

    def test_dev_prunes_all_prod_profiles(self):
        present = {"alpha", "bravo", "sandbox"}
        plan = compute_plan(_registry(), "dev", "/t", present)
        # default is the root (not under profiles/), so only 'sandbox' is kept
        assert plan.keep == ("sandbox",)
        assert set(plan.prune) == {"alpha", "bravo"}

    def test_missing_profile_detected(self):
        present = {"alpha"}  # bravo declared in registry but absent on disk
        plan = compute_plan(_registry(), "prod", "/t", present)
        assert plan.missing == ("bravo",)


class TestDiscover:
    def test_discover_profiles(self, tmp_path):
        src = _make_source(tmp_path, ["alpha", "bravo"])
        assert discover_profiles(src) == {"alpha", "bravo"}

    def test_discover_no_profiles_dir(self, tmp_path):
        assert discover_profiles(tmp_path / "nope") == set()


class TestRunSplitDryRun:
    def test_dry_run_makes_no_changes(self, tmp_path, capsys):
        src = _make_source(tmp_path, ["alpha", "bravo", "sandbox"])
        called = []
        run_split(
            _registry(), src,
            {"prod": tmp_path / "prod", "dev": tmp_path / "dev"},
            apply=False, runner=lambda argv: called.append(argv),
        )
        assert called == []                       # no copy/backup commands
        assert not (tmp_path / "prod").exists()   # nothing created
        assert discover_profiles(src) == {"alpha", "bravo", "sandbox"}  # source intact


class TestRunSplitApply:
    def test_apply_prunes_targets_not_source(self, tmp_path):
        src = _make_source(tmp_path, ["alpha", "bravo", "sandbox"])
        prod = tmp_path / "prod"
        dev = tmp_path / "dev"

        # Fake mirror: copy source tree into target so the prune step is real.
        import shutil as _sh

        def fake_runner(argv):
            # emulate `rsync -a --delete src/ target/` and `cp -a`
            if argv[0] == "rsync":
                target = Path(argv[-1].rstrip("/"))
                if target.exists():
                    _sh.rmtree(target)
                _sh.copytree(src, target)
            elif argv[0] == "cp":
                _sh.copytree(argv[-2], argv[-1])

        run_split(
            _registry(), src, {"prod": prod, "dev": dev},
            apply=True, runner=fake_runner,
        )

        # prod keeps only prod profiles
        assert discover_profiles(prod) == {"alpha", "bravo"}
        # dev keeps only dev-named profiles (default is root, not a dir)
        assert discover_profiles(dev) == {"sandbox"}
        # source is completely untouched
        assert discover_profiles(src) == {"alpha", "bravo", "sandbox"}
        # root-level config survived the mirror in both targets
        assert (prod / "config.yaml").exists()
        assert (dev / "config.yaml").exists()

    def test_missing_profile_aborts_apply(self, tmp_path):
        src = _make_source(tmp_path, ["alpha"])  # 'bravo' missing
        with pytest.raises(ValueError, match="don't exist on the source"):
            run_split(
                _registry(), src, {"prod": tmp_path / "prod"},
                apply=True, runner=lambda a: None,
            )

    def test_allow_missing_overrides(self, tmp_path):
        src = _make_source(tmp_path, ["alpha"])
        import shutil as _sh

        def fake_runner(argv):
            if argv[0] == "rsync":
                target = Path(argv[-1].rstrip("/"))
                _sh.copytree(src, target)

        # should not raise
        plans = run_split(
            _registry(), src, {"prod": tmp_path / "prod"},
            apply=True, allow_missing=True, runner=fake_runner,
        )
        assert plans[0].missing == ("bravo",)


class TestSafety:
    def test_target_equals_source_rejected(self, tmp_path):
        src = _make_source(tmp_path, ["alpha"])
        with pytest.raises(ValueError, match="must differ from source"):
            run_split(_registry(), src, {"prod": src}, apply=False)

    def test_target_inside_source_rejected(self, tmp_path):
        src = _make_source(tmp_path, ["alpha"])
        with pytest.raises(ValueError, match="must not live inside source"):
            run_split(_registry(), src, {"prod": src / "sub"}, apply=False)

    def test_missing_source_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="source data dir not found"):
            run_split(_registry(), tmp_path / "nope", {"prod": tmp_path / "p"}, apply=False)
