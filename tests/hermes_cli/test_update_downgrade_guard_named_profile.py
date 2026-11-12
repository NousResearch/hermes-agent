"""A named-profile update's refusal tells the user to reopen THAT profile for its snapshot."""
import subprocess

from hermes_cli import update_downgrade_guard as guard
from tests.hermes_cli import test_update_downgrade_guard as base

repo = base.repo  # the real git repo fixture: an old tag predating the runtime ledger


def _rev(root, ref):
    return subprocess.run(["git", "rev-parse", ref], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def test_named_profile_refusal_names_its_own_profile(repo, tmp_path, monkeypatch):
    root = tmp_path / "root"
    acme = root / "profiles" / "acme"
    monkeypatch.setenv("HERMES_HOME", str(acme))  # `hermes -p acme update`
    base._ledgered_store(acme)
    base._pre_cutover_snapshot(acme, "20261001-000000-pre-update")
    out = guard.downgrade_refusal(["git"], repo, _rev(repo, "old-release"), home=acme)
    assert f"{acme}: run `hermes -p acme` and enter `/snapshot restore 20261001-000000-pre-update`" in out
    assert "run `hermes` and" not in out
    # The default root's own update still points at bare `hermes`.
    base._ledgered_store(root)
    base._pre_cutover_snapshot(root, "20261002-000000-pre-update")
    monkeypatch.setenv("HERMES_HOME", str(root))
    out = guard.downgrade_refusal(["git"], repo, _rev(repo, "old-release"), home=root)
    assert f"{root}: run `hermes` and enter `/snapshot restore 20261002-000000-pre-update`" in out
