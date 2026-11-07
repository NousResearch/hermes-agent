"""Import probes must not complete an update or reject an unverified restore (#148264)."""

from pathlib import Path

import pytest

from hermes_cli import update_cmd, update_cmd_stash, update_cmd_validation
from tests.hermes_cli import test_update_target_identity as target_identity

git = target_identity.git
update_tree = target_identity.update_tree


def test_probe_disables_launch_preparation_before_bootstrap(probe_root, monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "0")
    (probe_root / ".git").mkdir()
    (probe_root / "pyproject.toml").write_text("", encoding="utf-8")
    # Exercise the real preparation entry point before the import-health body runs.
    # Reading the install stamp would start the stale-dependency resolution chain.
    repo = Path(__file__).resolve().parents[2]
    (probe_root / "hermes_bootstrap.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.append({str(repo)!r})\n"
        "from hermes_cli import steward, venv_sync\n"
        "def unexpected_preparation(*args, **kwargs):\n"
        "    raise RuntimeError('probe entered dependency preparation')\n"
        "steward.read_install_stamp = unexpected_preparation\n"
        f"venv_sync.prepare_launch(Path({str(probe_root)!r}), ['update'])\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    (probe_root / "consumer.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert update_cmd_validation._critical_module_import_failures(probe_root) == {}


@pytest.mark.parametrize("baseline,restored", [
    (None, {}), ({}, None), (None, {"consumer": ("RuntimeError", "unverified baseline")}),
])
def test_restore_does_not_compare_unverified_import_health(probe_root, monkeypatch, baseline, restored):
    git(probe_root, "init", "-q", "-b", "main")
    source = probe_root / "consumer.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git(probe_root, "add", ".")
    git(probe_root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "base")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    ref = update_cmd_stash._stash_local_changes_if_needed(["git"], probe_root)
    results = iter([baseline, restored])
    monkeypatch.setattr(update_cmd, "_critical_module_import_failures", lambda *a, **kw: next(results))

    assert update_cmd_stash._restore_stashed_changes(["git"], probe_root, ref)
    assert source.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert not git(probe_root, "stash", "list")


@pytest.mark.parametrize("moved", [True, False])
def test_interrupt_during_restore_resumes_gateways_and_finalizes_receipt(update_tree, monkeypatch, moved):
    from hermes_cli import main, update_receipt

    t = update_tree
    t.args.channel = "main"
    git(t.clone, "checkout", "-q", "main")
    if not moved:
        git(t.origin, "reset", "--hard", t.base)
    monkeypatch.setattr(main, "_sync_with_upstream_if_needed", update_cmd._sync_with_upstream_if_needed)
    local = t.clone / ".gitignore"
    local.write_text(local.read_text(encoding="utf-8") + "# local edit\n", encoding="utf-8")
    token = {"resume_needed": True}
    monkeypatch.setattr(update_cmd, "_pause_gateways_for_update", lambda opts: token)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(update_cmd, "_critical_module_import_failures", interrupt)
    with pytest.raises((SystemExit, KeyboardInterrupt)) as stopped:
        main.cmd_update(t.args)

    assert isinstance(stopped.value, SystemExit)
    assert stopped.value.code == 130
    assert token in t.resumed
    assert git(t.clone, "rev-parse", "HEAD") == (t.newer if moved else t.base)
    assert git(t.clone, "stash", "list")
    receipt = update_receipt.read_latest_receipt()
    assert receipt["outcome"] == ("interrupted" if moved else "failed")
    assert receipt["exit_code"] == 130
