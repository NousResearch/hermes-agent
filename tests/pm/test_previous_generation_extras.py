"""A rebuilt PM generation keeps the extras the generation it replaces held (#96180).

The recorded selection can be narrower than the environment it names: an SDK the operator put back
by hand after an update dropped it, or a ledger an interrupted migration wrote short. The next sync
replaces that environment, so an extra present there and missing from the ledger is gone for good
while the update reports success (Telegram cron delivery then fails on ``python-telegram-bot``).
"""
from __future__ import annotations

import json
import shutil

from pm.environments import install_state_dir, runtime_facts_path
from pm.lock import Facts
import pytest

from tests.hermes_cli import test_first_generation_carries_legacy_extras as _legacy


# The pre-PM carry suite's tree and real PM sync wiring, past its migration.
legacy_tree = pytest.fixture(_legacy.legacy_tree.__wrapped__)


# Real-shaped extras: an SDK extra names its package; ``vision`` is the empty alias it is upstream.
PYPROJECT = (
    '[project]\nname="hermes-agent"\nversion="1"\ndependencies=["Pillow==12.0.0"]\n'
    '[project.optional-dependencies]\nall=[]\ntelegram=["python-telegram-bot==22.8"]\n'
    'discord=["discord.py==2.6.4"]\nfal=["fal-client==0.14.1"]\nslack=["slack-sdk==3.45.0"]\nvision=[]\n')


def _committed(root, *anchors, recorded):
    shutil.rmtree(root / "venv")  # past the migration: only PM generations remain
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    venv = install_state_dir(root) / "environments" / "previous" / "venv"
    site = venv / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = fixture\n")
    for anchor in anchors:
        (site / anchor).mkdir()
    Facts(runtime_facts_path(root)).record_state("venv", "stamp:old", recorded, environment=venv)


def _recorded(root):
    return (Facts(runtime_facts_path(root)).get("venv") or {}).get("extras")


def test_a_sync_keeps_an_extra_the_committed_generation_held(legacy_tree):
    import pm

    _committed(legacy_tree, "telegram", "fal_client", recorded=["all"])

    pm.sync_venv(["slack"], explicit=True)

    assert {"all", "telegram", "fal", "slack"} <= set(_recorded(legacy_tree) or ()), _recorded(legacy_tree)


def test_a_dropped_extra_is_not_carried_back_from_the_committed_generation(legacy_tree):
    import pm
    from pm.extras import record_declined_extras

    _committed(legacy_tree, "telegram", recorded=["all"])
    record_declined_extras(add=["telegram"], project_root=legacy_tree)

    pm.sync_venv(["slack"], explicit=True)

    assert "telegram" not in set(_recorded(legacy_tree) or ()), _recorded(legacy_tree)


def test_a_ledger_that_matches_its_environment_is_unchanged(legacy_tree):
    import pm

    _committed(legacy_tree, "telegram", recorded=["all", "telegram"])

    pm.sync_venv(["slack"], explicit=True)

    assert sorted(_recorded(legacy_tree)) == ["all", "slack", "telegram"]
    assert json.loads(runtime_facts_path(legacy_tree).read_text())["packages"]["venv"]["extras"]


def test_an_empty_alias_whose_anchor_is_core_does_not_rebuild_a_healthy_generation(legacy_tree):
    """``vision = []`` anchors on Pillow, which is core: carrying it would change only the stamp."""
    import pm

    _committed(legacy_tree, "PIL", "telegram", recorded=["all", "telegram"])

    pm.sync_venv([], explicit=True)

    assert sorted(_recorded(legacy_tree)) == ["all", "telegram"]
