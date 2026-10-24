"""scripts/check_cron_import_chain.py: the deployment interlock.

The cron external worker imports the LIVE checkout, so a module-level error
anywhere in the shared delivery/agent chain fails every cron job on the
gateway delivery path at once. On 2026-10-24 that was 17 jobs, one line of
code, and an error message ("exited before ownership acknowledgement") that
named none of it.

This check imports the chain for real, in a subprocess, so a broken tree is a
30-second error at commit time rather than a 17-job outage at the next tick.
"""

import importlib.util
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_cron_import_chain.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_cron_import_chain", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_chain_file_list_matches_the_smoke_probe():
    """The interlock must cover exactly the chain the smoke probe imports."""
    mod = _load()
    from cron.import_smoke import SMOKE_CHAIN

    assert mod.CHAIN_FILES == tuple(
        name.replace(".", "/") + ".py" for name in SMOKE_CHAIN
    )


def test_a_chain_file_is_recognised():
    mod = _load()
    root = mod.ROOT
    assert mod._is_chain_file(root / "agent" / "auxiliary_client.py")
    assert mod._is_chain_file(root / "gateway" / "delivery.py")


def test_a_file_off_the_chain_is_not_recognised():
    mod = _load()
    root = mod.ROOT
    assert not mod._is_chain_file(root / "cron" / "scheduler.py")
    assert not mod._is_chain_file(root / "hermes_cli" / "kanban_ops.py")


def test_a_file_outside_the_repo_is_not_recognised(tmp_path):
    mod = _load()
    assert not mod._is_chain_file(tmp_path / "agent" / "auxiliary_client.py")


def test_the_healthy_tree_passes():
    """The live tree imports cleanly, so the interlock is green."""
    mod = _load()
    assert mod.main([]) == 0


def test_a_non_chain_path_is_a_no_op():
    """A change that cannot break the chain is not probed."""
    mod = _load()
    assert mod.main(["cron/import_smoke.py"]) == 0
