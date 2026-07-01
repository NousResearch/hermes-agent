"""/cost reprice slash subcommand — dry-run by default, --apply mutates (D-8)."""
import importlib

import pytest


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point the blackbox store at a temp HERMES_HOME and reload commands so its
    module-level `store` reference binds to the temp-backed module."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import plugins.blackbox.store as store_mod
    importlib.reload(store_mod)
    import plugins.blackbox.commands as commands
    importlib.reload(commands)
    return commands, store_mod


def _seed(store_mod, turn_id, provider, model, i=0, o=0, cr=0, cw=0):
    from plugins.blackbox.record import TurnRecord

    store_mod.insert_turn(
        TurnRecord(
            turn_id=turn_id,
            provider=provider,
            model=model,
            input_tokens=i,
            output_tokens=o,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
        )
    )


def test_reprice_dry_run_reports_but_writes_nothing(wired):
    commands, store_mod = wired
    _seed(store_mod, "t1", "openai", "claude-opus-4-8", i=1826, o=548, cr=282825)
    out = commands.handle_cost("reprice")
    assert "DRY-RUN" in out
    for key in ("scanned", "repriced", "zeroed", "still_unknown"):
        assert key in out
    assert store_mod.get_turn("t1")["cost_usd"] is None  # nothing written


def test_reprice_apply_mutates(wired):
    commands, store_mod = wired
    _seed(store_mod, "t1", "openai", "claude-opus-4-8", i=1826, o=548, cr=282825)
    out = commands.handle_cost("reprice --apply")
    assert "APPLIED" in out
    assert store_mod.get_turn("t1")["cost_usd"] is not None
