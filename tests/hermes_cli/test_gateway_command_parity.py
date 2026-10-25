"""The gateway command parity table is generated from the code and must be regenerated with it."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _generator():
    spec = importlib.util.spec_from_file_location("gen_gateway_command_parity",
                                                  ROOT / "scripts" / "gen_gateway_command_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_parity_table_matches_every_client_verdict():
    """A command whose gateway verdict, classic route, Ink/Desktop/ACP handling or port note
    changes without re-running the generator is drift; every registry command has a row."""
    gen = _generator()
    rendered = gen.render()
    assert gen.OUT.read_text(encoding="utf-8") == rendered, (
        "website/docs/developer-guide/gateway-command-parity.md is stale — run "
        "scripts/gen_gateway_command_parity.py")
    from hermes_cli.commands import COMMAND_REGISTRY
    rows = [line for line in rendered.splitlines() if line.startswith("| `/")]
    assert len(rows) == len(COMMAND_REGISTRY)
    # Per-session settings run their gateway handler for a local session (an admission-fenced control;
    # tests/gateway/test_session_setting_commands.py proves each lands on the session).
    for name in gen.TOGGLES:
        row = next(line for line in rows if line.startswith(f"| `/{name}`"))
        assert row.split(" | ")[1] == "control" and "per-session setting (gateway handler)" in row, row


def test_acp_column_is_the_table_acp_dispatch_reads(monkeypatch):
    """ACP's routes have one source (``acp_adapter.gateway_commands.ROUTES``): a route it changes
    changes the documented cell, and a command it drops falls back to the owner's ``slash.exec``."""
    from acp_adapter import gateway_commands
    gen = _generator()
    assert gen.acp_cell('undo', 'refused') == gateway_commands.ROUTES['undo'][0]
    monkeypatch.setitem(gateway_commands.ROUTES, 'undo', ('moved route', gateway_commands.ROUTES['undo'][1]))
    assert gen.acp_cell('undo', 'refused') == 'moved route'
    monkeypatch.delitem(gateway_commands.ROUTES, 'undo')
    assert gen.acp_cell('undo', 'refused') == 'slash.exec: refused'
