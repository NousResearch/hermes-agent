"""A sidecar live-session RPC (``@_rpc(..., live_session=True)``) resolves ``session_id`` in the
sidecar's own session map, which never holds a shared-gateway session: the parity table must
not list a client that calls one as served by the sidecar."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_sidecar_live_session_rpc_is_reported_broken_on_the_shared_gateway():
    spec = importlib.util.spec_from_file_location("gen_gateway_command_parity",
                                                  ROOT / "scripts" / "gen_gateway_command_parity.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    cell = gen._rpc_cell({"rollback.list", "rollback.diff"}, set())

    assert cell.startswith("**broken**"), cell
    assert "rollback.diff" in cell and "rollback.list" in cell
    assert "sidecar:" not in cell
