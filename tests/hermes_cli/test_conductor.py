"""Policy tests for the standalone scheme dispatcher in conductor."""
from __future__ import annotations

import pytest

from hermes_cli.conductor import (
    SchemeDispatcher,
    _dispatch,
    _is_scheme_cmd,
    run_hermes,
)


@pytest.fixture()
def dispatcher() -> SchemeDispatcher:
    each = SchemeDispatcher()
    each.register("c://cc", lambda raw: {"ok": True, "stdout": f"cctx:{raw}\n"})
    each.register(
        "pc://run",
        lambda raw, name: {"ok": True, "stdout": f"pc_run:{raw}\n"},
    )
    each.register("pc://", lambda raw: {"ok": True, "stdout": f"pc:{raw}\n"})
    each.register("mcp://", lambda raw: {"ok": True, "stdout": f"mcp:{raw}\n"})
    each.register("custom://", lambda raw: {"ok": True, "stdout": f"custom:{raw}\n"})
    return each


def test_register_dispatches_first_match(dispatcher: SchemeDispatcher) -> None:
    assert dispatcher.dispatch("c://cc +æ://ops")["stdout"] == "cctx:c://cc +æ://ops\n"
    assert dispatcher.dispatch("pc://run alpha")["stdout"] == "pc_run:pc://run\n"
    assert dispatcher.dispatch("pc://")["stdout"] == "pc:pc://\n"
    assert dispatcher.dispatch("mcp://tools")["stdout"] == "mcp:mcp://tools\n"


def test_register_overlaps_preserve_longest_prefix() -> None:
    each = SchemeDispatcher()
    each.register(
        "c://",
        lambda raw: {"ok": False, "stdout": "unexpected"},
    )
    each.register("c://cc", lambda raw: {"ok": True, "stdout": "expected"})
    result = each.dispatch("c://cc +æ://ops")
    assert result == {"ok": True, "stdout": "expected"}


def test_is_scheme_cmd_when_matched(dispatcher: SchemeDispatcher) -> None:
    assert dispatcher.is_scheme_cmd("mcp://tools") is True
    assert dispatcher.is_scheme_cmd("pc://run default") is True
    assert dispatcher.is_scheme_cmd("c://cc daollc://") is True


def test_is_scheme_cmd_unknown() -> None:
    assert _is_scheme_cmd("random command") is False
    assert _is_scheme_cmd("") is False


def test_dispatch_unknown_scheme() -> None:
    result = _dispatch("foo://bar")
    assert result["ok"] is False
    assert result["rc"] == 2
    assert "unsupported scheme" in result["stderr"]


def test_default_dispatcher_handles_builtin_schemes() -> None:
    builtin = {
        "c://cc +æ://ops",
        "pc://run alpha",
        "pc://",
        "mcp://tools",
        "vscode://",
        "reachy://",
        "NOUS://",
        "llc://",
        "daollc://",
        "commandprompt://",
        "home://",
        "fs://",
        "fs://stat C:/æ/hermes-fork",
        "fs://tree C:/æ/hermes-fork",
    }
    for command in builtin:
        result = _dispatch(command)
        assert result["ok"] is True, command
        assert result["stdout"]
        assert "surface" in result
    identity_result = _dispatch("+æ://identity")
    assert identity_result["ok"] is True
    assert identity_result["surface"]["kind"] == "bounded_private_client_mesh"
    assert identity_result["surface"]["conductor"] == "hermes-agent/conductor"
    action_result = _dispatch("+æ://conductor plan")
    assert action_result["ok"] is True
    assert action_result["surface"]["kind"] == "ae_engineering_hub"
    assert action_result["surface"]["action"] == "plan"


def test_run_hermes_scheme_scheme_property() -> None:
    result = run_hermes({"cmd": "mcp://tools"})
    assert result["scheme"] == "mcp"
    assert result["ok"] is True


def test_run_hermes_cli_verb_passthrough() -> None:
    result = run_hermes({"cmd": "viewport status"})
    # unknown verbs should not claim scheme dispatch; they fall through to CLI
    assert "unsupported scheme" not in result.get("stderr", "")


def test_run_hermes_missing_cmd() -> None:
    result = run_hermes({})
    assert result["ok"] is False
    assert result["rc"] == 2
    assert result["stdout"] == ""
    assert result["stderr"] == "missing cmd"
    assert result["surface"] == {"kind": "invalid"}


def test_run_hermes_preserves_run_pc_default() -> None:
    result = run_hermes({"cmd": "run pc://"})
    assert result["ok"] is True
    assert "pc://run default" in result["stdout"]
    assert result["surface"]["client"] == "default"


def test_policy_surface_keys_present_for_builtins() -> None:
    cases = {
        "pc://": ("private_client",),
        "pc://run beta": ("private_client_run", "beta"),
        "c://cc +æ://ops": ("cctx",),
        "NOUS://": ("provider",),
        "vscode://": ("viewport_host",),
        "reachy://": ("robot",),
        "mcp://tools": ("mcp",),
        "daollc://": ("dao",),
        "+æ://ops": ("dao",),
        "commandprompt://": ("commandprompt",),
        "home://": ("os_home",),
        "fs://": ("fs",),
        "fs://stat C:/æ/hermes-fork": ("fs",),
        "fs://read C:/æ/hermes-fork/AGENTS.md": ("fs",),
        "fs://tree C:/æ/hermes-fork": ("fs",),
    }
    for command, expected_kinds in cases.items():
        result = _dispatch(command)
        assert result["ok"] is True, command
        assert result["surface"]["kind"] in expected_kinds
        if len(expected_kinds) == 2:
            assert result["surface"].get("client") == expected_kinds[1]
    home = _dispatch("home://")
    assert set(home["surface"]["entrypoints"]) == {
        "terminal", "editor", "files", "victus", "nvidia", "vlc", "ffmpeg", "qr", "mesh"
    }
    assert home["surface"]["entrypoints"]["files"] == "fs://"


def test_cc_dispatch_routes_to_gpu_mcp() -> None:
    """+æ://cc is the command & control surface -> local GPU-MCP (CUDA + Rust/WASM)."""
    result = _dispatch("+æ://cc home://")
    assert result["ok"] is True
    assert result["scheme_detail"] == "+æ://cc"
    assert result["surface"]["kind"] == "mcp"
    assert result["surface"]["address"] == "mcp://gpu-mcp"
    assert result["surface"]["launch"] == "python environments/gpu_mcp.py"


def test_cc_longer_prefix_outranks_bare_ae() -> None:
    """+æ://cc must win over the +æ:// (dao) prefix for cc subcommands."""
    cc = _dispatch("+æ://cc home://")
    bare = _dispatch("+æ://ops")
    assert cc["scheme_detail"] == "+æ://cc"
    assert bare["surface"]["kind"] == "dao"


def test_glocal_agent_dispatch_routes_to_gpu_mcp() -> None:
    """æ://glocal-agent is the canonical name for the sovereign local agent
    primitive (+æ^glocal) -> local GPU-MCP (alias of +æ://cc home://)."""
    result = _dispatch("æ://glocal-agent home://")
    assert result["ok"] is True
    assert result["scheme_detail"] == "æ://glocal-agent"
    assert result["surface"]["kind"] == "mcp"
    assert result["surface"]["address"] == "mcp://gpu-mcp"
    assert result["surface"]["launch"] == "python environments/gpu_mcp.py"


def test_glocal_agent_short_form_defaults_to_home() -> None:
    result = _dispatch("æ://glocal-agent")
    assert result["ok"] is True
    assert result["scheme_detail"] == "æ://glocal-agent"
    assert result["surface"]["node"] == "home://"


def test_viewport_scheme_is_the_mandate() -> None:
    """viewport:// is the mandate: the local HTML/CSS/WASM surface is the control
    plane. vscode:// is its host. The v in vscode = viewport, not Visual Studio."""
    vp = _dispatch("viewport://home://")
    assert vp["ok"] is True
    assert vp["surface"]["kind"] == "viewport"
    assert vp["surface"]["v"] == "viewport"
    vs = _dispatch("vscode://")
    assert vs["ok"] is True
    assert vs["surface"]["kind"] == "viewport_host"
    assert vs["surface"]["v"] == "viewport"
