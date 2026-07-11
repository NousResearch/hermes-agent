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
        "+æ://ops": ("aectx",),
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
    assert result["surface"]["launch"] == "python -m gpu_mcp"


def test_cc_longer_prefix_outranks_bare_aectx() -> None:
    """+æ://cc must win over the +æ:// (aectx) catch-all for cc subcommands."""
    cc = _dispatch("+æ://cc home://")
    bare = _dispatch("+æ://ops")
    assert cc["scheme_detail"] == "+æ://cc"
    assert bare["surface"]["kind"] == "aectx"


def test_glocal_agent_dispatch_routes_to_gpu_mcp() -> None:
    """æ://glocal-agent is the canonical name for the sovereign local agent
    primitive (+æ^glocal) -> local GPU-MCP (alias of +æ://cc home://)."""
    result = _dispatch("æ://glocal-agent home://")
    assert result["ok"] is True
    assert result["scheme_detail"] == "æ://glocal-agent"
    assert result["surface"]["kind"] == "mcp"
    assert result["surface"]["address"] == "mcp://gpu-mcp"
    assert result["surface"]["launch"] == "python -m gpu_mcp"


def test_glocal_agent_short_form_defaults_to_home() -> None:
    result = _dispatch("æ://glocal-agent")
    assert result["ok"] is True
    assert result["scheme_detail"] == "æ://glocal-agent"
    assert result["surface"]["node"] == "home://"


def test_glocal_cloud_computer_is_hybrid_local_default() -> None:
    """+æ://glocal cloud computer resolves to a hybrid surface: local hands
    always, and a LOCAL brain by default (cloud is opt-in only)."""
    result = _dispatch("+æ://glocal cloud computer")
    assert result["ok"] is True
    assert result["scheme_detail"] == "+æ://glocal cloud computer"
    surf = result["surface"]
    assert surf["kind"] == "hybrid"
    # hands are ALWAYS the sovereign local gpu-mcp
    assert surf["hands"]["address"] == "mcp://gpu-mcp"
    assert surf["hands"]["launch"] == "python -m gpu_mcp"
    # brain defaults local — never silently cloud
    assert surf["brain"]["provider"] == "local"
    assert surf["brain"]["opt_in"] is False
    assert surf["brain"]["policy"] == "local-default; cloud-explicit-only"


def test_glocal_cloud_computer_opt_in_portal() -> None:
    """Explicit `portal` token flips the brain to Nous Portal (opt-in only)."""
    result = _dispatch("+æ://glocal cloud computer portal")
    assert result["surface"]["brain"]["provider"] == "nous-portal"
    assert result["surface"]["brain"]["opt_in"] is True


def test_mesh_qr_pairing_offer_accept() -> None:
    """+æ://mesh offer emits a QR; the scanned manifest accepts into a live
    pc://mesh/<name>/local route. Opt-in only; bad payload is rejected."""
    offer = _dispatch("+æ://mesh offer legion")
    assert offer["ok"] is True
    assert offer["surface"]["kind"] == "mesh_offer"
    assert offer["surface"]["route"] == "pc://mesh/legion/local"
    manifest = offer["surface"]["manifest"]
    assert manifest.startswith("ae://peer?host=legion")
    accept = _dispatch(f"+æ://mesh accept {manifest}")
    assert accept["ok"] is True
    peer = _dispatch("pc://mesh/legion/local")
    assert peer["ok"] is True
    assert peer["surface"]["address"] == "pc://mesh/legion/local"
    bad = _dispatch("+æ://mesh accept garbage")
    assert bad["ok"] is False





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


def test_viewport_hermes_agent_resolves_to_concrete_surface() -> None:
    """viewport://hermes-agent is the concrete instance: the Hermes Agent
    viewport (ae://glocal-agent primitive rendered local)."""
    r = _dispatch("viewport://hermes-agent")
    assert r["ok"] is True
    assert r["surface"]["node"] == "hermes-agent"
    assert r["surface"]["agent"] == "ae://glocal-agent"
    assert r["surface"]["control_surface"] == "mcp://gpu-mcp"
    assert r["surface"]["brain"] == "ollama://localhost:11434"
    assert r["surface"]["html"] == "templates/surfaces/index.html"


def test_ae_agentic_context_router() -> None:
    """æ:// is the agentic context router; bare +æ:// catch-all resolves to
    aectx (not dao). Longer prefixes still win over the catch-all."""
    bare = _dispatch("æ://pc://")
    assert bare["ok"] is True
    assert bare["surface"]["kind"] == "aectx"
    assert bare["surface"]["target"] == "pc://"
    assert bare["surface"]["active"] is True

    plus = _dispatch("+æ://ops")
    assert plus["surface"]["kind"] == "aectx"
    assert plus["surface"]["target"] == "ops"

    ga = _dispatch("æ://glocal-agent home://")
    assert ga["scheme_detail"] == "æ://glocal-agent"
    assert ga["surface"]["kind"] == "mcp"

    cc = _dispatch("+æ://cc home://")
    assert cc["scheme_detail"] == "+æ://cc"


def test_hermes_superagent_blocked_scalar_supremacy() -> None:
    """hermes-superagent:// is blocked at the chassis (scalar supremacy).

    No tier above the sovereign scalar; the language enforces the boundary so
    surfaces linking to it dead-end at the router. status must NOT resolve.
    """
    r = _dispatch("hermes-superagent://status")
    assert r["ok"] is False
    assert r["rc"] == 2
    assert r["surface"]["kind"] == "blocked"
    assert r["surface"]["reason"] == "scalar-supremacy"
    assert r["surface"]["route_through"] == "æ://"


def test_pc_reports_sovereign_mesh() -> None:
    """pc:// reports the canonical mesh; pc://<node> addresses a node on it."""
    bare = _dispatch("pc://")
    assert bare["surface"]["kind"] == "private_client"
    assert bare["surface"]["mesh"] == "pc://mesh/victus/local"
    assert bare["surface"]["node"] == "pc://mesh/victus/local"
    assert bare["surface"]["local_only"] is True

    node = _dispatch("pc://mesh/victus/local/vlc")
    assert node["surface"]["node"] == "mesh/victus/local/vlc"


def test_robot_scheme_with_reachy_flagship() -> None:
    """robot:// is the abstract embodiment scheme; reachy:// is the flagship.

    robot://reachy and reachy:// resolve to the same flagship surface; a
    generic robot://<model> resolves non-flagship. All ride the pc:// mesh.
    """
    reachy = _dispatch("reachy://")
    assert reachy["surface"]["kind"] == "robot"
    assert reachy["surface"]["model"] == "reachy"
    assert reachy["surface"]["flagship"] is True
    assert reachy["scheme_detail"] == "reachy://"
    assert reachy["surface"]["mesh"] == "pc://mesh/victus/local"

    via_robot = _dispatch("robot://reachy")
    assert via_robot["surface"]["model"] == "reachy"
    assert via_robot["surface"]["flagship"] is True

    generic = _dispatch("robot://arm42 pc://mesh/victus/local")
    assert generic["surface"]["kind"] == "robot"
    assert generic["surface"]["model"] == "arm42"
    assert generic["surface"]["flagship"] is False
    assert generic["surface"]["node"] == "pc://mesh/victus/local"


def test_desktop_surface_under_cc() -> None:
    """desktop:// is the generative desktop viewport, routed under +æ://cc.

    Native when the WindowsDesktop actuator is available; otherwise it
    degrades to intent-reporting. Either way it stays sovereign-scoped.
    """
    d = _dispatch("desktop://focus")
    assert d["surface"]["kind"] == "desktop"
    assert d["surface"]["action"] == "focus"
    assert d["surface"]["control"] == "+æ://cc"
    assert d["surface"]["node"] == "pc://mesh/victus/local"
    # native bridge present on Windows; intent-only fallback elsewhere
    assert d["surface"].get("native") in (True, False)

    bare = _dispatch("desktop://")
    assert bare["surface"]["action"] == "enumerate"
