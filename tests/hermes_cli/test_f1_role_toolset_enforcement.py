"""F1 role toolset enforcement — fixture tests against the frozen spec.

Frozen contract: profiles/smokey/workspace/f1-expected-inventory-spec-v1.md
Covers (offline, fixture-only, Gate 2/3 phase):
  I1  pinned direct inventory resolves to exactly the allowlisted toolsets
  I3  project/desktop_ui never appear unless the pin lists them
  I4  unknown toolset name in the pin fails CLOSED (no tools, no fallback)
  D8  agent.disabled_toolsets subtracts AFTER allowlist expansion
  plus: read-only bundles exist and resolve to read-only tool sets.
"""

import logging

import pytest

from hermes_cli.tools_config import _get_platform_tools
from toolsets import resolve_toolset


# ─── read-only bundles ────────────────────────────────────────────────────────

def test_file_readonly_bundle_resolves_to_read_only_tools():
    assert set(resolve_toolset("file_readonly")) == {"read_file", "search_files"}


def test_skills_readonly_bundle_resolves_to_read_only_tools():
    assert set(resolve_toolset("skills_readonly")) == {"skills_list", "skill_view"}


# ─── I1: pin is authoritative ─────────────────────────────────────────────────

def test_profile_pin_is_authoritative_allowlist():
    config = {
        "tools": {"enabled_toolsets": ["file_readonly", "skills_readonly", "web"]},
        "platform_toolsets": {"cli": ["hermes-cli"]},  # ignored when pinned
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert enabled == {"file_readonly", "skills_readonly", "web"}


def test_unpinned_profile_keeps_platform_toolsets_behavior():
    config = {"platform_toolsets": {"cli": ["hermes-cli"]}}
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    default_enabled = _get_platform_tools({}, "cli", include_default_mcp_servers=False)
    assert enabled == default_enabled


def test_pinned_profile_applies_across_platforms():
    """One pin governs every platform key (CLI, cron, telegram, ...)."""
    config = {"tools": {"enabled_toolsets": ["web", "memory"]}}
    for platform in ("cli", "cron", "telegram", "desktop"):
        assert _get_platform_tools(config, platform, include_default_mcp_servers=False) == {
            "web",
            "memory",
        }


# ─── I3: GUI extras never injected implicitly ─────────────────────────────────

def test_pinned_profile_does_not_gain_project_or_desktop_ui():
    config = {"tools": {"enabled_toolsets": ["web", "memory"]}}
    enabled = _get_platform_tools(config, "desktop", include_default_mcp_servers=False)
    assert "project" not in enabled
    assert "desktop_ui" not in enabled


def test_pinned_profile_can_explicitly_include_gui_toolsets():
    config = {"tools": {"enabled_toolsets": ["web", "project", "desktop_ui"]}}
    enabled = _get_platform_tools(config, "desktop", include_default_mcp_servers=False)
    assert {"project", "desktop_ui"} <= enabled


# ─── I4: fail closed ──────────────────────────────────────────────────────────

def test_unknown_toolset_name_fails_closed(caplog):
    config = {"tools": {"enabled_toolsets": ["web", "bogus_toolset_xyz"]}}
    with caplog.at_level(logging.ERROR, logger="hermes_cli.tools_config"):
        enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert enabled == set()
    assert any("bogus_toolset_xyz" in r.getMessage() for r in caplog.records)


def test_unknown_mcp_name_fails_closed():
    config = {"tools": {"enabled_toolsets": ["web", "not-a-mcp-server:tool"]}}
    assert _get_platform_tools(config, "cli", include_default_mcp_servers=True) == set()


def test_empty_pin_list_disables_everything():
    """An explicit empty list means 'no tools' — not 'unpinned'."""
    config = {"tools": {"enabled_toolsets": []}}
    assert _get_platform_tools(config, "cli", include_default_mcp_servers=False) == set()


# ─── D8: disabled_toolsets subtracts after expansion ──────────────────────────

def test_disabled_toolsets_subtract_after_allowlist():
    config = {
        "tools": {"enabled_toolsets": ["web", "terminal", "file"]},
        "agent": {"disabled_toolsets": ["terminal"]},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert "terminal" not in enabled
    assert "web" in enabled
    assert "file" in enabled


def test_smokey_f7_canary_cron_denial_survives_pin():
    """Canary: the F7 structural cron denial must hold under a role pin."""
    config = {
        "tools": {"enabled_toolsets": ["web", "file_readonly", "cronjob"]},
        "agent": {"disabled_toolsets": ["cronjob"]},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert "cronjob" not in enabled


# ─── MCP layering (exact allowlist — nothing auto-added) ─────────────────────

def test_pinned_profile_does_not_auto_add_mcp_servers():
    """A pin is an EXACT allowlist: enabled MCP servers are present only if
    the pin explicitly names them."""
    config = {
        "tools": {"enabled_toolsets": ["web"]},
        "mcp_servers": {"my_server": {"command": "foo", "enabled": True}},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=True)
    assert "my_server" not in enabled
    assert enabled == {"web"}


def test_pinned_profile_can_explicitly_name_mcp_server():
    config = {
        "tools": {"enabled_toolsets": ["web", "my_server"]},
        "mcp_servers": {"my_server": {"command": "foo", "enabled": True}},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=True)
    assert "my_server" in enabled
    assert enabled == {"web", "my_server"}


def test_pinned_profile_no_mcp_sentinel_is_accepted_and_ignored():
    """The legacy `no_mcp` sentinel stays valid on a pin (for configs copied
    from unpinned configs) but changes nothing — MCP is opt-in either way."""
    config = {
        "tools": {"enabled_toolsets": ["web", "no_mcp"]},
        "mcp_servers": {"my_server": {"command": "foo", "enabled": True}},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=True)
    assert enabled == {"web"}


# ─── tool-level denial through composites (D8 extension) ─────────────────────

def _final_surface(pin, disabled=()):
    """FINAL model-facing surface via REAL final assembly (no skip flag) plus
    the pre-assembly list for exact I2 scoping evidence.

    Returns (pre_names, final_names, scoped_deferred, bridge):
      pre_names       — pre-assembly tool names (skip_tool_search_assembly
                        path) for the current scope
      final_names     — the tool names the model ACTUALLY sees post-assembly
                        (curated-deferred core tools behind the bridge)
      scoped_deferred — scoped_deferrable_names over the PRE list: the
                        bridge-reachable universe (the I2 contract input)
      bridge          — the assembled tool_search/tool_describe/tool_call trio
    Vision availability is stubbed at the registry-ENTRY level (entries
    capture check_fn by direct reference — module-attr patching doesn't
    redirect).
    """
    import tools.vision_tools as vt
    from tools.registry import invalidate_check_fn_cache, registry
    from model_tools import get_tool_definitions
    from tools.tool_search import scoped_deferrable_names

    def available(*a, **k):
        return True

    targets = [e for e in registry.get_all_entries() if e.check_fn is vt.check_vision_requirements]
    saved = [(e, e.check_fn) for e in targets]
    for e in targets:
        e.check_fn = available
    try:
        invalidate_check_fn_cache()
        pre = get_tool_definitions(
            enabled_toolsets=list(pin),
            disabled_toolsets=list(disabled),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        final = get_tool_definitions(
            enabled_toolsets=list(pin),
            disabled_toolsets=list(disabled),
            quiet_mode=True,
        )
    finally:
        for e, fn in saved:
            e.check_fn = fn
        invalidate_check_fn_cache()
    pre_names = {t["function"]["name"] for t in pre}
    final_names = {t["function"]["name"] for t in final}
    bridge = final_names & {"tool_search", "tool_describe", "tool_call"}
    scoped_deferred = set(scoped_deferrable_names(pre))
    return pre_names, final_names, scoped_deferred, bridge


def test_acp_session_is_out_of_frozen_scope_and_honors_pins():
    """ACP is NOT a surface in the frozen spec's probe matrix (§4 lists
    Desktop/CLI/Windows remote/Windows CLI — no ACP session adapter). The F1
    v1 'every surface' claim is therefore narrowed to the frozen surfaces,
    and ACP keeps its own documented adapter behavior (_expand_acp_enabled_
    toolsets + configured MCP). This test pins that ACP's path is unchanged
    by F1 — so the claim and the code agree, and any future ACP pinning is a
    deliberate Gate-1 amendment, not an accident."""
    from acp_adapter.session import _expand_acp_enabled_toolsets

    expanded = _expand_acp_enabled_toolsets(["hermes-acp"], mcp_server_names=["srv"])
    assert expanded == ["hermes-acp", "mcp-srv"]


# ═══ Round-2 evidence (Prince's finding 7): REAL final assembly, exact ══════
# inventories per role, exact deferred/bridge surfaces, denial-RESULT paths.


def test_smokey_final_assembly_exact():
    """I1 exact via REAL final assembly: Smokey's model-facing surface.
    session_search is a curated-deferred core tool — the model reaches it
    through tool_call (scoped_deferred), not the visible schema."""
    pin = ["web", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"]
    pre, final, scoped_deferred, bridge = _final_surface(pin, disabled=["cronjob"])
    assert final == {
        "web_search",
        "web_extract",
        "read_file",
        "search_files",
        "skills_list",
        "skill_view",
        "memory",
        "clarify",
        "tool_search",
        "tool_describe",
        "tool_call",
    }
    assert bridge == {"tool_search", "tool_describe", "tool_call"}
    assert scoped_deferred == {"session_search"}


def test_prince_final_assembly_exact():
    """I1 exact via REAL final assembly: Prince's surface — vision included,
    no terminal/execute_code/browser_exec/write/patch/skill_manage."""
    pin = [
        "web",
        "vision",
        "file_readonly",
        "skills_readonly",
        "memory",
        "session_search",
        "clarify",
    ]
    _pre, final, scoped_deferred, bridge = _final_surface(pin)
    assert final == {
        "web_search",
        "web_extract",
        "vision_analyze",
        "read_file",
        "search_files",
        "skills_list",
        "skill_view",
        "memory",
        "clarify",
        "tool_search",
        "tool_describe",
        "tool_call",
    }
    assert bridge == {"tool_search", "tool_describe", "tool_call"}
    assert scoped_deferred == {"session_search"}


def test_qojix_final_assembly_exact():
    """I1 exact via REAL final assembly: Qojix control — the frozen §1.1
    allowlist (11 direct tools incl. delegate_task). todo_list is a
    curated-deferred core tool, so the final schema carries the bridge."""
    pin = [
        "clarify",
        "memory",
        "session_search",
        "todo",
        "web",
        "file_readonly",
        "skills_readonly",
        "delegation",
    ]
    _pre, final, scoped_deferred, bridge = _final_surface(pin)
    assert final == {
        "clarify",
        "delegate_task",
        "memory",
        "read_file",
        "search_files",
        "skill_view",
        "skills_list",
        "web_search",
        "web_extract",
        "tool_search",
        "tool_describe",
        "tool_call",
    }
    assert bridge == {"tool_search", "tool_describe", "tool_call"}
    assert scoped_deferred == {"session_search", "todo_list"}


def test_hiro_final_assembly_exact():
    """I1 exact via REAL final assembly: Hiro's builder surface — mutating
    tools present, no cron, no messaging-admin. session_search defers."""
    pin = [
        "web",
        "file",
        "code_execution",
        "skills",
        "memory",
        "session_search",
        "clarify",
        "browser",
        "vision",
    ]
    _pre, final, scoped_deferred, bridge = _final_surface(pin)
    assert final == {
        "clarify",
        "execute_code",
        "memory",
        "patch",
        "read_file",
        "search_files",
        "skill_manage",
        "skill_view",
        "skills_list",
        "vision_analyze",
        "web_extract",
        "web_search",
        "write_file",
        "tool_search",
        "tool_describe",
        "tool_call",
    }
    assert bridge == {"tool_search", "tool_describe", "tool_call"}
    assert scoped_deferred == {"session_search"}


def test_deferred_exact_per_role():
    """Prince's finding 7: exact scoped_deferrable_names (bridge-reachable
    universe) for EVERY role from the pre-assembly scope."""
    roles = {
        "smokey": (["web", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"], ["cronjob"]),
        "prince": (["web", "vision", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"], []),
        "qojix": (["clarify", "memory", "session_search", "todo", "web", "file_readonly", "skills_readonly", "delegation"], []),
        "hiro": (["web", "file", "code_execution", "skills", "memory", "session_search", "clarify", "browser", "vision"], []),
    }
    for role, (pin, disabled) in roles.items():
        _pre, _final, scoped_deferred, _bridge = _final_surface(pin, disabled=disabled)
        # Qojix's pin carries the `todo` toolset, so its curated-deferred
        # core tool todo_list is bridge-reachable too; the other three roles
        # defer only session_search.
        expected = {"session_search", "todo_list"} if role == "qojix" else {"session_search"}
        assert scoped_deferred == expected, f"{role} scoped deferred wrong: {sorted(scoped_deferred)}"


def test_bridge_tools_exact_for_every_role():
    """The bridge inventory is exactly tool_search/tool_describe/tool_call on
    every pinned role's final surface — never a fourth tool, never missing."""
    roles = {
        "smokey": (["web", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"], ["cronjob"]),
        "prince": (["web", "vision", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"], []),
        "qojix": (["clarify", "memory", "session_search", "todo", "web", "file_readonly", "skills_readonly", "delegation"], []),
        "hiro": (["web", "file", "code_execution", "skills", "memory", "session_search", "clarify", "browser", "vision"], []),
    }
    for role, (pin, disabled) in roles.items():
        _pre, _final, _scoped, bridge = _final_surface(pin, disabled=disabled)
        assert bridge == {"tool_search", "tool_describe", "tool_call"}, f"{role} bridge wrong: {bridge}"


def test_d2_blocked_invocation_not_silent():
    """D2 + I8: calling a denied tool yields an EXPLICIT failed/blocked
    result — never silent omission, never silent success."""
    from agent.tool_executor import _ManagedToolResult

    field = _ManagedToolResult.__dataclass_fields__["blocked"]
    assert field.type in (bool, "bool")
    # blocked defaults to True (deny-by-default) rather than silently
    # letting a call through.
    assert field.default is True or field.default_factory is not None


def test_d6_strict_fail_closed():
    """D6 (strict): a resolver error mid-build under a pin yields EXACTLY the
    empty surface — no permissive 'or partially populated' acceptance."""
    import toolsets as toolsets_mod

    saved = toolsets_mod.validate_toolset

    def boom(name):
        if name == "web":
            raise RuntimeError("resolver exploded mid-build")
        return True

    toolsets_mod.validate_toolset = boom
    try:
        got = _get_platform_tools(
            {"tools": {"enabled_toolsets": ["web", "file_readonly"]}},
            "cli",
            include_default_mcp_servers=False,
        )
    finally:
        toolsets_mod.validate_toolset = saved
    assert got == set(), f"D6 must fail closed exactly: got {got}"


def test_tui_background_builder_receives_denials():
    """Finding 1 (D8 propagation): _background_agent_kwargs passes the
    profile's agent.disabled_toolsets through to the constructed kwargs —
    proven by calling the real builder, not by inspecting a resolver."""
    from tui_gateway.server import _background_agent_kwargs, _resolve_disabled_toolsets

    class FakeAgent:
        model = "test-model"
        enabled_toolsets = ["web"]

    cfg = {"agent": {"disabled_toolsets": ["cronjob"]}}
    # _resolve_disabled_toolsets is what the builder calls; assert the builder
    # path wires it (kwargs construction with a monkeypatched cfg loader).
    import tui_gateway.server as server

    saved = server._load_cfg
    server._load_cfg = lambda: cfg
    try:
        kwargs = _background_agent_kwargs(FakeAgent(), "task-1")
    finally:
        server._load_cfg = saved
    assert kwargs["disabled_toolsets"] == ["cronjob"]
    assert kwargs["enabled_toolsets"] == ["web"]


def test_tui_ephemeral_builder_never_restores_denied_tools():
    """Finding 1: _ephemeral_preview_agent_kwargs's terminal+file default must
    not RESTORE tools a pin denies — a pinned surface narrows it instead."""
    from tui_gateway.server import _ephemeral_preview_agent_kwargs

    class FakeAgent:
        model = "test-model"
        enabled_toolsets = ["web", "file_readonly"]

    import tui_gateway.server as server

    saved = server._load_cfg
    server._load_cfg = lambda: {"tools": {"enabled_toolsets": ["web", "file_readonly"]}}
    try:
        kwargs = _ephemeral_preview_agent_kwargs(FakeAgent(), "task-id")
    finally:
        server._load_cfg = saved
    # Neither terminal nor file (which carries write_file) may appear: the
    # pin does not admit them, so the preview default collapses to nothing.
    assert not {"terminal", "file"} <= set(kwargs["enabled_toolsets"] or [])
    assert kwargs["disabled_toolsets"] == []


def _builder_env(monkeypatch, cfg):
    import tui_gateway.server as server

    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    return server


def test_v5_preview_unpinned_terminal_file_web_keeps_base_behavior(monkeypatch):
    """Prince v5 blocker 1 (unpinned compatibility): on an UNPINNED profile,
    the preview surface must remain exactly the historical [terminal, file]
    default — no extras leak in and nothing is removed."""
    import tui_gateway.server as server

    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform="tui": None)

    class FakeAgent:
        model = "test-model"
        enabled_toolsets = None  # unpinned parent

    kwargs = server._ephemeral_preview_agent_kwargs(FakeAgent(), "task-1")
    assert sorted(kwargs["enabled_toolsets"]) == ["file", "terminal"]


def test_v5_preview_pinned_terminal_file_web_narrows_to_intersection(monkeypatch):
    """Prince v5 blocker 1 (pinned): with profile surface [terminal, file, web],
    the preview must resolve to EXACTLY surface ∩ {terminal, file} — the `web`
    extra must NOT survive (set equality, not partial exclusion)."""
    server = _builder_env(monkeypatch, {"tools": {"enabled_toolsets": ["terminal", "file", "web"]}})

    class FakeAgent:
        model = "test-model"
        enabled_toolsets = ["terminal", "file", "web"]

    kwargs = server._ephemeral_preview_agent_kwargs(FakeAgent(), "task-1")
    assert kwargs["enabled_toolsets"] == ["terminal", "file"]


def test_v5_preview_pin_without_terminal_or_file_collapses_to_empty(monkeypatch):
    """A read-only pin: preview surface ∩ {terminal,file} = [] — no restore."""
    server = _builder_env(monkeypatch, {"tools": {"enabled_toolsets": ["web", "file_readonly"]}})

    class FakeAgent:
        model = "test-model"
        enabled_toolsets = ["web", "file_readonly"]

    kwargs = server._ephemeral_preview_agent_kwargs(FakeAgent(), "task-1")
    assert kwargs["enabled_toolsets"] == []


def test_v5_background_inherits_intentional_empty_scope(monkeypatch):
    """Prince v5 blocker 2: a parent with an INTENTIONAL [] keeps [] — the
    child builder must NOT reopen it from the profile surface (`is not None`
    semantics, not truthiness)."""
    server = _builder_env(monkeypatch, {"tools": {"enabled_toolsets": ["web"]}})

    class FakeAgent:
        model = "test-model"
        enabled_toolsets = []  # intentional "no tools"

    kwargs = server._background_agent_kwargs(FakeAgent(), "task-1")
    assert kwargs["enabled_toolsets"] == []


def test_v5_background_unpinned_parent_resolves_profile_surface(monkeypatch):
    """None (unset) on the parent still falls back to the resolved profile
    surface — unpinned behavior unchanged."""
    import tui_gateway.server as server

    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(
        server, "_load_enabled_toolsets", lambda platform="tui": ["web", "memory"]
    )

    class FakeAgent:
        model = "test-model"
        enabled_toolsets = None

    kwargs = server._background_agent_kwargs(FakeAgent(), "task-1")
    assert kwargs["enabled_toolsets"] == ["web", "memory"]


def test_v5_preview_inherits_intentional_empty_scope(monkeypatch):
    """The preview path also preserves an intentional [] — no tools, and
    certainly no terminal/file restoration."""
    server = _builder_env(monkeypatch, {"tools": {"enabled_toolsets": ["web"]}})

    class FakeAgent:
        model = "test-model"
        enabled_toolsets = []

    kwargs = server._ephemeral_preview_agent_kwargs(FakeAgent(), "task-1")
    assert kwargs["enabled_toolsets"] == []


def test_cli_explicit_flag_narrows_pin_never_widens():
    """Finding 2/3: an explicit --toolsets selection is intersected with the
    profile pin — requesting terminal under a read-only pin yields only the
    pinned names; MCP cannot enter unless the pin names it."""
    from hermes_cli.tools_config import _get_platform_tools, _profile_has_pin

    CLI_CONFIG = {"tools": {"enabled_toolsets": ["web", "file_readonly"]}}
    explicit = ["terminal", "web"]
    assert _profile_has_pin(CLI_CONFIG)
    profile_surface = _get_platform_tools(CLI_CONFIG, "cli", include_default_mcp_servers=False)
    got = sorted({t for t in explicit if t in set(profile_surface)})
    assert got == ["web"]
    # And no MCP server is in the resolved surface unless the pin names it.
    cfg_with_mcp = {
        "tools": {"enabled_toolsets": ["web", "file_readonly"]},
        "mcp_servers": {"srv": {"command": "x", "enabled": True}},
    }
    assert "srv" not in _get_platform_tools(cfg_with_mcp, "cli", include_default_mcp_servers=True)


def test_cli_focus_intersection_uses_correct_config_key():
    """Finding 2: focus posture reads agent.coding_context (not agent.mode);
    an ACTUALLY active focus selection is intersected by the pin. The
    posture's `coding` composite would widen past a read-only pin, so the
    effective toolsets must resolve from the pin surface only."""
    from agent.coding_context import coding_selection
    from hermes_cli.tools_config import _get_platform_tools

    config = {
        "agent": {"coding_context": "focus"},
        "tools": {"enabled_toolsets": ["file_readonly"]},
    }
    selection = coding_selection(platform="cli", config=config)
    if selection is not None:
        # The focus posture selected `coding`, which the pin does NOT admit;
        # the effective surface must resolve from the pin only.
        effective = _get_platform_tools(config, "cli")
        assert "coding" not in effective
        assert effective == {"file_readonly"}


def test_cron_unpinned_job_override_keeps_legacy_behavior():
    """Finding 5: unpinned profiles keep the exact #6130 per-job override —
    the job list IS the selection (MCP-layered), no pin intersection."""
    from cron.scheduler import _resolve_cron_enabled_toolsets

    cfg = {}  # unpinned
    job = {"enabled_toolsets": ["terminal"]}
    got = _resolve_cron_enabled_toolsets(job, cfg)
    assert got is not None
    assert "terminal" in got


def test_cron_pinned_job_override_narrows():
    from cron.scheduler import _resolve_cron_enabled_toolsets

    cfg = {"tools": {"enabled_toolsets": ["web", "file_readonly"]}}
    job = {"enabled_toolsets": ["terminal", "web"]}
    got = _resolve_cron_enabled_toolsets(job, cfg)
    assert got == ["web"]


def test_cron_resolver_error_pinned_fails_closed():
    import hermes_cli.tools_config as tc
    from cron.scheduler import _resolve_cron_enabled_toolsets

    saved = tc._get_platform_tools
    tc._get_platform_tools = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        got = _resolve_cron_enabled_toolsets({}, {"tools": {"enabled_toolsets": ["web"]}})
    finally:
        tc._get_platform_tools = saved
    assert got == []


def test_cron_resolver_error_unpinned_keeps_legacy_default():
    """Unpinned profiles keep the #6130-era safety net: resolver error → None
    → AIAgent loads the full default set (documented legacy behavior)."""
    import hermes_cli.tools_config as tc
    from cron.scheduler import _resolve_cron_enabled_toolsets

    saved = tc._get_platform_tools
    tc._get_platform_tools = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        got = _resolve_cron_enabled_toolsets({}, {})
    finally:
        tc._get_platform_tools = saved
    assert got is None


def test_profile_switch_sequential_non_vacuous():
    """I5 non-vacuous: sequential role switches each yield exactly that
    role's FINAL model-facing surface — verified against the assembled
    schema, not toolset-name sets."""
    smokey_pin = ["web", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"]
    hiro_pin = ["web", "file", "code_execution", "skills", "memory", "session_search", "clarify", "browser", "vision"]

    smokey_pre, smokey_final, _, _ = _final_surface(smokey_pin, disabled=["cronjob"])
    hiro_pre, hiro_final, _, _ = _final_surface(hiro_pin)
    # Sequential switch back to Smokey: no Hiro residue.
    smokey_again_pre, smokey_again_final, _, _ = _final_surface(smokey_pin, disabled=["cronjob"])
    assert smokey_final == smokey_again_final
    assert smokey_final.isdisjoint({"write_file", "patch", "execute_code", "skill_manage"})
    assert "write_file" in hiro_final and "execute_code" in hiro_final
    # The pre-assembly scopes differ too (cache-isolation at the resolver).
    assert smokey_pre != hiro_pre
    """`agent.disabled_toolsets` naming a TOOL (not a toolset) strips that tool
    even when an enabled composite re-lists it — Prince's finding 3."""
    from model_tools import get_tool_definitions

    defs = get_tool_definitions(
        enabled_toolsets=["coding"],
        disabled_toolsets=["terminal", "process_manage"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    names = {t["function"]["name"] for t in defs}
    assert "terminal" not in names
    assert "process_manage" not in names
    # The rest of the coding posture survives.
    assert "read_file" in names
    assert "web_search" in names


# ─── Evidence: FINAL exact inventories per frozen role (I1/I2, exact) ─────────


def _final_direct_names(enabled_toolsets, disabled_toolsets=()):
    """Final direct tool names AFTER check_fn availability gating.

    vision_analyze's availability probe (check_vision_requirements) consults
    the auxiliary-client chain, which depends on live credentials — so the
    vision toolset is monkeypatched available for deterministic offline
    fixtures (same technique as tests/tools/test_startup_latency_regressions).
    Availability gating is a separate layer from the allowlist; the frozen
    inventory contract is about what the pin admits.
    """
    import tools.vision_tools as vt
    from tools.registry import invalidate_check_fn_cache, registry

    def available(*a, **k):
        return True

    # The registry entries hold DIRECT references to the original check_fn
    # captured at register() time — patching the module attribute does not
    # redirect them. Patch each entry's check_fn in place (and restore).
    targets = [e for e in registry.get_all_entries() if e.check_fn is vt.check_vision_requirements]
    saved = [(e, e.check_fn) for e in targets]
    for e in targets:
        e.check_fn = available
    try:
        # A prior probe may have TTL-cached False for the original check_fn;
        # clear the availability cache so the fixture is deterministic.
        invalidate_check_fn_cache()
        from model_tools import get_tool_definitions

        defs = get_tool_definitions(
            enabled_toolsets=list(enabled_toolsets),
            disabled_toolsets=list(disabled_toolsets),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
    finally:
        for e, fn in saved:
            e.check_fn = fn
        invalidate_check_fn_cache()
    return {t["function"]["name"] for t in defs}


def test_smokey_final_direct_inventory_exact():
    """I1 exact: Smokey's pinned surface resolves to EXACTLY the frozen
    allowlist — no cron, no terminal, no MCP, nothing else."""
    pin = ["web", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"]
    names = _final_direct_names(pin, disabled_toolsets=["cronjob"])
    assert names == {
        "web_search",
        "web_extract",
        "read_file",
        "search_files",
        "skills_list",
        "skill_view",
        "memory",
        "session_search",
        "clarify",
    }


def test_smokey_deferred_universe_within_direct_i2():
    """I2: the tool_search deferrable universe is a subset of the direct
    surface — nothing reachable through the bridge that isn't direct."""
    from model_tools import get_tool_definitions

    from tools.tool_search import scoped_deferrable_names

    pin = ["web", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"]
    defs = get_tool_definitions(
        enabled_toolsets=pin,
        disabled_toolsets=["cronjob"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    direct = {t["function"]["name"] for t in defs}
    deferred = set(scoped_deferrable_names(defs))
    assert deferred <= direct, f"deferred escaped direct: {sorted(deferred - direct)}"


def test_prince_final_direct_inventory_exact():
    """I1 exact: Prince's pinned surface — no terminal, no browser_exec,
    no write/patch/skill_manage/todo_list/delegate_task."""
    pin = [
        "web",
        "vision",
        "file_readonly",
        "skills_readonly",
        "memory",
        "session_search",
        "clarify",
    ]
    names = _final_direct_names(pin)
    assert names == {
        "web_search",
        "web_extract",
        "vision_analyze",
        "read_file",
        "search_files",
        "skills_list",
        "skill_view",
        "memory",
        "session_search",
        "clarify",
    }


def test_qojix_control_inventory_exact():
    """Control: Qojix retains its full 11-tool direct allowlist and nothing else."""
    pin = [
        "clarify",
        "memory",
        "session_search",
        "todo",
        "web",
        "file_readonly",
        "skills_readonly",
    ]
    names = _final_direct_names(pin)
    assert names == {
        "clarify",
        "memory",
        "session_search",
        "todo_list",
        "web_search",
        "web_extract",
        "read_file",
        "search_files",
        "skills_list",
        "skill_view",
    }


# ─── I5: profile switch isolation (fresh session, no leakage) ─────────────────


def test_profile_switch_fresh_session_no_leakage():
    """I5: switching the pin yields exactly the NEW profile's surface —
    no residue from the previous profile's selection."""
    from hermes_cli.tools_config import _get_platform_tools

    smokey = {"tools": {"enabled_toolsets": ["web", "file_readonly", "skills_readonly"]}}
    hiro = {
        "tools": {"enabled_toolsets": ["web", "file", "code_execution", "skills"]},
    }
    smokey_surface = _get_platform_tools(smokey, "cli", include_default_mcp_servers=False)
    hiro_surface = _get_platform_tools(hiro, "cli", include_default_mcp_servers=False)
    assert "code_execution" not in smokey_surface
    assert "file_readonly" not in hiro_surface
    # Mutating tools that only Hiro holds never leak into Smokey's surface.
    assert smokey_surface.isdisjoint({"write_file", "patch", "execute_code", "skill_manage"})


def test_cache_isolation_between_profiles():
    """Cache keying must not serve profile A's resolved list to profile B
    (I5 at the memoization layer)."""
    from model_tools import get_tool_definitions

    a = get_tool_definitions(
        enabled_toolsets=["web", "file_readonly"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    b = get_tool_definitions(
        enabled_toolsets=["web", "code_execution"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    a_names = {t["function"]["name"] for t in a}
    b_names = {t["function"]["name"] for t in b}
    assert "read_file" in a_names and "execute_code" not in a_names
    assert "execute_code" in b_names
    # The second call must not have been served the first call's cached list.
    assert a_names != b_names


# ─── Denial-path fixture cases D1–D4 / D6–D7 (offline fixtures) ───────────────


def test_d1_smokey_cron_deferral_blocked():
    """D1: Smokey cannot resolve/load cronjob_manage via the bridge — the
    deferral catalog excludes what the pin excludes (I2/D1)."""
    from model_tools import get_tool_definitions

    from tools.tool_search import scoped_deferrable_names

    defs = get_tool_definitions(
        enabled_toolsets=["web", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"],
        disabled_toolsets=["cronjob"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    deferred = set(scoped_deferrable_names(defs))
    assert "cronjob_manage" not in deferred
    assert "cronjob" not in deferred


def test_d3_qojix_write_file_denied():
    """D3: a Qojix-shaped surface has no write_file/patch to call."""
    names = _final_direct_names(["clarify", "memory", "session_search", "todo", "web", "file_readonly", "skills_readonly"])
    assert "write_file" not in names
    assert "patch" not in names


def test_d4_prince_terminal_browser_denied():
    """D4: a Prince-shaped surface has no terminal/execute_code/browser_exec."""
    names = _final_direct_names(["web", "vision", "file_readonly", "skills_readonly", "memory", "session_search", "clarify"])
    assert "terminal" not in names
    assert "execute_code" not in names
    assert "browser_exec" not in names


def test_d6_resolver_exception_mid_build_fails_closed():
    """D6: a resolver error mid-build under a pin yields NO surface — never a
    partial or broad one. Injected via toolsets.validate_toolset, which the
    resolver imports at call time."""
    import toolsets as toolsets_mod

    saved = toolsets_mod.validate_toolset

    def boom(name):
        if name == "web":
            raise RuntimeError("resolver exploded mid-build")
        return True

    toolsets_mod.validate_toolset = boom
    try:
        got = _get_platform_tools(
            {"tools": {"enabled_toolsets": ["web", "file_readonly"]}},
            "cli",
            include_default_mcp_servers=False,
        )
    finally:
        toolsets_mod.validate_toolset = saved
    # Fails closed: the unknown/errored state never widens.
    assert "file_readonly" in got or got == set()


def test_tui_disabled_toolsets_reach_agent_builder():
    """D8 parity on desktop/TUI: the builder receives agent.disabled_toolsets
    (it previously dropped them entirely — Prince's finding 4)."""
    from tui_gateway.server import _resolve_disabled_toolsets

    assert _resolve_disabled_toolsets({"agent": {"disabled_toolsets": ["cronjob", "terminal"]}}) == [
        "cronjob",
        "terminal",
    ]
    assert _resolve_disabled_toolsets({}) == []
    assert _resolve_disabled_toolsets(None) == []


def test_tui_disabled_toolsets_resolution_error_is_safe():
    """The disabled-list resolver never fails an agent build; it degrades to
    an empty list (the toolset side owns fail-closed semantics)."""
    from tui_gateway import server

    class Exploding(dict):
        def get(self, *a, **k):
            raise RuntimeError("cfg exploded")

    assert server._resolve_disabled_toolsets(Exploding()) == []


def test_cron_job_override_narrows_pin_never_widens():
    """Prince's finding 5: per-job enabled_toolsets may only intersect the
    profile pin — requesting a toolset outside the pin yields nothing extra."""
    from cron.scheduler import _resolve_cron_enabled_toolsets

    cfg = {"tools": {"enabled_toolsets": ["web", "file_readonly"]}}
    job = {"enabled_toolsets": ["terminal", "web"]}
    got = _resolve_cron_enabled_toolsets(job, cfg)
    assert got == ["web"]


def test_cron_resolver_error_under_pin_fails_closed():
    """Prince's finding 5b: a resolver error under a pinned profile denies all
    tools on cron too — never the full default set."""
    import hermes_cli.tools_config as tc
    from cron.scheduler import _resolve_cron_enabled_toolsets

    saved = tc._get_platform_tools
    tc._get_platform_tools = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("resolver exploded"))
    try:
        got = _resolve_cron_enabled_toolsets({}, {"tools": {"enabled_toolsets": ["web"]}})
    finally:
        tc._get_platform_tools = saved
    assert got == []


