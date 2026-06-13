"""QuestFrame PCVR Toolkit bridge for Hermes."""

from __future__ import annotations

from . import core
from .cli import questframe_command, register_cli

_TOOLS = (
    ("questframe_status", core.STATUS_SCHEMA, core.handle_status, "Q"),
    ("questframe_setup", core.SETUP_SCHEMA, core.handle_setup, "Q"),
    ("questframe_fh6vr_preflight", core.PREFLIGHT_SCHEMA, core.handle_preflight, "Q"),
    (
        "questframe_rtx3060_profiles",
        core.RTX3060_PROFILES_SCHEMA,
        core.handle_rtx3060_profiles,
        "Q",
    ),
    (
        "questframe_rtx3060_selftest",
        core.RTX3060_SELFTEST_SCHEMA,
        core.handle_rtx3060_selftest,
        "Q",
    ),
    (
        "questframe_session_readiness",
        core.SESSION_READINESS_SCHEMA,
        core.handle_session_readiness,
        "Q",
    ),
    (
        "questframe_graphics_session",
        core.GRAPHICS_SESSION_SCHEMA,
        core.handle_graphics_session,
        "Q",
    ),
    ("questframe_frame_loop", core.FRAME_LOOP_SCHEMA, core.handle_frame_loop, "Q"),
    (
        "questframe_dibr_swapchain",
        core.DIBR_SWAPCHAIN_SCHEMA,
        core.handle_dibr_swapchain,
        "Q",
    ),
    (
        "questframe_fh6_capture_preflight",
        core.FH6_CAPTURE_PREFLIGHT_SCHEMA,
        core.handle_fh6_capture_preflight,
        "Q",
    ),
    (
        "questframe_live_capture_selftest",
        core.LIVE_CAPTURE_SELFTEST_SCHEMA,
        core.handle_live_capture_selftest,
        "Q",
    ),
    (
        "questframe_depth_surface_selftest",
        core.DEPTH_SURFACE_SELFTEST_SCHEMA,
        core.handle_depth_surface_selftest,
        "Q",
    ),
    (
        "questframe_depth_reader_selftest",
        core.DEPTH_READER_SELFTEST_SCHEMA,
        core.handle_depth_reader_selftest,
        "Q",
    ),
    (
        "questframe_depth_producer_selftest",
        core.DEPTH_PRODUCER_SELFTEST_SCHEMA,
        core.handle_depth_producer_selftest,
        "Q",
    ),
    (
        "questframe_companion_depth_producer_selftest",
        core.COMPANION_DEPTH_PRODUCER_SELFTEST_SCHEMA,
        core.handle_companion_depth_producer_selftest,
        "Q",
    ),
    (
        "questframe_color_depth_pairing_selftest",
        core.COLOR_DEPTH_PAIRING_SELFTEST_SCHEMA,
        core.handle_color_depth_pairing_selftest,
        "Q",
    ),
    (
        "questframe_openxr_presentation_selftest",
        core.OPENXR_PRESENTATION_SELFTEST_SCHEMA,
        core.handle_openxr_presentation_selftest,
        "Q",
    ),
    (
        "questframe_immersive_presentation_loop_selftest",
        core.IMMERSIVE_PRESENTATION_LOOP_SELFTEST_SCHEMA,
        core.handle_immersive_presentation_loop_selftest,
        "Q",
    ),
    (
        "questframe_cockpit_presence_selftest",
        core.COCKPIT_PRESENCE_SELFTEST_SCHEMA,
        core.handle_cockpit_presence_selftest,
        "Q",
    ),
    (
        "questframe_kofi_parity_selftest",
        core.KOFI_PARITY_SELFTEST_SCHEMA,
        core.handle_kofi_parity_selftest,
        "Q",
    ),
    (
        "questframe_pcvr_management_selftest",
        core.PCVR_MANAGEMENT_SELFTEST_SCHEMA,
        core.handle_pcvr_management_selftest,
        "Q",
    ),
    (
        "questframe_hermes_bridge_selftest",
        core.HERMES_BRIDGE_SELFTEST_SCHEMA,
        core.handle_hermes_bridge_selftest,
        "Q",
    ),
    (
        "questframe_hmd_controller_input_selftest",
        core.HMD_CONTROLLER_INPUT_SELFTEST_SCHEMA,
        core.handle_hmd_controller_input_selftest,
        "Q",
    ),
    (
        "questframe_vcc_health",
        core.VCC_HEALTH_SCHEMA,
        core.handle_vcc_health,
        "Q",
    ),
    (
        "questframe_support_report",
        core.SUPPORT_REPORT_SCHEMA,
        core.handle_support_report,
        "Q",
    ),
    ("questframe_unity_scan", core.UNITY_SCAN_SCHEMA, core.handle_unity_scan, "Q"),
)


def register(ctx) -> None:
    """Register QuestFrame tools, slash command, and CLI command."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="questframe",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            description=schema.get("description", ""),
            emoji=emoji,
        )

    ctx.register_command(
        "questframe",
        handler=core.handle_slash,
        description="Run QuestFrame PCVR and FH6VR bridge diagnostics.",
        args_hint=(
            "[status|preflight|profiles|rtx3060-selftest|session|"
            "graphics-session|frame-loop|dibr-swapchain|capture-preflight|"
            "live-capture-selftest|depth-surface-selftest|depth-reader-selftest|"
            "depth-producer-selftest|companion-depth-producer-selftest|"
            "color-depth-pairing-selftest|openxr-presentation-selftest|"
            "pcvr-management-selftest|hermes-bridge-selftest|"
            "hmd-controller-input-selftest|support-report|unity-scan]"
        ),
    )
    ctx.register_cli_command(
        name="questframe",
        help="QuestFrame PCVR Toolkit bridge",
        setup_fn=register_cli,
        handler_fn=questframe_command,
        description=(
            "Inspect the QuestFrame Hermes bridge, call the FH6VR C# launcher, "
            "and scan Unity/VCC projects for VRChat package risk."
        ),
    )
