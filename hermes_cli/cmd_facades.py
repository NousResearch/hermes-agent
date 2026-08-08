"""Thin command facades for the hermes CLI (extracted from hermes_cli/main.py).

Each facade is a lazy-importing delegation handler wired into the parser via
``build_*_parser(subparsers, cmd_X=cmd_X)`` inside ``main()``. Extracted from
the main.py god file (epic #78647, target #78631) — slice R5-S1, window
10956-11154 at golden pin b3e45a3d (byte-identical content at execution pin).
Zero behavior change: bodies moved verbatim; ``_require_tty`` is resolved via
in-method lazy import from hermes_cli.main to keep the 6 test patch sites on
``hermes_cli.main._require_tty`` working.
"""


import sys


def cmd_memory(args):
    sub = getattr(args, "memory_command", None)
    if sub == "off":
        from hermes_cli.config import load_config, save_config

        config = load_config()
        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = ""
        save_config(config)
        print("\n  ✓ Memory provider: built-in only")
        print("  Saved to config.yaml\n")
    elif sub == "reset":
        from hermes_constants import get_hermes_home, display_hermes_home

        mem_dir = get_hermes_home() / "memories"
        target = getattr(args, "target", "all")
        files_to_reset = []
        if target in {"all", "memory"}:
            files_to_reset.append(("MEMORY.md", "agent notes"))
        if target in {"all", "user"}:
            files_to_reset.append(("USER.md", "user profile"))

        # Check what exists
        existing = [
            (f, desc) for f, desc in files_to_reset if (mem_dir / f).exists()
        ]
        if not existing:
            print(
                f"\n  Nothing to reset — no memory files found in {display_hermes_home()}/memories/\n"
            )
            return

        print("\n  This will permanently erase the following memory files:")
        for f, desc in existing:
            path = mem_dir / f
            size = path.stat().st_size
            print(f"    ◆ {f} ({desc}) — {size:,} bytes")

        if not getattr(args, "yes", False):
            try:
                answer = input("\n  Type 'yes' to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.\n")
                return
            if answer != "yes":
                print("  Cancelled.\n")
                return

        for f, desc in existing:
            (mem_dir / f).unlink()
            print(f"  ✓ Deleted {f} ({desc})")

        print(
            "\n  Memory reset complete. New sessions will start with a blank slate."
        )
        print(f"  Files were in: {display_hermes_home()}/memories/\n")
    else:
        from hermes_cli.memory_setup import memory_command

        memory_command(args)


def cmd_acp(args):
    """Launch Hermes Agent as an ACP server."""
    try:
        from acp_adapter.entry import main as acp_main

        acp_argv = []
        if getattr(args, "acp_version", False):
            acp_argv.append("--version")
        if getattr(args, "check", False):
            acp_argv.append("--check")
        if getattr(args, "setup", False):
            acp_argv.append("--setup")
        if getattr(args, "setup_browser", False):
            acp_argv.append("--setup-browser")
        if getattr(args, "assume_yes", False):
            acp_argv.append("--yes")
        acp_main(acp_argv)
    except ImportError:
        print("ACP dependencies not installed.", file=sys.stderr)
        print("Install them with:  pip install -e '.[acp]'", file=sys.stderr)
        sys.exit(1)


def cmd_tools(args):
    action = getattr(args, "tools_action", None)
    if action in {"list", "disable", "enable"}:
        from hermes_cli.tools_config import tools_disable_enable_command

        tools_disable_enable_command(args)
    elif action == "post-setup":
        from hermes_cli.tools_config import run_post_setup_command

        sys.exit(run_post_setup_command(args))
    else:
        from hermes_cli.main import _require_tty
        _require_tty("tools")
        from hermes_cli.tools_config import tools_command

        tools_command(args)


def cmd_insights(args):
    try:
        from hermes_state import SessionDB
        from agent.insights import InsightsEngine

        db = SessionDB()
        engine = InsightsEngine(db)
        report = engine.generate(days=args.days, source=args.source)
        print(engine.format_terminal(report))
        db.close()
    except Exception as e:
        print(f"Error generating insights: {e}")


def cmd_monitoring(args):
    """Gateway monitoring status: health & diagnostics export posture."""
    from hermes_cli.config import load_config

    action = getattr(args, "monitoring_action", None) or "status"
    config = load_config()
    mon_raw = config.get("monitoring")
    mon: dict = mon_raw if isinstance(mon_raw, dict) else {}

    if action == "status":
        from agent.monitoring import otlp_exporter

        gh_raw = mon.get("gateway_health_export")
        gh: dict = gh_raw if isinstance(gh_raw, dict) else {}
        export_raw = mon.get("export")
        export_cfg: dict = export_raw if isinstance(export_raw, dict) else {}
        otlp_raw = export_cfg.get("otlp")
        otlp: dict = otlp_raw if isinstance(otlp_raw, dict) else {}

        print("Gateway monitoring")
        print(f"  Health export:  {'enabled' if gh.get('enabled') else 'disabled'} "
              f"(monitoring.gateway_health_export.enabled)")
        if gh.get("enabled"):
            print(f"    Metrics:            {'on' if gh.get('metrics_enabled', True) else 'off'} "
                  f"(interval {gh.get('export_interval_seconds', 60)}s)")
            print(f"    Diagnostic events:  {'on' if gh.get('diagnostic_events_enabled', True) else 'off'}")
            print(f"    Warning/error logs: {'on' if gh.get('warning_error_events_enabled', True) else 'off'} "
                  f"(interval {gh.get('logs_export_interval_seconds', 5)}s)")
            print("    Content safety:     always on "
                  "(rendered messages are never exported; not configurable)")
        endpoint = otlp.get("endpoint") or ""
        if otlp.get("enabled") and endpoint:
            print(f"  OTLP endpoint:  {endpoint}")
        else:
            print("  OTLP endpoint:  not configured (monitoring.export.otlp)")
        print(f"  OTel SDK:       {'installed' if otlp_exporter.is_available() else 'not installed'} "
              f"(optional extra: hermes-agent[otlp])")
        print("\n  Scope: gateway service health + redacted diagnostics only.")
        print("  No prompts, messages, tool args/results, usage analytics, or traces.")
        return

    print(f"Unknown monitoring action: {action}", file=sys.stderr)
    sys.exit(2)


def cmd_skills(args):
    # Route 'config' action to skills_config module
    if getattr(args, "skills_action", None) == "config":
        from hermes_cli.main import _require_tty
        _require_tty("skills config")
        from hermes_cli.skills_config import skills_command as skills_config_command

        skills_config_command(args)
    else:
        from hermes_cli.skills_hub import skills_command

        skills_command(args)


def cmd_pairing(args):
    from hermes_cli.pairing import pairing_command

    pairing_command(args)


def cmd_plugins(args):
    from hermes_cli.plugins_cmd import plugins_command

    plugins_command(args)


def cmd_mcp(args):
    from hermes_cli.mcp_config import mcp_command

    mcp_command(args)


def cmd_claw(args):
    from hermes_cli.claw import claw_command

    claw_command(args)
