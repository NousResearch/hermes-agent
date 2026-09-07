"""Streamlined setup flows: the Nous Portal one-shot (`hermes portal`), first-time quick setup,
Blank Slate setup and the `--quick` missing-items pass. Names from setup.py are imported lazily
per function so test patches on ``hermes_cli.setup`` take effect."""

import contextlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("hermes_cli.setup")

# (env-var name substring, platform label, emoji) — order matters: first match wins.
_MESSAGING_PLATFORMS = (("TELEGRAM", "Telegram", "📱"), ("DISCORD", "Discord", "💬"), ("SLACK", "Slack", "💼"))



def _blank_slate_done(config: dict, hermes_home, tools_line: str, *extra: str, intro: str | None = None) -> None:
    """Shared Blank Slate epilogue: success banner, the "enable later" hints, then the summary."""
    from hermes_cli.setup import _info, _print_setup_summary, print_success
    print()
    print_success("Blank Slate setup complete — minimal agent ready.")
    _info(*([intro] if intro else []), tools_line, "  Seed skills:         hermes skills opt-in --sync",
          "  Add MCP servers:     hermes mcp add", *extra, "  Tune agent settings: hermes setup agent", None)
    _print_setup_summary(config, hermes_home)


def _reload_config_into(config: dict, *, dict_only: bool = False) -> None:
    """Re-sync the in-memory config dict from disk after a sub-flow that saved via its own
    load/save cycle, so a later save_config(config) can't clobber it."""
    from hermes_cli.setup import load_config
    refreshed = load_config()
    if not dict_only or isinstance(refreshed, dict):
        config.clear()
        config.update(refreshed)


def _run_nous_flow(config: dict, *, context: str, cancel_exc: tuple, cancel_lines: tuple, print_error) -> bool:
    """Run ``_model_flow_nous`` (login, model pick, provider switch, Tool Gateway opt-in) — the
    single source of truth shared with ``hermes model``. False when cancelled or failed (the
    message is already printed)."""
    from hermes_cli.setup import _info
    try:
        from hermes_cli.model_setup_flows import _model_flow_nous
        _model_flow_nous(config)
        return True
    except cancel_exc:
        # _login_nous raises SystemExit(130)/(1) on cancel/failure; the expired-session re-login
        # path inside _model_flow_nous only catches Exception, so SystemExit would kill the CLI.
        _info(*cancel_lines)
    except Exception as exc:
        logger.debug("_model_flow_nous error during %s: %s", context, exc)
        print_error(exc)
    return False


def _run_portal_one_shot(config: dict) -> None:
    """One-shot Nous Portal setup (``hermes setup --portal`` / ``hermes portal``)."""
    from hermes_cli.setup import _info, _print_banner, print_error, print_info, print_success
    _print_banner("│     ⚕ Hermes Setup — Nous Portal (one-shot)             │")
    _info(None, "  One subscription, 300+ models, plus the Tool Gateway:",
          "    web search, image generation, TTS, browser automation",
          "    — all routed through your Nous Portal sub.", None,
          "  Sign up: https://portal.nousresearch.com/manage-subscription", None)

    def _on_error(exc: Exception) -> None:
        print()
        print_error(f"  Nous Portal setup encountered an error: {exc}")
        print_info("  You can retry later with `hermes portal`.")

    if not _run_nous_flow(config, context="`hermes portal`", cancel_exc=(KeyboardInterrupt, EOFError, SystemExit),
                          cancel_lines=(None, "  Setup cancelled.", "  You can retry later with `hermes portal`."),
                          print_error=_on_error):
        return

    # Re-sync from disk so a caller's later save_config(config) can't clobber the login save.
    with contextlib.suppress(Exception):
        _reload_config_into(config, dict_only=True)
    print()
    print_success("Portal setup complete.")
    _info("  Run `hermes portal info` to inspect routing.", "  Run `hermes` to start chatting.")


def _run_first_time_quick_setup(config: dict, hermes_home, is_existing: bool):
    """Streamlined first-time setup via Nous Portal: OAuth, model, terminal & messaging;
    everything else gets defaults."""
    from hermes_cli.setup import (
        _apply_default_agent_settings, _info, print_header, print_info, _print_setup_summary, print_success,
        print_warning, prompt_choice, save_config, setup_gateway, setup_terminal_backend
    )
    # Step 1: Nous Portal — OAuth login + model selection (provider set to "nous" by the save).
    print_header("Nous Portal", gap=True)
    _info("One subscription, 300+ models, plus the Tool Gateway:",
          "  web search, image generation, TTS, browser automation.",
          "Sign up: https://portal.nousresearch.com/manage-subscription", None)

    def _on_error(exc: Exception) -> None:
        print_warning(f"Nous Portal setup encountered an error: {exc}")
        print_info("You can try again later with: hermes model")

    _run_nous_flow(config, context="quick setup", cancel_exc=(KeyboardInterrupt, EOFError),
                   cancel_lines=(None, "Nous Portal setup cancelled."), print_error=_on_error)
    # The wizard's later save_config(config) must not clobber the login/model save.
    _reload_config_into(config)

    # Step 2: Terminal Backend; Step 3: defaults for everything else.
    setup_terminal_backend(config)
    _apply_default_agent_settings(config)
    save_config(config)

    # Step 4: Offer messaging gateway setup
    print()
    gateway_choice = prompt_choice("Connect a messaging platform? (Telegram, Discord, etc.)", [
        "Set up messaging now (recommended)", "Skip — set up later with 'hermes setup gateway'",
    ], 0)
    if gateway_choice == 0:
        setup_gateway(config)
        save_config(config)
    else:
        # Messaging skipped — still install/start the gateway service so cron jobs run and
        # platforms come alive as soon as tokens are added later (e.g. via `hermes import`).
        from hermes_cli.gateway import ensure_gateway_service
        ensure_gateway_service(context="setup")
    print()
    print_success("Setup complete! You're ready to go.")
    _info(None, "  Configure all settings:    hermes setup")
    if gateway_choice != 0:
        print_info("  Connect Telegram/Discord:  hermes setup gateway")
    _print_macos_fda_tip()
    print()
    _print_setup_summary(config, hermes_home)


def _print_macos_fda_tip() -> None:
    """One-time macOS tip: one Full Disk Access grant kills every per-folder prompt. Same
    prompt-free probe as doctor's check_macos_full_disk_access (the FDA-gated TCC dir never
    triggers a dialog); silent on non-macOS and when FDA is granted or indeterminate.

    every per-folder permission prompt, permanently (issue #52010 follow-up).
    """
    from hermes_cli.setup import _info
    if sys.platform != "darwin":
        return
    try:
        os.listdir(Path.home() / "Library" / "Application Support" / "com.apple.TCC")
        return  # already granted — nothing to teach
    except PermissionError:
        pass
    except OSError:
        return  # indeterminate — don't nag
    _info(None, "  macOS tip: silence ALL folder permission prompts with one switch —",
          "  System Settings → Privacy & Security → Full Disk Access → enable",
          "  your terminal (and Hermes.app if you use Desktop), or run:",
          "    open \"x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles\"",
          "  The grant is permanent — it survives every Hermes update.")


def _blank_slate_minimal_toolsets(config: dict):
    """Write the minimal toolset state for a Blank Slate install: only ``file``, ``terminal``,
    ``vision`` (``read_file`` can't read images) and ``skills`` (the seeded ``hermes-agent`` skill
    needs ``skill_view``) stay on. Two layers enforce it: ``platform_toolsets["cli"]`` (explicit,
    so defaults aren't re-expanded) and ``agent.disabled_toolsets`` (hard-suppression applied last
    in ``_get_platform_tools``, overriding the recovery that would re-add e.g. ``kanban``)."""
    keep = {"file", "terminal", "vision", "skills"}
    config.setdefault("platform_toolsets", {})["cli"] = sorted(keep)
    try:
        from toolsets import TOOLSETS
        from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_plugin_toolset_keys
        all_keys = {k for k, _, _ in CONFIGURABLE_TOOLSETS}
        all_keys.update(_get_plugin_toolset_keys())
        # Plain TOOLSETS entries catch recovered toolsets like ``kanban``. Skip "hermes-*" platform
        # composites, "includes" groupings, and posture toolsets (session-level picks by
        # agent/coding_context.py — disabling them would subtract terminal/read_file).
        for k, tdef in TOOLSETS.items():
            if k.startswith("hermes-") or (isinstance(tdef, dict) and (tdef.get("includes") or tdef.get("posture"))):
                continue
            # selections made by agent/coding_context.py — not permanent user-facing disables. Adding them
            # here causes model_tools to subtract their tools (terminal, read_file, …) from the minimal
            # Blank Slate surface (#57315).
            all_keys.add(k)
        disabled = sorted(all_keys - keep)
        if disabled:
            config.setdefault("agent", {})["disabled_toolsets"] = disabled
    except Exception as exc:
        logger.debug("blank-slate disabled_toolsets computation skipped: %s", exc)


def _blank_slate_minimize_config(config: dict):
    """Turn OFF every optional config feature; all opt back in via ``hermes setup agent``."""
    config.setdefault("agent", {})["max_turns"] = 90
    config.setdefault("compression", {})["enabled"] = False
    mem = config.setdefault("memory", {})
    mem["memory_enabled"] = False
    mem["user_profile_enabled"] = False
    config.setdefault("checkpoints", {})["enabled"] = False
    config.setdefault("smart_model_routing", {})["enabled"] = False
    config.setdefault("session_reset", {})["mode"] = "none"
    config.setdefault("display", {})["tool_progress"] = "all"


def _set_bundled_skills_opt_out(opt_out: bool, log_label: str, on_success=None, on_error=None) -> None:
    """Record the bundled-skills opt-out marker and sync (essential skills are always seeded);
    ``on_success(sync_result)`` / ``on_error(exc)`` report the outcome."""
    try:
        from tools.skills_sync import sync_skills
        from tools.skills_sync_bundled_ops import set_bundled_skills_opt_out
        set_bundled_skills_opt_out(opt_out)
        result = sync_skills(quiet=True)
        if on_success is not None:
            on_success(result)
    except Exception as exc:
        logger.debug("blank-slate %s error: %s", log_label, exc)
        if on_error is not None:
            on_error(exc)


def _run_blank_slate_setup(config: dict, hermes_home, is_existing: bool):
    """Blank Slate setup — essentials only, everything else OFF; then finish now or walk through
    opting capabilities back in. Nothing is enabled that the user did not explicitly choose."""
    from hermes_cli.setup import (
        _info, print_header, print_info, print_success, prompt_choice, save_config, setup_model_provider,
        setup_terminal_backend
    )
    print_header("Blank Slate Setup", gap=True)
    _info("Everything starts OFF. First we force-enable only what's required",
          "to run an agent, then you choose whether to stop there or walk",
          "through enabling more — opting in to exactly what you want.", "",
          "Forced on: Provider & Model, File Operations, Terminal, Vision, Skills.",
          "Everything else (web, browser, code exec, memory,",
          "delegation, cron, plugins, MCP, …) starts disabled. The",
          "essential `hermes-agent` skill is always kept so the agent",
          "can help you drive and configure Hermes itself.", None)

    # Step 1: Provider & Model (REQUIRED — the agent cannot run without it)
    print_header("Step 1 — Provider & Model (required)")
    setup_model_provider(config)
    save_config(config)

    # Step 2: Terminal backend (where commands run — a core decision)
    print_header("Step 2 — Terminal Backend")
    setup_terminal_backend(config)

    # Step 3: Lock in the minimal toolset + minimized config knobs
    _blank_slate_minimal_toolsets(config)
    _blank_slate_minimize_config(config)
    save_config(config)
    print()
    print_success("Minimal baseline applied:")
    print_info("  Toolsets: file, terminal, vision, skills (everything else off)")
    print_info("  Compression, memory, checkpoints, smart routing: off")

    # The fork: stop here, or walk through enabling things
    print_header("How far do you want to go?", gap=True)
    path = prompt_choice("Your minimal agent is ready. What next?", [
        "Start with everything disabled — finish now (most minimal)",
        "Walk through all configurations — opt in to tools, skills, plugins, MCP",
    ], 0)
    if path != 0:
        _blank_slate_walkthrough(config, hermes_home)
        return
    save_config(config)
    # Blank Slate means no bundled skills; record the opt-out so future `hermes update` runs
    # don't re-inject them.
    _set_bundled_skills_opt_out(True, "skill opt-out")
    _blank_slate_done(config, hermes_home, "  Enable tools:        hermes tools", "  Enable plugins:      hermes plugins",
                      intro="Enable anything later, on demand:")


def _blank_slate_walkthrough(config: dict, hermes_home):
    """Opt-in walkthrough for Blank Slate: skills, tools, plugins, MCP, gateway."""
    from hermes_cli.setup import (
        _info, print_header, print_info, print_success, print_warning, prompt_yes_no, save_config, setup_gateway,
    )
    # Bundled skills — default to NONE, offer to seed all
    print_header("Bundled Skills", gap=True)
    print_info("Blank Slate ships with NO bundled skills by default.")
    seed_skills = prompt_yes_no("Seed the full bundled skill catalog? (No = start with zero skills)", default=False)

    def _seeded(result) -> None:
        copied = len(result.get("copied", [])) if isinstance(result, dict) else 0
        print_success(f"Seeded {copied} bundled skills.")

    def _opted_out(_result) -> None:
        _info("No skills seeded (except the essential `hermes-agent`",
              "skill). A .no-bundled-skills marker keeps future",
              "`hermes update` runs from re-injecting them. Opt back in any",
              "time with `hermes skills opt-in --sync`.")

    # Seeding first clears any stale opt-out marker; declining sets it (essential skills still seed).
    _set_bundled_skills_opt_out(
        not seed_skills, "skill handling", on_success=_seeded if seed_skills else _opted_out,
        on_error=lambda exc: print_warning(f"Skill setup step encountered an error: {exc}"),
    )

    # Walk through enabling additional tools
    print_header("Tools", gap=True)
    _info("Pick exactly which additional toolsets to turn on.",
          "(file and terminal are already on; leave the rest off if you want", " the most minimal agent.)")
    if prompt_yes_no("Open the tool selector to enable more tools?", default=False):
        try:
            from hermes_cli.tools_config import tools_command
            tools_command(first_install=False, config=config)
            _reload_config_into(config)  # tools_command saves via its own load/save cycle
        except Exception as exc:
            logger.debug("blank-slate tools_command error: %s", exc)
            print_warning(f"Tool selector encountered an error: {exc}")
    else:
        print_info("Keeping the minimal toolset. Add tools later with `hermes tools`.")

    # Built-in plugins and MCP servers (off unless chosen)
    for header, question, yes_msg, no_msg in (
        ("Plugins", "Review and enable built-in plugins now?",
         "Manage plugins with `hermes plugins list` / `hermes plugins install`.",
         "No plugins enabled. Add later with `hermes plugins`."),
        ("MCP Servers", "Add an MCP server now?",
         "Add servers with `hermes mcp add <name> --url ... | --command ...`.",
         "No MCP servers configured. Add later with `hermes mcp add`."),
    ):
        print_header(header, gap=True)
        print_info(yes_msg if prompt_yes_no(question, default=False) else no_msg)

    # Optional messaging gateway
    print()
    if prompt_yes_no("Connect a messaging platform (Telegram, Discord, …)?", default=False):
        setup_gateway(config)
    save_config(config)
    _blank_slate_done(config, hermes_home, "  Enable more tools:   hermes tools")


def _run_quick_setup(config: dict, hermes_home):
    """Quick setup — only configure items that are missing."""
    from hermes_cli.setup import (
        color, Colors, _info, print_header, print_info, _print_setup_summary, print_success,
        _prompt_and_save_env_var, _prompt_api_key, _section_rule, prompt_checklist, save_config,
    )
    from hermes_cli.config import (get_missing_env_vars, get_missing_config_fields, check_config_version)
    print_header("Quick Setup — Missing Items Only", gap=True)

    # Check what's missing
    missing_env = get_missing_env_vars(required_only=False)
    missing_required = [v for v in missing_env if v.get("is_required")]
    missing_optional = [v for v in missing_env if not v.get("is_required")]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()
    if not (missing_required or missing_optional or missing_config or current_ver < latest_ver):
        print_success("Everything is configured! Nothing to do.")
        _info(None, "Run 'hermes setup' and choose 'Full Setup' to reconfigure,",
              "or pick a specific section from the menu.")
        return
    if missing_required:
        _info(None, f"{len(missing_required)} required setting(s) missing:")
        for var in missing_required:
            print(f"     • {var['name']}")
        print()
        for var in missing_required:
            print()
            print(color(f"  {var['name']}", Colors.CYAN))
            print_info(f"  {var.get('description', '')}")
            if var.get("url"):
                print_info(f"  Get key at: {var['url']}")
            _prompt_and_save_env_var(var, f"  Saved {var['name']}", f"  Skipped {var['name']}")
    missing_tools = [v for v in missing_optional if v.get("category") == "tool"]
    missing_messaging = [v for v in missing_optional if v.get("category") == "messaging" and not v.get("advanced")]
    if missing_tools:  # checklist, then the API-key screen for each pick
        print_header("Tool API Keys", gap=True)
        labels = [var.get("description", var["name"]) + (f" → {', '.join(var['tools'][:2])}" if var.get("tools") else "")
                  for var in missing_tools]
        for idx in prompt_checklist("Which tools would you like to configure?", labels):
            _prompt_api_key(missing_tools[idx])
    if missing_messaging:  # checklist, then prompt for each selected platform's vars
        print_header("Messaging Platforms", gap=True)
        _info("Connect Hermes to messaging apps to chat from anywhere.",
              "You can configure these later with 'hermes setup gateway'.")
        # Group by platform in first-seen order; vars matching no platform are dropped.
        grouped: dict[str, list] = {}
        emojis = {}
        for var in missing_messaging:
            match = next(((plat, emoji) for needle, plat, emoji in _MESSAGING_PLATFORMS if needle in var["name"]), None)
            if match:
                grouped.setdefault(match[0], []).append(var)
                emojis[match[0]] = match[1]
        platform_order = list(grouped)
        labels = [f"{emojis[p]} {p}" for p in platform_order]
        for idx in prompt_checklist("Which platforms would you like to set up?", labels):
            plat = platform_order[idx]
            _section_rule(f"{emojis[plat]} {plat}")
            for var in grouped[plat]:
                print_info(f"  {var.get('description', '')}")
                if var.get("url"):
                    print_info(f"  {var['url']}")
                _prompt_and_save_env_var(var, "  ✓ Saved", "  Skipped")
                print()

    # Handle missing config fields
    if missing_config:
        _info(None, f"Adding {len(missing_config)} new config option(s) with defaults...")
        for field in missing_config:
            print_success(f"  Added {field['key']} = {field['default']}")
        config["_config_version"] = latest_ver
        save_config(config)
    _print_setup_summary(config, hermes_home)
