"""Streamlined setup flows: the Nous Portal one-shot (`hermes portal`), first-time quick setup,
Blank Slate setup and the `--quick` missing-items pass. Names from setup.py are imported lazily
per function so test patches on ``hermes_cli.setup`` take effect."""

import contextlib
import logging
import os
import sys
from pathlib import Path

from agent.i18n import t

logger = logging.getLogger("hermes_cli.setup")

# (env-var name substring, platform label, emoji) — order matters: first match wins.
_MESSAGING_PLATFORMS = (("TELEGRAM", "Telegram", "📱"), ("DISCORD", "Discord", "💬"), ("SLACK", "Slack", "💼"))
_PORTAL_SIGNUP_URL = "https://portal.nousresearch.com/manage-subscription"



def _blank_slate_done(config: dict, hermes_home, tools_line: str, *extra: str, intro: str | None = None) -> None:
    """Shared Blank Slate epilogue: success banner, the "enable later" hints, then the summary."""
    from hermes_cli.setup import _info, _print_setup_summary, print_success
    print()
    print_success(t("setup.blank.complete"))
    _info(*([intro] if intro else []), tools_line, t("setup.blank.hint_seed_skills"),
          t("setup.blank.hint_mcp"), *extra, t("setup.blank.hint_agent"), None)
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
    from hermes_cli.setup import (
        _box_line, _info, _lines, _print_banner, _record_setup_completed, print_error, print_info, print_success
    )
    _print_banner(_box_line(t("setup.portal.banner"), indent=5))
    _info(None, *_lines("setup.portal.intro"), None, t("setup.portal.sign_up", url=_PORTAL_SIGNUP_URL), None)

    def _on_error(exc: Exception) -> None:
        from hermes_cli.auth_error_copy import provider_setup_failure_lines
        print()
        lead, *rest = provider_setup_failure_lines(exc, retry_command="hermes portal")
        print_error(f"  {lead}")
        for line in rest:
            print_info(f"  {line}")

    if not _run_nous_flow(config, context="`hermes portal`", cancel_exc=(KeyboardInterrupt, EOFError, SystemExit),
                          cancel_lines=(None, *_lines("setup.portal.cancelled")),
                          print_error=_on_error):
        return

    # Re-sync from disk so a caller's later save_config(config) can't clobber the login save.
    with contextlib.suppress(Exception):
        _reload_config_into(config, dict_only=True)
    print()
    print_success(t("setup.portal.complete"))
    _info(*_lines("setup.portal.next_steps"))
    _record_setup_completed(config)


def _run_first_time_quick_setup(config: dict, hermes_home, is_existing: bool):
    """Streamlined first-time setup via Nous Portal: OAuth, model, terminal & messaging;
    everything else gets defaults."""
    from hermes_cli.setup import (
        _apply_default_agent_settings, _info, _lines, print_header, print_info, _print_setup_summary, print_success,
        print_warning, prompt_choice, save_config, setup_gateway, setup_terminal_backend
    )
    # Step 1: Nous Portal — OAuth login + model selection (provider set to "nous" by the save).
    print_header("Nous Portal", gap=True)
    _info(*_lines("setup.quick.portal_intro"), t("setup.quick.sign_up", url=_PORTAL_SIGNUP_URL), None)

    def _on_error(exc: Exception) -> None:
        from hermes_cli.auth_error_copy import provider_setup_failure_lines
        lead, *rest = provider_setup_failure_lines(exc, retry_command="hermes model")
        print_warning(lead)
        for line in rest:
            print_info(line)

    _run_nous_flow(config, context="quick setup", cancel_exc=(KeyboardInterrupt, EOFError),
                   cancel_lines=(None, t("setup.quick.portal_cancelled")), print_error=_on_error)
    # The wizard's later save_config(config) must not clobber the login/model save.
    _reload_config_into(config)

    # Step 2: Terminal Backend; Step 3: defaults for everything else.
    setup_terminal_backend(config)
    _apply_default_agent_settings(config)
    save_config(config)

    # Step 4: Offer messaging gateway setup
    print()
    gateway_choice = prompt_choice(t("setup.quick.messaging_question"), [
        t("setup.quick.messaging_now"), t("setup.quick.messaging_skip"),
    ], 0)
    if gateway_choice == 0:
        setup_gateway(config)
        save_config(config)
    else:
        # Messaging is optional; service persistence still requires explicit consent.
        from hermes_cli.gateway_setup_service import ensure_gateway_service
        ensure_gateway_service(context="setup", interactive=True, config=config)
    print()
    print_success(t("setup.quick.complete"))
    _info(None, t("setup.quick.hint_all_settings"))
    if gateway_choice != 0:
        print_info(t("setup.quick.hint_gateway"))
    _print_macos_fda_tip()
    print()
    _print_setup_summary(config, hermes_home)


def _print_macos_fda_tip() -> None:
    """One-time macOS tip: one Full Disk Access grant kills every per-folder prompt. Same
    prompt-free probe as doctor's check_macos_full_disk_access (the FDA-gated TCC dir never
    triggers a dialog); silent on non-macOS and when FDA is granted or indeterminate.

    every per-folder permission prompt, permanently (issue #52010 follow-up).
    """
    from hermes_cli.setup import _info, _lines
    if sys.platform != "darwin":
        return
    try:
        os.listdir(Path.home() / "Library" / "Application Support" / "com.apple.TCC")
        return  # already granted — nothing to teach
    except PermissionError:
        pass
    except OSError:
        return  # indeterminate — don't nag
    _info(None, *_lines("setup.quick.macos_fda_tip"),
          "    open \"x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles\"",
          t("setup.quick.macos_fda_permanent"))


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
        _info, _lines, print_header, print_info, print_success, prompt_choice, save_config, setup_model_provider,
        setup_terminal_backend
    )
    print_header(t("setup.blank.header"), gap=True)
    _info(*_lines("setup.blank.intro"), "", *_lines("setup.blank.forced_on"), None)

    # Step 1: Provider & Model (REQUIRED — the agent cannot run without it)
    print_header(t("setup.blank.step_provider"))
    setup_model_provider(config)
    save_config(config)

    # Step 2: Terminal backend (where commands run — a core decision)
    print_header(t("setup.blank.step_terminal"))
    setup_terminal_backend(config)

    # Step 3: Lock in the minimal toolset + minimized config knobs
    _blank_slate_minimal_toolsets(config)
    _blank_slate_minimize_config(config)
    save_config(config)
    print()
    print_success(t("setup.blank.baseline_applied"))
    print_info(t("setup.blank.baseline_toolsets"))
    print_info(t("setup.blank.baseline_features"))

    # The fork: stop here, or walk through enabling things
    print_header(t("setup.blank.fork_header"), gap=True)
    path = prompt_choice(t("setup.blank.fork_question"), [
        t("setup.blank.fork_finish"), t("setup.blank.fork_walkthrough"),
    ], 0)
    if path != 0:
        _blank_slate_walkthrough(config, hermes_home)
        return
    save_config(config)
    # Blank Slate means no bundled skills; record the opt-out so future `hermes update` runs
    # don't re-inject them.
    _set_bundled_skills_opt_out(True, "skill opt-out")
    _blank_slate_done(config, hermes_home, t("setup.blank.hint_tools"), t("setup.blank.hint_plugins"),
                      intro=t("setup.blank.enable_later"))


def _blank_slate_walkthrough(config: dict, hermes_home):
    """Opt-in walkthrough for Blank Slate: skills, tools, plugins, MCP, gateway."""
    from hermes_cli.setup import (
        _info, _lines, print_header, print_info, print_success, print_warning, prompt_yes_no, save_config,
        setup_gateway,
    )
    # Bundled skills — default to NONE, offer to seed all
    print_header(t("setup.blank.skills_header"), gap=True)
    print_info(t("setup.blank.skills_none_default"))
    seed_skills = prompt_yes_no(t("setup.blank.skills_seed_prompt"), default=False)

    def _seeded(result) -> None:
        copied = len(result.get("copied", [])) if isinstance(result, dict) else 0
        print_success(t("setup.blank.skills_seeded", count=copied))

    def _opted_out(_result) -> None:
        _info(*_lines("setup.blank.skills_opted_out"))

    # Seeding first clears any stale opt-out marker; declining sets it (essential skills still seed).
    _set_bundled_skills_opt_out(
        not seed_skills, "skill handling", on_success=_seeded if seed_skills else _opted_out,
        on_error=lambda exc: print_warning(t("setup.blank.skills_error", error=exc)),
    )

    # Walk through enabling additional tools
    print_header(t("setup.section.tools"), gap=True)
    _info(*_lines("setup.blank.tools_intro"))
    if prompt_yes_no(t("setup.blank.tools_prompt"), default=False):
        try:
            from hermes_cli.tools_config import tools_command
            tools_command(first_install=False, config=config)
            _reload_config_into(config)  # tools_command saves via its own load/save cycle
        except Exception as exc:
            logger.debug("blank-slate tools_command error: %s", exc)
            print_warning(t("setup.blank.tools_error", error=exc))
    else:
        print_info(t("setup.blank.tools_keep_minimal"))

    # Built-in plugins and MCP servers (off unless chosen): setup.blank.<kind>_{header,prompt,yes,no}
    for kind in ("plugins", "mcp"):
        print_header(t(f"setup.blank.{kind}_header"), gap=True)
        answer = "yes" if prompt_yes_no(t(f"setup.blank.{kind}_prompt"), default=False) else "no"
        print_info(t(f"setup.blank.{kind}_{answer}"))

    # Optional messaging gateway
    print()
    if prompt_yes_no(t("setup.blank.messaging_prompt"), default=False):
        setup_gateway(config)
    save_config(config)
    _blank_slate_done(config, hermes_home, t("setup.blank.hint_more_tools"))


def _run_quick_setup(config: dict, hermes_home):
    """Quick setup — only configure items that are missing."""
    from hermes_cli.setup import (
        color, Colors, _info, _lines, print_header, print_info, _print_setup_summary, print_success,
        _prompt_and_save_env_var, _prompt_api_key, _section_rule, prompt_checklist, save_config,
    )
    from hermes_cli.config import (get_missing_env_vars, get_missing_config_fields, check_config_version)
    print_header(t("setup.missing.header"), gap=True)

    # Check what's missing
    missing_env = get_missing_env_vars(required_only=False)
    missing_required = [v for v in missing_env if v.get("is_required")]
    missing_optional = [v for v in missing_env if not v.get("is_required")]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()
    if not (missing_required or missing_optional or missing_config or current_ver < latest_ver):
        print_success(t("setup.missing.nothing_to_do"))
        _info(None, *_lines("setup.missing.reconfigure_hint"))
        return
    if missing_required:
        _info(None, t("setup.missing.required_count", count=len(missing_required)))
        for var in missing_required:
            print(f"     • {var['name']}")
        print()
        for var in missing_required:
            print()
            print(color(f"  {var['name']}", Colors.CYAN))
            print_info(f"  {var.get('description', '')}")
            if var.get("url"):
                print_info(t("setup.missing.get_key_at", url=var["url"]))
            _prompt_and_save_env_var(var, t("setup.missing.saved_var", name=var["name"]),
                                     t("setup.missing.skipped_var", name=var["name"]))
    missing_tools = [v for v in missing_optional if v.get("category") == "tool"]
    missing_messaging = [v for v in missing_optional if v.get("category") == "messaging" and not v.get("advanced")]
    if missing_tools:  # checklist, then the API-key screen for each pick
        print_header(t("setup.missing.tools_header"), gap=True)
        labels = [var.get("description", var["name"]) + (f" → {', '.join(var['tools'][:2])}" if var.get("tools") else "")
                  for var in missing_tools]
        for idx in prompt_checklist(t("setup.missing.tools_question"), labels):
            _prompt_api_key(missing_tools[idx])
    if missing_messaging:  # checklist, then prompt for each selected platform's vars
        print_header(t("setup.section.messaging"), gap=True)
        _info(*_lines("setup.missing.messaging_intro"))
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
        for idx in prompt_checklist(t("setup.missing.platforms_question"), labels):
            plat = platform_order[idx]
            _section_rule(f"{emojis[plat]} {plat}")
            for var in grouped[plat]:
                print_info(f"  {var.get('description', '')}")
                if var.get("url"):
                    print_info(f"  {var['url']}")
                _prompt_and_save_env_var(var, t("setup.api_key.saved"), t("setup.missing.skipped"))
                print()

    # Handle missing config fields
    if missing_config:
        _info(None, t("setup.missing.adding_config", count=len(missing_config)))
        for field in missing_config:
            print_success(t("setup.missing.added_field", key=field["key"], default=field["default"]))
        config["_config_version"] = latest_ver
        save_config(config)
    _print_setup_summary(config, hermes_home)
