"""``hermes config`` / ``hermes config show``: the read-only configuration summary.

Split out of ``hermes_cli.config``; config names are read through the module at call time
(``_cfg.<name>``) so ``hermes_cli.config.<name>`` patch seams hold.
"""

from typing import Any, Dict

from hermes_cli import config as _cfg
from hermes_cli import managed_scope
from hermes_cli.colors import Colors, color


def _section(title: str) -> None:
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def _show_managed_banner() -> None:
    """Surface administrator-pinned settings so the user knows why a config.yaml value may not
    be the effective one."""
    managed_keys = managed_scope.managed_config_keys()
    managed_env = managed_scope.load_managed_env()
    if not managed_keys and not managed_env:
        return
    print()
    print(color(
        f"  ⚷ Some settings are managed by your administrator ({managed_scope.get_managed_dir()}) "
        f"and cannot be changed", Colors.YELLOW, Colors.BOLD))
    for label, keys in (("config", managed_keys), ("env", managed_env)):
        if keys:
            print(color(f"    Managed {label} keys: {', '.join(sorted(keys))}", Colors.YELLOW))


_SHOW_CONFIG_API_KEYS = (
    ("OPENROUTER_API_KEY", "OpenRouter"),
    ("VOICE_TOOLS_OPENAI_KEY", "OpenAI (STT/TTS)"),
    ("EXA_API_KEY", "Exa"),
    ("PARALLEL_API_KEY", "Parallel"),
    ("FIRECRAWL_API_KEY", "Firecrawl"),
    ("TAVILY_API_KEY", "Tavily"),
    ("PERPLEXITY_API_KEY", "Perplexity"),
    ("BROWSERBASE_API_KEY", "Browserbase"),
    ("BROWSER_USE_API_KEY", "Browser Use"),
    ("FAL_KEY", "FAL"))


def _show_model_section(config: dict[str, Any]) -> None:
    _section("Model")
    print(f"  Model:        {_cfg.redact_config_value(config.get('model', 'not set'))}")
    cfg_max_turns = config.get('agent', {}).get('max_turns', _cfg.DEFAULT_CONFIG['agent']['max_turns'])
    print(f"  Max turns:    {cfg_max_turns}")
    # Read the .env FILE directly so a stale HERMES_MAX_ITERATIONS ghost is caught even when the
    # gateway bridge already overrode os.environ.
    try:
        env_ghost = _cfg.load_env().get("HERMES_MAX_ITERATIONS")
    except Exception:
        env_ghost = None
    if env_ghost is not None and str(env_ghost).strip() != str(cfg_max_turns).strip():
        print(color(f"                ⚠ .env has stale HERMES_MAX_ITERATIONS={env_ghost} "
                    f"(run 'hermes doctor --fix' to remove)", Colors.YELLOW))


def _show_display_section(config: dict[str, Any]) -> None:
    _section("Display")
    display = config.get('display', {})
    try:
        from hermes_cli.personality import active_personality_name
        active_personality = active_personality_name(config) or 'none'
    except Exception:
        active_personality = display.get('personality') or 'none'
    on_off = lambda flag: 'on' if flag else 'off'
    print(f"  Personality:  {active_personality}")
    print(f"  Reasoning:    {on_off(display.get('show_reasoning', True))}")
    print(
        f"  Bell:         complete={on_off(display.get('bell_on_complete', False))}, "
        f"prompt={on_off(display.get('bell_on_prompt', False))}")
    ump = display.get('user_message_preview', {})
    ump = ump if isinstance(ump, dict) else {}
    print(f"  User preview: first {ump.get('first_lines', 2)} line(s), last {ump.get('last_lines', 2)} line(s)")


def _show_terminal_section(config: dict[str, Any]) -> None:
    _section("Terminal")
    terminal = config.get('terminal', {})
    print(f"  Backend:      {terminal.get('backend', 'local')}")
    print(f"  Working dir:  {terminal.get('cwd', '.')}")
    print(f"  Timeout:      {terminal.get('timeout', 60)}s")

    configured = lambda *names: 'configured' if all(_cfg.get_env_value(n) for n in names) else '(not set)'
    from hermes_cli.config_defaults import DEFAULT_SANDBOX_IMAGE as default_img, DEFAULT_VERCEL_IMAGE as _DEFAULT_VERCEL_IMAGE
    backend_lines = {
        'docker': lambda: [f"  Docker image: {terminal.get('docker_image', default_img)}"],
        'singularity': lambda: [f"  Image:        {terminal.get('singularity_image', 'docker://' + default_img)}"],
        'modal': lambda: [
            f"  Modal image:  {terminal.get('modal_image', default_img)}",
            f"  Modal token:  {configured('MODAL_TOKEN_ID')}"],
        'daytona': lambda: [
            f"  Daytona image: {terminal.get('daytona_image', default_img)}",
            f"  API key:      {configured('DAYTONA_API_KEY')}"],
        'vercel_sandbox': lambda: [
            f"  Vercel image:   {terminal.get('vercel_runtime') or terminal.get('vercel_image') or _DEFAULT_VERCEL_IMAGE}",
            f"  Vercel auth:    {'configured' if _cfg.get_env_value('VERCEL_OIDC_TOKEN') or (_cfg.get_env_value('VERCEL_TOKEN') and _cfg.get_env_value('VERCEL_PROJECT_ID') and _cfg.get_env_value('VERCEL_TEAM_ID')) else '(not set)'}",
        ],
        'ssh': lambda: [
            f"  SSH host:     {_cfg.get_env_value('TERMINAL_SSH_HOST') or '(not set)'}",
            f"  SSH user:     {_cfg.get_env_value('TERMINAL_SSH_USER') or '(not set)'}"]}
    for line in backend_lines.get(terminal.get('backend'), list)():
        print(line)


def _show_compression_section(config: dict[str, Any]) -> None:
    _section("Context Compression")
    compression = config.get('compression', {})
    enabled = compression.get('enabled', True)
    print(f"  Enabled:      {'yes' if enabled else 'no'}")
    if not enabled:
        return
    print(f"  Threshold:    {compression.get('threshold', 0.50) * 100:.0f}%")
    tt = compression.get('threshold_tokens')
    try:
        if tt is not None and int(tt) > 0:
            print(f"  Token cap:    {int(tt):,} tokens (takes lower of ratio vs absolute)")
    except (TypeError, ValueError):
        pass
    print(f"  Target ratio: {compression.get('target_ratio', 0.20) * 100:.0f}% of threshold preserved")
    print(f"  Protect last: {compression.get('protect_last_n', 20)} messages")
    print(f"  Protect first: {compression.get('protect_first_n', 3)} non-system head messages")
    aux_comp = config.get('auxiliary', {}).get('compression', {})
    print(f"  Model:        {aux_comp.get('model', '') or '(auto)'}")
    comp_provider = aux_comp.get('provider', 'auto')
    if comp_provider and comp_provider != 'auto':
        print(f"  Provider:     {comp_provider}")


def _show_aux_overrides(config: dict[str, Any]) -> None:
    aux_tasks = {"Vision": config.get('auxiliary', {}).get('vision', {})}
    overrides = {
        label: (t.get('provider', 'auto'), t.get('model', ''))
        for label, t in aux_tasks.items()
        if t.get('provider', 'auto') != 'auto' or t.get('model', '')}
    if not overrides:
        return
    _section("Auxiliary Models (overrides)")
    for label, (prov, mdl) in overrides.items():
        parts = [f"provider={prov}"] + ([f"model={mdl}"] if mdl else [])
        print(f"  {label:12s}  {', '.join(parts)}")


def _show_skill_settings() -> None:
    try:
        from agent.skill_utils import discover_all_skill_config_vars, resolve_skill_config_values
        skill_vars = discover_all_skill_config_vars()
        if not skill_vars:
            return
        resolved = resolve_skill_config_values(skill_vars)
        _section("Skill Settings")
        for var in skill_vars:
            value = resolved.get(var["key"], "")
            display_val = str(value) if value else color("(not set)", Colors.DIM)
            skill_tag = color(f"[{var.get('skill', '')}]", Colors.DIM)
            print(f"  {var['key']:<20s} {display_val}  {skill_tag}")
    except Exception:
        pass


def show_config():
    """Display current configuration."""
    config = _cfg.load_config()

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│              ☤ Hermes Configuration                    │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    _show_managed_banner()

    _section("Paths")
    print(f"  Config:       {_cfg.get_config_path()}")
    print(f"  Secrets:      {_cfg.get_env_path()}")
    print(f"  Install:      {_cfg.get_project_root()}")

    _section("API Keys")
    for env_key, name in _SHOW_CONFIG_API_KEYS:
        print(f"  {name:<14} {_cfg.redact_key(_cfg.get_env_value(env_key))}")
    from hermes_cli.auth import get_anthropic_key
    print(f"  {'Anthropic':<14} {_cfg.redact_key(get_anthropic_key())}")

    _show_model_section(config)
    _show_display_section(config)
    _show_terminal_section(config)

    _section("Timezone")
    tz = config.get('timezone', '')
    print(f"  Timezone:     {tz or color('(server-local)', Colors.DIM)}")

    _show_compression_section(config)
    _show_aux_overrides(config)

    _section("Messaging Platforms")
    for label, env_key in (("Telegram", "TELEGRAM_BOT_TOKEN"), ("Discord", "DISCORD_BOT_TOKEN")):
        state = 'configured' if _cfg.get_env_value(env_key) else color('not configured', Colors.DIM)
        print(f"  {label + ':':<13} {state}")

    _show_skill_settings()

    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  hermes config edit     # Edit config file", Colors.DIM))
    print(color("  hermes config set <key> <value>", Colors.DIM))
    print(color("  hermes setup           # Run setup wizard", Colors.DIM))
    print()
