"""hermes memory setup|status — configure memory provider plugins."""

from __future__ import annotations

import os
import sys
import shlex
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.secret_prompt import masked_secret_prompt

_CANCELLED = -1

# pip name → import name mapping for packages where they differ
_IMPORT_NAMES = {
    "honcho-ai": "honcho",
    "mem0ai": "mem0",
    "hindsight-client": "hindsight_client",
    "hindsight-all": "hindsight"}


def _provider_extras(provider_name: str, manifest: dict, plugin_dir=None) -> list[str]:
    """The pyproject extras a provider needs on THIS install.

    ``plugin.yaml``'s ``extra:`` names the baseline. Hindsight's
    ``local_embedded`` mode needs the daemon+embedder wheel on top
    (hindsight-local, #70636) — mode lives in its config file, which the
    manifest can't express.

    A provider manifest with legacy ``pip_dependencies`` (no ``extra:``,
    no pyproject of its own — the third-party plugin shape, e.g. a
    directory-installed external provider) is bridged: pm materializes
    the specs into a generated pyproject.toml, and the provider becomes
    a workspace member installed by the venv sync. Returns [] either
    way — no EXTRA to sync, the union carries the deps."""
    extras = []
    declared = manifest.get("extra")
    if isinstance(declared, str) and declared:
        extras.append(declared)
    elif plugin_dir is not None and manifest.get("pip_dependencies"):
        try:
            from pm.workspace import materialize_legacy_pyproject

            materialize_legacy_pyproject(Path(plugin_dir))
        except Exception:
            pass
    # hindsight local_embedded needs hindsight-all, whose protobuf floor
    # conflicts with mem0/modal — it cannot be a venv extra. It stays a
    # provider-owned install until plugin side-venvs land (plan step 5).
    return extras


def _curses_select(
    title: str, items: list[tuple[str, str]], default: int = 0, *, cancel_returns: int | None = None
) -> int:
    """Interactive single-select with arrow keys."""
    from hermes_cli.curses_ui import curses_radiolist

    if cancel_returns is None:
        cancel_returns = default
    display_items = [f"{label} - {desc}" if desc else label for label, desc in items]
    result = curses_radiolist(title, display_items, selected=default, cancel_returns=cancel_returns)
    _clear_interactive_transition()
    return result


def _print_cancelled_setup() -> None:
    print("\n  Cancelled. No changes saved.\n")


def _clear_interactive_transition() -> None:
    """Clear stale curses content before entering a follow-up setup screen."""
    if not sys.stdout.isatty():
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    """Prompt for a value with optional default and secret masking."""
    suffix = f" [{default}]" if default else ""
    if secret:
        val = masked_secret_prompt(f"  {label}{suffix}: ")
    else:
        sys.stdout.write(f"  {label}{suffix}: ")
        sys.stdout.flush()
        val = sys.stdin.readline().strip()
    return val or (default or "")


def _install_dependencies(provider_name: str, *, force: bool = False) -> None:
    """Install pip dependencies declared in ``plugin.yaml``.

    With ``force`` every declared dependency goes to the installer even if it imports (the resolver
    no-ops when nothing drifted) — how ``hermes update`` heals a provider after a venv rebuild.

    When ``force`` is true, every declared dependency is handed to the installer even if its import
    currently succeeds — the resolver then reinstalls anything missing or version-drifted and no-ops on
    satisfied ranges. This is how ``hermes update`` heals the active memory provider after a venv
    rebuild/sync removed or downgraded its bridge packages (#53272, #70636).
    """
    import subprocess
    from plugins.memory import find_provider_dir

    plugin_dir = find_provider_dir(provider_name)
    if not plugin_dir:
        return
    yaml_path = plugin_dir / "plugin.yaml"
    if not yaml_path.exists():
        return
    try:
        import yaml
        with open(yaml_path, encoding="utf-8-sig") as f:
            meta = yaml.safe_load(f) or {}
    except Exception:
        return

    extras = _provider_extras(provider_name, meta, plugin_dir=plugin_dir)
    if not extras:
        return

    import pm

    missing = [e for e in extras if force or not pm.available(e)]
    if not missing:
        return

    print(f"\n  Installing dependencies: {', '.join(missing)}")
    try:
        pm.sync_venv(missing, explicit=True)
        print(f"  ✓ Installed {', '.join(missing)}")
    except Exception as e:
        print(f"  ⚠ Install failed: {e}")
        print("  Run manually: hermes pm install")

    # Also show external (non-pip) dependencies that are missing.
    for dep in meta.get("external_dependencies", []):
        check_cmd = dep.get("check", "")
        install_cmd = dep.get("install", "")
        if check_cmd:
            try:
                subprocess.run(shlex.split(check_cmd), check=True, capture_output=True, timeout=5)
            except Exception:
                if install_cmd:
                    print(f"\n  ⚠ '{dep.get('name', '')}' not found. Install with:")
                    print(f"    {install_cmd}")


def _schema_of(provider) -> list:
    return provider.get_config_schema() if hasattr(provider, "get_config_schema") else []


def _get_available_providers() -> list:
    """Discover memory providers from plugins/memory/ as ``(name, setup_hint, provider)`` tuples."""
    try:
        from plugins.memory import discover_memory_providers, load_memory_provider
        raw = discover_memory_providers()
    except Exception:
        raw = []

    results = []
    for name, desc, available in raw:
        try:
            provider = load_memory_provider(name)
            if not provider:
                continue
        except Exception:
            continue
        schema = _schema_of(provider)
        has_secrets = any(f.get("secret") for f in schema)
        has_non_secrets = any(not f.get("secret") for f in schema)
        if has_secrets and has_non_secrets:
            setup_hint = "API key / local"
        elif has_secrets:
            setup_hint = "requires API key"
        elif not schema:
            setup_hint = "no setup needed"
        else:
            setup_hint = "local"
        results.append((name, setup_hint, provider))
    return results


def _find_provider(providers: list, provider_name: str):
    return next((p for p in providers if p[0] == provider_name), None)


def _post_setup_hook(provider, config: dict) -> bool:
    """Normalize the ``memory`` block; True when the provider's ``post_setup`` took over (it owns
    config, connection test and activation), so the caller must stop."""
    if not isinstance(config.get("memory"), dict):
        config["memory"] = {}
    if hasattr(provider, "post_setup"):
        provider.post_setup(str(get_hermes_home()), config)
        return True
    return False


def cmd_setup_provider(provider_name: str) -> None:
    """Run memory setup for a specific provider, skipping the picker."""
    from hermes_cli.config import load_config, save_config

    match = _find_provider(_get_available_providers(), provider_name)
    if not match:
        print(f"\n  Memory provider '{provider_name}' not found.")
        print("  Run 'hermes memory setup' to see available providers.\n")
        return
    name, _, provider = match

    _clear_interactive_transition()
    _install_dependencies(name)
    config = load_config()
    if _post_setup_hook(provider, config):
        return
    # Fallback: generic schema-based setup (same as cmd_setup)
    config["memory"]["provider"] = name
    save_config(config)
    print(f"\n  Memory provider: {name}")
    print("  Activation saved to config.yaml\n")


def _prompt_schema_fields(name: str, schema: list, provider_config: dict, env_writes: dict) -> bool:
    """Walk a provider's config schema, prompting per field. False when the user cancelled."""
    print(f"\n  Configuring {name}:\n")
    for field in schema:
        key = field["key"]
        desc = field.get("description", key)
        default = field.get("default")
        # Dynamic default: look up default from another field's value
        default_from = field.get("default_from")
        if default_from and isinstance(default_from, dict):
            ref_value = provider_config.get(default_from.get("field", ""), "")
            ref_map = default_from.get("map", {})
            if ref_value and ref_value in ref_map:
                default = ref_map[ref_value]
        is_secret = field.get("secret", False)
        choices = field.get("choices")
        env_var = field.get("env_var")
        url = field.get("url")

        when = field.get("when")
        if when and isinstance(when, dict) and not all(provider_config.get(k) == v for k, v in when.items()):
            continue

        if choices and not is_secret:
            current = provider_config.get(key, default)
            current_idx = choices.index(current) if current and current in choices else 0
            sel = _curses_select(
                f"  {desc}", [(c, "") for c in choices], default=current_idx, cancel_returns=_CANCELLED
            )
            if sel == _CANCELLED:
                _print_cancelled_setup()
                return False
            provider_config[key] = choices[sel]
        elif is_secret:
            existing = os.environ.get(env_var, "") if env_var else ""
            if existing:
                masked = f"...{existing[-4:]}" if len(existing) > 4 else "set"
                val = _prompt(f"{desc} (current: {masked}, blank to keep)", secret=True)
            else:
                if url:
                    print(f"  Get yours at {url}")
                val = _prompt(desc, secret=True)
            if val and env_var:
                env_writes[env_var] = val
        else:
            effective_default = provider_config.get(key) or default
            val = _prompt(desc, default=str(effective_default) if effective_default else None)
            if val:
                provider_config[key] = val
                if env_var and env_var not in env_writes:
                    env_writes[env_var] = val
    return True


def cmd_setup(args) -> None:
    """Interactive memory provider setup wizard."""
    from hermes_cli.config import load_config, save_config

    providers = _get_available_providers()
    if not providers:
        print("\n  No memory provider plugins detected.")
        print("  Install a plugin to ~/.hermes/plugins/ and try again.\n")
        return

    items = [(name, f"— {desc}") for name, desc, _ in providers]
    items.append(("Built-in only", "— MEMORY.md / USER.md (default)"))
    builtin_idx = len(items) - 1
    selected = _curses_select("Memory provider setup", items, default=builtin_idx, cancel_returns=_CANCELLED)
    if selected == _CANCELLED:
        _print_cancelled_setup()
        return

    config = load_config()
    if not isinstance(config.get("memory"), dict):
        config["memory"] = {}
    if selected >= len(providers):
        config["memory"]["provider"] = ""
        save_config(config)
        print("\n  ✓ Memory provider: built-in only")
        print("  Saved to config.yaml\n")
        return

    name, _, provider = providers[selected]
    _clear_interactive_transition()
    _install_dependencies(name)
    if _post_setup_hook(provider, config):
        return

    provider_config = config["memory"].get(name, {})
    if not isinstance(provider_config, dict):
        provider_config = {}
    env_writes: dict = {}
    schema = _schema_of(provider)
    if schema and not _prompt_schema_fields(name, schema, provider_config, env_writes):
        return

    # Write activation key to config.yaml
    config["memory"]["provider"] = name
    save_config(config)

    if provider_config and hasattr(provider, "save_config"):
        try:
            provider.save_config(provider_config, str(get_hermes_home()))
        except Exception as e:
            print(f"  Failed to write provider config: {e}")
    if env_writes:
        _write_env_vars(env_writes)

    print(f"\n  Memory provider: {name}")
    print("  Activation saved to config.yaml")
    if provider_config:
        print("  Provider config saved")
    if env_writes:
        print("  API keys saved to .env")
    print("\n  Start a new session to activate.\n")


def _write_env_vars(
    env_writes: dict, hermes_home: str | os.PathLike[str] | None = None) -> None:
    """Persist memory-provider env vars through the canonical ``.env`` writer.

    Delegates to ``hermes_cli.config.save_env_value`` so every key flows
    through the same input-validation gate as every other ``.env`` writer:
    the ``_ENV_VAR_NAME_RE`` regex (no malformed identifiers), the
    ``_ENV_VAR_NAME_DENYLIST`` (no ``LD_PRELOAD`` / ``PYTHONPATH`` /
    ``HERMES_HOME`` / etc.), CR/LF stripping on the value, and the atomic
    0o600-from-creation write (no TOCTOU permission window).

    Validation failures (``ValueError`` from ``save_env_value`` — a
    denylisted name or an identifier rejected by ``_ENV_VAR_NAME_RE``) are
    surfaced and skipped rather than aborting the wizard, so a single bad
    key from one schema field doesn't take down the rest of the batch.
    Non-validation errors (filesystem failures, permission errors) are
    intentionally NOT caught — those indicate the wizard cannot safely
    persist any subsequent key either and should propagate.

    ``hermes_home`` may be supplied by plugin ``post_setup`` hooks that
    already received an explicit home directory (e.g. a non-default
    profile). It is applied through the context-local Hermes home override
    so ``save_env_value`` still owns the validation, sanitization, and
    atomic-write path without mutating global ``os.environ``.
    """
    from hermes_cli.config import save_env_value
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(hermes_home) if hermes_home is not None else None
    try:
        for key, val in env_writes.items():
            try:
                save_env_value(key, val)
            except ValueError as exc:
                print(f"  Skipping {key}: {exc}")
    finally:
        if token is not None:
            reset_hermes_home_override(token)


def _mark(enabled) -> str:
    return "enabled ✓" if enabled else "disabled ✗"


def cmd_status(args) -> None:
    """Show current memory provider config."""
    from hermes_cli.config import load_config

    config = load_config()
    mem_config = config.get("memory", {})
    provider_name = mem_config.get("provider", "")

    # Memory tool enablement for the CLI platform via the canonical resolver, respecting the
    # check_fn gate when both stores are disabled.
    from hermes_cli.tools_config import _get_platform_tools
    from tools.memory_tool import check_memory_requirements
    cli_tools = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    memory_tool_enabled = ("memory" in cli_tools) and check_memory_requirements()

    print("\nMemory status\n" + "─" * 40)
    print("  Built-in (MEMORY.md / USER.md):")
    print(f"    Memory injection:   {_mark(mem_config.get('memory_enabled', True))}")
    print(f"    User profile:       {_mark(mem_config.get('user_profile_enabled', True))}")
    print(f"    Memory tool:        {_mark(memory_tool_enabled)}")
    print(f"  Provider:  {provider_name or '(none — built-in only)'}")

    providers = _get_available_providers()
    match = _find_provider(providers, provider_name)
    provider = match[2] if match else None

    if provider_name:
        provider_config = mem_config.get(provider_name, {})
        display_config = provider_config
        if provider and hasattr(provider, "get_status_config"):
            try:
                display_config = provider.get_status_config(provider_config)
            except Exception as e:
                display_config = dict(provider_config) if isinstance(provider_config, dict) else provider_config
                if isinstance(display_config, dict):
                    display_config["status_config_error"] = str(e)
        if display_config:
            print(f"\n  {provider_name} config:")
            for key, val in display_config.items():
                print(f"    {key}: {val}")

        if provider:
            print("\n  Plugin:    installed ✓")
            if provider.is_available():
                print("  Status:    available ✓")
            else:
                print("  Status:    not available ✗")
                # All fields with env_var (secret and non-secret)
                required_fields = [f for f in _schema_of(provider) if f.get("env_var")]
                if required_fields:
                    print("  Missing:")
                    for f in required_fields:
                        env_var = f.get("env_var", "")
                        url = f.get("url", "")
                        is_set = bool(os.environ.get(env_var))
                        line = f"    {'✓' if is_set else '✗'} {env_var}"
                        if url and not is_set:
                            line += f"  → {url}"
                        print(line)
                print("  Note: systemd/gateway services do not inherit ~/.hermes/.env —")
                print("        set any variables above in the service environment.")
        else:
            print("\n  Plugin:    NOT installed ✗")
            print(f"  Install the '{provider_name}' memory plugin to ~/.hermes/plugins/")

    if providers:
        print("\n  Installed plugins:")
        for pname, desc, _ in providers:
            active = " ← active" if pname == provider_name else ""
            print(f"    • {pname}  ({desc}){active}")
    print()


def memory_command(args) -> None:
    """Route memory subcommands."""
    if getattr(args, "memory_command", None) == "setup":
        provider = getattr(args, "provider", None)
        if provider:
            cmd_setup_provider(provider)
        else:
            cmd_setup(args)
    else:
        cmd_status(args)
