"""hermes memory setup|status — configure memory provider plugins."""

from __future__ import annotations

import os
import re
import sys
import shlex

from hermes_constants import get_hermes_home
from hermes_cli.secret_prompt import masked_secret_prompt

_CANCELLED = -1

# pip name → import name mapping for packages where they differ
# (Intel-macOS slim stack, #81421: the slim meta-packages don't expose
# a top-level import at their pip name, so map them to the modules they
# ship — otherwise the missing-dep probe re-installs them every refresh)
_IMPORT_NAMES = {
    "honcho-ai": "honcho",
    "mem0ai": "mem0",
    "hindsight-client": "hindsight_client",
    "hindsight-all": "hindsight",
    "hindsight-all-slim": "hindsight_api",
    "hindsight-api-slim": "hindsight_api",
    "hindsight-embed": "hindsight_embed"}


def _provider_pip_dependencies(provider_name: str, declared: list) -> list:
    """Return the pip deps a provider actually needs on THIS install.

    ``plugin.yaml`` declares the baseline bridge packages; some providers add mode-dependent extras
    at setup time that the manifest can't express.

    Hindsight's ``local_embedded`` mode installs ``hindsight-all`` (daemon + embedder + client) during
    ``hermes memory setup`` — if the update-time refresh only reinstalled the declared ``hindsight-client``,
    the embedded daemon would stay broken after a venv rebuild stripped ``hindsight-embed`` (#70636).

    The bare full bundle is NOT portable to Intel macOS: the current full
    local-ML dependency set pulls MLX packages that have no x86_64 wheels,
    so the resolver backtracks to ancient ``hindsight-all``/``hindsight-api``
    releases whose overlapping ``hindsight_api`` files override the working
    slim API and crash the daemon with "Unknown embeddings provider: onnx"
    (#81421).  On that platform the local-embedded mode installs the thin
    slim stack instead (see ``_hindsight_local_embedded_deps``).
    """
    deps = list(declared or [])
    if provider_name == "hindsight":
        try:
            import json
            cfg_path = get_hermes_home() / "hindsight" / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
            # "local" is a legacy alias for "local_embedded"
            mode = cfg.get("mode", "")
            if mode in {"local", "local_embedded"}:
                deps += _hindsight_local_embedded_deps()
        except Exception:
            pass
    return deps


def _hindsight_local_embedded_deps() -> list:
    """Hindsight local-embedded pip specs for THIS platform.

    The bare full bundle is not portable to Intel macOS: its current
    local-ML dependency set pulls MLX packages that have no x86_64 wheels,
    so the resolver backtracks to ancient releases that break the
    configured ONNX runtime (#81421).  On that platform install the thin
    slim stack instead — ``hindsight-all-slim`` + ``hindsight-api-slim[local-onnx]``
    + ``hindsight-embed`` (the embed manager drives the configured ONNX
    embeddings provider and is declared explicitly so a venv rebuild strips
    it no more than hindsight-embed did in #70636).  Apple Silicon (arm64)
    and every other OS keep the full ``hindsight-all`` bundle.

    This is the single source of truth for the local-embedded spec list —
    both ``_provider_pip_dependencies`` and the Hindsight plugin's
    ``HindsightMemoryProvider.post_setup`` wizard call it, so the wizard can
    never drift from the refresh/heal path (#81421, #81530).
    """
    if _is_intel_macos():
        return [
            "hindsight-all-slim",
            "hindsight-api-slim[local-onnx]",
            # The embed manager must be declared explicitly: it is the
            # component that drives the configured ONNX embeddings
            # provider, and a venv rebuild strips it exactly like
            # hindsight-embed did in #70636.
            "hindsight-embed",
        ]
    return ["hindsight-all"]


def _is_intel_macos() -> bool:
    """True on macOS running on an Intel (x86_64) CPU.

    The full Hindsight local-ML bundle is not installable there: its
    current dependencies include MLX packages that ship no x86_64 wheels,
    so the resolver silently backtracks to ancient releases that break
    the configured ONNX runtime (#81421).  Apple Silicon (arm64) and
    every other OS keep the full bundle.
    """
    import platform
    return platform.system() == "Darwin" and platform.machine() in {
        "x86_64",
        "i386",
        "i686",
    }


# ---------------------------------------------------------------------------
# Curses-based interactive picker (same pattern as hermes tools)
# ---------------------------------------------------------------------------
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
        with open(yaml_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
    except Exception:
        return

    pip_deps = _provider_pip_dependencies(provider_name, meta.get("pip_dependencies", []))
    if not pip_deps:
        return

    missing = []
    for dep in pip_deps:
        if force:
            missing.append(dep)
            continue
        dep_name = re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*", dep)
        base = dep_name.group(0) if dep_name else dep
        import_name = _IMPORT_NAMES.get(base, base.replace("-", "_").split("[")[0])
        try:
            __import__(import_name)
        except ImportError:
            missing.append(dep)
    if not missing:
        # Even when every slim dep already imports, the configured local
        # runtime on Intel macOS may still be stale — the #81421 failure
        # mode is an ancient hindsight_api release that is importable but
        # no longer exposes LocalSTEmbeddings, so the probe above would
        # skip reinstall while the daemon still crashes. Smoke-check even
        # on the refresh path (gated inside the helper).
        if provider_name == "hindsight":
            _maybe_run_intel_macos_local_embedded_smoke_check()
        return

    print(f"\n  Installing dependencies: {', '.join(missing)}")

    # install_specs routes to the durable target on sealed hosted images (HERMES_LAZY_INSTALL_TARGET)
    # and is venv-scoped on normal installs.
    from tools.lazy_deps import install_specs

    manual_cmd = f"uv pip install {' '.join(missing)}"
    install_succeeded = False
    try:
        outcome = install_specs(missing, timeout=120)
        if outcome.ok:
            print(f"  ✓ Installed {', '.join(missing)}")
            install_succeeded = True
        elif outcome.blocked:
            print(f"  ⚠ Cannot install {', '.join(missing)}: {outcome.reason}")
        else:
            print(f"  ⚠ Failed to install {', '.join(missing)}")
            stderr = (outcome.stderr or "")[:200]
            if stderr:
                print(f"    {stderr}")
            print(f"  Run manually: {manual_cmd}")
    except Exception as e:
        print(f"  ⚠ Install failed: {e}")
        print(f"  Run manually: {manual_cmd}")

    # Post-install smoke check (#81421): only after the install actually
    # reports success. Pip may resolve and report ok while still
    # backtracking the slim runtime to an ancient API release whose
    # ``hindsight_api`` no longer exposes ``LocalSTEmbeddings`` — the
    # daemon then crashes with "Unknown embeddings provider: onnx" while
    # the update claims success. Deliberately OUTSIDE the install
    # try/except: its RuntimeError must propagate to the caller (the
    # wizard / update flow), not be swallowed as a failed-install notice.
    # A failed or blocked install never reaches it, so the "Run manually:"
    # guidance and the smoke-check error can't contradict each other
    # (#81530 follow-up).
    if install_succeeded and provider_name == "hindsight":
        _maybe_run_intel_macos_local_embedded_smoke_check()

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


def _maybe_run_intel_macos_local_embedded_smoke_check() -> None:
    """Post-install/refresh smoke check for the Intel-macOS slim runtime.

    On Intel macOS + local/local_embedded, pip reports success even when
    the resolver backtracked the slim stack to an ancient ``hindsight_api``
    release whose files shadow the working slim runtime — the daemon then
    crashes with "Unknown embeddings provider: onnx" while the update
    claims success (#81421).  This verifies the configured local runtime
    is actually usable, prints an actionable failure banner, and raises
    so ``hermes update`` cannot report a healed install that is still
    broken.

    It is gated to (Intel macOS + local mode); every other path returns
    immediately and costs nothing on refresh.
    """
    import json as _json

    try:
        cfg_path = get_hermes_home() / "hindsight" / "config.json"
        cfg = _json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    except Exception:
        cfg = {}
    if cfg.get("mode", "") not in {"local", "local_embedded"}:
        return
    if not _is_intel_macos():
        return
    smoke_errors = _smoke_import_hindsight_local()
    if not smoke_errors:
        return
    print(
        "  ✗ Hindsight slim runtime smoke validation failed (#81421). The "
        "daemon will not start with this stack; re-running the install "
        "will not help until the resolver stops backtracking. Check `pip "
        "show hindsight-api-slim` and `pip show hindsight-embed` for "
        "overlapping API files."
    )
    for err in smoke_errors:
        print(f"    {err}")
    raise RuntimeError(
        "Hindsight slim runtime smoke validation failed on Intel macOS "
        "— pip (or an existing importable release) reports success but "
        "the configured local runtime is not usable. See the warnings "
        "printed above; the heal is NOT considered successful. (#81421)"
    )


def _smoke_import_hindsight_local() -> list:
    """Validate the Hindsight local runtime is usable; returns a list of
    failure descriptions (empty on success).

    Runs the probe in a clean subprocess so cached modules from the
    updating process cannot mask the newly installed environment (#81421).
    The subprocess exits 0 with JSON ``{"errors": [...]}`` on stdout;
    a non-zero exit is itself treated as a failure (the script crashed
    before it could report).

    The probe covers more than bare imports: the #81421 regression is
    "pip says ok but the daemon crashes with Unknown embeddings provider:
    onnx", which means a fully importable API surface can still expose a
    runtime that rejects the configured embeddings provider.  We import
    the slim API backend (``hindsight_api``), the embed manager
    (``hindsight_embed``), and confirm the configured local embeddings
    class (``hindsight_api.LocalSTEmbeddings`` — the SentenceTransformers /
    ONNX provider that backs ``local_embedded``) resolves.  We do not
    instantiate it (that triggers a model download we can't do in CI);
    resolving the class is enough to detect the backtrack failure mode.
    """
    import json as _json
    import subprocess as _sp
    import sys as _sys

    script = (
        "import json\n"
        "errors = []\n"
        "api_module = None\n"
        "for mod in ('hindsight_api', 'hindsight_embed'):\n"
        "    try:\n"
        "        m = __import__(mod)\n"
        "    except Exception as e:\n"
        "        errors.append(f'{mod}: {type(e).__name__}: {e}')\n"
        "        continue\n"
        "    if mod == 'hindsight_api':\n"
        "        api_module = m\n"
        "if api_module is not None and not hasattr(api_module, 'LocalSTEmbeddings'):\n"
        "    errors.append(\n"
        "        'hindsight_api.LocalSTEmbeddings: AttributeError: the slim runtime '\n"
        "        'shipped an API release that does not expose the configured ONNX '\n"
        "        'embeddings provider — this is the #81421 backtrack failure mode'\n"
        "    )\n"
        "print(json.dumps({'errors': errors}))\n"
    )
    try:
        completed = _sp.run(
            [_sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as e:
        return [f"subprocess invocation failed: {type(e).__name__}: {e}"]
    if completed.returncode != 0:
        stderr = (completed.stderr or "")[:200].strip()
        return [f"smoke subprocess exited {completed.returncode}: {stderr or '<no stderr>'}"]
    try:
        payload = _json.loads(completed.stdout)
    except Exception as e:
        return [f"smoke subprocess output not parseable JSON: {type(e).__name__}: {e}"]
    if not isinstance(payload, dict) or not isinstance(payload.get("errors"), list):
        return ["smoke subprocess output shape unexpected"]
    return [str(err) for err in payload["errors"]]


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

    ``save_env_value`` applies the shared gate (name regex, ``LD_PRELOAD``/``PYTHONPATH``/``HERMES_HOME``
    denylist, CR/LF stripping, atomic 0o600 writes). ``ValueError`` is reported and skipped so one
    bad key doesn't sink the batch; filesystem errors propagate. ``hermes_home`` is applied via the
    context-local override, not ``os.environ``.
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
