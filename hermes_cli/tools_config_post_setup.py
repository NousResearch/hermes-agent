"""Post-setup install hooks and installed-state predicates for `hermes tools` provider rows."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Set

from hermes_cli.cli_output import (
    print_error as _print_error, print_info as _print_info, print_success as _print_success,
    print_warning as _print_warning)
from hermes_cli.config import get_env_value
from hermes_cli.tools_config_cua import (
    _cua_driver_install_ready, _pip_install, _post_setup_no_window_flags, _run_text, install_cua_driver,
)

logger = logging.getLogger("hermes_cli.tools_config")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _info_lines(*lines: str) -> None:
    """Print each line as a 4-space-indented info row."""
    for line in lines:
        _print_info(f"    {line}")


def _ensure_browser_use_cli(*, verbose_hints: bool = False) -> None:
    """Install the Browser Use CLI if it isn't already runnable.
    Primary driver engine for EVERY browser backend except Camofox (Firefox-based, no CDP surface).
    MANAGED-FIRST: a browser-use on the user's PATH does NOT satisfy this check — only the
    Hermes-managed ``$HERMES_HOME/bin`` copy does."""
    _print_info("    Ensuring browser-use CLI (managed install)...")
    try:
        from tools.browser_use_cli import install_cli
        ok, message = install_cli()
    except Exception as exc:  # pragma: no cover — defensive
        ok, message = False, f"install failed: {exc}"
    if ok:
        _print_success(f"    {message}")
    else:
        for line in str(message).splitlines():
            _print_warning(f"    {line[:200]}")
        _print_info("    Falling back to zero-install runs via `uvx browser-use`" if shutil.which("uvx")
                    else "    Install manually: uv tool install browser-use  (https://docs.astral.sh/uv/)")
    if verbose_hints:
        _info_lines("Local Chrome needs remote debugging: chrome://inspect/#remote-debugging",
                    "Cloud browsers: browser-use auth login  (or set BROWSER_USE_API_KEY)")


def _post_setup_lightpanda() -> None:
    # Browser Use mode spawns ``lightpanda serve``; built-in tools go through agent-browser. No Chromium needed.
    _ensure_browser_use_cli()
    from tools.browser_lightpanda import LIGHTPANDA_INSTALL_HINT, find_lightpanda_binary

    lightpanda_bin = find_lightpanda_binary()
    if lightpanda_bin:
        _print_success(f"    Lightpanda found: {lightpanda_bin}")
    else:
        _print_warning("    lightpanda binary not found on PATH, ~/.lightpanda or ~/.local/bin")
        _print_info(f"    {LIGHTPANDA_INSTALL_HINT}")
        if os.name == "nt":
            _print_info("    Lightpanda has no native Windows build; run Hermes under WSL2.")


def _install_chromium(install_cmd: list[str]) -> None:
    """Run the agent-browser Chromium install command and report the outcome."""
    _print_info("    Installing Chromium (~170MB one-time download)...")
    try:
        result = _run_text(install_cmd, cwd=str(PROJECT_ROOT), timeout=600, creationflags=_post_setup_no_window_flags())
        if result.returncode == 0:
            _print_success("    Chromium installed")
            # Invalidate the cached "missing" flag so later check_browser_requirements() calls see the install.
            import tools.browser_tool as _bt
            _bt._cached_chromium_installed = None
            return
        _print_warning("    Chromium install failed:")
        for line in (result.stderr or result.stdout or "").strip().splitlines()[-3:]:
            _print_info(f"      {line[:200]}")
    except subprocess.TimeoutExpired:
        _print_warning("    Chromium install timed out (>10min)")
    except Exception as exc:
        _print_warning(f"    Chromium install failed: {exc}")
    _print_info("    Run manually: npx agent-browser install --with-deps")


def _post_setup_agent_browser(post_setup_key: str) -> None:
    """``agent_browser`` (local Chromium) and ``browserbase`` (cloud rows) hooks.
    agent-browser is not a root package.json dependency — it resolves lazily via npx (or a
    global/Hermes-managed install), so there is no ``npm install`` step here."""
    # Every non-Camofox backend drives through the Browser Use CLI — install it here too.
    _ensure_browser_use_cli()
    try:
        # Lazy import so the tools_config UI doesn't pull in browser_tool at import time.
        # agent-browser resolves lazily via npx on the default install (#43564), invisible to the
        # PATH/node_modules probes above. Mirror the rung hermes_cli.doctor uses so this probe can't diverge
        # from it, including the Termux carve-out (bare npx is too fragile to advertise as ready there — see
        # check_browser_requirements).
        # agent-browser is no longer a root package.json dependency (#43564) — it resolves lazily via npx
        # for most installs, which a bare PATH + node_modules probe can't see. Mirror the local-CLI tail of
        # :func:`tools.browser_tool_install.check_browser_requirements` (same cascade, same Termux carve-out) so the
        # setup/status surfaces can't diverge from what browser tools actually find at runtime;
        # validate=False keeps this a cheap existence check with no subprocess spawn.
        # agent-browser is no longer a root package.json dependency (#43564) — it resolves lazily via npx
        # (or a global/Hermes-managed install) instead of a local `npm install`, so there's no node_modules/
        # population step here anymore.
        from tools.browser_tool import AGENT_BROWSER_NPX_SPEC
        from tools.browser_tool_install import (
            _chromium_installed, _running_in_docker, _find_agent_browser, _resolve_npx_bin,
            _is_npx_agent_browser_sentinel)
    except Exception as exc:  # pragma: no cover — defensive
        _print_warning(f"    Could not check Chromium status: {exc}")
        return

    # Reuse the runtime resolution cascade (PATH -> Homebrew/Hermes-managed node -> npx) rather than
    # a bare shutil.which — Hermes-managed-Node-only setups resolve agent-browser/npx only that way.
    try:
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        _print_warning("    npx not found - browser tools require Node.js: https://nodejs.org")
        return

    # Only the local provider needs Chromium on disk; cloud providers host their own.
    if post_setup_key != "agent_browser":
        return

    # Without Chromium the CLI hangs on first use until the command timeout fires. Skip inside
    # Docker — the image bakes Chromium in, and runtime users usually can't write PLAYWRIGHT_BROWSERS_PATH.
    if _chromium_installed():
        _print_success("    Chromium browser already installed, nothing to do")
        return

    if _running_in_docker():
        _print_warning("    Chromium is missing but you're running in Docker.")
        _info_lines("Pull the latest image to get the bundled Chromium:",
                    "  docker pull ghcr.io/nousresearch/hermes-agent:latest")
        return

    if _is_npx_agent_browser_sentinel(browser_cmd):
        # Re-resolve npx via the same cascade _find_agent_browser used — a bare shutil.which("npx")
        # would silently diverge and hand subprocess.run a None argument.
        npx_bin = _resolve_npx_bin()
        if not npx_bin:
            _print_warning("    npx not found - install Chromium manually: npx agent-browser install --with-deps")
            return
        install_cmd = [npx_bin, "--ignore-scripts", "-y", AGENT_BROWSER_NPX_SPEC, "install", "--with-deps"]
    else:
        install_cmd = [browser_cmd, "install", "--with-deps"]
    _install_chromium(install_cmd)


def _post_setup_camofox() -> None:
    from hermes_constants import find_node_executable

    camofox_dir = PROJECT_ROOT / "node_modules" / "@askjo" / "camofox-browser"
    _npm_bin = find_node_executable("npm")
    if camofox_dir.exists():
        _print_success("    Camofox already installed, nothing to do")
    elif _npm_bin:
        _print_info("    Installing Camofox browser server...")
        # Absolute npm path so the .cmd shim executes on Windows; --workspaces=false avoids resolving apps/desktop.
        result = _run_text([_npm_bin, "install", "--silent", "--workspaces=false"], timeout=None,
                           cwd=str(PROJECT_ROOT), creationflags=_post_setup_no_window_flags())
        if result.returncode == 0:
            _print_success("    Camofox installed")
        else:
            _print_warning("    npm install failed - run manually: npm install --workspaces=false")
    if camofox_dir.exists():
        _info_lines("Start the Camofox server:", "  npx @askjo/camofox-browser",
                    "First run downloads the Camoufox engine (~300MB)",
                    "Or use Docker: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")
    elif not _npm_bin:
        _print_warning("    Node.js not found. Install Camofox via Docker:")
        _print_info("      docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")


_KITTENTTS_WHEEL_URL = "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl"

# pip-only post-setup hooks: module (import probe), label, installing (progress line), args, manual
# (fallback command), on_install (fresh-install notes), always. Also feeds _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.
def _pip_hook(module, label, installing, args, manual, on_install=(), always=()) -> dict:
    return {"module": module, "label": label, "installing": installing, "args": args, "manual": manual,
            "on_install": on_install, "always": always}


_PIP_POST_SETUP_HOOKS: dict = {
    "faster_whisper": _pip_hook(
        "faster_whisper", "faster-whisper", "Installing faster-whisper (model ~150MB downloads on first use)...",
        ["-U", "faster-whisper", "--quiet"], "uv pip install -U faster-whisper",
        on_install=("Model sizes: tiny, base (default), small, medium, large-v3",
                    "Change via stt.local.model in ~/.hermes/config.yaml")),
    "kittentts": _pip_hook(
        "kittentts", "kittentts", "Installing kittentts (~25-80MB model, CPU-only)...",
        ["-U", _KITTENTTS_WHEEL_URL, "soundfile", "--quiet"], f"uv pip install -U '{_KITTENTTS_WHEEL_URL}' soundfile",
        on_install=("Voices: Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo",
                    "Models: KittenML/kitten-tts-nano-0.8-int8 (25MB), micro (41MB), mini (80MB)")),
    "piper": _pip_hook(
        "piper", "piper-tts", "Installing piper-tts (~14MB wheel, voices downloaded on first use)...",
        ["-U", "piper-tts", "--quiet"], "uv pip install -U piper-tts",
        always=("Default voice: en_US-lessac-medium (downloaded on first TTS call)",
                "Full voice list: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md",
                "Switch voices by setting tts.piper.voice in ~/.hermes/config.yaml")),
    "ddgs": _pip_hook(
        "ddgs", "ddgs", "Installing ddgs (DuckDuckGo search package)...", ["-U", "ddgs", "--quiet"],
        "uv pip install -U ddgs",
        always=("No API key required. DuckDuckGo enforces server-side rate limits.",
                "Pair with an extract provider if you also need web_extract."))}


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _post_setup_pip(spec: dict) -> None:
    """Run one ``_PIP_POST_SETUP_HOOKS`` entry."""
    label = spec["label"]
    lines = list(spec["always"])
    if _importable(spec["module"]):
        _print_success(f"    {label} is already installed")
    else:
        _print_info(f"    {spec['installing']}")
        try:
            result = _pip_install(spec["args"], timeout=300)
        except subprocess.TimeoutExpired:
            _print_warning(f"    {label} install timed out (>5min)")
            _info_lines(f"Run manually: {spec['manual']}")
            return
        if result.returncode != 0:
            _print_warning(f"    {label} install failed:")
            _info_lines(f"  {(result.stderr or '').strip()[:300]}", f"Run manually: {spec['manual']}")
            return
        _print_success(f"    {label} installed")
        lines = list(spec["on_install"]) + lines
    _info_lines(*lines)


def _post_setup_spotify() -> None:
    # Full `hermes auth spotify` flow: no client_id yet → interactive wizard (persists to ~/.hermes/.env)
    # then PKCE; existing app → OAuth only.
    from types import SimpleNamespace
    try:
        from hermes_cli.auth import login_spotify_command
    except Exception as exc:
        _print_warning(f"    Could not load Spotify auth: {exc}")
        _info_lines("Run manually: hermes auth spotify")
        return
    _print_info("    Starting Spotify login...")
    try:
        login_spotify_command(SimpleNamespace(
            client_id=None, redirect_uri=None, scope=None, no_browser=False, timeout=None))
        _print_success("    Spotify authenticated")
    except SystemExit as exc:
        # User aborted the wizard or OAuth failed — don't fail the toolset enable.
        _print_warning(f"    Spotify login did not complete: {exc}")
        _info_lines("Run later: hermes auth spotify")
    except Exception as exc:
        _print_warning(f"    Spotify login failed: {exc}")
        _info_lines("Run manually: hermes auth spotify")


def _post_setup_langfuse() -> None:
    if _importable("langfuse"):
        _print_success("    langfuse SDK already installed")
    else:
        _print_info("    Installing langfuse SDK...")
        result = _pip_install(["langfuse", "--quiet"], timeout=120)
        if result.returncode == 0:
            _print_success("    langfuse SDK installed")
        else:
            _print_warning("    langfuse SDK install failed — run manually: uv pip install langfuse")
    # The bundled observability/langfuse plugin is opt-in (standalone plugins don't load until enabled).
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _save_enabled_set
        enabled = _get_enabled_set()
        if "observability/langfuse" in enabled or "langfuse" in enabled:
            _print_success("    Plugin observability/langfuse already enabled")
        else:
            enabled.add("observability/langfuse")
            _save_enabled_set(enabled)
            _print_success("    Plugin observability/langfuse enabled")
    except Exception as exc:
        _print_warning(f"    Could not enable plugin automatically: {exc}")
        _info_lines("Run manually: hermes plugins enable observability/langfuse")
    _info_lines("Restart Hermes for tracing to take effect.", "Verify: hermes plugins list")


def _post_setup_xai_grok() -> None:
    """Shared xAI credential bootstrap for any picker row that talks to xAI (TTS, STT, Video Gen, x_search
    …). Accepts a SuperGrok-tier OAuth token (preferred — billed to the existing subscription) or a raw
    XAI_API_KEY; the rows declare empty env_vars so the auth UX lives here."""
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
        oauth_logged_in = bool(get_xai_oauth_auth_status().get("logged_in"))
    except Exception:
        oauth_logged_in = False
    if oauth_logged_in:
        _print_success("    xAI will use your xAI Grok OAuth (SuperGrok / Premium+) credentials")
        return
    if get_env_value("XAI_API_KEY"):
        _print_success("    xAI will use your existing XAI_API_KEY")
        return

    _print_info("    xAI needs credentials. Choose one:")
    try:
        from hermes_cli.setup import prompt_choice, prompt as _setup_prompt
        from hermes_cli.setup_tts import _run_xai_oauth_login_from_setup
        from hermes_cli.config import save_env_value
    except Exception as exc:
        _print_warning(f"    Could not load setup helpers: {exc}")
        _info_lines("Run later: hermes auth add xai-oauth   (or set XAI_API_KEY)")
        return

    idx = prompt_choice(
        "    How do you want xAI to authenticate?", default=0,
        choices=["Sign in with xAI Grok OAuth (SuperGrok / Premium+) — browser login",
                 "Paste an xAI API key (console.x.ai)",
                 "Skip — configure later via `hermes auth add xai-oauth`"])
    if idx == 0:
        if _run_xai_oauth_login_from_setup():
            _print_success("    Logged in — xAI will use these OAuth credentials")
        else:
            _print_warning("    xAI Grok OAuth login did not complete. Run later: hermes auth add xai-oauth")
    elif idx == 1:
        api_key = _setup_prompt("    xAI API key", password=True)
        if api_key:
            save_env_value("XAI_API_KEY", api_key)
            _print_success("    XAI_API_KEY saved")
        else:
            _print_warning("    No API key provided. Run later: hermes auth add xai-oauth")
    else:
        _print_info("    xAI will remain inactive until credentials are configured.")


# post_setup key -> hook. Unknown keys are a silent no-op (callers validate against valid_post_setup_keys()).
_POST_SETUP_HOOKS: dict = {
    "lightpanda": _post_setup_lightpanda,
    "agent_browser": lambda: _post_setup_agent_browser("agent_browser"),
    "browserbase": lambda: _post_setup_agent_browser("browserbase"),
    "browser_use_cli": lambda: _ensure_browser_use_cli(verbose_hints=True),
    "camofox": _post_setup_camofox,
    "cua_driver": lambda: install_cua_driver(upgrade=False),
    "spotify": _post_setup_spotify,
    "langfuse": _post_setup_langfuse,
    "xai_grok": _post_setup_xai_grok,
    **{key: (lambda spec=spec: _post_setup_pip(spec)) for key, spec in _PIP_POST_SETUP_HOOKS.items()},
}


def _run_post_setup(post_setup_key: str):
    """Run post-setup hooks for tools that need extra installation steps."""
    _POST_SETUP_HOOKS.get(post_setup_key, lambda: None)()


def valid_post_setup_keys() -> Set[str]:
    """Return the set of post-setup keys declared by any visible provider (``TOOL_CATEGORIES`` plus
    plugin-registered providers). This is the allowlist ``post-setup`` and the dashboard endpoint
    validate against, so a caller cannot drive ``_run_post_setup`` with an arbitrary key."""
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES, _plugin_browser_providers, _plugin_image_gen_providers,
        _plugin_video_gen_providers, _plugin_web_search_providers)

    keys: Set[str] = set()
    for cat in TOOL_CATEGORIES.values():
        keys.update(ps for prov in cat.get("providers", []) if (ps := prov.get("post_setup")))
    for builder in (_plugin_web_search_providers, _plugin_image_gen_providers,
                    _plugin_video_gen_providers, _plugin_browser_providers):
        try:
            keys.update(ps for prov in builder() if (ps := prov.get("post_setup")))
        except Exception:  # pragma: no cover — defensive; plugins optional
            continue
    return keys


def run_post_setup_command(args) -> int:
    """``hermes tools post-setup <key>`` — non-interactive runner the dashboard spawns so the GUI can drive
    backend setup without re-implementing install logic. Exit code: 0 ok, 2 unknown key."""
    key = getattr(args, "post_setup_key", None)
    if not key:
        _print_error("Usage: hermes tools post-setup <key>")
        return 2
    valid = valid_post_setup_keys()
    if key not in valid:
        _print_error(f"Unknown post-setup key: {key!r}. Valid keys: {', '.join(sorted(valid)) or '(none)'}")
        return 2
    _print_info(f"Running post-setup hook: {key}")
    try:
        _run_post_setup(key)
    except Exception as exc:  # pragma: no cover — defensive
        _print_error(f"Post-setup failed: {exc}")
        return 1
    _print_success(f"Post-setup '{key}' complete")
    return 0


# post_setup_key -> predicate(): True when the install side-effect is already satisfied. Used by
# `_toolset_needs_configuration_prompt` to force provider setup when a no-key provider still needs a
# binary/dependency install (otherwise toggling the toolset on silently skips the hook). Only add an
# entry when the post_setup is the ONLY install side-effect for a no-key provider and the check is
# local, bounded, and import-light.
_POST_SETUP_INSTALLED: dict = {"cua_driver": lambda: _cua_driver_install_ready()}


def _post_setup_already_installed(post_setup_key: str) -> bool:
    """Return True when the post_setup install side-effect is satisfied (or no check is registered)."""
    predicate = _POST_SETUP_INSTALLED.get(post_setup_key)
    try:
        return predicate is None or bool(predicate())
    except Exception:
        return True


def _module_installed(module_name: str) -> bool:
    """Cheap importable-without-importing check (no heavy side effects)."""
    import importlib.util
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


# Python deps installed via ``hermes tools`` aren't in the managed runtime's locked ``all`` sync, so a
# runtime replacement snapshots this static allowlist before the old site-packages disappears and
# restores it afterward. Derived from the pip hooks (minus ``--quiet``) so install args can't drift.
_RESTORABLE_PYTHON_TOOL_DEPENDENCIES: dict[str, tuple[str, tuple[str, ...]]] = {
    **{key: (spec["module"], tuple(a for a in spec["args"] if a != "--quiet"))
       for key, spec in _PIP_POST_SETUP_HOOKS.items()},
    "langfuse": ("langfuse", ("langfuse",))}


def active_restorable_python_tool_dependencies() -> list[str]:
    """Return ``hermes tools`` Python dependencies present in this runtime."""
    return [
        name for name, (module_name, _install_args) in _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.items()
        if _module_installed(module_name)]


def restorable_python_tool_dependency(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Return the import probe and pip arguments for an allowlisted tool."""
    return _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.get(name)


def _agent_browser_installed() -> bool:
    """True when everything ``_run_post_setup("agent_browser")`` installs is present: the agent-browser CLI
    *and* the Chromium build it drives (or the Lightpanda engine, which needs no Chromium), so "Run
    setup" flips to installed only when re-running it would be a no-op."""
    from hermes_cli.nous_subscription import _local_browser_runnable

    # The hook runs in a spawned process; this probe runs in the long-lived web-server/CLI process whose
    # browser_tool may have cached a stale "Chromium missing" result. Drop the cache so the pill flips to Ready.
    if (bt := sys.modules.get("tools.browser_tool")) is not None:
        bt._cached_chromium_installed = None
    return _local_browser_runnable()


def _camofox_installed() -> bool:
    """True when the Camofox npm package ``_run_post_setup("camofox")`` installs is in node_modules."""
    return (PROJECT_ROOT / "node_modules" / "@askjo" / "camofox-browser").exists()


def _lightpanda_installed() -> bool:
    """True when a lightpanda binary is on PATH or in a known install dir."""
    try:
        from tools.browser_lightpanda import find_lightpanda_binary
        return find_lightpanda_binary() is not None
    except Exception:
        return False


def _cloud_agent_browser_installed() -> bool:
    """Installed-check for the ``browserbase`` hook: cloud providers host their own Chromium, so
    presence of the agent-browser CLI is the whole contract."""
    from hermes_cli.nous_subscription import _has_agent_browser
    return _has_agent_browser()


# post_setup_key -> predicate(): True when the install side-effect is satisfied. Used by
# ``provider_readiness_status`` to mark a keyless post_setup row "ready" vs "needs_setup"; mirrors the
# installed-checks the hooks perform. ``xai_grok`` is absent — a credential bootstrap handled as an
# auth check. Late-bound lambdas so tests can monkeypatch the underlying predicates.
_POST_SETUP_READY: dict = {
    **{key: (lambda m=module: _module_installed(m)) for key, (module, _args) in _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.items()},
    "agent_browser": lambda: _agent_browser_installed(),
    "browserbase": lambda: _cloud_agent_browser_installed(),
    "camofox": lambda: _camofox_installed(),
    "lightpanda": lambda: _lightpanda_installed(),
    "cua_driver": lambda: _cua_driver_install_ready()}
