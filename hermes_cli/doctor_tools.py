"""External-tool checks for hermes doctor: terminal backends, git/rg, Node + agent-browser, npm audit, tool availability.
Split out of ``hermes_cli/doctor.py``, which re-exports every name so ``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from hermes_cli.doctor_platform import _system_package_install_cmd
from hermes_cli.doctor_report import Finding, _fail_and_issue, check_bool, check_info, check_ok, check_warn, doctor_check
from hermes_cli.vercel_auth import describe_vercel_auth
from hermes_constants import agent_browser_runnable, is_termux as _is_termux


def _safe_which(cmd: str) -> str | None:
    """shutil.which wrapper resilient to platform monkeypatching in tests."""
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def _run_ok(cmd: list[str], timeout: int, **kw) -> bool:
    """True when *cmd* exits 0 within *timeout*; a timeout counts as failure."""
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, **kw).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _termux_browser_setup_steps(node_installed: bool) -> list[str]:
    steps = [] if node_installed else ["pkg install nodejs"]
    steps += ["npm install -g agent-browser", "agent-browser install"]
    return [f"{i}) {step}" for i, step in enumerate(steps, 1)]


_TERMUX_INSTALL_ALL_FALLBACK_NOTES = (
    "Termux install profile: use .[termux-all] for broad compatibility (installer default on Termux).",
    "Matrix E2EE extra is excluded on Termux (python-olm currently fails to build).",
    "Local faster-whisper extra is excluded on Termux (ctranslate2/av build path unavailable).",
    "STT fallback: use Groq Whisper (set GROQ_API_KEY) or OpenAI Whisper (set VOICE_TOOLS_OPENAI_KEY).",
)


def _is_kanban_worker_env_gate(item: dict) -> bool:
    """Return True when Kanban is unavailable only because this is not a worker process."""
    tools = item.get("tools") or []
    return (item.get("name") == "kanban" and not os.environ.get("HERMES_KANBAN_TASK")
            and bool(tools) and all(str(tool).startswith("kanban_") for tool in tools))


def _doctor_tool_availability_detail(toolset: str) -> str:
    """Optional explanatory suffix for toolsets whose doctor status needs context."""
    if toolset == "kanban" and not os.environ.get("HERMES_KANBAN_TASK"):
        return "(runtime-gated; loaded only for dispatcher-spawned workers)"
    return ""


def _doctor_web_capability_rows() -> list[tuple[str, str, str]]:
    """Return ``(status, label, detail)`` rows (status ``ok``/``warn``) for web search/extract readiness.

    Uses the same active-provider resolvers as the tools but reports ``is_available()``
    readiness, so an explicitly selected but unconfigured backend does not look healthy.

    See #78412.
    """
    rows: list[tuple[str, str, str]] = []
    try:
        from agent.web_search_registry import get_active_extract_provider, get_active_search_provider
        from tools.web_tools import _ensure_web_plugins_loaded, _provider_is_ready
        # Fresh process: bundled web providers only register during plugin discovery (idempotent, cheap).
        _ensure_web_plugins_loaded()
    except Exception:
        return rows
    for capability, getter in (("web search", get_active_search_provider), ("web extract", get_active_extract_provider)):
        try:
            provider = getter()
        except Exception:
            provider = None
        if provider is None:
            rows.append(("warn", capability, "(no provider selected or registered)"))
            continue
        name = getattr(provider, "name", None) or type(provider).__name__
        rows.append(("ok", capability, f"({name})") if _provider_is_ready(provider)
                    else ("warn", capability, f"({name} selected; provider not configured)"))
    return rows


def _apply_doctor_tool_availability_overrides(available: list[str], unavailable: list[dict]) -> tuple[list[str], list[dict]]:
    """Adjust runtime-gated tool availability for doctor diagnostics."""
    from hermes_cli.doctor_state import _honcho_is_configured_for_doctor
    updated_available, updated_unavailable = list(available), []
    for item in unavailable:
        if _is_kanban_worker_env_gate(item):
            gated = "kanban"
        elif item.get("name") == "honcho" and _honcho_is_configured_for_doctor():
            gated = "honcho"
        else:
            updated_unavailable.append(item)
            continue
        if gated not in updated_available:
            updated_available.append(gated)
    return updated_available, updated_unavailable


def _enabled_cli_toolsets_for_doctor() -> set[str] | None:
    """Return toolsets enabled for the CLI, or None if config resolution fails."""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools
        return {str(toolset) for toolset in _get_platform_tools(load_config() or {}, "cli")}
    except Exception:
        return None


def _missing_api_key_toolsets_for_summary(unavailable: list[dict]) -> list[dict]:
    """Filter unavailable API-key toolsets to those enabled for the CLI."""
    api_key_unavailable = [item for item in unavailable if item.get("missing_vars") or item.get("env_vars")]
    enabled_toolsets = _enabled_cli_toolsets_for_doctor()
    return api_key_unavailable if enabled_toolsets is None else [i for i in api_key_unavailable if str(i.get("name") or "") in enabled_toolsets]


@doctor_check()
def _check_git_and_rg(should_fix: bool, f: Finding) -> None:
    check_bool(_safe_which("git"), "git", ("git not found", "(optional)"))
    if not check_bool(_safe_which("rg"), ("ripgrep (rg)", "(faster file search)"),
                      ("ripgrep (rg) not found", "(file search uses grep fallback)")):
        check_info(f"Install for faster search: {_system_package_install_cmd('ripgrep')}")


_BUILTIN_TERMINAL_BACKENDS = {"local", "docker", "singularity", "modal", "managed_modal", "daytona", "vercel_sandbox", "ssh"}


def _check_docker_backend(terminal_env: str, running_in_container: bool, issues: list[str]) -> None:
    if terminal_env == "docker":
        if not _safe_which("docker"):
            _fail_and_issue("docker not found", "(required for TERMINAL_ENV=docker)", "Install Docker or change TERMINAL_ENV", issues)
        else:
            # `docker version` hits /version, which socket proxies (tecnativa) allow by default; `docker info`
            # needs /info and is commonly blocked, giving a false "daemon not running". The backend itself
            # probes with `docker version` too (environments/docker.py).
            _require(_run_ok(["docker", "version"], timeout=10), ("docker", "(daemon running)"), ("docker daemon not running", ""),
                     "Start Docker daemon", issues)
    elif _safe_which("docker"):
        check_ok("docker", "(optional)")
    elif _is_termux():
        check_info("Docker backend is not available inside Termux (expected on Android)")
    elif not running_in_container:  # in-container case already explained by the caller
        check_warn("docker not found", "(optional)")


def _check_ssh_backend(issues: list[str]) -> None:
    ssh_host = os.getenv("TERMINAL_SSH_HOST")
    if not ssh_host:
        return _fail_and_issue("TERMINAL_SSH_HOST not set", "(required for TERMINAL_ENV=ssh)", "Set TERMINAL_SSH_HOST in .env", issues)
    ssh_user, ssh_port, ssh_key = (os.getenv(f"TERMINAL_SSH_{k}") for k in ("USER", "PORT", "KEY"))
    cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
    if ssh_port:
        cmd += ["-p", ssh_port]
    if ssh_key:
        cmd += ["-i", os.path.expanduser(ssh_key)]
    cmd += [f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host, "echo ok"]
    _require(_run_ok(cmd, timeout=15, text=True, encoding='utf-8', errors='replace'),
             f"SSH connection to {ssh_host}", (f"SSH connection to {ssh_host}", ""), f"Check SSH configuration for {ssh_host}", issues)


def _require(cond, ok, bad, issue: str, issues: list[str]) -> None:
    """``check_ok(*ok)`` when *cond*, else ``check_fail(*bad)`` and record *issue*."""
    if not check_bool(cond, ok, bad, fail=True):
        issues.append(issue)


def _check_daytona_backend(issues: list[str]) -> None:
    _require(os.getenv("DAYTONA_API_KEY"), ("Daytona API key", "(configured)"),
             ("DAYTONA_API_KEY not set", "(required for TERMINAL_ENV=daytona)"), "Set DAYTONA_API_KEY environment variable", issues)
    try:
        from daytona import Daytona  # noqa: F401 — SDK presence check
        check_ok("daytona SDK", "(installed)")
    except ImportError:
        _fail_and_issue("daytona SDK not installed", "(pip install daytona)", "Install daytona SDK: pip install daytona", issues)


def _check_vercel_backend(issues: list[str]) -> None:
    from tools.terminal_tool_backends import _SUPPORTED_VERCEL_RUNTIMES
    runtime = os.getenv("TERMINAL_VERCEL_RUNTIME", "node24").strip() or "node24"
    supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
    _require(runtime in _SUPPORTED_VERCEL_RUNTIMES, ("Vercel runtime", f"({runtime})"),
             ("Vercel runtime unsupported", f"({runtime}; use {supported})"), f"Set TERMINAL_VERCEL_RUNTIME to one of: {supported}", issues)
    _require(os.getenv("TERMINAL_CONTAINER_DISK", "51200").strip() in {"", "0", "51200"},
             ("Vercel disk setting", "(uses platform default)"), ("Vercel custom disk unsupported", "(reset terminal.container_disk to 51200)"),
             "Vercel Sandbox does not support custom container_disk; use the shared default 51200", issues)
    _require(importlib.util.find_spec("vercel") is not None, ("vercel SDK", "(installed)"),
             ("vercel SDK not installed", "(pip install 'hermes-agent[vercel]')"),
             "Install the Vercel optional dependency: pip install 'hermes-agent[vercel]'", issues)
    auth_status = describe_vercel_auth()
    if auth_status.ok:
        check_ok("Vercel auth", f"({auth_status.label})")
    elif auth_status.label.startswith("partial"):
        _fail_and_issue("Vercel auth incomplete", f"({auth_status.label})", "Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID together", issues)
    else:
        _fail_and_issue("Vercel auth not configured", f"({auth_status.label})", "Configure Vercel Sandbox auth with VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID", issues)
    for line in auth_status.detail_lines:
        check_info(f"Vercel auth {line}")
    persistent = os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"1", "true", "yes", "on"}
    check_info("Vercel persistence: snapshot filesystem only; live processes do not survive sandbox recreation"
               if persistent else "Vercel persistence: ephemeral filesystem")


def _check_plugin_backend(terminal_env: str, issues: list[str]) -> None:
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        from agent.terminal_env_registry import get_provider
        provider = get_provider(terminal_env)
    except Exception:
        provider = None
    if provider is None:
        return _fail_and_issue(f"Unknown terminal backend '{terminal_env}'", "(no built-in or plugin backend by that name)",
                               "Fix terminal.backend in config.yaml, or install/enable the plugin that provides it", issues)
    for ok, label, detail in provider.doctor_checks():
        _require(ok, (label, detail), (label, detail), detail.strip("()"), issues)


_BACKEND_CHECKS = {"ssh": _check_ssh_backend, "daytona": _check_daytona_backend, "vercel_sandbox": _check_vercel_backend}


@doctor_check()
def _check_terminal_backend(should_fix: bool, f: Finding) -> None:
    """Docker/SSH/Daytona/Vercel/plugin terminal backends, gated on TERMINAL_ENV."""
    terminal_env = os.getenv("TERMINAL_ENV", "local")
    try:
        from hermes_constants import is_container as _is_container
        running_in_container = _is_container()
    except Exception:
        running_in_container = False
    # In our container docker-in-docker isn't set up, so local is intended: skip the noisy "docker not found"
    # warning. An explicit TERMINAL_ENV=docker (mounted docker.sock) still gets checked.
    if running_in_container and terminal_env != "docker":
        check_info("Running inside a container — using local terminal backend (docker-in-docker is not configured by default)")
        terminal_env = "local"
    _check_docker_backend(terminal_env, running_in_container, f.issues)
    if terminal_env in _BACKEND_CHECKS:
        _BACKEND_CHECKS[terminal_env](f.issues)
    elif terminal_env not in _BUILTIN_TERMINAL_BACKENDS:
        _check_plugin_backend(terminal_env, f.issues)


def _check_agent_browser(should_fix: bool) -> bool:
    """agent-browser resolution; returns True when browser tools will find a usable install.

    Mirrors ``tools.browser_tool_install._find_agent_browser``'s own cascade (lazy npx or a global/Hermes-managed
    install) so doctor can't diverge from the tools; validate=False keeps it a cheap, side-effect-free check.
    """
    try:
        # agent-browser is no longer a root package.json dependency (#43564) — it resolves lazily via npx
        # (or a global/Hermes-managed install) at first use.
        from tools.browser_tool_install import _find_agent_browser, _is_npx_agent_browser_sentinel
        resolved = _find_agent_browser(validate=False)
    except Exception:
        resolved = None
    if resolved and _is_npx_agent_browser_sentinel(resolved):
        check_ok("agent-browser", "(resolves via npx on first use)")
        if should_fix:
            # Can't tell whether npx's cache is warm — fire the same warm-up `hermes update` does.
            from tools.browser_tool_install import warm_agent_browser_npx_cache
            check_info("  Warmed npx cache for agent-browser" if warm_agent_browser_npx_cache()
                       else "  Could not warm npx cache (offline or npx unavailable)")
        return True
    if resolved and agent_browser_runnable(resolved):
        check_ok("agent-browser", "(browser automation)")
        return True
    if resolved:
        # Almost always a dangling global symlink left by npm postinstall after `hermes update` wiped node_modules.
        check_warn("agent-browser found but not runnable", f"(broken symlink at {resolved}? run: npx agent-browser --version)")
    elif _is_termux():
        _termux_browser_hints("agent-browser is not installed (expected in the tested Termux path)",
                              "Install it manually later with: npm install -g agent-browser && agent-browser install", node_installed=True)
    else:
        check_warn("agent-browser not installed", "(requires npm/npx on PATH)")
    return False


def _termux_browser_hints(*lines: str, node_installed: bool) -> None:
    for line in lines:
        check_info(line)
    check_info("Termux browser setup:")
    for step in _termux_browser_setup_steps(node_installed=node_installed):
        check_info(step)


def _check_chromium() -> None:
    """Playwright Chromium presence, using the exact predicate browser_tool uses to hide browser_* tools.

    Lazy import: browser_tool is ~150KB; an import failure is a separate bug surfaced elsewhere. Camofox, a
    CDP override, a cloud provider, or Lightpanda all bypass the local Chromium requirement (no warning).
    """
    from hermes_cli.doctor import PROJECT_ROOT
    try:
        from tools.browser_tool import _is_camofox_mode
        from tools.browser_tool_cloud import _get_cloud_provider
        from tools.browser_tool_cdp import _get_cdp_override_raw
        from tools.browser_tool_install import _chromium_installed
        from tools.browser_tool_lightpanda_fallback import _using_lightpanda_engine
    except Exception:
        return
    if _is_camofox_mode() or bool(_get_cdp_override_raw()) or _get_cloud_provider() is not None or _using_lightpanda_engine():
        return
    if not check_bool(_chromium_installed(), ("Playwright Chromium", "(browser engine)"),
                      ("Playwright Chromium not installed", "(browser_* tools will be hidden from the agent)")):
        with_deps = "" if sys.platform == "win32" else "--with-deps "
        check_info(f"Install with: cd {PROJECT_ROOT} && npx playwright install {with_deps}chromium")


def _check_lightpanda() -> None:
    """Lightpanda engine (browser.engine / AGENT_BROWSER_ENGINE); independent of Node since Browser Use mode spawns ``lightpanda serve`` itself."""
    try:
        from tools.browser_tool_lightpanda_fallback import _using_lightpanda_engine, lightpanda_engine_status
        from tools.browser_lightpanda import LIGHTPANDA_INSTALL_HINT, find_lightpanda_binary
    except Exception:
        return
    # _using_lightpanda_engine() is a cached config read — a failure there is exceptional, not hidden.
    if not _using_lightpanda_engine():
        return
    try:
        used, reason = lightpanda_engine_status()
    except Exception as e:
        used, reason = False, f"status check failed: {e}"
    if not used:
        check_warn("browser.engine=lightpanda is shadowed", f"({reason})")
        check_info("Fix: pick Lightpanda in `hermes tools` → Browser Automation, or set browser.engine: auto")
    elif not check_bool(find_lightpanda_binary(), ("Lightpanda", f"({reason})"),
                        ("Lightpanda selected but binary not found", "(browser tools will fail until it is installed)")):
        check_info(LIGHTPANDA_INSTALL_HINT)


@doctor_check()
def _check_node_and_browser(should_fix: bool, f: Finding) -> None:
    """Node.js, agent-browser resolution, Playwright Chromium, Lightpanda engine."""
    if _safe_which("node"):
        check_ok("Node.js")
        if _check_agent_browser(should_fix) and not _is_termux():  # Chromium check is not a tested Termux path
            _check_chromium()
    elif _is_termux():
        _termux_browser_hints("Node.js not found (browser tools are optional in the tested Termux path)",
                              "Install Node.js on Termux with: pkg install nodejs", node_installed=False)
    else:
        check_warn("Node.js not found", "(optional, needed for browser tools)")
    _check_lightpanda()


def _plural(n: int) -> str:
    return "vulnerability" if n == 1 else "vulnerabilities"


def _audit_one(npm_bin: str, npm_dir, label: str, audit_extra: list[str], issues: list[str]) -> None:
    """Run one `npm audit --json` and report; any failure is silently skipped.

    Workspace-scoped (`--workspace <name>`) advisories are build-time tooling (esbuild/vite), not runtime
    code. `npm audit fix --workspace` crashes on current npm (arborist "edgesOut") and the root-level fix can
    crash on the same tree ("isDescendantOf"), so no manual fix command is offered — they clear via a lockfile bump.
    """
    import json
    try:
        # Resolved absolute path so Windows can execute npm.cmd (CreateProcessW can't run bare .cmd names).
        audit_result = subprocess.run([npm_bin, "audit", "--json", *audit_extra], cwd=str(npm_dir),
                                      capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
        audit_data = json.loads(audit_result.stdout) if audit_result.stdout.strip() else {}
        counts = audit_data.get("metadata", {}).get("vulnerabilities", {})
        critical, high, moderate = (counts.get(k, 0) for k in ("critical", "high", "moderate"))
        total = critical + high + moderate
        workspace_scoped = bool(audit_extra) and audit_extra[0] == "--workspace"
        if total == 0:
            check_ok(f"{label} deps", "(no known vulnerabilities)")
        elif critical > 0 or high > 0:
            flag = " --workspaces=false" if audit_extra == ["--workspaces=false"] else ""
            remedy = "build-tool advisory; clears via lockfile bump" if workspace_scoped else f"run: cd {npm_dir} && npm audit fix{flag}"
            check_warn(f"{label} deps", f"({critical} critical, {high} high, {moderate} moderate — {remedy})")
            if workspace_scoped:
                check_info("  ^ build-time tooling (not runtime); if manual npm remediation "
                           "errors with an arborist crash it's a known npm bug — clears via a lockfile bump")
            issues.append(f"{label} has {total} npm {_plural(total)}")
        else:
            check_ok(f"{label} deps", f"({moderate} moderate {_plural(moderate)})")
    except Exception:
        pass


@doctor_check()
def _check_npm_audit(should_fix: bool, f: Finding) -> None:
    """npm audit per Node package tree (root, web/ui-tui workspaces, WhatsApp bridge).

    PROJECT_ROOT is audited with --workspaces=false so the apps/* glob (Electron, node-pty, ...) is never
    resolved for a routine check; web and ui-tui via --workspace. The WhatsApp bridge may live under a writable
    HERMES_HOME mirror rather than the (possibly read-only) Docker install tree, hence the shared resolver.
    """
    from hermes_cli.doctor import PROJECT_ROOT
    npm_bin = _safe_which("npm")
    if npm_bin:
        try:
            # Each entry: (cwd, label, extra_audit_args) PROJECT_ROOT is audited with --workspaces=false so
            # that the apps/* glob (which pulls in Electron, node-pty, etc.) is never resolved for a routine
            # security check. The web and ui-tui workspaces are audited separately via --workspace flags.
            # See #38772. The WhatsApp bridge may live under a writable HERMES_HOME mirror instead of the
            # (possibly read-only) install tree in Docker — resolve it through the shared helper so we audit
            # the dir that actually holds node_modules. See #49561.
            from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
            whatsapp_bridge_dir = resolve_whatsapp_bridge_dir()
        except Exception:
            whatsapp_bridge_dir = PROJECT_ROOT / "scripts" / "whatsapp-bridge"
        for npm_dir, label, audit_extra in (
            (PROJECT_ROOT, "Browser tools (agent-browser)", ["--workspaces=false"]),
            (PROJECT_ROOT, "web workspace", ["--workspace", "web"]),
            (PROJECT_ROOT, "ui-tui workspace", ["--workspace", "ui-tui"]),
            (whatsapp_bridge_dir, "WhatsApp bridge", []),
        ):
            # Workspace-scoped audits check the root node_modules; standalone dirs check their own.
            if ((PROJECT_ROOT if audit_extra else npm_dir) / "node_modules").exists():
                _audit_one(npm_bin, npm_dir, label, audit_extra, f.issues)
    if _is_termux():
        check_info("Termux compatibility fallbacks:")
        for note in _TERMUX_INSTALL_ALL_FALLBACK_NOTES:
            check_info(note)


@doctor_check("Could not check tool availability", "({e})")
def _check_tool_availability(should_fix: bool, f: Finding) -> None:
    from hermes_cli.doctor import PROJECT_ROOT
    sys.path.insert(0, str(PROJECT_ROOT))
    from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
    available, unavailable = _apply_doctor_tool_availability_overrides(*check_tool_availability())
    # Web is split into search/extract readiness rows so an explicitly
    # selected but unconfigured backend cannot look healthy.
    web_rows = []
    # See #78412.
    if "web" in available or any(item.get("name") == "web" for item in unavailable):
        web_rows = _doctor_web_capability_rows()
        if web_rows:
            available = [tid for tid in available if tid != "web"]
            unavailable = [item for item in unavailable if item.get("name") != "web"]
    for tid in available:
        check_ok(TOOLSET_REQUIREMENTS.get(tid, {}).get("name", tid), _doctor_tool_availability_detail(tid))
    for status, label, detail in web_rows:
        (check_ok if status == "ok" else check_warn)(label, detail)
    for item in unavailable:
        env_vars = item.get("missing_vars") or item.get("env_vars") or []
        check_warn(item["name"], f"(missing {', '.join(env_vars)})" if env_vars else "(system dependency not met)")
    # Only toolsets enabled for the CLI count toward the summary; default-off or
    # disabled toolsets may warn above but must not pollute it.
    api_disabled = _missing_api_key_toolsets_for_summary(unavailable)
    if api_disabled or any(status != "ok" for status, _, _ in web_rows):
        f.issues.append("Run 'hermes setup' to configure missing API keys for full tool access")
