"""Terminal-backend setup wizard (local/docker/singularity/modal/daytona/vercel/ssh/plugin).
setup.py names are resolved through the module object so test patches on ``hermes_cli.setup.<name>``
take effect; setup.py re-exports the public entry points."""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from tools import tool_backend_helpers
from hermes_cli import nous_subscription

logger = logging.getLogger("hermes_cli.setup")

_SANDBOX_IMAGE = "nikolaik/python-nodejs:python3.11-nodejs20"
_RUN_KW = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")


def _prompt_vercel_sandbox_settings(config: dict):
    """Prompt for Vercel Sandbox settings without exposing unsupported disk sizing."""
    terminal = config.setdefault("terminal", {})
    _setup._info(None, "Vercel Sandbox settings:", "  Filesystem persistence uses Vercel snapshots.",
                 "  Snapshots restore files only; live processes do not continue after sandbox recreation.")
    from tools.terminal_tool_backends import _SUPPORTED_VERCEL_RUNTIMES
    current_runtime = terminal.get("vercel_runtime") or "node24"
    supported_label = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
    runtime = _setup.prompt(f"  Runtime ({supported_label})", current_runtime).strip() or current_runtime
    if runtime not in _SUPPORTED_VERCEL_RUNTIMES:
        _setup.print_warning(f"Unsupported Vercel runtime '{runtime}', keeping {current_runtime}.")
        runtime = current_runtime if current_runtime in _SUPPORTED_VERCEL_RUNTIMES else "node24"
    terminal["vercel_runtime"] = runtime
    _setup.save_env_value("TERMINAL_VERCEL_RUNTIME", runtime)
    persist_label = "yes" if terminal.get("container_persistent", True) else "no"
    persist = _setup.prompt("  Persist filesystem with snapshots? (yes/no)", persist_label).lower()
    terminal["container_persistent"] = persist in {"yes", "true", "y", "1"}
    # (key, prompt label, default, parser) — unparseable input leaves the value untouched.
    for key, label, default, parse in (
        ("container_cpu", "  CPU cores", 1, float),
        ("container_memory", "  Memory in MB (5120 = 5GB)", 5120, int)):
        try:
            terminal[key] = parse(_setup.prompt(label, str(terminal.get(key, default))))
        except ValueError:
            pass
    if terminal.get("container_disk", 51200) not in {0, 51200}:
        _setup.print_warning(
            "Vercel Sandbox does not support custom disk sizing; resetting container_disk to 51200.")
    terminal["container_disk"] = 51200
    _setup._info(None, "Vercel authentication:", "  Use a long-lived Vercel access token plus project/team IDs.")
    linked = _read_nearest_vercel_project()
    if linked:
        _setup.print_info("  Found defaults in nearest .vercel/project.json.")
    _setup.remove_env_value("VERCEL_OIDC_TOKEN")
    # (label, env var, linked-project fallback key, secret) — prompted in order, saved when non-empty.
    for label, env_var, linked_key, secret in (
        ("    Vercel access token", "VERCEL_TOKEN", None, True),
        ("    Vercel project ID", "VERCEL_PROJECT_ID", "projectId", False),
        ("    Vercel team ID", "VERCEL_TEAM_ID", "orgId", False)):
        default = _setup.get_env_value(env_var) or (linked.get(linked_key, "") if linked_key else "")
        value = _setup.prompt(label, default, password=secret)
        if value:
            _setup.save_env_value(env_var, value)


def _read_nearest_vercel_project(start: Path | None = None) -> dict[str, str]:
    """Read project/team defaults from the nearest Vercel link file."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        project_file = directory / ".vercel" / "project.json"
        if not project_file.exists():
            continue
        try:
            data = json.loads(project_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {key: data[key] for key in ("projectId", "orgId")
                if isinstance(data.get(key), str) and data[key].strip()}
    return {}


def _prompt_secret_env(label: str, env_var: str, *, confirm_msg: str = "") -> None:
    """Prompt for a secret and persist it to .env when non-empty."""
    value = _setup.prompt(label, password=True)
    if value:
        _setup.save_env_value(env_var, value)
        if confirm_msg:
            _setup.print_success(confirm_msg)


def _existing_secret_keeps(env_var: str, label: str, question: str) -> bool:
    """True when ``env_var`` is already set and the user declines to update it."""
    if not _setup.get_env_value(env_var):
        return False
    _setup.print_info(f"  {label}: already configured")
    return not _setup.prompt_yes_no(question, False)


def _pip_install_vercel(package):
    """uv when Hermes has one ($HERMES_HOME/bin is never on PATH, so which() misses it and
    bootstrapping mid-wizard is fine), else pip — a `uv venv` venv may not even have pip."""
    import subprocess
    from hermes_cli.managed_uv import ensure_uv
    uv_bin = ensure_uv()
    cmd = ([uv_bin, "pip", "install", "--python", sys.executable, package] if uv_bin
           else [sys.executable, "-m", "pip", "install", package])
    return subprocess.run(cmd, **_RUN_KW)


def _ensure_sdk(package: str, manual_hint: str, *, show_stderr: bool = False, install=None) -> None:
    """Import *package*; if missing, install it (default: the venv pip ladder)."""
    try:
        __import__(package)
    except ImportError:
        _setup.print_info(f"Installing {package} SDK...")
        if install is None:
            from hermes_cli.tools_config import _pip_install
            install = lambda pkg: _pip_install([pkg])  # noqa: E731
        result = install(package)
        if result.returncode == 0:
            _setup.print_success(f"{package} SDK installed")
        else:
            _setup.print_warning(f"Install failed — run manually: {manual_hint}")
            if show_stderr and result.stderr:
                _setup.print_info(f"  Error: {result.stderr.strip().splitlines()[-1]}")


def _report_binary(found: str | None, missing: str, install_hint: str, found_prefix: str = "Found: ") -> None:
    if found:
        _setup.print_info(f"{found_prefix}{found}")
    else:
        _setup.print_warning(missing)
        _setup.print_info(install_hint)


def _setup_backend_local(config: dict) -> None:
    _setup.print_success("Terminal backend: Local")
    _setup.print_info("Commands run directly on this machine.")
    # Gateway cwd defaults to home; sudo stays off. Both configurable via `hermes setup terminal`.
    config["terminal"].setdefault("cwd", str(Path.home()))


def _setup_backend_docker(config: dict) -> None:
    _setup.print_success("Terminal backend: Docker")
    _report_binary(shutil.which("docker"), "Docker not found in PATH!",
                   "Install Docker: https://docs.docker.com/get-docker/", "Docker found: ")
    # Image and resource limits use defaults; tune via `hermes setup terminal`.
    config["terminal"].setdefault("docker_image", _SANDBOX_IMAGE)
    _setup._info(None, "Docker sandboxes can be protected with the egress credential firewall.",
                 "It routes sandbox traffic through iron-proxy so containers receive "
                 "proxy tokens instead of real API keys.",
                 "   Docker only for now; Modal, SSH, Daytona, and Singularity are not wired yet.")
    if _setup.prompt_yes_no("  Enable egress firewall for Docker sandboxes?", False):
        proxy_cfg = config.setdefault("proxy", {})
        proxy_cfg["enabled"] = True
        proxy_cfg.setdefault("enforce_on_docker", True)
        _setup.print_success("Egress firewall enabled in config")
        _setup.print_info(
            "Run `hermes egress setup` then `hermes egress start` to mint tokens and launch the proxy.")
    else:
        _setup.print_info("Skipping egress firewall. You can enable it later with `hermes egress setup`.")


def _setup_backend_singularity(config: dict) -> None:
    _setup.print_success("Terminal backend: Singularity/Apptainer")
    _report_binary(shutil.which("apptainer") or shutil.which("singularity"),
                   "Singularity/Apptainer not found in PATH!",
                   "Install: https://apptainer.org/docs/admin/main/installation.html")
    config["terminal"].setdefault("singularity_image", f"docker://{_SANDBOX_IMAGE}")


def _setup_backend_modal(config: dict) -> None:
    _setup.print_success("Terminal backend: Modal")
    _setup.print_info("Serverless cloud sandboxes. Each session gets its own container.")
    from tools.managed_tool_gateway import is_managed_tool_gateway_ready
    from tools.tool_backend_helpers import normalize_modal_mode
    managed_modal_available = bool(
        tool_backend_helpers.managed_nous_tools_enabled()
        and nous_subscription.get_nous_subscription_features(config).nous_auth_present
        and is_managed_tool_gateway_ready("modal"))
    modal_mode = normalize_modal_mode(_setup.cfg_get(config, "terminal", "modal_mode"))
    use_managed_modal = False
    if managed_modal_available:
        # Default to the configured mode; when unset, to "direct" only if Modal creds exist.
        default_idx = {"managed": 0, "direct": 1}.get(modal_mode, 1 if _setup.get_env_value("MODAL_TOKEN_ID") else 0)
        use_managed_modal = _setup.prompt_choice(
            "Select how Modal execution should be billed:",
            ["Use my Nous subscription", "Use my own Modal account"], default_idx) == 0
    if use_managed_modal:
        config["terminal"]["modal_mode"] = "managed"
        _setup.print_info("Modal execution will use the managed Nous gateway and bill to your subscription.")
        if _setup.get_env_value("MODAL_TOKEN_ID") or _setup.get_env_value("MODAL_TOKEN_SECRET"):
            _setup.print_info(
                "Direct Modal credentials are still configured, but this backend is pinned to managed mode.")
        return
    config["terminal"]["modal_mode"] = "direct"
    _setup.print_info("Requires a Modal account: https://modal.com")
    _ensure_sdk("modal", "uv pip install modal")
    _setup._info(None, "Modal authentication:", "  Get your token at: https://modal.com/settings")
    if _existing_secret_keeps("MODAL_TOKEN_ID", "Modal token", "  Update Modal credentials?"):
        return
    _prompt_secret_env("    Modal Token ID", "MODAL_TOKEN_ID")
    _prompt_secret_env("    Modal Token Secret", "MODAL_TOKEN_SECRET")


def _setup_backend_daytona(config: dict) -> None:
    _setup.print_success("Terminal backend: Daytona")
    _setup._info("Persistent cloud development environments.",
                 "Each session gets a dedicated sandbox with filesystem persistence.",
                 "Sign up at: https://daytona.io")
    _ensure_sdk("daytona", "uv pip install daytona", show_stderr=True)
    print()
    had_key = bool(_setup.get_env_value("DAYTONA_API_KEY"))
    if not _existing_secret_keeps("DAYTONA_API_KEY", "Daytona API key", "  Update API key?"):
        _prompt_secret_env("    Daytona API key", "DAYTONA_API_KEY",
                           confirm_msg="    Updated" if had_key else "    Configured")
    config["terminal"].setdefault("daytona_image", _SANDBOX_IMAGE)


def _setup_backend_vercel(config: dict) -> None:
    _setup.print_success("Terminal backend: Vercel Sandbox")
    _setup._info("Cloud microVM sandboxes with snapshot-backed filesystem persistence.",
                 "Requires the optional SDK: pip install 'hermes-agent[vercel]'")
    _ensure_sdk("vercel", "pip install 'hermes-agent[vercel]'", show_stderr=True, install=_pip_install_vercel)
    _prompt_vercel_sandbox_settings(config)


def _setup_backend_ssh(config: dict) -> None:
    _setup.print_success("Terminal backend: SSH")
    _setup.print_info("Run commands on a remote machine via SSH.")
    # (label, env var, fallback default when .env is empty); the port is only saved when not 22.
    fields = (
        ("  SSH host (hostname or IP)", "TERMINAL_SSH_HOST", ""),
        ("  SSH user", "TERMINAL_SSH_USER", os.getenv("USER", "")),
        ("  SSH port", "TERMINAL_SSH_PORT", "22"),
        ("  SSH private key path", "TERMINAL_SSH_KEY", str(Path.home() / ".ssh" / "id_rsa")))
    values = []
    for label, env_var, default in fields:
        value = _setup.prompt(label, _setup.get_env_value(env_var) or default)
        values.append(value)
        if value and (env_var != "TERMINAL_SSH_PORT" or value != "22"):
            _setup.save_env_value(env_var, value)
    host, user, port, ssh_key = values
    if host and _setup.prompt_yes_no("  Test SSH connection?", True):
        _setup.print_info("  Testing connection...")
        import subprocess
        ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", *(["-i", ssh_key] if ssh_key else []),
                   *(["-p", port] if port and port != "22" else []), f"{user}@{host}" if user else host, "echo ok"]
        result = subprocess.run(ssh_cmd, timeout=10, **_RUN_KW)
        if result.returncode == 0:
            _setup.print_success("  SSH connection successful!")
        else:
            _setup.print_warning(f"  SSH connection failed: {result.stderr.strip()}")
            _setup.print_info("  Check your SSH key and host settings.")


def _setup_backend_plugin(config: dict, backend: str) -> None:
    try:
        from agent.terminal_env_registry import get_provider
        provider = get_provider(backend)
        _setup.print_success(f"Terminal backend: {provider.display_name}")
        for line in provider.setup_instructions():
            _setup.print_info(line)
        provider.post_setup()
    except Exception as exc:
        _setup.print_warning(f"Backend plugin setup hook failed: {exc}")


_BUILTIN_TERMINAL_BACKENDS = [
    ("local", "Local - run directly on this machine (default)"),
    ("docker", "Docker - isolated container with configurable resources"),
    ("modal", "Modal - serverless cloud sandbox"), ("ssh", "SSH - run on a remote machine"),
    ("daytona", "Daytona - persistent cloud development environment"),
    ("vercel_sandbox", "Vercel Sandbox - cloud microVM with snapshot filesystem persistence")]
_TERMINAL_BACKEND_SETUP = {
    "local": _setup_backend_local, "docker": _setup_backend_docker, "singularity": _setup_backend_singularity,
    "modal": _setup_backend_modal, "daytona": _setup_backend_daytona, "vercel_sandbox": _setup_backend_vercel,
    "ssh": _setup_backend_ssh}
# Backend -> env var mirrored from config after setup (config.yaml is the source of truth, but
# terminal_tool reads these from .env).
_BACKEND_ENV_MIRROR = {"modal": ("TERMINAL_MODAL_MODE", "modal_mode", "auto"),
                       "vercel_sandbox": ("TERMINAL_VERCEL_RUNTIME", "vercel_runtime", "node24")}


def setup_terminal_backend(config: dict):
    """Configure the terminal execution backend."""
    import platform as _platform
    _setup.print_header("Terminal Backend")
    _setup._info("Choose where Hermes runs shell commands and code.",
                 "This affects tool execution, file access, and isolation.",
                 f"   Guide: {_setup._DOCS_BASE}/user-guide/configuration#terminal-backend-configuration", None)
    current_backend = _setup.cfg_get(config, "terminal", "backend", default="local")
    backends = list(_BUILTIN_TERMINAL_BACKENDS)
    if _platform.system() == "Linux":
        backends.append(("singularity", "Singularity/Apptainer - HPC-friendly container"))
    # Plugin-registered backends (~/.hermes/plugins/). Fail-soft: a broken plugin must not take
    # the wizard down.
    plugin_backend_names = []
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent — plugin state may not be loaded yet
        from agent.terminal_env_registry import list_providers
        for provider in list_providers():
            pname = provider.name.strip().lower()
            backends.append((pname, f"{provider.display_name} - {provider.description}"))
            plugin_backend_names.append(pname)
    except Exception:
        pass
    terminal_choices = [label for _, label in backends] + [f"Keep current ({current_backend})"]
    terminal_idx = _setup.prompt_choice("Select terminal backend:", terminal_choices, len(backends))
    if terminal_idx == len(backends):
        _setup.print_info(f"Keeping current backend: {current_backend}")
        return
    selected_backend = backends[terminal_idx][0] if 0 <= terminal_idx < len(backends) else None
    config.setdefault("terminal", {})["backend"] = selected_backend
    # Plugin names shadow only the ssh built-in (dispatch order of the original chain).
    handler = _TERMINAL_BACKEND_SETUP.get(selected_backend)
    if handler is not None and (selected_backend != "ssh" or selected_backend not in plugin_backend_names):
        handler(config)
    elif selected_backend in plugin_backend_names:
        _setup_backend_plugin(config, selected_backend)
    _setup.save_env_value("TERMINAL_ENV", selected_backend)
    if selected_backend in _BACKEND_ENV_MIRROR:
        env_var, key, default = _BACKEND_ENV_MIRROR[selected_backend]
        _setup.save_env_value(env_var, config["terminal"].get(key, default))
    _setup.save_config(config)
    print()
    _setup.print_success(f"Terminal backend set to: {selected_backend}")


import hermes_cli.setup as _setup  # noqa: E402  (bottom: hermes_cli.setup imports this module)
