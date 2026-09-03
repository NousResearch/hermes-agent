"""Tests for CLI/TUI CWD resolution in load_cli_config().

Rules:
- Backend selection: an exported TERMINAL_ENV wins unless config.yaml
  explicitly pins terminal.backend / terminal.env_type.
- Local backend CLI/TUI: always os.getcwd(), ignoring config and inherited env.
- Non-local with placeholder: pop cwd for backend default.
- Non-local with explicit path: keep as-is.
"""


_CWD_PLACEHOLDERS = (".", "auto", "cwd")


def _resolve_cwd(terminal_config: dict, defaults: dict, env: dict,
                 file_pins_backend: bool = False):
    """Mirror the CWD resolution logic from cli.py load_cli_config().

    ``file_pins_backend`` mirrors whether config.yaml explicitly set
    ``terminal.backend`` / ``terminal.env_type``. Only an explicit pin
    outranks an exported TERMINAL_ENV.
    """
    env_backend = env.get("TERMINAL_ENV", "").strip()
    if env_backend and not file_pins_backend:
        terminal_config["env_type"] = env_backend
    effective_backend = terminal_config.get("env_type", "local")

    if effective_backend == "local":
        terminal_config["cwd"] = "/fake/getcwd"
        defaults["terminal"]["cwd"] = terminal_config["cwd"]
    elif terminal_config.get("cwd") in _CWD_PLACEHOLDERS:
        terminal_config.pop("cwd", None)

    # Bridge: TERMINAL_CWD always exported in CLI, skipped in gateway
    _is_gateway = env.get("_HERMES_GATEWAY") == "1"
    if "cwd" in terminal_config:
        if _is_gateway:
            pass  # don't touch env
        else:
            env["TERMINAL_CWD"] = str(terminal_config["cwd"])

    return env.get("TERMINAL_CWD", "")


class TestLocalBackendCli:
    """Local backend always uses os.getcwd()."""

    def test_explicit_config_ignored(self):
        env = {}
        tc = {"cwd": "/explicit/path", "env_type": "local"}
        d = {"terminal": {"cwd": "/explicit/path"}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"

    def test_inherited_env_overwritten(self):
        env = {"TERMINAL_CWD": "/parent/hermes"}
        tc = {"cwd": "/home/user", "env_type": "local"}
        d = {"terminal": {"cwd": "/home/user"}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"

    def test_placeholder_resolved(self):
        env = {}
        tc = {"cwd": "."}
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"

    def test_env_and_no_config_file(self):
        env = {"TERMINAL_CWD": "/stale/value"}
        tc = {"cwd": ".", "env_type": "local"}
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"


class TestNonLocalBackends:
    """Non-local backends use config or per-backend defaults."""

    def test_placeholder_popped(self):
        env = {}
        tc = {"cwd": ".", "env_type": "docker"}
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env) == ""

    def test_explicit_path_kept(self):
        env = {}
        tc = {"cwd": "/srv/app", "env_type": "ssh"}
        d = {"terminal": {"cwd": "/srv/app"}}
        assert _resolve_cwd(tc, d, env) == "/srv/app"

    def test_auto_placeholder_popped(self):
        env = {}
        tc = {"cwd": "auto", "env_type": "modal"}
        d = {"terminal": {"cwd": "auto"}}
        assert _resolve_cwd(tc, d, env) == ""


class TestEnvSelectedBackend:
    """An exported TERMINAL_ENV selects the backend when config doesn't pin one.

    Regression: with no ``terminal:`` section in config.yaml, effective_backend
    was read from merged DEFAULTS ("local") even under TERMINAL_ENV=ssh, so the
    local branch force-exported TERMINAL_CWD=os.getcwd() — an agent-host path.
    On an ssh target that path doesn't exist, so the tool wrapper's `cd` failed
    with exit 126 before any command ran.
    """

    def test_ssh_env_preserves_wrapper_cwd(self):
        env = {"TERMINAL_ENV": "ssh", "TERMINAL_CWD": "/srv/project"}
        tc = {"cwd": ".", "env_type": "local"}  # merged default
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env) == "/srv/project"

    def test_ssh_env_without_cwd_pops_for_remote_default(self):
        env = {"TERMINAL_ENV": "ssh"}
        tc = {"cwd": ".", "env_type": "local"}
        d = {"terminal": {"cwd": "."}}
        # Popped, so terminal_tool falls back to the remote home (~).
        assert _resolve_cwd(tc, d, env) == ""

    def test_explicit_config_pin_still_wins(self):
        env = {"TERMINAL_ENV": "ssh", "TERMINAL_CWD": "/srv/project"}
        tc = {"cwd": ".", "env_type": "local"}
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env, file_pins_backend=True) == "/fake/getcwd"

    def test_local_env_still_uses_getcwd(self):
        env = {"TERMINAL_ENV": "local", "TERMINAL_CWD": "/stale/value"}
        tc = {"cwd": ".", "env_type": "local"}
        d = {"terminal": {"cwd": "."}}
        assert _resolve_cwd(tc, d, env) == "/fake/getcwd"


class TestGatewayLazyImport:
    """Gateway lazy import of cli.py must not clobber TERMINAL_CWD."""

    def test_gateway_cwd_preserved(self):
        env = {"_HERMES_GATEWAY": "1", "TERMINAL_CWD": "/home/user/project"}
        tc = {"cwd": "/home/user", "env_type": "local"}
        d = {"terminal": {"cwd": "/home/user"}}
        result = _resolve_cwd(tc, d, env)
        assert result == "/home/user/project"

    def test_cli_overwrites_stale_env(self):
        env = {"TERMINAL_CWD": "/stale/from/dotenv"}
        tc = {"cwd": "/home/user", "env_type": "local"}
        d = {"terminal": {"cwd": "/home/user"}}
        result = _resolve_cwd(tc, d, env)
        assert result == "/fake/getcwd"
