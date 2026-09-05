"""Configuration management for Hermes Agent: config.yaml / .env loading, saving,
validation, migration, and the ``hermes config`` command."""

import copy
import difflib
import json
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

import yaml

from hermes_cli.cli_output import line_input
from hermes_cli.colors import Colors, color
from hermes_cli import managed_scope
from hermes_cli.default_soul import DEFAULT_SOUL_MD, is_legacy_template_soul
from hermes_cli.secret_prompt import masked_secret_prompt
# Re-export from hermes_constants — canonical definition lives there.
from hermes_constants import get_hermes_home, get_process_hermes_home  # noqa: F401
from utils import atomic_replace, atomic_yaml_write, fast_safe_load

logger = logging.getLogger(__name__)

# (config_path, mtime_ns, size) tuples already warned about, so concurrent CLI/gateway
# loads of a broken config.yaml don't spam stderr. A changed file (new mtime) warns again.
_CONFIG_PARSE_WARNED: set = set()

# path -> (mtime_ns, size, error message) of active parse failures. Written by
# _warn_config_parse_failure() (the single funnel for every load-path parse failure) and
# probed by get_active_config_parse_failure() so provider auto-resolution can refuse to
# adopt a paid provider from env keys while the user's REAL config is unreadable.
_CONFIG_PARSE_FAILURES: dict = {}


class InvalidUserConfigError(RuntimeError):
    """Raised when a run that cannot repair config finds invalid user YAML."""


def _backup_corrupt_config(config_path: Path) -> Optional[Path]:
    """Copy an unparseable ``config.yaml`` to a timestamped ``.corrupt.*.bak``; None on skip/failure.
    Symlinks are not followed (never clobber whatever a malicious symlink points at). A sibling
    backup of the same size means this corruption was already snapshotted — skip to avoid churn.

    Returns the backup path on success, else ``None``. See #21541.
    """
    try:
        if config_path.is_symlink():
            return None
        st = config_path.stat()
        if st.st_size == 0:
            return None
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup_path = config_path.with_name(f"{config_path.name}.corrupt.{ts}.bak")
        for existing in config_path.parent.glob(f"{config_path.name}.corrupt.*.bak"):
            try:
                if existing.stat().st_size == st.st_size:
                    return None
            except OSError:
                continue
        if backup_path.exists():
            return None
        shutil.copy2(config_path, backup_path)
        return backup_path
    except Exception:
        return None


_PARSE_FAILURE_FALLBACK_MSG = {
    "last-known-good": (
        "Keeping the previously loaded config for this process — "
        "edits to config.yaml are being IGNORED until the YAML is fixed."),
    "refuse-write": (
        "REFUSING to write config.yaml so the existing file is preserved. "
        "Fix the YAML (hermes config edit) and retry.")}
_PARSE_FAILURE_DEFAULTS_MSG = (
    "Falling back to default config — every user override (auxiliary providers, fallback chain, "
    "model settings) is being IGNORED. Fix the YAML and restart.")


def _warn_config_parse_failure(
    config_path: Path, exc: Exception, *, fallback: str = "defaults") -> None:
    """Surface a config.yaml parse failure to log and stderr (once per file signature).
    Silent fallback to ``DEFAULT_CONFIG`` drops every user override, so this must be loud.

    ``fallback`` selects the message wording: ``"defaults"`` (fresh process, nothing else to serve) or
    ``"last-known-good"`` (in-process retention of the previously loaded config — see the codex#31188 port
    in ``_load_config_impl``).
    """
    try:
        st = config_path.stat()
        key = (str(config_path), st.st_mtime_ns, st.st_size)
        _CONFIG_PARSE_FAILURES[str(config_path)] = (st.st_mtime_ns, st.st_size, str(exc))
    except OSError:
        key = (str(config_path), 0, 0)
    if key in _CONFIG_PARSE_WARNED:
        return
    _CONFIG_PARSE_WARNED.add(key)
    backup_path = _backup_corrupt_config(config_path)
    msg = f"Failed to parse {config_path}: {exc}. " + _PARSE_FAILURE_FALLBACK_MSG.get(
        fallback, _PARSE_FAILURE_DEFAULTS_MSG)
    if backup_path is not None:
        msg += f" A copy of the corrupted file was saved to {backup_path}."
    logger.warning(msg)
    try:
        sys.stderr.write(f"⚠️  hermes config: {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def get_active_config_parse_failure() -> Optional[str]:
    """Return the recorded parse error while the ACTIVE config.yaml is still byte-identical
    (mtime_ns + size) to the file that failed to parse; else None."""
    try:
        path = get_config_path()
        mtime_ns, size, err = _CONFIG_PARSE_FAILURES[str(path)]
        st = path.stat()
        return err if (st.st_mtime_ns, st.st_size) == (mtime_ns, size) else None
    except Exception:
        return None


_IS_WINDOWS = platform.system() == "Windows"
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Env var names that influence how the next subprocess executes — never writable through
# ``save_env_value``: dynamic loader (LD_*/DYLD_*: attacker code loads before main()),
# interpreter init (PYTHON*, NODE_*: Hermes restarts through them), PATH (fix tool lookup
# with absolute paths instead), git rewrites (fire on every plugin install/update),
# implicitly-invoked commands (BROWSER/EDITOR/VISUAL/PAGER = RCE on next $EDITOR), SHELL,
# and Hermes runtime-location / security-policy flags (config.yaml is the supported surface).
#
# ``HERMES_*`` overall is NOT blocked — many integration credentials use that prefix
# (HERMES_LANGFUSE_PUBLIC_KEY, HERMES_SPOTIFY_CLIENT_ID, ...). The denylist is name-by-name so
# it cannot break provider setup wizards. Enforced on *write* only: pre-existing/out-of-band
# ``.env`` values keep working; the dashboard's writable surface just cannot escalate.
_ENV_VAR_NAME_DENYLIST: frozenset[str] = frozenset({
    # Loader / linker
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_DEBUG",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_FALLBACK_FRAMEWORK_PATH",
    # Python / Node
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
    "PYTHONEXECUTABLE", "PYTHONNOUSERSITE", "NODE_OPTIONS", "NODE_PATH",
    # General / git
    "PATH", "SHELL", "BROWSER", "EDITOR", "VISUAL", "PAGER",
    "GIT_SSH_COMMAND", "GIT_EXEC_PATH", "GIT_SHELL",
    # Hermes runtime location
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_CONFIG", "HERMES_ENV",
    "HERMES_CONFIG_PATH", "HERMES_ENV_PATH",
    # MCP catalog trust root; package-manager wrappers may still set it in the process env.
    "HERMES_OPTIONAL_MCPS",
    # Local ACP subprocess selection (executable/argv authority).
    "HERMES_COPILOT_ACP_COMMAND", "HERMES_COPILOT_ACP_ARGS",
    # Security policy / approval-routing context — set via their dedicated controls only.
    "HERMES_YOLO_MODE", "HERMES_ACCEPT_HOOKS", "HERMES_REDACT_SECRETS",
    "HERMES_INTERACTIVE", "HERMES_EXEC_ASK", "HERMES_GATEWAY_SESSION",
    "HERMES_CRON_SESSION", "HERMES_SINGLE_QUERY_SESSION",
    "HERMES_SESSION_KEY", "HERMES_SESSION_PLATFORM"})


def _env_var_policy_name(key: str, *, is_windows: Optional[bool] = None) -> str:
    """Name used for env policy comparisons: Windows env names are case-insensitive, POSIX not.
    The override keeps both semantics testable on any host."""
    windows = _IS_WINDOWS if is_windows is None else is_windows
    return key.upper() if windows else key


def validate_env_var_name_for_write(key: str) -> None:
    """Validate an env name before a generic persistence write (exposed for batch callers)."""
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    if _env_var_policy_name(key) in _ENV_VAR_NAME_DENYLIST:
        raise ValueError(
            f"Environment variable {key!r} is on the writer denylist. "
            "Names that influence subprocess execution (LD_PRELOAD, PYTHONPATH, PATH, EDITOR, ...) "
            "or Hermes runtime location and security policy (HERMES_HOME, HERMES_YOLO_MODE, ...) "
            "cannot be persisted via the env writer. If you really need this, edit ~/.hermes/.env "
            "directly.")


# Serializes all config read/write paths and guards the module-level caches below. libyaml's
# C extension is not thread-safe for concurrent safe_load() on one file, and tool threads
# (approval, browser, setup flows) load/save config concurrently during long agent runs.
# RLock because save_config internally calls read_raw_config.
_CONFIG_LOCK = threading.RLock()
# path -> last successfully loaded (expanded) config; served after a parse failure so a
# mid-edit broken YAML never silently drops user overrides (e.g. approvals.deny rules).
_LAST_EXPANDED_CONFIG_BY_PATH: Dict[str, Any] = {}
# path -> (user_mtime_ns, user_size, managed_mtime_ns, managed_size, merged, env_ref_snapshot).
# load_config() returns a deepcopy of the cached value while the signature matches (skips
# safe_load + merge + normalize + expand, ~13 ms). Writers use atomic_yaml_write (fresh inode
# -> new mtime_ns) so no explicit invalidation is needed. The managed-file signature is folded
# in so editing the managed-scope config.yaml invalidates, and the env snapshot invalidates
# when a referenced ${VAR} changes value (late .env load, in-process rotation).
# (path, mtime_ns, size) -> cached expanded config dict. load_config() returns a deepcopy of the cached
# value when the file hasn't changed since the last load, skipping yaml.safe_load + _deep_merge +
# _normalize_* + _expand_env_vars (~13 ms/call). save_config() + migrate_config() write via
# atomic_yaml_write which produces a fresh inode, so stat() sees a new mtime_ns and the next load
# repopulates automatically — no explicit invalidation hook. See #58514.
_LOAD_CONFIG_CACHE: Dict[str, Tuple[int, int, int, int, Dict[str, Any], Dict[str, Optional[str]]]] = {}
# path -> (mtime_ns, size, raw yaml dict) for read_raw_config() (no defaults merged in).
_RAW_CONFIG_CACHE: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}

# Env var names written to .env that aren't in OPTIONAL_ENV_VARS (managed by setup/provider
# flows directly). Also the set reload_env() may remove from os.environ.
_EXTRA_ENV_KEYS = frozenset({
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    "DISCORD_HOME_CHANNEL", "DISCORD_HOME_CHANNEL_NAME",
    "TELEGRAM_HOME_CHANNEL", "TELEGRAM_HOME_CHANNEL_NAME",
    "SLACK_HOME_CHANNEL", "SLACK_HOME_CHANNEL_NAME",
    "SIGNAL_ACCOUNT", "SIGNAL_HTTP_URL", "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
    "SIGNAL_HOME_CHANNEL", "SIGNAL_HOME_CHANNEL_NAME", "SMS_HOME_CHANNEL", "SMS_HOME_CHANNEL_NAME",
    "DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET", "DINGTALK_HOME_CHANNEL", "DINGTALK_HOME_CHANNEL_NAME",
    "FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_ENCRYPT_KEY", "FEISHU_VERIFICATION_TOKEN",
    "FEISHU_HOME_CHANNEL", "FEISHU_HOME_CHANNEL_NAME", "YUANBAO_HOME_CHANNEL", "YUANBAO_HOME_CHANNEL_NAME",
    "WECOM_BOT_ID", "WECOM_SECRET", "WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET",
    "WECOM_CALLBACK_AGENT_ID", "WECOM_CALLBACK_TOKEN", "WECOM_CALLBACK_ENCODING_AES_KEY",
    "WECOM_CALLBACK_HOST", "WECOM_CALLBACK_PORT", "WECOM_HOME_CHANNEL", "WECOM_HOME_CHANNEL_NAME",
    "WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN", "WEIXIN_BASE_URL", "WEIXIN_CDN_BASE_URL",
    "WEIXIN_HOME_CHANNEL", "WEIXIN_HOME_CHANNEL_NAME", "WEIXIN_DM_POLICY", "WEIXIN_GROUP_POLICY",
    "WEIXIN_ALLOWED_USERS", "WEIXIN_GROUP_ALLOWED_USERS", "WEIXIN_ALLOW_ALL_USERS",
    "BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD", "BLUEBUBBLES_HOME_CHANNEL", "BLUEBUBBLES_HOME_CHANNEL_NAME",
    "QQ_APP_ID", "QQ_CLIENT_SECRET", "QQBOT_HOME_CHANNEL", "QQBOT_HOME_CHANNEL_NAME",
    "QQ_HOME_CHANNEL", "QQ_HOME_CHANNEL_NAME",  # legacy aliases (pre-rename, still read for back-compat)
    "QQ_ALLOWED_USERS", "QQ_GROUP_ALLOWED_USERS", "QQ_ALLOW_ALL_USERS", "QQ_MARKDOWN_SUPPORT",
    "QQ_STT_API_KEY", "QQ_STT_BASE_URL", "QQ_STT_MODEL",
    "IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS", "IRC_SERVER_PASSWORD",
    "IRC_NICKSERV_PASSWORD", "TERMINAL_ENV", "TERMINAL_SSH_KEY", "TERMINAL_SSH_PORT",
    # Deprecated (replaced by display.tool_progress) but STILL READ by the gateway as a
    # back-compat fallback. The boolean HERMES_TOOL_PROGRESS variant is unsupported (its only
    # consumer, the v3->4 migration, is below the v12 support floor); doctor flags it as ignored.
    "HERMES_TOOL_PROGRESS_MODE",
    "WHATSAPP_MODE", "WHATSAPP_ENABLED",
    "MATTERMOST_HOME_CHANNEL", "MATTERMOST_HOME_CHANNEL_NAME", "MATTERMOST_REPLY_MODE",
    "MATRIX_PASSWORD", "MATRIX_ENCRYPTION", "MATRIX_DEVICE_ID", "MATRIX_HOME_ROOM",
    "MATRIX_REQUIRE_MENTION", "MATRIX_FREE_RESPONSE_ROOMS", "MATRIX_AUTO_THREAD", "MATRIX_DM_AUTO_THREAD",
    "MATRIX_RECOVERY_KEY",
    # Langfuse observability plugin tuning keys + standard SDK vars (activation is via
    # plugins.enabled; credentials gate the plugin at runtime).
    "HERMES_LANGFUSE_ENV", "HERMES_LANGFUSE_RELEASE", "HERMES_LANGFUSE_SAMPLE_RATE",
    "HERMES_LANGFUSE_MAX_CHARS", "HERMES_LANGFUSE_CAPTURE", "HERMES_LANGFUSE_DEBUG",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
    # ACP (Agent Client Protocol) keys — profile-isolable so profiles can use different backends.
    "HERMES_ACP_AUTH_METHOD", "HERMES_ACP_AUTO_APPROVE", "HERMES_COPILOT_ACP_COMMAND",
    "HERMES_COPILOT_ACP_ARGS", "COPILOT_CLI_PATH", "COPILOT_ACP_BASE_URL"})


# ---- Managed mode (NixOS declarative config) ----

_MANAGED_TRUE_VALUES = ("true", "1", "yes")
_NIX_MANAGED_SYSTEMS = {"nixos", "home-manager"}
# Only the NixOS module ever wrote a bare "true" or an empty marker.
_LEGACY_MANAGED_SYSTEM = "nixos"
# Nix store root; identifies `nix run` / `nix profile install` installs (which don't set
# HERMES_MANAGED). Module-level so tests can patch it without touching /nix/store.
_NIX_STORE = Path("/nix/store")
# Homebrew is no longer a supported distribution: these markers fall through to git/unknown
# detection instead of blocking config writes.
_IGNORED_MANAGED_VALUES = frozenset({"brew", "homebrew"})


def get_managed_system() -> Optional[str]:
    """Return the package manager owning this install, if any.
    Signals: HERMES_MANAGED env var (systemd service) or a ``.managed`` marker file in
    HERMES_HOME (NixOS activation script — interactive shells don't see the service env)."""
    marker = os.getenv("HERMES_MANAGED", "").strip().lower() or None
    managed_marker = get_hermes_home() / ".managed"
    if marker is None and managed_marker.exists():
        try:
            marker = managed_marker.read_text(encoding="utf-8", errors="replace").strip().lower()
        except OSError:
            marker = ""
    if marker is None or marker in _IGNORED_MANAGED_VALUES:
        return None
    if marker == "" or marker in _MANAGED_TRUE_VALUES:
        return _LEGACY_MANAGED_SYSTEM
    return marker


def is_managed() -> bool:
    """Check if Hermes is running in package-manager-managed mode."""
    return get_managed_system() is not None


# Nix installs arrive by several routes (nix run, nix profile, system flake, home-manager) and
# the running process cannot tell which, so the text names the routes instead of one command.
_NIX_UPDATE_MSG = (
    "Update Hermes through the Nix source that installed it "
    "(e.g. nix profile upgrade, or update your flake input and rebuild with nixos-rebuild or home-manager switch)"
)


def get_managed_update_command() -> Optional[str]:
    """Return the preferred upgrade command for a managed install."""
    return _NIX_UPDATE_MSG if get_managed_system() in _NIX_MANAGED_SYSTEMS else None


# "apt" is the Termux APT distribution identifier, not a generic Debian/Ubuntu signal; another
# APT distribution needs its own method. "home-manager" is listed because the managed marker can
# return it and a stamp must name every method this function returns.
_SUPPORTED_INSTALL_METHODS = frozenset({"apt", "docker", "nix", "nixos", "home-manager", "git", "unknown"})


def _install_method_stamp(path: Path) -> Optional[str]:
    try:
        method = path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    return method if method in _SUPPORTED_INSTALL_METHODS else None


def detect_install_method(project_root: Optional[Path] = None) -> str:
    """Detect how Hermes was installed: apt/docker/nix/nixos/home-manager/git/unknown.
    Order: code-scoped ``<install tree>/.install_method`` stamp (authoritative) -> legacy
    ``$HERMES_HOME/.install_method`` -> managed marker -> /nix/store path -> .git dir -> unknown.
    The stamp lives next to the code because HERMES_HOME is shared data: a container and a host
    install can bind-mount the same home, so a home-scoped ``docker`` stamp would make the host
    ``hermes update`` refuse to run. A legacy ``docker`` value is therefore ignored unless we are
    really inside a container, and being in a container alone never implies 'docker'.

    The supported installs self-identify via the code-scoped stamp: - the curl installer
    (scripts/install.sh, the README/website install command) git-clones the repo and stamps ``git`` next to
    the code; - the published ``nousresearch/hermes-agent`` image bakes a ``docker`` stamp into
    ``/opt/hermes`` at build time. An unsupported manual install dropped into a container (no stamp) falls
    through to the ``.git`` checks and behaves like any off-path install. See issue #34397.
    """
    # The stamp is a property of the running code tree (parent of hermes_cli/), NOT of $HERMES_HOME,
    # so it survives two installs sharing a home.
    root = project_root if project_root is not None else get_project_root()
    method = _install_method_stamp(root / ".install_method")
    if method:
        return method

    method = _install_method_stamp(get_hermes_home() / ".install_method")
    if method and not (method == "docker" and not _running_in_container()):
        return method

    managed = get_managed_system()
    if managed:
        return managed.lower().replace(" ", "-")

    # Code under /nix/store/ is the hallmark of a nix-built install.
    try:
        resolved = root.resolve()
        if resolved != _NIX_STORE and _NIX_STORE in resolved.parents:
            return "nix"
    except OSError:
        pass

    # A .git directory, or a ``gitdir:`` pointer file for worktrees.
    git_path = root / ".git"
    try:
        if git_path.is_dir() or git_path.read_text(encoding="utf-8").strip().startswith("gitdir:"):
            return "git"
    except OSError:
        pass
    return "unknown"


def _running_in_container() -> bool:
    """Import-safe wrapper around ``hermes_constants.is_container``."""
    try:
        from hermes_constants import is_container

        return is_container()
    except Exception:
        return False


def is_nix_install_method(method: str) -> bool:
    """True for every install method Nix owns ("nix", "nixos", "home-manager")."""
    return method == "nix" or method in _NIX_MANAGED_SYSTEMS


_UPDATE_COMMAND_BY_METHOD = {
    "docker": "docker pull nousresearch/hermes-agent:latest",
    "apt": "pkg upgrade hermes-agent",  # "apt" == Termux APT by contract; uses Termux's `pkg`.
}


def recommended_update_command_for_method(method: str) -> str:
    """Return the update command or guidance for a given install method."""
    if is_nix_install_method(method):
        return _NIX_UPDATE_MSG
    return _UPDATE_COMMAND_BY_METHOD.get(method, "hermes update")


def recommended_update_command() -> str:
    """Return the best update command for the current installation.
    Managed state wins over the code-scoped stamp: a managed install can carry a stale stamp
    naming an update path the managed guard refuses."""
    return get_managed_update_command() or recommended_update_command_for_method(
        detect_install_method(get_project_root()))


# Shared by ``cmd_update`` and ``_cmd_update_check`` (hermes_cli/main.py) so the wording never
# forks. The published image excludes ``.git``, so the git update path can never succeed there
# and the generic "reinstall via install.sh" fallback would install a NEW host-side Hermes.
_DOCKER_UPDATE_MESSAGE = """\
✗ ``hermes update`` doesn't apply inside the Docker container.

Hermes Agent runs as a published image (nousresearch/hermes-agent), not a
git checkout — the container has no working tree to pull into.  Update by
pulling a fresh image and restarting your container instead:

  docker pull nousresearch/hermes-agent:latest
  # then restart whatever started the container, e.g.:
  docker compose up -d --force-recreate hermes-agent
  # or, for ad-hoc runs, exit the current container and `docker run` again

Verify the new version after restart:
  docker run --rm nousresearch/hermes-agent:latest --version

Notes:
  • If you pinned a specific tag (e.g. ``:v0.14.0``) the ``:latest`` tag
    won't move your container — pull the newer tag you actually want, or
    switch to ``:latest`` / ``:main`` for rolling updates.  See available
    tags at https://hub.docker.com/r/nousresearch/hermes-agent/tags
  • Your config and session history live under ``$HERMES_HOME`` (``/opt/data``
    in the container, typically bind-mounted from the host) and persist
    across image upgrades — re-pulling doesn't lose any state.
  • Running a fork?  Build your own image with this repo's ``Dockerfile``
    and replace the ``docker pull`` step with your build/push pipeline."""


def format_docker_update_message() -> str:
    """Return the user-facing message for ``hermes update`` inside Docker."""
    return _DOCKER_UPDATE_MESSAGE


def format_managed_message(action: str = "modify this Hermes installation") -> str:
    """Build a user-facing error for managed installs."""
    managed_system = get_managed_system() or "a package manager"
    return (
        f"Cannot {action}: this Hermes installation is managed by {managed_system}.\n"
        "Use your package manager to upgrade or reinstall Hermes.")


def managed_error(action: str = "modify configuration"):
    """Print user-friendly error for managed mode."""
    print(format_managed_message(action), file=sys.stderr)


def get_container_exec_info() -> Optional[dict]:
    """Read container mode metadata from HERMES_HOME/.container-mode.
    Written by the NixOS activation script when container.enable = true; tells the host CLI to
    exec into the container instead of running locally. None when container mode is off, when
    already inside the container, or when HERMES_DEV=1 is set. Only FileNotFoundError is
    swallowed; other errors (permissions, malformed data) propagate."""
    if os.environ.get("HERMES_DEV") == "1":
        return None

    from hermes_constants import is_container
    if is_container():
        return None

    try:
        info = {}
        with open(get_hermes_home() / ".container-mode", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    info[key.strip()] = value.strip()
    except FileNotFoundError:
        return None

    return {
        "backend": info.get("backend", "docker"),
        "container_name": info.get("container_name", "hermes-agent"),
        "exec_user": info.get("exec_user", "hermes"),
        "hermes_bin": info.get("hermes_bin", "/data/current-package/bin/hermes")}


# ---- Config paths / HERMES_HOME skeleton ----

def get_config_path() -> Path:
    """Get the main config file path."""
    return get_hermes_home() / "config.yaml"


def require_parseable_user_config(*, ignore_user_config: bool = False) -> None:
    """Reject an existing invalid config before a non-interactive agent run.
    Interactive surfaces keep ``load_config()``'s recovery behavior so the operator can repair
    the file; a one-shot run has no such chance, and defaults there could silently pick a hosted
    provider and spend against ``.env`` credentials. Missing/empty files stay valid first-run
    states; ``--ignore-user-config`` / HERMES_IGNORE_USER_CONFIG=1 remain authoritative."""
    if ignore_user_config or os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1":
        return

    config_path = get_config_path()
    try:
        with open(config_path, encoding="utf-8") as f:
            data = fast_safe_load(f)
    except FileNotFoundError:
        return
    except Exception as exc:
        parse_error = exc
    else:
        if data is None or isinstance(data, dict):
            return
        parse_error = TypeError(f"top-level YAML value must be a mapping, got {type(data).__name__}")

    backup_path = _backup_corrupt_config(config_path)
    message = (
        f"Refusing non-interactive startup because {config_path} is invalid: "
        f"{parse_error}. Repair the file or pass --ignore-user-config to "
        "intentionally run with built-in defaults.")
    if backup_path is not None:
        message += f" A copy was saved to {backup_path}."
    logger.error(message)
    raise InvalidUserConfigError(message) from parse_error


def get_env_path() -> Path:
    """Get the .env file path (for API keys)."""
    return get_hermes_home() / ".env"


def get_project_root() -> Path:
    """Get the project installation directory."""
    return Path(__file__).parent.parent.resolve()


def _resolve_hermes_uid_gid() -> tuple[Optional[int], Optional[int]]:
    """Read HERMES_UID / HERMES_GID (set by Docker deployments); (None, None) if unset/invalid/Windows.
    The entrypoint chowns HERMES_HOME once, but subdirs created at runtime (``profiles/<name>/``)
    need the same chown or they land root:root and block later uid-mapped workers.

    Docker containers running Hermes commonly set these to map the in-container user to a host user so
    volume-mounted state files end up with the right ownership. See #34107.
    """
    if sys.platform == "win32":
        return None, None

    def _env_int(name: str) -> Optional[int]:
        try:
            return int(os.environ.get(name, "").strip() or None)
        except (TypeError, ValueError):
            return None

    return _env_int("HERMES_UID"), _env_int("HERMES_GID")


def _chown_to_hermes_uid(path) -> None:
    """Chown ``path`` to ``HERMES_UID:HERMES_GID`` when set; EPERM/ENOENT are non-fatal (the
    entrypoint's startup chown -R fixes ownership on the next restart).

    Used by :func:`_secure_dir` to keep ownership consistent across all directories created by
    :func:`ensure_hermes_home` on Docker deployments. See #34107.
    """
    uid, gid = _resolve_hermes_uid_gid()
    if uid is None and gid is None:
        return
    try:
        os.chown(path, uid if uid is not None else -1, gid if gid is not None else -1)
    except (OSError, AttributeError, NotImplementedError):
        pass


def _secure_dir(path):
    """chmod a directory owner-only (0700) and apply HERMES_UID/GID ownership. No-op when managed.
    HERMES_HOME_MODE (e.g. 0701) overrides the mode so a web server can traverse HERMES_HOME to
    a served subdirectory without directory listings.

    Also applies ``HERMES_UID``/``HERMES_GID``-based ownership when those env vars are set (#34107 — Docker
    deployments need this so profile subdirs created at runtime by kanban workers don't land as root:root
    and block subsequent uid-mapped workers).
    """
    if is_managed():
        return
    try:
        mode = int(os.environ.get("HERMES_HOME_MODE", "").strip() or "700", 8)
    except ValueError:
        mode = 0o700
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass
    _chown_to_hermes_uid(path)


def _is_container() -> bool:
    """Detect Docker/Podman/LXC (or HERMES_CONTAINER / HERMES_SKIP_CHMOD opt-out).
    Volume-mounted config is not forced to 0o600 in containers: gateway and dashboard may run
    as different UIDs, or the mount itself needs broader permissions."""
    if (os.environ.get("HERMES_CONTAINER") or os.environ.get("HERMES_SKIP_CHMOD")
            or os.path.exists("/.dockerenv")):
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8") as f:
            cgroup_content = f.read()
        return any(marker in cgroup_content for marker in ("docker", "lxc", "kubepods"))
    except (OSError, IOError):
        return False


def _secure_file(path):
    """chmod a file 0600. Skipped when managed (activation sets 0640 group-readable) or in a
    container (mounts often need broader permissions)."""
    if is_managed() or _is_container():
        return
    try:
        if os.path.exists(str(path)):
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _ensure_default_soul_md(home: Path) -> None:
    """Seed DEFAULT_SOUL_MD on first run; upgrade a legacy comment-only scaffold in place.
    A SOUL.md the user actually customized is never touched."""
    soul_path = home / "SOUL.md"
    if soul_path.exists():
        try:
            existing = soul_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        if not is_legacy_template_soul(existing):
            return
    soul_path.write_text(DEFAULT_SOUL_MD, encoding="utf-8")
    _secure_file(soul_path)


# Home paths whose directory skeleton was created this process. Only successful passes are
# recorded, so a raised managed-mode/missing-profile error keeps re-checking on later loads.
_HERMES_HOME_ENSURED: set = set()
_HERMES_HOME_SUBDIRS = (
    "cron", "sessions", "logs", "logs/curator", "memories",
    "pairing", "hooks", "image_cache", "audio_cache", "skills")


def ensure_hermes_home():
    """Ensure the ~/.hermes directory skeleton exists with secure permissions.
    Memoized per home path: this runs on EVERY ``load_config()`` and the ~14 mkdir/chmod syscalls
    made repeated loads the dominant cost of hot read paths."""
    home = get_hermes_home()
    key = str(home)

    # Named profiles must be created explicitly. Check tombstones BEFORE the memo so a stale
    # empty shell cannot skip the deleted-profile guard.
    from hermes_constants import assert_named_profile_home_live
    assert_named_profile_home_live(home)
    if key in _HERMES_HOME_ENSURED and home.is_dir():
        return
    if is_managed():
        # Activation creates the dirs; verify, then seed SOUL.md. logs/curator may be unknown to
        # the activation script (inside an already-secured logs/). umask(0o007) => SOUL.md is 0660.
        old_umask = os.umask(0o007)
        try:
            if not home.is_dir():
                raise RuntimeError(f"HERMES_HOME {home} does not exist.")
            for subdir in ("cron", "sessions", "logs", "memories"):
                if not (home / subdir).is_dir():
                    raise RuntimeError(f"{home / subdir} does not exist.")
            (home / "logs" / "curator").mkdir(parents=True, exist_ok=True)
            _ensure_default_soul_md(home)
        finally:
            os.umask(old_umask)
    else:
        home.mkdir(parents=True, exist_ok=True)
        _secure_dir(home)
        for subdir in _HERMES_HOME_SUBDIRS:
            d = home / subdir
            d.mkdir(parents=True, exist_ok=True)
            _secure_dir(d)
        _ensure_default_soul_md(home)

    _HERMES_HOME_ENSURED.add(key)


# ---- Config loading/saving ----

from hermes_cli.config_defaults import DEFAULT_CONFIG, OPTIONAL_ENV_VARS  # noqa: E402,F401
from hermes_cli.config_providers import (  # noqa: E402,F401  (re-exported; callers/tests use hermes_cli.config.<name>)
    _API_MODE_ALIASES, _CAMEL_ALIASES, _KNOWN_PROVIDER_KEYS, _PROVIDER_NORMALIZE_WARNED,
    _canonical_api_mode, _coerce_ssl_verify, _custom_provider_entry_to_provider_config,
    _entries_for_route, _normalize_custom_provider_entry, _normalize_provider_models,
    _pick_provider_base_url, _route_model_cfg, _warn_once_per_provider,
    apply_custom_provider_extra_headers_to_client_kwargs,
    apply_custom_provider_tls_to_client_kwargs, coerce_provider_id, find_provider_entry,
    get_compatible_custom_providers, get_custom_provider_context_length,
    get_custom_provider_extra_headers, get_custom_provider_model_capability,
    get_custom_provider_tls_settings, is_provider_enabled, normalize_extra_headers,
    providers_dict_to_custom_providers, stringify_provider_map)
# Back-compat re-exports — :mod:`hermes_cli.personality` owns personality/overlay semantics.
from hermes_cli.personality import (  # noqa: E402,F401
    NEUTRAL_PERSONALITY_NAMES as _NEUTRAL_PERSONALITY_NAMES,
    prompt_text as _prompt_text,
    render_personality_prompt,
    resolve_ephemeral_system_prompt as resolve_ephemeral_system_prompt_from_config)

# ---- Config Migration System ----

# Env vars introduced per config version; migration only mentions vars new since the user's
# previous version.
ENV_VARS_BY_VERSION: Dict[int, List[str]] = {
    3: ["FIRECRAWL_API_KEY", "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID", "FAL_KEY"],
    4: ["VOICE_TOOLS_OPENAI_KEY", "ELEVENLABS_API_KEY"],
    5: ["WHATSAPP_ENABLED", "WHATSAPP_MODE", "WHATSAPP_ALLOWED_USERS",
        "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS"],
    10: ["TAVILY_API_KEY"],
    11: ["TERMINAL_MODAL_MODE"]}

# Intentionally empty: the LLM provider is required but handled by the setup wizard's provider
# selection step, so no single env var is universally required.
REQUIRED_ENV_VARS = {}


def get_missing_env_vars(required_only: bool = False) -> List[Dict[str, Any]]:
    """Check which environment variables are missing."""
    groups = [(REQUIRED_ENV_VARS, True)]
    if not required_only:
        groups.append((OPTIONAL_ENV_VARS, False))
    return [
        {"name": var_name, **info, "is_required": is_required}
        for table, is_required in groups
        for var_name, info in table.items()
        if not get_env_value(var_name)]


def _split_key_path(key: str) -> list[str]:
    """Split a dotted config-key path, honoring backslash-escaped dots (``a\\.b`` -> ``a.b``).
    Backslashes before any other character are preserved verbatim.

    ``hermes config set`` uses ``.`` as the nesting separator, so a key that itself contains a literal dot
    (e.g. provider names like ``qwen3.5-397b-wafer``) was silently split into bogus nested segments
    (#84064).
    """
    parts: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(key):
        ch = key[i]
        if ch == "\\" and key[i + 1:i + 2] == ".":
            current.append(".")
            i += 2
            continue
        if ch == ".":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    parts.append("".join(current))
    return parts


def _greedy_literal_match(container: dict, parts: list) -> Optional[Tuple[str, int]]:
    """Return ``(literal_key, n_consumed)`` for the longest dotted literal key present in
    *container*, or None. With no multi-segment literal this is the historic plain-split walk.

    Dots in config key names are the norm, not the exception — model IDs (``grok-4.6``, ``glm-5.3``), Matrix
    room IDs (``!room:chat.example.cc``), and versioned provider names all embed dots. Users typing
    ``providers.myprov.models.grok-4.6.context_length`` do not know the escape syntax exists, so when
    navigating an EXISTING mapping we prefer an existing literal key equal to the dot-join of the next N
    path segments (longest match wins) over blindly splitting. See #84064 / #80006 / 91095 / #91607 /
    #99124.
    """
    if not isinstance(container, dict) or not parts:
        return None
    return next(
        ((".".join(parts[:n]), n) for n in range(len(parts), 0, -1) if ".".join(parts[:n]) in container),
        None)


def _phantom_sibling(container: dict, part: str) -> Optional[str]:
    """Existing literal dotted key that creating an intermediate mapping ``part`` would shadow
    (``grok-4`` beside ``grok-4.5``) — the write would produce a phantom sibling the runtime never
    reads, so callers fail loudly instead.

    Called when a write is about to CREATE a new intermediate mapping named ``part``. See #84064.
    """
    if not isinstance(container, dict):
        return None
    prefix = part + "."
    return next((k for k in container if isinstance(k, str) and k.startswith(prefix)), None)


def _set_nested(config, dotted_key: str, value):
    """Set a value at a dotted key path, creating intermediate dicts on demand.
    Numeric segments index lists; the index must already exist (lists are never grown).

    Guards against #17876: before this fix the code unconditionally replaced any non-dict value (including
    lists) with ``{}``, silently destroying list-typed config like ``custom_providers`` whenever a caller
    used an indexed path.
    Dotted key names (#84064 family): when navigating an existing mapping, an existing literal key equal to
    the dot-join of the next N segments is preferred over blind splitting (see ``_greedy_literal_match``),
    so ``models.grok-4.6.supports_vision`` lands on the real ``grok-4.6`` entry. And when a write WOULD
    create a new intermediate mapping that shadows an existing dotted sibling (``grok-4`` beside
    ``grok-4.5``), it raises ``ValueError`` instead of silently writing a phantom the runtime never reads.
    """
    parts = _split_key_path(dotted_key)
    current = config
    i = 0
    while i < len(parts):
        remaining = parts[i:]
        at_leaf = len(remaining) == 1
        if isinstance(current, list):
            part = remaining[0]
            if at_leaf:
                current[int(part)] = value
                return
            try:
                current = current[int(part)]
            except (TypeError, ValueError):
                raise TypeError(
                    f"Cannot navigate into list at key {dotted_key!r}: "
                    f"segment {part!r} is not a numeric index")
            i += 1
        elif isinstance(current, dict):
            match = _greedy_literal_match(current, remaining)
            if match is not None:
                key, consumed = match
                if i + consumed == len(parts):
                    current[key] = value
                    return
                # Preserve dicts and lists; replace scalar with a fresh dict.
                if not isinstance(current.get(key), (dict, list)):
                    current[key] = {}
                current = current[key]
                i += consumed
                continue
            part = remaining[0]
            if at_leaf:
                current[part] = value
                return
            shadowed = _phantom_sibling(current, part)
            if shadowed is not None:
                escaped = shadowed.replace(".", "\\.")
                raise ValueError(
                    f"Refusing to create nested key {part!r} in {dotted_key!r}: the mapping "
                    f"already contains a literal key {shadowed!r} that contains a dot. If you "
                    f"meant that key, escape its dots with a backslash (e.g. {escaped}).")
            current = current.setdefault(part, {})
            i += 1
        else:
            raise TypeError(f"Cannot navigate into {type(current).__name__} at key {dotted_key!r}")


def clear_model_endpoint_credentials(
    model_cfg: Dict[str, Any], *, clear_api_key: bool = True, clear_api_mode: bool = True,
    clear_base_url: bool = False) -> Dict[str, Any]:
    """Remove stale inline endpoint credentials from a model config.
    ``model.api_key`` is valid only for explicit custom endpoints; built-in providers resolve
    credentials from env/auth.json/the pool. Leftovers keep secrets in config.yaml and can
    contaminate later custom resolution paths."""
    if not isinstance(model_cfg, dict):
        return model_cfg
    if clear_api_key:
        model_cfg.pop("api_key", None)
        model_cfg.pop("api", None)
    if clear_api_mode:
        model_cfg.pop("api_mode", None)
    if clear_base_url:
        model_cfg.pop("base_url", None)
    return model_cfg


_MISSING = object()


def _locate_nested(config, parts: list):
    """Walk *parts* through nested dicts/lists (escape-aware, greedy-literal like ``_set_nested``).
    Returns ``(parents, container, key)`` where ``container[key]`` is the addressed leaf and
    ``parents`` lists the ``(container, key)`` hops above it, or ``None`` when any hop is missing,
    a list index is non-numeric/out of range, or a scalar is hit before the path is consumed."""
    parents = []
    current = config
    i = 0
    while True:
        remaining = parts[i:]
        if isinstance(current, list):
            try:
                key = int(remaining[0])
                current[key]
            except (TypeError, ValueError, IndexError):
                return None
            consumed = 1
        elif isinstance(current, dict):
            match = _greedy_literal_match(current, remaining)
            if match is None:
                return None
            key, consumed = match
        else:
            return None
        i += consumed
        if i == len(parts):
            return parents, current, key
        parents.append((current, key))
        current = current[key]


def _get_nested(config, dotted_key: str):
    """Return a dotted-path value (``_MISSING`` when absent); same navigation as ``_set_nested``
    so ``models.grok-4.6.context_length`` reads the real ``grok-4.6`` entry.

    Mirrors ``_set_nested``'s navigation: honors backslash-escaped dots and prefers an existing literal
    dotted key over blind splitting, so ``config get providers.p.models.grok-4.6.context_length`` reads the
    real ``grok-4.6`` entry instead of reporting the key unset (#84064).
    """
    loc = _locate_nested(config, _split_key_path(dotted_key))
    if loc is None:
        return _MISSING
    _, container, key = loc
    return container[key]


def _unset_nested(config, dotted_key: str) -> bool:
    """Remove a dotted-path value; True if it existed. Empty dict containers left behind are
    dropped, while user-authored empty lists and non-empty sibling branches are preserved.

    Same escape-aware, greedy-literal navigation as ``_set_nested`` / ``_get_nested`` (#84064): unsetting an
    unescaped dotted key removes the real literal entry rather than a phantom sibling.
    """
    loc = _locate_nested(config, _split_key_path(dotted_key))
    if loc is None:
        return False
    parents, current, key = loc
    del current[key]
    # ``parent[part] is current`` for every hop, so each now-empty dict container is dropped.
    for parent, part in reversed(parents):
        if current != {}:
            break
        del parent[part]
        current = parent
    return True


_ENV_CONFIG_KEYS = frozenset({
    'OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'VOICE_TOOLS_OPENAI_KEY',
    'EXA_API_KEY', 'PARALLEL_API_KEY', 'FIRECRAWL_API_KEY', 'FIRECRAWL_API_URL',
    'FIRECRAWL_GATEWAY_URL', 'TOOL_GATEWAY_DOMAIN', 'TOOL_GATEWAY_SCHEME',
    'TOOL_GATEWAY_USER_TOKEN', 'TAVILY_API_KEY', 'PERPLEXITY_API_KEY', 'API_SERVER_KEY',
    'BROWSERBASE_API_KEY', 'BROWSERBASE_PROJECT_ID', 'BROWSER_USE_API_KEY',
    'FAL_KEY', 'TELEGRAM_BOT_TOKEN', 'DISCORD_BOT_TOKEN',
    'TERMINAL_SSH_HOST', 'TERMINAL_SSH_USER', 'TERMINAL_SSH_KEY',
    'SUDO_PASSWORD', 'SLACK_BOT_TOKEN', 'SLACK_APP_TOKEN',
    'GITHUB_TOKEN', 'HONCHO_API_KEY'})


def _is_env_config_key(key: str) -> bool:
    """Return whether `hermes config set` routes this key to .env."""
    if "." in key:
        return False
    key_upper = key.upper()
    return (
        key_upper in _ENV_CONFIG_KEYS
        or key_upper.endswith(('_API_KEY', '_TOKEN', '_SECRET'))
        or key_upper.startswith('TERMINAL_SSH'))


def _format_config_get_value(value, *, as_json: bool) -> str:
    """Format a config value for command-line output."""
    if as_json:
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, sort_keys=False).rstrip()
    return str(value)


def get_missing_config_fields() -> List[Dict[str, Any]]:
    """Check which config fields are missing or outdated (recursive)."""
    missing = []

    def _check(defaults: dict, current: dict, prefix: str = ""):
        for key, default_value in defaults.items():
            if key.startswith('_'):
                continue
            full_key = key if not prefix else f"{prefix}.{key}"
            if key not in current:
                missing.append({"key": full_key, "default": default_value,
                                "description": f"New config option: {full_key}"})
            elif isinstance(default_value, dict) and isinstance(current.get(key), dict):
                _check(default_value, current[key], full_key)

    _check(DEFAULT_CONFIG, load_config())
    return missing


def get_missing_skill_config_vars() -> List[Dict[str, Any]]:
    """Return skill-declared config vars (``skills.config.<key>``) that are missing or empty."""
    try:
        from agent.skill_utils import discover_all_skill_config_vars, SKILL_CONFIG_PREFIX
    except Exception:
        return []

    try:
        all_vars = discover_all_skill_config_vars()
    except Exception as e:
        # A malformed SKILL.md must never break `hermes update`; this prompting is a nicety.
        logger.debug("discover_all_skill_config_vars failed: %s", e)
        return []
    if not all_vars:
        return []

    config = load_config()
    values = ((var, cfg_get(config, *f"{SKILL_CONFIG_PREFIX}.{var['key']}".split("."))) for var in all_vars)
    return [var for var, v in values if v is None or (isinstance(v, str) and not v.strip())]


def _coerce_config_version(value: Any) -> int:
    """Return a safe integer config version, treating invalid values as legacy."""
    if isinstance(value, bool):
        return 0
    try:
        version = int(value)
    except (TypeError, ValueError):
        return 0
    return max(version, 0)


def check_config_version(*, raise_on_parse_error: bool = False) -> Tuple[int, int]:
    """Return ``(current_version, latest_version)`` from the raw on-disk config.
    Reads the raw file rather than ``load_config()``: the deep-merge would make a file lacking
    ``_config_version`` inherit the latest version, hiding that the schema was never migrated.
    Invalid YAML gets a parse warning, not an automatic schema rewrite. Tolerant runtime status
    callers keep the historical latest/latest fallback for malformed YAML; mutation and explicit
    validation paths set ``raise_on_parse_error`` so a parse failure or a non-mapping root cannot
    be mistaken for an up-to-date config."""
    latest = _coerce_config_version(DEFAULT_CONFIG.get("_config_version", 1)) or 1
    config_path = get_config_path()
    if not config_path.exists():
        return latest, latest

    try:
        with open(config_path, encoding="utf-8") as f:
            config = fast_safe_load(f)
    except Exception as e:
        _warn_config_parse_failure(config_path, e)
        if raise_on_parse_error:
            raise InvalidUserConfigError(
                f"Cannot inspect {config_path}: config.yaml is not valid YAML ({e})"
            ) from e
        return latest, latest

    if config is None:
        config = {}  # empty file / bare document: valid first-run state
    if not isinstance(config, dict):
        # A list/scalar root parses fine but is just as unusable as broken YAML: save_config()
        # would refuse it later, after .env was already rewritten. Strict callers see it up front.
        if raise_on_parse_error:
            raise InvalidUserConfigError(
                f"Cannot inspect {config_path}: config.yaml top-level value must be "
                f"a mapping, got {type(config).__name__}"
            )
        config = {}
    return _coerce_config_version(config.get("_config_version")), latest


# ---- Config structure validation ----

# DEFAULT_CONFIG is the single source of truth for documented roots; the set is derived so new
# defaults are accepted automatically. These optional/legacy roots are valid on disk but
# intentionally absent from DEFAULT_CONFIG (omitted when unused / alternate schema forms).
_EXTRA_KNOWN_ROOT_KEYS = {
    "custom_providers",  # legacy list form; modern equivalent is providers: {}
    "fallback_model",    # optional single dict or chain list; omitted when disabled
    "mcp_servers",       # MCP server definitions written by setup/tools flows
    "image_gen",         # agent/image_gen_registry.py
    "video_gen",         # agent/video_gen_registry.py
    "plugins",           # plugin enable/disable lists (hermes_cli/plugins_cmd.py)
    "smart_model_routing",   # written by the setup wizard
    "platform_toolsets",     # written by the setup wizard
    "known_plugin_toolsets", # hermes_cli/tools_config.py toolset-save flow
    "known_builtin_toolsets",  # ditto — builtin toolsets a platform's checklist has offered
    "tool_gateway_declined_tools",  # per-tool Tool Gateway offer declines
    # Top-level forms read/bridged by gateway/config.py:
    "session_reset", "group_sessions_per_user", "thread_sessions_per_user",
    "stt_echo_transcripts", "reset_triggers", "always_log_local", "filter_silence_narration",
    "multiplex_profiles", "profile_routes", "platforms", "require_mention",
    "unauthorized_dm_behavior", "signal",
    "timeouts",          # unified timeout resolution section (agent/deadline.py)
}
_KNOWN_ROOT_KEYS = frozenset(DEFAULT_CONFIG.keys()) | _EXTRA_KNOWN_ROOT_KEYS

# Valid fields inside a custom_providers list entry (key_env is read at runtime by
# runtime_provider.py and auxiliary_client.py).
_VALID_CUSTOM_PROVIDER_FIELDS = {
    "name", "base_url", "api_key", "api_mode", "model", "models",
    "context_length", "rate_limit_delay", "extra_body",
    "ssl_ca_cert", "ssl_verify", "key_env"}

# Fields that look like they should be inside custom_providers, not at root
_CUSTOM_PROVIDER_LIKE_FIELDS = {"base_url", "api_key", "rate_limit_delay", "api_mode"}


@dataclass
class ConfigIssue:
    """A detected config structure problem."""
    severity: str  # "error", "warning"
    message: str
    hint: str


def _issue(issues: List["ConfigIssue"], severity: str, message: str, hint: str) -> None:
    issues.append(ConfigIssue(severity, message, hint))


def _require_fields(
    issues: List["ConfigIssue"], entry: Dict[str, Any], label: str,
    fields: Tuple[Tuple[str, str], ...], suffix: str = "") -> None:
    """Append a warning for every falsy ``field`` of *entry* (message: ``<label> is missing '<f>' field``)."""
    for field, hint in fields:
        if not entry.get(field):
            _issue(issues, "warning", f"{label} is missing '{field}' field{suffix}", hint)


_CP_REQUIRED_FIELDS = (
    ("name", "Add a name, e.g.: name: my-provider"),
    ("base_url", "Add the API endpoint URL, e.g.: base_url: https://api.example.com/v1"))
_FB_REQUIRED_FIELDS = (
    ("provider", "Add: provider: openrouter (or another provider)"),
    ("model", "Add: model: <model-name>"))
_FB_SINGLE_REQUIRED_FIELDS = (
    ("provider", "Add: provider: openrouter (or another provider)"),
    ("model", "Add: model: anthropic/claude-sonnet-4 (or another model)"))


def _validate_voice(config: Dict[str, Any], issues: List[ConfigIssue]) -> None:
    voice_cfg = config.get("voice")
    if not (isinstance(voice_cfg, dict) and "submit_mode" in voice_cfg):
        return
    submit_mode = voice_cfg.get("submit_mode")
    normalized = submit_mode.strip().lower() if isinstance(submit_mode, str) else None
    if normalized not in {"direct", "draft"}:
        _issue(issues, "error", f"voice.submit_mode must be 'direct' or 'draft', got {submit_mode!r}",
               "Set voice.submit_mode to direct (submit immediately) or draft (edit before sending)")


def _validate_entry_list(
    entries: list, label: str, issues: List[ConfigIssue], fields, *, non_dict: Tuple[str, str, str],
) -> None:
    """Validate each list entry: ``non_dict`` = (severity, message-with-{i}-and-{type}, hint) for
    non-dict items; dict items get ``_require_fields`` with *fields*."""
    severity, message, hint = non_dict
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            _issue(issues, severity, message.format(i=i, type=type(entry).__name__), hint)
        else:
            _require_fields(issues, entry, f"{label}[{i}]", fields)


def _validate_custom_providers(cp: Any, issues: List[ConfigIssue]) -> None:
    """custom_providers must be a list of dicts, not a dict."""
    if isinstance(cp, dict):
        _issue(issues, "error",
               "custom_providers is a dict — it must be a YAML list (items prefixed with '-')",
               "Change to:\n  custom_providers:\n    - name: my-provider\n      base_url: https://...\n"
               "      api_key: ...")
        suspicious = set(cp.keys()) & _CUSTOM_PROVIDER_LIKE_FIELDS
        if suspicious:
            _issue(issues, "warning",
                   f"Root-level keys {sorted(suspicious)} look like custom_providers entry fields",
                   "These should be indented under a '- name: ...' list entry, not at root level")
    elif isinstance(cp, list):
        _validate_entry_list(cp, "custom_providers", issues, _CP_REQUIRED_FIELDS, non_dict=(
            "warning", "custom_providers[{i}] is not a dict (got {type})",
            "Each entry should have at minimum: name, base_url"))


def _validate_fallback_model(fb: Any, issues: List[ConfigIssue]) -> None:
    """fallback_model: single dict OR list of dicts (chain)."""
    if isinstance(fb, list):
        _validate_entry_list(fb, "fallback_model", issues, _FB_REQUIRED_FIELDS, non_dict=(
            "error", "fallback_model[{i}] should be a dict, got {type}", "Each entry needs provider + model"))
    elif not isinstance(fb, dict):
        _issue(issues, "error",
               f"fallback_model should be a dict with 'provider' and 'model', got {type(fb).__name__}",
               "Change to:\n  fallback_model:\n    provider: openrouter\n    model: anthropic/claude-sonnet-4")
    elif fb:
        _require_fields(issues, fb, "fallback_model", _FB_SINGLE_REQUIRED_FIELDS,
                        suffix=" — fallback will be disabled")


def _validate_web_backends(config: Dict[str, Any], issues: List[ConfigIssue]) -> None:
    """A stale web backend selection otherwise fails only at the first web_search/web_extract
    call with a generic "no registered provider" error; warn at startup instead."""
    # See #99199.
    web_cfg = config.get("web")
    if not isinstance(web_cfg, dict):
        return
    try:
        from tools.tool_backend_helpers import removed_backend_note
    except Exception:
        return
    seen: set = set()
    for _key in ("backend", "search_backend", "extract_backend"):
        _val = str(web_cfg.get(_key) or "").strip().lower()
        if not _val or _val in seen:
            continue
        seen.add(_val)
        note = removed_backend_note("web", _val)
        if note:
            _issue(issues, "warning",
                   f"web.{_key} is set to '{_val}', but {note} — "
                   "web_search/web_extract will fail until it is changed",
                   "Run 'hermes tools' and pick a different Web Search & Extract provider")


def validate_config_structure(config: Optional[Dict[str, Any]] = None) -> List["ConfigIssue"]:
    """Validate config.yaml structure and return detected issues (accepts a pre-loaded dict).
    Catches common YAML mistakes that otherwise surface as confusing runtime errors."""
    if config is None:
        try:
            config = load_config()
        except Exception:
            return [ConfigIssue("error", "Could not load config.yaml", "Run 'hermes setup' to create a valid config")]

    issues: List[ConfigIssue] = []
    _validate_voice(config, issues)
    cp = config.get("custom_providers")
    fb = config.get("fallback_model")
    for value, validator in ((cp, _validate_custom_providers), (fb, _validate_fallback_model)):
        if value is not None:
            validator(value, issues)

    if isinstance(cp, dict) and "fallback_model" not in config and "fallback_model" in (cp or {}):
        _issue(issues, "error", "fallback_model appears inside custom_providers instead of at root level",
               "Move fallback_model to the top level of config.yaml (no indentation)")

    if cp and not config.get("model"):
        _issue(issues, "warning",
               "custom_providers defined but no 'model' section — Hermes won't know which provider to use",
               "Add a model section:\n  model:\n    provider: custom\n    default: your-model-name\n"
               "    base_url: https://...")

    # Only provider-like fields are flagged as misplaced roots. Arbitrary unknown top-level keys
    # are deliberately NOT warned about: top-level scalars are bridged into os.environ so users
    # can feed skills/external apps env-style keys — a closed-world allowlist cannot enumerate those.
    for key in config:
        if not key.startswith("_") and key not in _KNOWN_ROOT_KEYS and key in _CUSTOM_PROVIDER_LIKE_FIELDS:
            _issue(issues, "warning",
                   f"Root-level key '{key}' looks misplaced — should it be under 'model:' or inside a 'custom_providers' entry?",
                   f"Move '{key}' under the appropriate section")

    _validate_web_backends(config, issues)
    return issues


def print_config_warnings(config: Optional[Dict[str, Any]] = None) -> None:
    """Print config structure warnings to stderr at startup; nothing if config is healthy."""
    try:
        issues = validate_config_structure(config)
    except Exception:
        issues = []
    if not issues:
        return

    lines = ["\033[33m⚠ Config issues detected in config.yaml:\033[0m"]
    for ci in issues:
        marker = "\033[31m✗\033[0m" if ci.severity == "error" else "\033[33m⚠\033[0m"
        lines.append(f"  {marker} {ci.message}")
    lines.append("  \033[2mRun 'hermes doctor' for fix suggestions.\033[0m")
    sys.stderr.write("\n".join(lines) + "\n\n")


def warn_deprecated_cwd_env_vars() -> None:
    """Warn if MESSAGING_CWD / TERMINAL_CWD is set in .env (canonical: terminal.cwd in config.yaml).
    Reads the file rather than ``os.environ`` because runtime bridges and session restoration
    legitimately set ``TERMINAL_CWD``."""
    try:
        env_map = load_env()
    except Exception:
        return

    lines: list[str] = []
    for name in ("MESSAGING_CWD", "TERMINAL_CWD"):
        val = str(env_map.get(name) or "").strip()
        if val:
            lines.append(f"  \033[33m⚠\033[0m {name}={val} found in .env — this is deprecated.")
    if lines:
        from hermes_constants import display_hermes_home

        hint_path = display_hermes_home()
        lines.insert(0, "\033[33m⚠ Deprecated .env settings detected:\033[0m")
        lines.append(
            "  \033[2mMove to config.yaml instead:  "
            "terminal:\\n    cwd: /your/project/path\033[0m")
        lines.append(f"  \033[2mThen remove the old entries from {hint_path}/.env\033[0m")
        sys.stderr.write("\n".join(lines) + "\n\n")


def _persist_migration(config: Dict[str, Any]) -> None:
    """Persist a migrated config under THE migration write invariant: a migration may only
    persist values that DIFFER from the schema default, plus explicit removals/renames of user
    data. Every migration step MUST write through here (``save_config`` with default-stripping
    ON, no ``merge_existing``) so the invariant cannot regress one migration at a time."""
    save_config(config)


def _prompt_and_save_env(name: str, info: Dict[str, Any], prompt: str, results: Dict[str, Any]) -> bool:
    """Prompt for one env var (masked when ``info['password']``), save it, record it; False if skipped."""
    value = masked_secret_prompt(prompt) if info.get("password") else line_input(prompt).strip()
    if not value:
        return False
    save_env_value(name, value)
    results["env_added"].append(name)
    print(f"  ✓ Saved {name}")
    return True


def _ask_yes_no(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer in {"y", "yes"}


def migrate_config(interactive: bool = True, quiet: bool = False) -> Dict[str, Any]:
    """Migrate config to latest version, prompting for new required fields."""
    results = {"env_added": [], "config_added": [], "warnings": []}

    # Validate config.yaml before any migration side effect: sanitize_env_file() rewrites .env,
    # which must not happen when the migration will be refused for malformed YAML.
    current_ver, latest_ver = check_config_version(raise_on_parse_error=True)

    try:
        fixes = sanitize_env_file()
        if fixes and not quiet:
            print(f"  ✓ Normalized .env line formatting ({fixes} line(s) changed)")
    except Exception:
        pass  # best-effort; never block migration on sanitize failure

    # Auto-migration support floor (v12): an EXPLICIT on-disk ``_config_version`` below the
    # floor is NOT migrated and NOT rewritten — surface a message and leave the file untouched
    # (deep-merge supplies defaults at read time). A config with NO version key is a fresh
    # minimal config, not an ancient install: it gets the normal ladder and a version stamp.
    # Missing/unparseable files never trip the floor gate.
    # Imported lazily because the steps call back into this module.
    from hermes_cli.config_migrations import (
        SUPPORT_FLOOR_VERSION, run_migrations, support_floor_message)

    try:
        has_explicit_version = "_config_version" in read_user_config_raw()
    except Exception:
        has_explicit_version = False
    floor_refused = (
        has_explicit_version and current_ver < SUPPORT_FLOOR_VERSION and current_ver < latest_ver)
    if floor_refused:
        msg = support_floor_message()
        results["warnings"].append(msg)
        # stderr so it is visible even on quiet startup paths.
        sys.stderr.write(f"⚠ hermes config: {msg}\n")
        if not quiet:
            print(f"  ⚠ {msg}")
    else:
        run_migrations(current_ver, results, quiet)

    _disable_suspicious_mcp_servers(results, quiet)
    _warn_invalid_platform_toolsets(results, quiet)

    if current_ver < latest_ver and not quiet and not floor_refused:
        print(f"Config version: {current_ver} → {latest_ver}")

    missing_env = get_missing_env_vars(required_only=True)
    if missing_env and not quiet:
        print("\n⚠️  Missing required environment variables:")
        for var in missing_env:
            print(f"   • {var['name']}: {var['description']}")
    if interactive and missing_env:
        print("\nLet's configure them now:\n")
        for var in missing_env:
            if var.get("url"):
                print(f"  Get your key at: {var['url']}")
            if not _prompt_and_save_env(var["name"], var, f"  {var['prompt']}: ", results):
                results["warnings"].append(f"Skipped {var['name']} - some features may not work")
            print()

    if interactive and not quiet:
        _offer_new_optional_env_vars(current_ver, latest_ver, results)

    # New default keys are NOT materialised to disk (load_config() deep-merges DEFAULT_CONFIG at
    # read time); this list only feeds the "N new config option(s)" display.
    results["config_added"].extend(field["key"] for field in get_missing_config_fields())

    if current_ver < latest_ver and not floor_refused:
        config = read_raw_config()
        config["_config_version"] = latest_ver
        _persist_migration(config)

    missing_skill_config = get_missing_skill_config_vars()
    if missing_skill_config and interactive and not quiet:
        _offer_skill_config_vars(missing_skill_config, results)

    return results


def _disable_suspicious_mcp_servers(results: Dict[str, Any], quiet: bool) -> None:
    """Post-migration: disable exfiltration-shaped MCP stdio entries (hand-edited or from older
    installs). The stanza is preserved for auditability but marked disabled."""
    config = read_raw_config()
    # Preserve the stanza for auditability but mark it disabled so the next startup will not spawn it.
    # (#45620)
    raw_mcp_servers = config.get("mcp_servers")
    if not isinstance(raw_mcp_servers, dict):
        return
    try:
        from hermes_cli.mcp_security import validate_mcp_server_entry
    except Exception:
        return
    mcp_touched = False
    for server_name, entry in raw_mcp_servers.items():
        issues = validate_mcp_server_entry(server_name, entry) if isinstance(entry, dict) else None
        if not issues:
            continue
        entry["enabled"] = False
        mcp_touched = True
        results["warnings"].append(f"Disabled suspicious MCP server '{server_name}'")
        if not quiet:
            for issue in issues:
                print(f"  ⚠ {issue}")
            print(f"  ⚠ Disabled MCP server '{server_name}' pending review")
    if mcp_touched:
        config["mcp_servers"] = raw_mcp_servers
        _persist_migration(config)


def _warn_invalid_platform_toolsets(results: Dict[str, Any], quiet: bool) -> None:
    """Surface invalid toolset names in platform_toolsets: ``resolve_toolset()`` returns [] for an
    unknown name, silently disabling the affected tools. Best-effort; never blocks migration."""
    try:
        from toolsets import validate_toolset
        from hermes_cli.toolset_validation import validate_platform_toolsets
        from hermes_cli.toolset_scope import toolset_allowed_for_platform

        for w in validate_platform_toolsets(
                read_raw_config().get("platform_toolsets"), validate_toolset, toolset_allowed_for_platform):
            results["warnings"].append(w)
            if not quiet:
                print(f"  ⚠ {w}")
    except Exception as _ts_val_err:
        logger.debug("platform_toolsets validation skipped: %s", _ts_val_err)


def _offer_list(heading: str, items: List[str], question: str) -> bool:
    """Print a bulleted offer list and ask; False (with the "set later" hint) when declined."""
    print(heading)
    for item in items:
        print(f"    • {item}")
    print()
    if not _ask_yes_no(question):
        print("  Set later with: hermes config set <key> <value>")
        return False
    print()
    return True


def _offer_new_optional_env_vars(current_ver: int, latest_ver: int, results: Dict[str, Any]) -> None:
    """Interactively offer env vars that are NEW since the user's previous config version."""
    new_var_names: set = set()
    for ver in range(current_ver + 1, latest_ver + 1):
        new_var_names.update(ENV_VARS_BY_VERSION.get(ver, []))
    new_and_unset = [
        (name, OPTIONAL_ENV_VARS[name])
        for name in sorted(new_var_names)
        if not get_env_value(name) and name in OPTIONAL_ENV_VARS]
    if not new_and_unset or not _offer_list(
        f"\n  {len(new_and_unset)} new optional key(s) in this update:",
        [f"{name} — {info.get('description', '')}" for name, info in new_and_unset],
        "  Configure new keys? [y/N]: "):
        return
    for name, info in new_and_unset:
        print(f"  {info.get('description', name)}")
        if info.get("url"):
            print(f"  Get your key at: {info['url']}")
        _prompt_and_save_env(name, info, f"  {info.get('prompt', name)} (Enter to skip): ", results)
        print()


def _offer_skill_config_vars(missing_skill_config: List[Dict[str, Any]], results: Dict[str, Any]) -> None:
    """Prompt for skill-declared settings that are missing/empty and persist the answers."""
    if not _offer_list(
        f"\n  {len(missing_skill_config)} skill setting(s) not configured:",
        [f"{v['key']} — {v['description']} (from skill: {v.get('skill', 'unknown')})" for v in missing_skill_config],
        "  Configure skill settings? [y/N]: "):
        return
    config = read_raw_config()
    try:
        from agent.skill_utils import SKILL_CONFIG_PREFIX
    except Exception:
        SKILL_CONFIG_PREFIX = "skills.config"
    for var in missing_skill_config:
        default = var.get("default", "")
        default_hint = f" (default: {default})" if default else ""
        value = line_input(f"  {var['prompt']}{default_hint}: ").strip() or str(default or "")
        if value:
            _set_nested(config, f"{SKILL_CONFIG_PREFIX}.{var['key']}", value)
            results["config_added"].append(var["key"])
            print(f"  ✓ Saved {var['key']} = {value}")
        else:
            results["warnings"].append(
                f"Skipped {var['key']} — skill '{var.get('skill', '?')}' may ask for it later")
        print()
    _persist_migration(config)


def _merge_partial_save(raw: dict, override: dict) -> dict:
    """Merge *override* over *raw* for partial ``save_config`` writes.
    Omitted top-level sections are preserved; shared dict sections deep-merge so one nested key
    can change without dropping siblings on disk. Key REMOVALS are not supported here —
    migrations go through ``_persist_migration`` with a full ``read_raw_config()`` dict."""
    result = copy.deepcopy(override)
    for key, value in raw.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
        elif isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(value, result[key])
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*: dict-over-dict recurses (so overriding one leaf
    keeps sibling defaults), and ``None`` over a dict section is ignored.

    An empty section key in config.yaml (``terminal:`` with no value) parses as YAML ``None``; treating that
    as an override would replace the entire default dict with ``None`` and crash every downstream consumer
    that expects a mapping (#58277).
    """
    result = base.copy()
    for key, value in override.items():
        over_dict = isinstance(result.get(key), dict)
        if over_dict and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif not (over_dict and value is None):
            result[key] = value
    return result


def _strip_dotted_keys(cfg: dict, dotted_keys: set) -> Tuple[dict, set]:
    """Remove dotted leaf keys from *cfg* in place -> ``(cfg, keys_actually_present)``.
    ``save_config`` drops managed-scope leaves this way so a bulk write never persists a user
    value that would lose to the managed layer on the next load."""
    stripped: set = set()
    for dotted in dotted_keys:
        *parents, leaf = dotted.split(".")
        node = cfg_get(cfg, *parents)
        if isinstance(node, dict) and leaf in node:
            del node[leaf]
            stripped.add(dotted)
    return cfg, stripped


_ENV_REF_RE = re.compile(r"\${([^}]+)}")


def _env_ref_lookup(name: str) -> Optional[str]:
    """Resolve the env var behind a ``${VAR}`` / ``${env:VAR}`` ref — plain ``os.environ`` outside
    a profile secret scope (legacy behavior for the default profile).

    Inside a scope (a multiplexed gateway turn, a secondary profile's config load, a cron job) the read goes
    through ``agent.secret_scope.get_secret`` so the ref resolves against *that* profile's ``.env``: under
    multiplexing a miss is a miss, never another profile's ``os.environ`` value (#84079 — every profile
    "had" the default profile's ``${MATRIX_ACCESS_TOKEN}`` and fanned out). Same policy as
    ``gateway.config._getenv`` and ``get_env_value``.
    """
    try:
        from agent.secret_scope import current_secret_scope, get_secret as _get_secret
    except Exception:
        return os.environ.get(name)
    if current_secret_scope() is None:
        return os.environ.get(name)
    return _get_secret(name)


def _env_expand_match(m: re.Match) -> str:
    """Expand one ``${VAR}`` (legacy bare name) or ``${env:VAR}`` (Cursor-style SecretRef).
    Other SecretRef sources (``file:``, ``bitwarden:``, ``vault:``...) are NOT resolved here:
    external backends inject their values into the environment at startup (the ``secrets:``
    block), so a config ref only ever needs the env shape. Unresolved refs stay verbatim so
    callers can detect them."""
    raw = m.group(0)
    inner = m.group(1).strip()
    name = _env_ref_var_name(inner)
    if name is None:
        if not inner.startswith("env:") and _is_non_env_secret_ref(inner):
            logger.warning(
                "Config ref %r uses source %r which is not resolvable in "
                "config.yaml — external secret sources inject env vars at "
                "startup, so reference the variable as ${env:NAME} instead",
                raw, inner.split(":", 1)[0])
        return raw  # non-env source, or empty ``${env:}``
    val = _env_ref_lookup(name)
    if val is not None:
        return val
    if inner.startswith("env:"):
        logger.warning(
            "Config ref %r: %s is not set (check ~/.hermes/.env); "
            "keeping the literal placeholder", raw, name)
    return raw


def _is_non_env_secret_ref(ref: str) -> bool:
    """True for a SecretRef body with a non-``env`` source (``bitwarden:FOO``, ``vault:...``)."""
    return ":" in ref and re.match(r"^[a-z][a-z0-9_-]*:", ref) is not None


def _env_ref_var_name(ref: str) -> Optional[str]:
    """Env-var name a ``${...}`` body reads, or None for a non-env source / empty ``env:``."""
    ref = ref.strip()
    if ref.startswith("env:"):
        return ref[len("env:"):].strip() or None
    if _is_non_env_secret_ref(ref):
        return None
    return ref


def _expand_env_vars(obj):
    """Recursively expand ``${VAR}`` / ``${env:VAR}`` in string values (keys/non-strings untouched)."""
    if isinstance(obj, str):
        return _ENV_REF_RE.sub(_env_expand_match, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def _env_ref_snapshot(obj, snapshot=None):
    """Map each env-sourced ``${...}`` ref in *obj* to its current value.
    Stored with cached ``load_config()`` results so a cache hit can detect that the expansion was
    made against a different environment (load before ``load_hermes_dotenv()``, in-process
    rotation) — file mtime/size alone cannot see either.

    See #58514.
    """
    if snapshot is None:
        snapshot = {}
    if isinstance(obj, str):
        for raw in _ENV_REF_RE.findall(obj):
            name = _env_ref_var_name(raw)
            if name is not None:
                snapshot[name] = _env_ref_lookup(name)
    elif isinstance(obj, dict):
        for value in obj.values():
            _env_ref_snapshot(value, snapshot)
    elif isinstance(obj, list):
        for item in obj:
            _env_ref_snapshot(item, snapshot)
    return snapshot


def _items_by_unique_name(items):
    """Return a name-indexed dict only when all items have unique string names."""
    if not isinstance(items, list):
        return None
    indexed = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        name = item["name"]
        if name in indexed:
            return None
        indexed[name] = item
    return indexed


def _preserve_env_ref_templates(current, raw, loaded_expanded=None):
    """Restore raw ``${VAR}`` templates where the value is otherwise unchanged, so persisting a
    loaded (expanded) config never writes the plaintext secret back to ``config.yaml``."""
    if isinstance(current, str) and isinstance(raw, str) and _ENV_REF_RE.search(raw):
        if current in (raw, loaded_expanded) or _expand_env_vars(raw) == current:
            return raw
        return current

    if isinstance(current, dict) and isinstance(raw, dict):
        return {
            key: _preserve_env_ref_templates(
                value, raw.get(key),
                loaded_expanded.get(key) if isinstance(loaded_expanded, dict) else None)
            for key, value in current.items()}

    if isinstance(current, list) and isinstance(raw, list):
        # Match named objects (e.g. custom_providers) by name so reordering keeps templates;
        # with duplicate names fall back to positional matching rather than shadowing an entry.
        current_by_name = _items_by_unique_name(current)
        raw_by_name = _items_by_unique_name(raw)
        loaded_by_name = _items_by_unique_name(loaded_expanded)
        if current_by_name is not None and raw_by_name is not None:
            return [
                _preserve_env_ref_templates(
                    item, raw_by_name.get(item.get("name")),
                    loaded_by_name.get(item.get("name")) if loaded_by_name is not None else None)
                for item in current]
        return [
            _preserve_env_ref_templates(
                item,
                raw[index] if index < len(raw) else None,
                loaded_expanded[index]
                if isinstance(loaded_expanded, list) and index < len(loaded_expanded)
                else None)
            for index, item in enumerate(current)]

    return current


def _explicit_config_paths(config: Dict[str, Any]) -> Set[Tuple[str, ...]]:
    """Leaf paths explicitly present in a RAW (un-normalized) config, so values injected by
    normalisation are never mistaken for user-set ones. Feeds ``_strip_default_values``."""
    paths: Set[Tuple[str, ...]] = set()

    def _walk(value: Any, path: Tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                _walk(child, path + (key,))
        elif path:
            paths.add(path)

    _walk(config, ())
    return paths


def _strip_default_values(
    config: Dict[str, Any], defaults: Dict[str, Any] = DEFAULT_CONFIG,
    preserve_keys: Optional[Set[Tuple[str, ...]]] = None) -> Dict[str, Any]:
    """Return *config* without keys whose values match *defaults*.
    Paths in *preserve_keys* (explicitly present in the user's raw config) are always kept even
    when equal to the default. Dicts whose every child is stripped are removed entirely so
    default-only subtrees never bloat ``config.yaml``."""
    preserve_keys = {("_config_version",)} | set(preserve_keys or ())

    def _strip(value: Any, default: Any, path: Tuple[str, ...]) -> Any:
        if path in preserve_keys:
            return copy.deepcopy(value)
        if isinstance(value, dict) and value:
            default_dict = default if isinstance(default, dict) else {}
            stripped = {k: _strip(v, default_dict.get(k), path + (k,)) for k, v in value.items()}
            return {k: v for k, v in stripped.items() if v is not None} or None
        return None if value == default else copy.deepcopy(value)

    return _strip(config, defaults, ()) or {}


def split_model_config_default(raw_default: Any) -> tuple[str, str]:
    """Canonicalize ``model.default``/``model.model`` -> ``(model, provider)``; a dict value pairs
    the model string with the provider it must be routed through."""
    if isinstance(raw_default, dict):
        provider = str(raw_default.get("provider") or "").strip()
        model = raw_default.get("model") or raw_default.get("default")
        return (str(model or "").strip(), provider)
    return (str(raw_default or "").strip(), "")


def _normalize_root_model_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize the ``model`` section at the single load/save chokepoint.
    Root-level ``provider``/``base_url``/``context_length`` (older layouts) are moved under
    ``model`` only when the corresponding ``model.*`` key is empty — never overriding. ``api_base``
    (the OpenAI-SDK/LiteLLM name users reach for) is an alias for ``base_url``; the runtime reads
    only ``model.base_url``. A dict-valued ``default``/``model``/``name`` is flattened so no reader
    sees a nested dict, and the id is canonicalized to ``default``.

    Also aliases ``api_base`` → ``base_url`` (issue #8919). ``api_base`` is the intuitive name OpenAI-SDK /
    LiteLLM users reach for, and ``hermes config set`` blindly accepts any dotted key — so
    ``model.api_base`` got written, confirmed, and then silently ignored by the runtime resolver (which
    reads only ``model.base_url``), causing requests to fall back to OpenRouter. We migrate the alias to the
    canonical key (fallback-only — never override an explicit ``base_url``) and drop the alias so it can't
    confuse later loads.
    Finally, canonicalizes the model-id key to ``model.default`` (issue #34500). The runtime resolver and
    ~14 other readers select the chat model via ``model.default``; ``model.model`` was already aliased
    inline at some sites but ``model.name`` was not, so a custom-provider config like ``model: {name: <id>,
    provider: <custom>}`` resolved to an empty model and the API request went out with ``model=`` (HTTP 400
    from OpenAI-compatible backends) — while display paths (``hermes status``/``dump``) read ``name`` and
    *showed* the model, making the failure silent. Normalizing here (the single load/save chokepoint) means
    every reader, present and future, sees a populated ``default`` and the stale alias is migrated out of
    config.yaml on the next save. Precedence: ``default`` > ``model`` > ``name`` (never overrides an
    explicit ``default``, so existing configs are unaffected).
    """
    model_in = config.get("model")
    needs_model_work = isinstance(model_in, dict) and (
        model_in.get("api_base")
        or model_in.get("model") or model_in.get("name")
        or any(isinstance(model_in.get(k), dict) for k in ("default", "model", "name")))
    has_root = any(config.get(k) for k in ("provider", "base_url", "context_length", "api_base"))
    if not has_root and not needs_model_work:
        return config

    config = dict(config)
    model = config.get("model")
    model = dict(model) if isinstance(model, dict) else {"default": model} if model else {}
    config["model"] = model

    # Flatten ``{provider: <p>, model: <m>}``. The nested provider wins over the merged default
    # ``"auto"`` (which runtime resolution treats as authoritative) but never over a configured one.
    for _key in ("default", "model", "name"):
        _val = model.get(_key)
        if isinstance(_val, dict):
            _nested_model = _val.get("model") or _val.get("default")
            _nested_provider = str(_val.get("provider") or "").strip()
            model[_key] = str(_nested_model or "").strip()
            if _nested_provider:
                _outer_provider = str(model.get("provider") or "").strip()
                if not _outer_provider or _outer_provider == "auto":
                    model["provider"] = _nested_provider

    for key in ("provider", "base_url", "context_length"):
        root_val = config.get(key)
        if root_val and not model.get(key):
            model[key] = root_val
        config.pop(key, None)

    for alias_val in (config.get("api_base"), model.get("api_base")):
        if alias_val and not model.get("base_url"):
            model["base_url"] = alias_val
    config.pop("api_base", None)
    model.pop("api_base", None)

    # ``model``/``name`` are last-resort aliases (in that order), then dropped.
    alias = model.get("model") or model.get("name")
    if not model.get("default") and alias:
        model["default"] = alias
    if model.get("default"):
        model.pop("model", None)
        model.pop("name", None)

    return config


def _normalize_max_turns_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Move legacy root-level ``max_turns`` under ``agent``; the schema default is injected only
    when the user set max_turns somewhere (so save_config can otherwise omit it)."""
    config = dict(config)
    agent_config = dict(config.get("agent") or {})
    if "max_turns" in config and "max_turns" not in agent_config:
        agent_config["max_turns"] = config["max_turns"]
    config["agent"] = agent_config
    config.pop("max_turns", None)
    return config


def _normalize_providers_string(config: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a JSON-string-typed ``providers`` value into a dict.

    ``providers`` is an open-dict top-level key (``_OPEN_DICT_TOP_LEVEL_KEYS``),
    so config validation accepts *any* shape under it and never complains when a
    serialization round-trip (e.g. a dashboard/config write-back through
    ``json.dumps``) persists the whole section as a single JSON string scalar:

        providers: '{"custom": {"base_url": "...", "models": {...}}}'

    Every consumer guards with ``isinstance(providers, dict)`` and silently
    degrades to "no custom providers configured" — per-model ``supports_vision``
    overrides (``agent.image_routing``) and custom-provider identity lookups
    (``hermes_cli.runtime_provider``) never resolve, with zero warnings. Decode
    the string here, at the single normalization chokepoint shared by the user
    and managed configs, so every downstream reader sees the dict shape it
    expects. Runs before ``_expand_env_vars`` so ``${VAR}`` refs inside the
    decoded block still expand normally. A non-dict or malformed payload falls
    back to ``{}`` (with a one-time warning) rather than surviving as a string.
    """
    raw = config.get("providers")
    if not isinstance(raw, str):
        return config
    config = dict(config)
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        decoded = None
    if isinstance(decoded, dict):
        config["providers"] = decoded
    else:
        logger.warning(
            "Ignoring malformed 'providers' config: expected a mapping but got "
            "a %s-typed value; treating it as empty. Custom providers and "
            "per-model overrides will not resolve until this is fixed.",
            "JSON string decoding to non-dict" if decoded is not None else "string",
        )
        config["providers"] = {}
    return config


def _canonicalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The load/save normalization pipeline: max_turns relocation, model-section canon, providers-string decode."""
    return _normalize_providers_string(
        _normalize_root_model_keys(_normalize_max_turns_config(config))
    )


# Sentinel for an unlimited turn budget. ``sys.maxsize`` survives the str->int round-trip through
# the HERMES_MAX_ITERATIONS env bridge, works in every ``<``/``>=``/``max - used`` comparison in
# the iteration budget without an "unlimited" special case, and is unreachable in practice.
TURN_LIMIT_UNLIMITED = sys.maxsize

# Spellings that mean "no limit" (compared lowercased, whitespace-stripped).
_UNLIMITED_SPELLINGS = frozenset({
    "none", "null", "unlimited", "infinite", "infinity", "inf", "∞", "-1", "0"})


def resolve_turn_limit(raw: Any, default: int = TURN_LIMIT_UNLIMITED) -> int:
    """Normalize a raw ``agent.max_turns`` value into an int iteration cap (always >= 1)."""
    # bool is a subclass of int; reject explicitly so True/False don't become 1/0.
    if raw is None or isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        n = int(raw)
    elif isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            return default
        if s in _UNLIMITED_SPELLINGS:
            return TURN_LIMIT_UNLIMITED
        try:
            n = int(s)
        except ValueError:
            try:
                n = int(float(s))
            except ValueError:
                logger.debug("resolve_turn_limit: unparseable value %r → default %d", raw, default)
                return default
    else:
        # Unknown type (list, dict, …) — don't crash the agent over a bad config.
        logger.debug("resolve_turn_limit: unsupported type %s (%r) → default %d", type(raw).__name__, raw, default)
        return default
    return TURN_LIMIT_UNLIMITED if n <= 0 else n


def cfg_get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Traverse nested dict keys safely, returning ``default`` on any miss.
    Explicit ``None`` values are returned as-is (``dict.get`` semantics: ``default`` only when the
    key is absent). Named ``cfg_get`` to avoid shadowing the ubiquitous ``cfg_path`` local."""
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _read_raw_config_impl(*, want_deepcopy: bool) -> Dict[str, Any]:
    with _CONFIG_LOCK:
        try:
            config_path = get_config_path()
            st = config_path.stat()
            cache_key = (st.st_mtime_ns, st.st_size)
        except (FileNotFoundError, OSError):
            return {}

        path_key = str(config_path)
        cached = _RAW_CONFIG_CACHE.get(path_key)
        if cached is not None and cached[:2] == cache_key:
            return copy.deepcopy(cached[2]) if want_deepcopy else cached[2]

        try:
            with open(config_path, encoding="utf-8") as f:
                data = fast_safe_load(f) or {}
        except Exception as e:
            _warn_config_parse_failure(config_path, e)
            return {}

        if not isinstance(data, dict):
            data = {}
        # The cache stores its own deepcopy. The readonly path returns THAT object (identity
        # invariant: later cache hits return the same dict); the mutable path returns the parse.
        cached_copy = copy.deepcopy(data)
        _RAW_CONFIG_CACHE[path_key] = (cache_key[0], cache_key[1], cached_copy)
        return data if want_deepcopy else cached_copy


def read_raw_config() -> Dict[str, Any]:
    """Read config.yaml as-is (no defaults merged, no migration); ``{}`` if missing/unparseable.
    Cached on (mtime_ns, size); returns a deepcopy since callers mutate before ``save_config()``."""
    return _read_raw_config_impl(want_deepcopy=True)


def read_user_config_raw(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read a user ``config.yaml`` EXACTLY as written (no defaults/overlay/expansion, no cache).
    ONLY legal for write-back round-trips and raw-file diagnostics — behavioral reads must use
    load_config()/load_config_readonly()."""
    if config_path is None:
        config_path = get_config_path()
    try:
        with open(config_path, encoding="utf-8") as f:
            data = fast_safe_load(f) or {}
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def read_raw_config_readonly() -> Dict[str, Any]:
    """``read_raw_config()`` without the per-call deepcopy, for callers that ONLY READ.
    **Mutating the result corrupts the in-process cache for every subsequent caller.** Meant for
    per-turn policy checks that were paying a full config deepcopy 2-3x per agent turn."""
    return _read_raw_config_impl(want_deepcopy=False)


def _refuse_overwrite(config_path: Path, reason: str, exc: Exception, fix: str) -> RuntimeError:
    return RuntimeError(f"Refusing to overwrite {config_path}: existing config.yaml {reason} ({exc}). {fix}")


_FIX_PERMS = "Fix the file permissions or move it aside first."
_FIX_YAML = "Fix the file or restore from a .corrupt.*.bak backup first."


def require_readable_config_before_write(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Refuse to replace an existing config.yaml that cannot be read or parsed; return the mapping.
    Guards two collapse-to-empty failure modes that would let a read-then-write caller silently
    wipe user overrides: an unreadable file (permissions / broken mount) and an unparseable or
    non-mapping root — bare-``except`` loaders treat both as ``{}``, so a subsequent write would
    replace the recoverable file with only the caller's partial dict. Fails closed."""
    if config_path is None:
        config_path = get_config_path()
    try:
        config_path.stat()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise _refuse_overwrite(config_path, "cannot be accessed", exc, _FIX_PERMS) from exc

    try:
        with open(config_path, encoding="utf-8") as f:
            loaded = fast_safe_load(f)
    except OSError as exc:
        raise _refuse_overwrite(config_path, "cannot be read", exc, _FIX_PERMS) from exc
    except Exception as exc:
        _warn_config_parse_failure(config_path, exc, fallback="refuse-write")
        raise _refuse_overwrite(config_path, "is not valid YAML", exc, _FIX_YAML) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        exc = TypeError(f"top-level YAML must be a mapping, got {type(loaded).__name__}")
        _warn_config_parse_failure(config_path, exc, fallback="refuse-write")
        raise RuntimeError(
            f"Refusing to overwrite {config_path}: top-level YAML must be a mapping, got "
            f"{type(loaded).__name__}. Fix the file or restore from a .corrupt.*.bak backup first."
        ) from exc
    return loaded


def atomic_config_write(config_path: Path, data: Any, **kwargs: Any) -> None:
    """Fail-closed atomic write for ``config.yaml`` (``require_readable_config_before_write`` first)."""
    require_readable_config_before_write(config_path)
    atomic_yaml_write(config_path, data, **kwargs)


def load_config() -> Dict[str, Any]:
    """Load the merged configuration (DEFAULT_CONFIG + config.yaml + managed scope, env-expanded).
    Cached on the file signature; returns a deepcopy since most call sites mutate the result.
    Read-only hot paths should use ``load_config_readonly()`` to skip the deepcopy."""
    return _load_config_impl(want_deepcopy=True)


def load_config_readonly() -> Dict[str, Any]:
    """``load_config()`` without the defensive deepcopy (~half of the 265us cache-hit cost).
    **Mutating the returned dict (or any nested structure) corrupts the in-process cache for
    every subsequent caller** — only for code paths that never write to the result."""
    return _load_config_impl(want_deepcopy=False)


def _ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return ``parent[key]`` as a dict, replacing a missing or non-dict value with ``{}``."""
    child = parent.get(key)
    if not isinstance(child, dict):
        child = {}
        parent[key] = child
    return child


def write_platform_config_field(
    platform_key: str, field_key: str, value: Any, *, raw: bool = False) -> None:
    """Persist one scalar field under ``platforms.<platform_key>``.
    ``raw=True`` (CLI setup flows) edits only the user's raw file; dashboard routes use the
    default loaded-config path to keep their profile-scoped ``load_config`` behavior."""
    config = read_raw_config() if raw else load_config()
    platforms = _ensure_dict(config, "platforms")
    _ensure_dict(platforms, platform_key)[field_key] = value
    save_config(config)


# ``terminal.<key>`` -> env var read by tools.terminal_tool. Every key maps to ``TERMINAL_<KEY>``
# except ``backend`` (historically ``TERMINAL_ENV``).
TERMINAL_CONFIG_ENV_MAP = {
    "backend": "TERMINAL_ENV",
    **{
        key: f"TERMINAL_{key.upper()}"
        for key in (
            "modal_mode", "degraded_mode", "cwd", "temp_dir", "timeout", "lifetime_seconds",
            "docker_image", "docker_forward_env", "singularity_image", "modal_image",
            "daytona_image", "vercel_runtime", "ssh_host", "ssh_user", "ssh_port", "ssh_key",
            "container_cpu", "container_memory", "container_disk", "container_persistent",
            "docker_volumes", "docker_env", "docker_mount_cwd_to_workspace", "docker_network",
            "docker_extra_args", "docker_shm_size", "docker_run_as_host_user", "docker_snap_compat",
            "docker_persist_across_processes", "docker_shared_container_key",
            "docker_orphan_reaper", "sandbox_dir", "persistent_shell")}}


def _terminal_env_value(value: Any) -> str:
    return json.dumps(value) if isinstance(value, (list, dict)) else str(value)


def _terminal_config_value_is_bridgeable(key: str, value: Any) -> bool:
    """Return whether a terminal config value owns its mirrored env var."""
    return not (key == "cwd" and str(value or "").strip() in {".", "auto", "cwd"})


def terminal_config_owned_env_vars(terminal_config: Any) -> Set[str]:
    """Return env vars explicitly owned by a raw ``terminal`` config section."""
    if not isinstance(terminal_config, dict):
        return set()
    return {
        env_var
        for key, env_var in TERMINAL_CONFIG_ENV_MAP.items()
        if key in terminal_config
        and _terminal_config_value_is_bridgeable(key, terminal_config[key])}


def terminal_config_env_var_for_key(key: str) -> Optional[str]:
    """Return the env var mirrored by a ``terminal.*`` config key."""
    return TERMINAL_CONFIG_ENV_MAP.get(key[len("terminal."):]) if key.startswith("terminal.") else None


def _is_ssh_remote_tilde_cwd(backend: str, cwd: str) -> bool:
    """Whether the remote SSH shell must expand *cwd* itself: ``~`` expanded on the Hermes host
    would name the host/container home instead of the SSH user's."""
    return (backend or "").strip().lower() == "ssh" and (cwd == "~" or cwd.startswith("~/"))


def apply_terminal_config_to_env(
    *, env: Optional[Dict[str, str]] = None, config: Optional[Dict[str, Any]] = None,
    override: Optional[bool] = None) -> Dict[str, str]:
    """Bridge ``terminal.*`` config into the env vars terminal tools read.
    ``tools.terminal_tool`` is environment-driven because it also runs in child processes (TUI,
    dashboard PTY, gateway workers); this gives those launch paths the same bridge as the CLI
    without importing ``cli.py``. Explicit keys in the user's raw ``terminal`` section override
    matching env values; merged defaults only backfill missing env vars."""
    target = os.environ if env is None else env

    raw_terminal_cfg = read_raw_config().get("terminal")
    file_has_terminal_config = isinstance(raw_terminal_cfg, dict)
    raw_terminal_cfg = raw_terminal_cfg if file_has_terminal_config else {}
    should_override = file_has_terminal_config if override is None else override

    cfg = config if config is not None else load_config_readonly()
    terminal_cfg = cfg.get("terminal", {}) if isinstance(cfg, dict) else {}
    if not isinstance(terminal_cfg, dict):
        return target

    # A caller-supplied config is its own source of explicit keys; otherwise only keys present
    # in raw config.yaml may override existing env values (DEFAULT_CONFIG keys are backfill-only).
    explicit_keys = terminal_cfg.keys() if config is not None else raw_terminal_cfg.keys()
    backend_sources = (terminal_cfg.get("backend"), target.get("TERMINAL_ENV"))
    if not (config is not None or "backend" in raw_terminal_cfg):
        backend_sources = backend_sources[::-1]  # env wins when the file did not set backend
    terminal_backend = str(backend_sources[0] or backend_sources[1] or "")

    for cfg_key, env_var in TERMINAL_CONFIG_ENV_MAP.items():
        if cfg_key not in terminal_cfg:
            continue
        value = terminal_cfg[cfg_key]
        if not _terminal_config_value_is_bridgeable(cfg_key, value):
            continue
        if cfg_key == "cwd":
            raw_cwd = str(value or "").strip()
            if isinstance(value, str) and not _is_ssh_remote_tilde_cwd(terminal_backend, raw_cwd):
                value = os.path.expanduser(value)
        if (should_override and cfg_key in explicit_keys) or env_var not in target:
            target[env_var] = _terminal_env_value(value)
    return target


def _load_config_cache_sig(config_path: Path) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int, int, int]]]:
    """Return ``(user_sig, cache_sig)`` for ``_LOAD_CONFIG_CACHE``.
    The managed config file's (mtime, size) is folded in ((0, 0) = none) so editing it invalidates
    the merged result. ``cache_sig`` is None only when neither file exists (nothing to cache on)."""
    try:
        st = config_path.stat()
        user_sig: Optional[Tuple[int, int]] = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        user_sig = None
    managed_dir = managed_scope.get_managed_dir()
    try:
        mst = (managed_dir / "config.yaml").stat() if managed_dir else None
        managed_sig = (mst.st_mtime_ns, mst.st_size) if mst else (0, 0)
    except OSError:
        managed_sig = (0, 0)
    if user_sig is None and managed_sig == (0, 0):
        return None, None
    return user_sig, (*(user_sig or (0, 0)), *managed_sig)


def _last_known_good_fallback(config_path: Path, path_key: str, cache_sig, exc: Exception) -> Optional[Dict[str, Any]]:
    """Warn about a parse failure and return the last-known-good config, or None (-> defaults).
    A parse failure must not silently replace the effective config with defaults — that drops
    EVERY user override, including security-critical ``approvals.deny`` rules, when a gateway
    user mid-edits config.yaml into broken YAML. Keep serving the last good config until fixed."""
    # Falling through to DEFAULT_CONFIG here drops EVERY user override — including security-critical
    # ``approvals.deny`` rules, which are supposed to block commands even under yolo. Within a running
    # process we still have the last successfully loaded config — keep serving it until the file is fixed.
    # See #31188.
    lkg = _LAST_EXPANDED_CONFIG_BY_PATH.get(path_key)
    _warn_config_parse_failure(
        config_path, exc, fallback="last-known-good" if lkg is not None else "defaults")
    if lkg is None:
        return None
    # save_config() stores the pre-expansion dict (templates preserved); the load path stores the
    # expanded one. Expand defensively — idempotent when already expanded.
    lkg_copy: Dict[str, Any] = _expand_env_vars(copy.deepcopy(lkg))
    if cache_sig is not None:
        # Cache under the corrupt file's signature (empty env snapshot: always valid) so repeated
        # loads don't re-parse; fixing the file changes the signature and reloads normally.
        _LOAD_CONFIG_CACHE[path_key] = (*cache_sig, lkg_copy, {})
    return lkg_copy


def _merge_managed_overlay(expanded: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """Apply the managed-scope overlay; returns ``(merged, managed_config_or_falsy)``.
    Managed wins at the leaf and is applied AFTER user expansion so a user ``${VAR}`` cannot shadow
    a managed literal: managed values expand only against the process environment. This
    deliberately inverts the usual env-over-config precedence for the keys the managed layer pins
    (docs/design/managed-scope.md §4.1)."""
    managed_config = managed_scope.load_managed_config()
    if not managed_config:
        return expanded, managed_config
    # Same canonicalization as the user config BEFORE merging (parity with
    # managed_scope.apply_managed_overlay) so the merged result never exposes a nested dict.
    managed_normalized = _normalize_root_model_keys(managed_config)
    if isinstance(managed_normalized.get("model"), str):
        managed_normalized = dict(managed_normalized)
        managed_normalized["model"] = {"default": managed_normalized["model"]}
    return _deep_merge(expanded, _expand_env_vars(managed_normalized)), managed_config


def _load_config_impl(*, want_deepcopy: bool) -> Dict[str, Any]:
    with _CONFIG_LOCK:
        ensure_hermes_home()
        config_path = get_config_path()
        path_key = str(config_path)

        user_sig, cache_sig = _load_config_cache_sig(config_path)

        cached = _LOAD_CONFIG_CACHE.get(path_key)
        if cached is not None and cache_sig is not None and cached[:4] == cache_sig:
            # Signatures match, but the cached expansion is only valid if every ${VAR} it was
            # expanded against still has the same value — otherwise a load before
            # load_hermes_dotenv() pins unexpanded literals for the process lifetime.
            # Without this, a load_config() that ran before load_hermes_dotenv() pins unexpanded literals
            # (e.g. auxiliary.<task>.api_key) for the life of the process (#58514).
            env_snapshot = cached[5] if len(cached) > 5 else {}
            if all(_env_ref_lookup(k) == v for k, v in env_snapshot.items()):
                return copy.deepcopy(cached[4]) if want_deepcopy else cached[4]

        config = copy.deepcopy(DEFAULT_CONFIG)

        if user_sig is not None:
            try:
                with open(config_path, encoding="utf-8") as f:
                    user_config = fast_safe_load(f) or {}

                if "max_turns" in user_config:
                    agent_user_config = dict(user_config.get("agent") or {})
                    if agent_user_config.get("max_turns") is None:
                        agent_user_config["max_turns"] = user_config["max_turns"]
                    user_config["agent"] = agent_user_config
                    user_config.pop("max_turns", None)

                config = _deep_merge(config, user_config)
            except Exception as e:
                lkg_copy = _last_known_good_fallback(config_path, path_key, cache_sig, e)
                if lkg_copy is not None:
                    return copy.deepcopy(lkg_copy) if want_deepcopy else lkg_copy

        normalized = _canonicalize_config(config)
        expanded, managed_config = _merge_managed_overlay(_expand_env_vars(normalized))
        _LAST_EXPANDED_CONFIG_BY_PATH[path_key] = copy.deepcopy(expanded)
        if cache_sig is not None:
            # The cache stores its own deepcopy so load_config() callers can mutate freely while
            # load_config_readonly() callers all see the same stable object. The env snapshot
            # records the values this expansion was made against so later loads detect drift.
            cached_copy = copy.deepcopy(expanded)
            env_snapshot = _env_ref_snapshot(normalized)
            if managed_config:
                _env_ref_snapshot(managed_config, env_snapshot)
            _LOAD_CONFIG_CACHE[path_key] = (*cache_sig, cached_copy, env_snapshot)
            # Readonly path returns the same object later calls will see (identity invariant).
            if not want_deepcopy:
                return cached_copy
        else:
            _LOAD_CONFIG_CACHE.pop(path_key, None)
        # First-load result is a fresh dict (not aliased to the cache); safe to return directly.
        return expanded


_SECURITY_COMMENT = """
# ── Security ──────────────────────────────────────────────────────────
# Secret redaction is ON by default — strings that look like API keys,
# tokens, and passwords are masked in tool output, logs, and chat
# responses before the model or user ever sees them. Set redact_secrets
# to false to disable (e.g. when developing the redactor itself).
# tirith pre-exec scanning is enabled by default when the tirith binary
# is available. Configure via security.tirith_* keys or env vars
# (TIRITH_ENABLED, TIRITH_BIN, TIRITH_TIMEOUT, TIRITH_FAIL_OPEN).
#
# security:
#   redact_secrets: true
#   tirith_enabled: true
#   tirith_path: "tirith"
#   tirith_timeout: 5
#   tirith_fail_open: true
"""

_FALLBACK_COMMENT = """
# ── Fallback Model ────────────────────────────────────────────────────
# Automatic provider failover when primary is unavailable.
# Uncomment and configure to enable. Triggers on rate limits (429),
# overload (529), service errors (503), or connection failures.
#
# Supported providers:
#   openrouter   (OPENROUTER_API_KEY)  — routes to any model
#   openai-codex (OAuth — hermes auth) — OpenAI Codex
#   nous         (OAuth — hermes auth) — Nous Portal
#   zai          (ZAI_API_KEY)         — Z.AI / GLM
#   kimi-coding  (KIMI_API_KEY)        — Kimi / Moonshot
#   kimi-coding-cn (KIMI_CN_API_KEY)   — Kimi / Moonshot (China)
#   minimax      (MINIMAX_API_KEY)     — MiniMax
#   minimax-cn   (MINIMAX_CN_API_KEY)  — MiniMax (China)
#   bedrock      (AWS IAM / boto3)     — AWS Bedrock (Converse API)
#
# For custom OpenAI-compatible endpoints, add base_url and key_env.
#
# fallback_model:
#   provider: openrouter
#   model: anthropic/claude-sonnet-4
"""


def _strip_managed_keys_for_save(config: Dict[str, Any]) -> Dict[str, Any]:
    """Drop every leaf the managed layer pins (bulk safety net; single-key ``config set``
    hard-rejects) and tell the user what was not saved."""
    managed_keys = managed_scope.managed_config_keys()
    if not managed_keys:
        return config
    config, _stripped = _strip_dotted_keys(copy.deepcopy(config), managed_keys)
    if _stripped:
        print(
            f"Note: {len(_stripped)} managed setting(s) were not saved "
            f"(managed by your administrator): {', '.join(sorted(_stripped))}", file=sys.stderr)
    return config


def _commented_sections_for_save(normalized: Dict[str, Any]) -> Optional[str]:
    """Commented-out example blocks for features that are off/unconfigured."""
    parts = []
    if (normalized.get("security") or {}).get("redact_secrets") is None:
        parts.append(_SECURITY_COMMENT)
    fb = normalized.get("fallback_model", {})
    fb_entries = fb if isinstance(fb, list) else [fb]
    if not any(isinstance(e, dict) and e.get("provider") and e.get("model") for e in fb_entries):
        parts.append(_FALLBACK_COMMENT)
    return "".join(parts) or None


def save_config(
    config: Dict[str, Any], *, strip_defaults: bool = True,
    preserve_keys: Optional[Set[Tuple[str, ...]]] = None, merge_existing: bool = False):
    """Save configuration to ~/.hermes/config.yaml.
    Schema defaults are not written unless the user explicitly set them (the path exists in the
    raw config before normalisation), so config.yaml is never contaminated with defaults that
    would hide future default changes. ``merge_existing`` deep-merges the on-disk raw config
    under *config* so partial callers cannot drop sections they omitted."""
    with _CONFIG_LOCK:
        if is_managed():
            managed_error("save configuration")
            return

        config = _strip_managed_keys_for_save(config)

        ensure_hermes_home()
        config_path = get_config_path()
        require_readable_config_before_write(config_path)
        # Explicit user paths come from the RAW dict BEFORE normalisation (which may inject
        # agent.max_turns) so _strip_default_values keeps exactly what the user set.
        _raw_for_paths = read_raw_config()
        if merge_existing and _raw_for_paths:
            config = _merge_partial_save(_raw_for_paths, config)

        current_normalized = _canonicalize_config(config)
        normalized = current_normalized
        if _raw_for_paths:
            normalized = _preserve_env_ref_templates(
                normalized, _canonicalize_config(_raw_for_paths),
                _LAST_EXPANDED_CONFIG_BY_PATH.get(str(config_path)))

        if strip_defaults:
            # ``_strip_default_values`` always preserves ``_config_version`` itself.
            effective_preserve_keys = _explicit_config_paths(_raw_for_paths) | set(preserve_keys or ())
            normalized = _strip_default_values(normalized, DEFAULT_CONFIG, preserve_keys=effective_preserve_keys)

        atomic_yaml_write(config_path, normalized, extra_content=_commented_sections_for_save(normalized))
        _secure_file(config_path)
        _RAW_CONFIG_CACHE.pop(str(config_path), None)
        _LAST_EXPANDED_CONFIG_BY_PATH[str(config_path)] = copy.deepcopy(current_normalized)


def _parse_env_value(raw_value: str) -> str:
    """Parse the small .env value subset Hermes writes itself (bare, 'single', or "double" with
    ``\\"`` / ``\\\\`` escapes)."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        quoted = value[1:-1]
        parsed: list[str] = []
        i = 0
        while i < len(quoted):
            escaped = quoted[i] == "\\" and quoted[i + 1:i + 2] in ('"', "\\")
            parsed.append(quoted[i + 1] if escaped else quoted[i])
            i += 2 if escaped else 1
        return "".join(parsed)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


# load_env() memo keyed on (path, mtime, size). Editing .env bumps mtime -> rebuild;
# invalidate_env_cache() is the explicit knob for writers on coarse-mtime filesystems.
_env_cache: Optional[Tuple[Tuple[str, Optional[float], Optional[int]], Dict[str, str]]] = None


def load_env() -> Dict[str, str]:
    """Load ~/.hermes/.env as a dict (memoised; ``get_env_value()`` runs hundreds of times per
    interactive menu render). Each assignment's value is opaque data for boundary discovery."""
    global _env_cache
    env_path = get_env_path()

    try:
        st = env_path.stat()
        cache_key = (str(env_path), st.st_mtime, st.st_size)
    except FileNotFoundError:
        cache_key = (str(env_path), None, None)
    except Exception:
        cache_key = None
    if cache_key is not None and _env_cache is not None and _env_cache[0] == cache_key:
        return dict(_env_cache[1])

    env_vars: Dict[str, str] = {}
    for line in _read_env_lines(env_path) if env_path.exists() else ():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            # Bash-compatible ``export KEY=...`` parses as ``KEY``.
            key, _, value = line.removeprefix('export ').partition('=')
            env_vars[key.strip()] = _parse_env_value(value)
    if cache_key is not None:
        _env_cache = (cache_key, dict(env_vars))
    return env_vars


def invalidate_env_cache() -> None:
    """Clear the load_env() memo so the next call sees a write even on coarse-mtime filesystems."""
    global _env_cache
    _env_cache = None


def _sanitize_env_lines(lines: list) -> list:
    """Normalize .env line endings/whitespace without changing assignment semantics.
    Content after the first ``=`` is opaque value data: a known variable name embedded in a value
    must never be reinterpreted as another assignment, so concatenated lines stay on one line."""
    sanitized: list[str] = []
    for line in lines:
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        # Blank lines and comments are preserved verbatim.
        sanitized.append((raw if not stripped or stripped.startswith("#") else stripped) + "\n")
    return sanitized


def sanitize_env_file() -> int:
    """Rewrite ~/.hermes/.env with normalized line formatting; returns the number of changed lines."""
    env_path = get_env_path()
    if not env_path.exists():
        return 0
    with open(env_path, encoding="utf-8-sig", errors="replace") as f:
        original_lines = f.readlines()
    sanitized = _sanitize_env_lines(original_lines)
    if sanitized == original_lines:
        return 0
    fixes = abs(len(sanitized) - len(original_lines)) or sum(
        1 for a, b in zip(original_lines, sanitized) if a != b)
    _write_env_lines(env_path, sanitized, preserve_mode=False)
    invalidate_env_cache()
    return fixes


def _read_env_lines(env_path: Path) -> list:
    """Read ``.env`` lines, normalized. Explicit UTF-8 (Windows defaults to cp1252) with BOM
    tolerance (Notepad adds one)."""
    with open(env_path, encoding="utf-8-sig", errors="replace") as f:
        return _sanitize_env_lines(f.readlines())


def _write_env_lines(env_path: Path, lines: list, *, preserve_mode: bool) -> None:
    """Atomically replace ``.env`` (tmp file + fsync + rename).
    ``preserve_mode`` keeps the original file mode (e.g. 0640 for Docker volume mounts) instead of
    letting ``_secure_file`` tighten to 0600; a new file is always secured."""
    original_mode = None
    try:
        original_mode = stat.S_IMODE(env_path.stat().st_mode) if preserve_mode else None
    except OSError:
        pass
    fd, tmp_path = tempfile.mkstemp(dir=str(env_path.parent), suffix=".tmp", prefix=".env_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, env_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    if original_mode is not None:
        try:
            os.chmod(env_path, original_mode)
        except OSError:
            pass
    else:
        _secure_file(env_path)


def _check_non_ascii_credential(key: str, value: str) -> str:
    """Strip non-ASCII characters from a credential (HTTP header values must be ASCII) and warn.
    Lookalike glyphs typically come from copy-pasting out of a PDF or rich-text editor."""
    if value.isascii():
        return value

    bad_chars = [f"  position {i}: {ch!r} (U+{ord(ch):04X})" for i, ch in enumerate(value) if ord(ch) > 127]
    sanitized = value.encode("ascii", errors="ignore").decode("ascii")

    print(
        f"\n  Warning: {key} contains non-ASCII characters that will break API requests.\n"
        f"  This usually happens when copy-pasting from a PDF, rich-text editor,\n"
        f"  or web page that substitutes lookalike Unicode glyphs for ASCII letters.\n\n"
        + "\n".join(f"  {line}" for line in bad_chars[:5])
        + ("\n  ... and more" if len(bad_chars) > 5 else "")
        + "\n\n  The non-ASCII characters have been stripped automatically.\n"
        "  If authentication fails, re-copy the key from the provider's dashboard.\n",
        file=sys.stderr)
    return sanitized


def _quote_env_value(value: str) -> str:
    """Quote .env values containing characters with special dotenv meaning. Any whitespace
    (including internal runs) is quoted so ``set -a; . file`` word-splitting keeps paths intact."""
    if value == "":
        return value
    if not ("#" in value or '"' in value or "'" in value or any(c.isspace() for c in value)):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _env_line_defines_key(line: str, key: str, *, is_windows: Optional[bool] = None) -> bool:
    """True when a .env line assigns ``key`` — plain, ``export``-prefixed, or ``KEY = value``.
    Must match exactly the shapes ``load_env()`` parses; otherwise a hand-added line is invisible
    to save (duplicate appended) and remove (line survives -> the value resurrects on next load).

    ``load_env()`` accepts the bash-compatible ``export KEY=value`` form (#6659), so the writers must
    recognise the same shape.
    """
    stripped = line.strip()
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()
    assigned_key, separator, _value = stripped.partition("=")
    if not separator:
        return False
    # load_env() strips whitespace around the parsed name, so `KEY = value` IS a live assignment. The
    # writers must match the same shape, or a hand-edited spaced line is invisible to save (duplicate
    # appended) and remove (line survives -> value resurrects on next load). #67488.
    return _env_var_policy_name(
        assigned_key.strip(), is_windows=is_windows
    ) == _env_var_policy_name(key, is_windows=is_windows)


def _publish_env_value(key: str, value: Optional[str]) -> None:
    """Publish a just-persisted ``.env`` change to the live process.
    Under a multiplexed gateway a routed profile's write must not land in the SHARED
    ``os.environ`` where every profile sees it; the installed scope mapping is updated instead so
    same-turn reads see the change. All other callers keep the legacy ``os.environ`` publish.

    ``save_env_value`` / ``remove_env_value`` already target the right file (``get_env_path()`` honors the
    profile-home override), but the in-process mirror historically went straight to ``os.environ``. See
    #77490, #88441.
    """
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope() if is_multiplex_active() else None
    except Exception:
        scope = None
    target = scope if isinstance(scope, dict) else (None if scope is not None else os.environ)
    if target is not None:
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value


def _env_write_blocked(key: str, action: str) -> bool:
    """Shared write-lock check for ``.env`` writers; prints the refusal and returns True when blocked.
    Two distinct locks: ``is_managed()`` (package-manager install) and the managed *scope*
    (administrator-pinned env key — the managed .env wins at load anyway)."""
    if is_managed():
        managed_error(f"{action} {key}")
        return True

    if managed_scope.is_env_managed(key):
        print(
            f"Cannot {action} {key}: it is managed by your administrator ({_managed_source('.env')}) "
            f"and cannot be changed.", file=sys.stderr)
        return True
    return False


def _managed_source(filename: str):
    """``<managed dir>/<filename>`` for refusal messages, or a generic label without a managed dir."""
    managed_dir = managed_scope.get_managed_dir()
    return (managed_dir / filename) if managed_dir else "the managed scope"


def save_env_value(key: str, value: str):
    """Save or update a value in ~/.hermes/.env (also matching ``export KEY=`` lines, so a save
    never appends a second line that a later delete would resurrect)."""
    if _env_write_blocked(key, "set"):
        return
    validate_env_var_name_for_write(key)
    value = value.replace("\n", "").replace("\r", "")
    value = _check_non_ascii_credential(key, value)
    ensure_hermes_home()
    env_path = get_env_path()

    lines = _read_env_lines(env_path) if env_path.exists() else []
    serialized_value = _quote_env_value(value)

    idx = next((i for i, line in enumerate(lines) if _env_line_defines_key(line, key)), None)
    if idx is not None:
        lines[idx] = f"{key}={serialized_value}\n"
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={serialized_value}\n")

    _write_env_lines(env_path, lines, preserve_mode=env_path.exists())
    _publish_env_value(key, value)
    invalidate_env_cache()


def custom_endpoint_key_env(identity: str) -> str:
    """Env var name holding a custom endpoint's API key.
    ``identity`` is the endpoint's own id (Desktop endpoint id, or ``host:port`` for CLI setup),
    so two endpoints on one host get separate slots. The fixed ``HERMES_CUSTOM_`` prefix keeps the
    name POSIX-valid when the slug starts with a digit (``save_env_value`` rejects those)."""
    slug = re.sub(r"[^A-Z0-9]+", "_", str(identity or "").upper()).strip("_")
    return f"HERMES_CUSTOM_{slug}_API_KEY" if slug else "HERMES_CUSTOM_API_KEY"


def remove_env_value(key: str) -> bool:
    """Remove a key from ~/.hermes/.env and os.environ; True if it was found and removed."""
    if _env_write_blocked(key, "remove"):
        return False
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    env_path = get_env_path()
    if not env_path.exists():
        _publish_env_value(key, None)
        return False

    lines = _read_env_lines(env_path)
    new_lines = [line for line in lines if not _env_line_defines_key(line, key)]
    found = len(new_lines) < len(lines)
    if found:
        _write_env_lines(env_path, new_lines, preserve_mode=True)
    _publish_env_value(key, None)
    invalidate_env_cache()
    return found


def _write_anthropic_slots(token: str, api_key: str, save_fn=None, *, token_first: bool = True):
    """Write both Anthropic credential slots (one holds the value, the other is cleared)."""
    writer = save_fn or save_env_value
    order = (("ANTHROPIC_TOKEN", token), ("ANTHROPIC_API_KEY", api_key))
    for name, value in order if token_first else reversed(order):
        writer(name, value)


def save_anthropic_oauth_token(value: str, save_fn=None):
    """Persist an Anthropic OAuth/setup token and clear the API-key slot."""
    _write_anthropic_slots(value, "", save_fn)


def use_anthropic_claude_code_credentials(save_fn=None):
    """Use Claude Code's own credential files instead of persisting env tokens."""
    _write_anthropic_slots("", "", save_fn)


def save_anthropic_api_key(value: str, save_fn=None):
    """Persist an Anthropic API key and clear the OAuth/setup-token slot."""
    _write_anthropic_slots("", value, save_fn, token_first=False)


def save_env_value_secure(key: str, value: str) -> Dict[str, Any]:
    """Save via the unified credential lifecycle (also refreshes any config.yaml mirror of the old
    value and lifts a prior env-source suppression)."""
    from hermes_cli.credential_lifecycle import save_provider_env_credential

    # Route through the unified credential lifecycle so a rotation via the secret-capture path also
    # refreshes any config.yaml mirror of the old value and lifts a prior env-source suppression (#62269 fix
    # family).
    save_provider_env_credential(key, value)
    return {"success": True, "stored_as": key, "validated": False}


def reload_env() -> int:
    """Re-read ~/.hermes/.env into os.environ; returns count of vars changed.
    Removes deleted vars only when known to Hermes (OPTIONAL_ENV_VARS and _EXTRA_ENV_KEYS) so
    unrelated environment is never clobbered."""
    env_vars = load_env()
    count = 0
    for key, value in env_vars.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            count += 1
    for key in (set(OPTIONAL_ENV_VARS) | _EXTRA_ENV_KEYS) - set(env_vars):
        if key in os.environ:
            del os.environ[key]
            count += 1
    return count


def _scoped_environ_get(key: str) -> Optional[str]:
    """Read ``key`` from ``os.environ`` through ``agent.secret_scope.get_secret`` so an active
    profile scope (multiplexed gateway turn) never leaks another profile's raw value. Falls back to
    a plain environ read when the scope module is unavailable; ``UnscopedSecretError`` propagates."""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret as _get_secret
    except Exception:
        return os.environ.get(key)
    try:
        return _get_secret(key)
    except UnscopedSecretError:
        raise
    except Exception:
        return os.environ.get(key)


def get_env_value(key: str) -> Optional[str]:
    """Get a value from ``os.environ`` (scope-aware) or ``~/.hermes/.env``.

    The ``os.environ`` read routes through ``agent.secret_scope.get_secret`` so that, under an active
    profile scope (multiplexed gateway turn), this is scope-checked rather than leaking another profile's
    raw ``os.environ`` value. ``get_secret`` encodes the whole policy: global vars pass through; scope is
    authoritative under multiplexing (miss -> None, no environ fallthrough); when multiplexing is off it
    behaves exactly like the legacy ``os.environ`` read. Its siblings ``get_env_value_prefer_dotenv`` and
    ``gateway.config._getenv`` already work this way — this was the last scope-blind reader of the trio
    (#67027).
    """
    val = _scoped_environ_get(key)
    return load_env().get(key) if val is None else val


def get_env_value_prefer_dotenv(key: str) -> Optional[str]:
    """Resolve a Hermes-managed credential preferring ``~/.hermes/.env`` over ``os.environ``, so a
    deliberate .env edit beats a stale value inherited from the parent shell."""
    return load_env().get(key) or _scoped_environ_get(key)


# ---- Config display ----

def redact_key(key: str) -> str:
    """Redact an API key for display."""
    from agent.redact import mask_secret
    return mask_secret(key, empty=color("(not set)", Colors.DIM))


# Key names (case-insensitive, exact match) whose VALUE is a credential and must be masked
# before printing any config dict. Exact-match so ``token_count`` / ``secret_santa`` stay visible.
_SECRET_CONFIG_KEYS = frozenset({
    "api_key", "apikey", "key", "token", "access_token", "refresh_token", "id_token",
    "secret", "client_secret", "password", "passwd", "auth", "authorization",
    "private_key", "bearer", "jwt"})


def redact_config_value(value: Any, _depth: int = 0) -> Any:
    """Copy of ``value`` with credential-shaped keys masked. ``print`` bypasses the logging
    redactor and opaque tokens miss the vendor-prefix regexes, so structural masking is required."""
    from agent.redact import mask_secret

    if _depth > 20:  # bound recursion for pathological/cyclic configs
        return value
    if isinstance(value, dict):
        return {
            k: mask_secret(v)
            if isinstance(k, str) and k.lower() in _SECRET_CONFIG_KEYS and isinstance(v, str) and v
            else redact_config_value(v, _depth + 1)
            for k, v in value.items()}
    if isinstance(value, list):
        return [redact_config_value(v, _depth + 1) for v in value]
    return value


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


def _show_model_section(config: Dict[str, Any]) -> None:
    _section("Model")
    print(f"  Model:        {redact_config_value(config.get('model', 'not set'))}")
    cfg_max_turns = config.get('agent', {}).get('max_turns', DEFAULT_CONFIG['agent']['max_turns'])
    print(f"  Max turns:    {cfg_max_turns}")
    # Read the .env FILE directly so a stale HERMES_MAX_ITERATIONS ghost is caught even when the
    # gateway bridge already overrode os.environ.
    try:
        env_ghost = load_env().get("HERMES_MAX_ITERATIONS")
    except Exception:
        env_ghost = None
    if env_ghost is not None and str(env_ghost).strip() != str(cfg_max_turns).strip():
        print(color(f"                ⚠ .env has stale HERMES_MAX_ITERATIONS={env_ghost} "
                    f"(run 'hermes doctor --fix' to remove)", Colors.YELLOW))


def _show_display_section(config: Dict[str, Any]) -> None:
    _section("Display")
    display = config.get('display', {})
    try:
        from hermes_cli.personality import active_personality_name
        active_personality = active_personality_name(config) or 'none'
    except Exception:
        active_personality = display.get('personality') or 'none'
    on_off = lambda flag: 'on' if flag else 'off'  # noqa: E731
    print(f"  Personality:  {active_personality}")
    print(f"  Reasoning:    {on_off(display.get('show_reasoning', True))}")
    print(
        f"  Bell:         complete={on_off(display.get('bell_on_complete', False))}, "
        f"prompt={on_off(display.get('bell_on_prompt', False))}")
    ump = display.get('user_message_preview', {})
    ump = ump if isinstance(ump, dict) else {}
    print(f"  User preview: first {ump.get('first_lines', 2)} line(s), last {ump.get('last_lines', 2)} line(s)")


def _show_terminal_section(config: Dict[str, Any]) -> None:
    _section("Terminal")
    terminal = config.get('terminal', {})
    print(f"  Backend:      {terminal.get('backend', 'local')}")
    print(f"  Working dir:  {terminal.get('cwd', '.')}")
    print(f"  Timeout:      {terminal.get('timeout', 60)}s")

    configured = lambda *names: 'configured' if all(get_env_value(n) for n in names) else '(not set)'  # noqa: E731
    default_img = 'nikolaik/python-nodejs:python3.11-nodejs20'
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
            f"  Vercel runtime: {terminal.get('vercel_runtime', 'node24')}",
            f"  Vercel auth:    {'configured' if get_env_value('VERCEL_OIDC_TOKEN') or (get_env_value('VERCEL_TOKEN') and get_env_value('VERCEL_PROJECT_ID') and get_env_value('VERCEL_TEAM_ID')) else '(not set)'}",
        ],
        'ssh': lambda: [
            f"  SSH host:     {get_env_value('TERMINAL_SSH_HOST') or '(not set)'}",
            f"  SSH user:     {get_env_value('TERMINAL_SSH_USER') or '(not set)'}"]}
    for line in backend_lines.get(terminal.get('backend'), list)():
        print(line)


def _show_compression_section(config: Dict[str, Any]) -> None:
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


def _show_aux_overrides(config: Dict[str, Any]) -> None:
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
    config = load_config()

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│              ⚕ Hermes Configuration                    │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    _show_managed_banner()

    _section("Paths")
    print(f"  Config:       {get_config_path()}")
    print(f"  Secrets:      {get_env_path()}")
    print(f"  Install:      {get_project_root()}")

    _section("API Keys")
    for env_key, name in _SHOW_CONFIG_API_KEYS:
        print(f"  {name:<14} {redact_key(get_env_value(env_key))}")
    from hermes_cli.auth import get_anthropic_key
    print(f"  {'Anthropic':<14} {redact_key(get_anthropic_key())}")

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
        state = 'configured' if get_env_value(env_key) else color('not configured', Colors.DIM)
        print(f"  {label + ':':<13} {state}")

    _show_skill_settings()

    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  hermes config edit     # Edit config file", Colors.DIM))
    print(color("  hermes config set <key> <value>", Colors.DIM))
    print(color("  hermes setup           # Run setup wizard", Colors.DIM))
    print()


def edit_config():
    """Open config file in user's editor."""
    if is_managed():
        managed_error("edit configuration")
        return
    config_path = get_config_path()
    if not config_path.exists():
        save_config(DEFAULT_CONFIG, strip_defaults=False)
        print(f"Created {config_path}")

    # Windows lands on notepad even without Git Bash/nano; POSIX prefers nano/vim, which headless
    # servers are more likely to have.
    candidates = (['notepad', 'code', 'vim', 'vi', 'nano'] if sys.platform == "win32"
                  else ['nano', 'vim', 'vi', 'code', 'notepad'])
    editor = os.getenv('EDITOR') or os.getenv('VISUAL') or next(
        (cmd for cmd in candidates if shutil.which(cmd)), None)
    if not editor:
        print("No editor found. Config file is at:")
        print(f"  {config_path}")
        return

    print(f"Opening {config_path} in {editor}...")
    subprocess.run([editor, str(config_path)])


# ---- Cron model-drift guard helpers ----

_CRON_DRIFT_AXIS_BY_KEY = {
    "model": "model", "model.default": "model", "model.model": "model", "model.name": "model",
    "model.provider": "provider", "provider": "provider"}


def _cron_model_drift_axis_for_config_key(key: str) -> Optional[str]:
    """Return the cron drift guard axis affected by a config key, if any."""
    return _CRON_DRIFT_AXIS_BY_KEY.get(str(key or "").strip().lower())


def _cron_section(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the ``cron`` mapping of *config* (loading the merged config when None), else None."""
    if config is None:
        try:
            config = load_config()
        except Exception:
            return None
    cron_config = config.get("cron") if isinstance(config, dict) else None
    return cron_config if isinstance(cron_config, dict) else None


def cron_model_drift_guard_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Whether cron must fail closed on unpinned inference drift.
    Only the literal YAML boolean ``false`` disables this spend-safety guard; missing, malformed,
    or non-boolean values stay fail-closed. With *config* omitted the merged config is loaded so
    CLI warnings honor the same user/managed setting as the scheduler."""
    cron_config = _cron_section(config)
    return cron_config is None or cron_config.get("model_drift_guard", True) is not False


_CRON_MODEL_IMPACT_JOB_LIMIT = 50
_CRON_MODEL_IMPACT_ID_LIMIT = 256
_CRON_MODEL_IMPACT_NAME_LIMIT = 120


def _model_assignment_text(value: Any) -> str:
    """Return a trimmed scalar model/provider value, or empty for malformed data."""
    return value.strip() if isinstance(value, str) else ""


def resolve_cron_model_drift_defaults(
    config: Any, *, environ: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """Resolve the global ``(provider, model)`` cron compares against snapshots.
    Mirrors the scheduler's precedence: a truthy configured model wins over ``HERMES_MODEL``; the
    environment is only a fallback. Per-job and cron fleet defaults are handled by the caller
    because they suppress a drift axis rather than changing the global assignment."""
    env = os.environ if environ is None else environ
    provider = ""
    model_config = config.get("model") if isinstance(config, dict) else None
    if isinstance(model_config, dict):
        provider = _model_assignment_text(model_config.get("provider"))
        model_config = model_config.get("default") or model_config.get("model") or model_config.get("name")
    configured_model = _model_assignment_text(model_config)
    return provider, configured_model or _model_assignment_text(env.get("HERMES_MODEL", ""))


def cron_model_drift_axes(
    job: Any, *, current_provider: Any = "", current_model: Any = "", config: Any = None
) -> List[str]:
    """Return the unpinned axes that the fail-closed cron guard would block."""
    if not isinstance(job, dict) or not cron_model_drift_guard_enabled(config):
        return []

    current = {
        "provider": _model_assignment_text(current_provider).lower(),
        "model": _model_assignment_text(current_model).lower()}
    # A cron.model / cron.model_provider fleet default covers its axis: that axis no longer follows
    # the global assignment at fire time, so the guard never engages and a warning would be false.
    fleet = _cron_section(config) or {}
    drifted: List[str] = []
    for axis, fleet_key in (("provider", "model_provider"), ("model", "model")):
        if _model_assignment_text(fleet.get(fleet_key)) or _model_assignment_text(job.get(axis)):
            continue
        snapshot = _model_assignment_text(job.get(f"{axis}_snapshot")).lower()
        if snapshot and current[axis] and snapshot != current[axis]:
            drifted.append(axis)
    return drifted


def _is_control_char(char: str) -> bool:
    return unicodedata.category(char).startswith("C")


def _valid_cron_impact_job_id(value: Any) -> str:
    job_id = value.strip() if isinstance(value, str) else ""
    if len(job_id) > _CRON_MODEL_IMPACT_ID_LIMIT or any(map(_is_control_char, job_id)):
        return ""
    return job_id


def _cron_impact_job_name(value: Any, job_id: str) -> str:
    if isinstance(value, str):
        printable = "".join(char for char in value if not _is_control_char(char))
        name = " ".join(printable.split())[:_CRON_MODEL_IMPACT_NAME_LIMIT].rstrip()
        if name:
            return name
    return f"Job {job_id}"[:_CRON_MODEL_IMPACT_NAME_LIMIT].rstrip()


def _cron_model_impact_result(available: bool, guard_enabled: bool) -> Dict[str, Any]:
    return {
        "available": available,
        "guard_enabled": guard_enabled,
        "affected_count": 0,
        "truncated": False,
        "jobs": []}


def build_cron_model_impact(
    *, current_provider: Any = "", current_model: Any = "", config: Any = None, jobs: Any = None
) -> Dict[str, Any]:
    """Build a bounded, profile-local summary of jobs blocked by model drift.
    Job-store inspection is best effort: the model assignment has already succeeded when Desktop
    requests this, so an unreadable store is reported as unavailable rather than failing."""
    guard_enabled = cron_model_drift_guard_enabled(config)
    if jobs is None:
        try:
            from cron.jobs import load_jobs

            jobs = load_jobs()
        except Exception:
            return _cron_model_impact_result(False, guard_enabled)
    if not isinstance(jobs, list):
        return _cron_model_impact_result(False, guard_enabled)

    result = _cron_model_impact_result(True, guard_enabled)
    if not guard_enabled:
        return result

    from cron.jobs import is_job_runnable

    seen_ids: Set[str] = set()
    for job in jobs:
        if not isinstance(job, dict) or not is_job_runnable(job) or job.get("no_agent"):
            continue
        job_id = _valid_cron_impact_job_id(job.get("id"))
        if not job_id or job_id in seen_ids:
            continue
        seen_ids.add(job_id)
        axes = cron_model_drift_axes(
            job, current_provider=current_provider, current_model=current_model, config=config)
        if not axes:
            continue
        result["affected_count"] += 1
        if len(result["jobs"]) < _CRON_MODEL_IMPACT_JOB_LIMIT:
            result["jobs"].append({
                "id": job_id,
                "name": _cron_impact_job_name(job.get("name"), job_id),
                "drifted_axes": axes})

    result["truncated"] = result["affected_count"] > len(result["jobs"])
    return result


def warn_unpinned_cron_jobs_after_model_config_change(
    key: str, value: Any, config: Optional[Dict[str, Any]] = None) -> None:
    """Warn when a global model/provider change will trip cron's drift guard."""
    axis = _cron_model_drift_axis_for_config_key(key)
    if axis is None:
        return

    new_value = _model_assignment_text(value)
    if not new_value:
        return
    impact = build_cron_model_impact(
        current_provider=new_value if axis == "provider" else "",
        current_model=new_value if axis == "model" else "", config=config, jobs=None)
    affected = impact["affected_count"]
    if affected <= 0:
        return

    noun, verb = ("job", "has") if affected == 1 else ("jobs", "have")
    print(
        f"⚠️  {affected} enabled unpinned cron {noun} {verb} stored "
        f"{axis}_snapshot values that differ from the new global {axis}. "
        "They will fail closed on their next run instead of silently using the changed "
        "model/provider. Inspect with `hermes cron list`, then pin the intended values with "
        "`hermes cron edit <job_id> --provider <provider> --model <model>`.")


def _default_value_for_key(dotted_key: str):
    """Return the leaf value declared for *dotted_key* in ``DEFAULT_CONFIG`` (None for dicts/misses)."""
    node = cfg_get(DEFAULT_CONFIG, *_split_key_path(dotted_key))
    return None if isinstance(node, dict) else node


# Top-level keys that accept arbitrary user-supplied child keys (schema declares the dict, the
# user populates it): any path below is accepted without deep checking.
_OPEN_DICT_TOP_LEVEL_KEYS = frozenset({
    "providers", "credential_pool_strategies", "mcp_servers", "hooks", "quick_commands",
    "personalities", "command_allowlist", "model_catalog", "channel_prompts", "server_actions",
    "secrets", "goals", "loops"})

# Top-level keys whose sub-keys are partially schema-defined (e.g. a PlatformConfig dataclass) but
# where users may add fields DEFAULT_CONFIG doesn't enumerate: validate the FIRST segment only.
_SCHEMA_DEFINED_DICT_KEYS = frozenset({
    # Platform configs — PlatformConfig dataclass + dynamic extras
    "discord", "telegram", "slack", "whatsapp", "signal", "mattermost",
    "matrix", "feishu", "wecom", "weixin", "bluebubbles", "qqbot", "yuanbao",
    "email", "sms", "dingtalk",
    # MCP server template / dynamic auth dicts
    "sessions", "checkpoints",
    # Plugin enable/disable lists + index_url override; absent from DEFAULT_CONFIG.
    "plugins"})

# Top-level keys that can be ANY user-supplied name.
_DYNAMIC_TOP_LEVEL_KEYS = frozenset({
    "custom_providers",  # list-shaped, but indexed by position
})

# Containers whose immediate child IS a user-supplied platform name (``platforms.<name>.<field>``),
# both top-level and under ``gateway``; anything below the name is accepted (open ``extra``).
_PLATFORM_CONTAINER_KEYS = frozenset({"platforms"})


# Top-level keys whose sub-keys are accepted without deep checking.
_OPEN_SUBKEY_TOP_LEVEL_KEYS = _OPEN_DICT_TOP_LEVEL_KEYS | _DYNAMIC_TOP_LEVEL_KEYS | _SCHEMA_DEFINED_DICT_KEYS


def _known_top_level_keys() -> set[str]:
    """Return the union of known top-level config keys for validation."""
    return set(DEFAULT_CONFIG) | _OPEN_SUBKEY_TOP_LEVEL_KEYS


def _suggest_closest_key(key: str, candidates: set[str], cutoff: float = 0.6) -> Optional[str]:
    """Closest candidate key name for a typo'd ``key``, or None."""
    return next(iter(difflib.get_close_matches(key, sorted(candidates), n=1, cutoff=cutoff)), None)


def _validate_config_key(key: str) -> tuple[bool, Optional[str]]:
    """Validate a dotted config-key path against the known schema -> ``(is_known, suggestion)``.

    Headline case from #34067: ``gateway.discord.gateway_restart_notification`` was silently written, even
    though ``gateway`` only has 4 known sub-keys (``strict``, ``media_delivery_allow_dirs``,
    ``trust_recent_files``, ``trust_recent_files_seconds``). The correct path is
    ``discord.gateway_restart_notification`` (platform configs live at the top level, not under a
    ``platforms`` namespace).
    """
    if not key:
        return False, None

    segments = _split_key_path(key)
    top = segments[0]

    # A leading underscore on the FIRST segment marks an intentionally non-schema internal key
    # (test harnesses/tooling); only the first segment is exempt so ``agent._max_turns`` is caught.
    if top.startswith("_") or top in _PLATFORM_CONTAINER_KEYS:
        return True, None

    known = _known_top_level_keys()
    if top not in known:
        suggestion = _suggest_closest_key(top, known)
        if suggestion is None:
            return False, None
        rest = ".".join(segments[1:])
        return False, f"{suggestion}.{rest}" if rest else suggestion

    if top in _OPEN_SUBKEY_TOP_LEVEL_KEYS:
        return True, None

    # Walk DEFAULT_CONFIG: a nested ``platforms`` container or a scalar leaf hit before the path is
    # consumed both accept (the latter matches set_config_value's leaf->dict replacement); an
    # unknown sub-key fails with a same-level "did you mean" suggestion.
    node: Any = DEFAULT_CONFIG.get(top)
    consumed = [top]
    for seg in segments[1:]:
        if seg in _PLATFORM_CONTAINER_KEYS or not isinstance(node, dict):
            return True, None
        if seg not in node:
            sibling = _suggest_closest_key(seg, set(node.keys()))
            return False, ".".join(consumed + [sibling]) if sibling is not None else None
        consumed.append(seg)
        node = node[seg]
    return True, None


def _looks_structured_value(value: str) -> bool:
    """True when *value* plausibly encodes a YAML/JSON list or mapping. Deliberately conservative:
    a bare leading ``-`` is not a trigger (``-5``, ``--flag`` must stay strings)."""
    stripped = value.lstrip()
    if stripped[:1] in ('[', '{'):
        return True
    if '\n' not in value:
        return False
    for line in value.splitlines():
        item = line.strip()
        if item == '-' or item.startswith('- '):
            return True
        # ``key: value`` / ``key:`` mapping-entry shape (no whitespace in the key).
        head, sep, _rest = item.partition(': ')
        if sep and head and ' ' not in head and not head.startswith('#'):
            return True
        if item.endswith(':') and ' ' not in item[:-1] and item[:-1]:
            return True
    return False


def _coerce_int(value: str):
    """int(value) for a clean integer literal (signs/whitespace/underscores OK), else None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: str):
    """``float(value)`` only when the conversion preserves its decimal value; NaN/inf rejected.
    Decimal-looking identifiers more precise than a binary float must stay strings."""
    try:
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")) or Decimal(value) != Decimal(str(f)):
            return None
    except (TypeError, ValueError, InvalidOperation):
        return None
    return f


_SCALAR_WORDS = {
    'true': True, 'yes': True, 'on': True,
    'false': False, 'no': False, 'off': False,
    # YAML null. Many DEFAULT_CONFIG leaves are "null/absent = off"; without this,
    # ``config set X null`` stored the truthy string "null" and the feature could never be cleared.
    'null': None, 'none': None, '~': None}


def _coerce_config_set_value(key: str, value: str) -> Any:
    """Auto-coerce a ``hermes config set`` string to bool/None/int/float/list/dict.
    String-typed settings (per ``DEFAULT_CONFIG``) are preserved verbatim so enum members such as
    ``approvals.mode="off"`` never become booleans. List/mapping literals are parsed so
    isinstance-gated readers see real structures; the trigger is conservative."""
    if isinstance(_default_value_for_key(key), str):
        return value
    stripped = value.strip()
    lower = stripped.lower()
    if lower in _SCALAR_WORDS:
        return _SCALAR_WORDS[lower]
    for coerce in (_coerce_int, _coerce_float):
        coerced = coerce(stripped)
        if coerced is not None:
            return coerced
    if not _looks_structured_value(value):
        return value
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        print(
            f"Warning: value for '{key}' looks like a list/mapping but is "
            f"not valid YAML/JSON; storing as string. Most isinstance-gated "
            f"readers will ignore a string here.", file=sys.stderr)
        return value
    if isinstance(parsed, (list, dict)):
        return parsed
    print(
        f"Warning: value for '{key}' looks like a list/mapping but "
        f"parsed as {type(parsed).__name__}; storing as string.", file=sys.stderr)
    return value


def _redirect_platform_display_key(key: str) -> tuple[str, Optional[str]]:
    """Canonicalize ``platforms.<name>.<display_setting>`` -> ``display.platforms.<name>.<setting>``.
    The gateway resolves per-platform display settings (streaming, show_reasoning, ...) from
    ``display.platforms``; the top-level ``platforms.<name>`` block holds only connection config.
    Only known display settings (``OVERRIDEABLE_KEYS``) are redirected. Returns ``(key, note)``;
    the gateway import is guarded so the CLI works where the gateway package is unavailable.

    Before #71047 a write such as ``hermes config set platforms.telegram.streaming false`` landed on a key
    the gateway never reads: ``config get`` echoed the new value back while the runtime kept the old
    ``display.platforms`` one — a silent no-op that looks like a duplicated key to the user.
    """
    segs = _split_key_path(key)
    if len(segs) != 3 or segs[0] != "platforms":
        return key, None
    try:
        from gateway.display_config import OVERRIDEABLE_KEYS as _display_keys
    except Exception:
        return key, None
    if segs[2] not in _display_keys:
        return key, None
    canonical = f"display.platforms.{segs[1]}.{segs[2]}"
    return canonical, f"  (note: per-platform display setting — saved as {canonical})"


def _exit_if_key_managed(key: str, action: str) -> None:
    """A key pinned by the managed layer cannot be set/unset (the next load would reinstate it):
    hard-reject and name the source. Distinct from ``is_managed()``; env-shaped keys route to the
    .env writers, which carry their own guard."""
    if managed_scope.is_key_managed(key):
        print(
            f"Cannot {action} '{key}': it is managed by your administrator ({_managed_source('config.yaml')}) "
            f"and cannot be changed. Contact your administrator to modify it.", file=sys.stderr)
        sys.exit(1)


def _guard_section_overwrite(key: str, value: Any, user_config: Dict[str, Any], force: bool) -> str:
    """Refuse (or with ``force`` allow) a single-segment key overwriting a mapping with a scalar.
    Bare ``model`` is a documented shorthand — redirected to ``model.default`` so siblings survive.
    Returns the (possibly redirected) key."""
    existing = user_config.get(key)
    if "." in key or not isinstance(existing, dict):
        return key
    if key == "model":
        if force:
            print(
                f"⚠ Replacing entire 'model' section with a scalar "
                f"(discarding {len(existing)} existing sub-key(s))")
            return key
        print(
            f"✓ Redirecting bare 'model' to 'model.default' "
            f"(preserving {len(existing)} existing model sub-key(s))")
        return "model.default"
    if force:
        return key
    sub = [k for k in existing if isinstance(k, str)]
    err = [
        f"✗ Cannot set '{key}' to a scalar — '{key}' is a "
        f"configuration section with {len(sub)} sub-key(s)."]
    if sub:
        err.append(f"  Sub-keys: {', '.join(sub[:8])}")
        if len(sub) > 8:
            err.append(f"  ... and {len(sub) - 8} more")
    err += [
        "  Use a dotted path to set a specific leaf key:",
        f"    hermes config set {key}.<sub-key> <value>",
        "  Or use --force to replace the entire section:",
        f"    hermes config set --force {key} {value!r}"]
    print("\n".join(err), file=sys.stderr)
    sys.exit(1)


def _touch_skin_file(key: str, value: Any) -> None:
    """``display.skin`` set means "apply NOW": bump the skin file's mtime so the gateway watcher's
    (name, mtime) signature moves even when the name is unchanged. Best-effort."""
    if key == "display.skin" and isinstance(value, str) and value:
        try:
            skin_file = get_hermes_home() / "skins" / f"{value}.yaml"
            if skin_file.exists():
                skin_file.touch()
        except Exception:
            pass


def _exit_invalid(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def _write_user_config(config_path: Path, user_config: Dict[str, Any]) -> None:
    """Write only the user's raw config back (never the merged defaults)."""
    ensure_hermes_home()
    atomic_yaml_write(config_path, user_config, sort_keys=False)


def _print_unknown_key_notice(key: str, suggestion: Optional[str]) -> None:
    print(color(
        f"⚠ '{key}' is not a recognized config key — it was saved anyway, "
        "but Hermes may not read it.", Colors.YELLOW))
    if suggestion:
        print(color(f"  Did you mean: {suggestion}", Colors.YELLOW))
    print(color(
        "  (Custom top-level keys are supported and bridged to the "
        "environment for skills/external tools. Use --force to skip "
        "this notice.)", Colors.DIM))


def set_config_value(key: str, value: str, force: bool = False):
    """Set a configuration value at a dotted ``key``; ``value`` is auto-coerced to bool/int/float.
    ``force`` skips the unknown-key warning AND authorizes replacing a mapping section with a
    scalar. Without it, scalar writes over mappings are refused and bare ``model`` is redirected
    to ``model.default``."""
    if is_managed():
        managed_error("set configuration values")
        return
    # Empty segments (``"agent."``) would write config["agent"][""] into a live schema section.
    if key != key.strip() or not key.strip():
        _exit_invalid(f"✗ Invalid config key: {key!r} (empty or surrounding whitespace).")
    if "" in _split_key_path(key):
        _exit_invalid(
            f"✗ Invalid config key: {key!r} — contains an empty path segment "
            "(leading, trailing, or doubled '.').")
    _exit_if_key_managed(key, "set")
    if _is_env_config_key(key):
        # Unified lifecycle: also rotates any config.yaml mirror of the old value.
        from hermes_cli.credential_lifecycle import save_provider_env_credential

        # Unified lifecycle: also rotates any config.yaml mirror of the old value so a stale
        # higher-precedence copy can't win (#62269).
        save_provider_env_credential(key.upper(), value)
        print(f"✓ Set {key} in {get_env_path()}")
        return

    # Canonicalize per-platform display keys BEFORE validation/coercion so both see the path the
    # runtime reads. Unknown keys are still written (top-level scalars are bridged into os.environ
    # for skills/external apps) but get a post-write "did you mean" hint.
    key, _redirect_note = _redirect_platform_display_key(key)
    if _redirect_note:
        # Unknown-key notice (#34067): the key is still written (arbitrary keys are supported — top-level
        # scalars are bridged into os.environ for skills and external apps), but a plausible-but-wrong
        # dotted path like ``gateway.discord.gateway_restart_notification`` previously reported bare success
        # and left the user debugging behavior that never changed. Warn after the write so the user gets
        # immediate feedback plus a "did you mean" hint, without blocking legitimate unknown keys.
        print(_redirect_note)
    is_known, suggestion = _validate_config_key(key)

    # Read the RAW user config (not merged) so defaults are never dumped back; fail-closed.
    config_path = get_config_path()
    user_config = require_readable_config_before_write(config_path)
    value = _coerce_config_set_value(key, value)
    # A scalar ``model`` shorthand must become a dict before writing sub-keys, or _set_nested
    # replaces it with an empty dict and the model id is lost.
    _model_val = user_config.get("model")
    if key.strip().lower().startswith("model.") and isinstance(_model_val, str) and _model_val:
        user_config["model"] = {"default": _model_val}
    key = _guard_section_overwrite(key, value, user_config, force)
    try:
        _set_nested(user_config, key, value)
    except ValueError as e:
        _exit_invalid(f"✗ {e}")
    # api_base -> base_url alias at set-time too (mirrors _normalize_root_model_keys).
    if key.strip().lower() in ("model.api_base", "api_base"):
        # Normalize the api_base → base_url alias at set-time too (issue #8919), so a fresh `hermes config
        # set model.api_base ...` lands on the canonical key the runtime resolver actually reads, instead of
        # being silently ignored.
        user_config = _normalize_root_model_keys(user_config)
        key = "model.base_url"
        print("  (note: 'api_base' is an alias — saved as model.base_url)")
    _write_user_config(config_path, user_config)

    # Keep .env in sync: terminal_tool reads TERMINAL_ENV etc. directly from env vars.
    env_var = terminal_config_env_var_for_key(key)
    if env_var and key != "terminal.cwd":
        save_env_value(env_var, _terminal_env_value(value))

    _touch_skin_file(key, value)

    # Mask the echoed value when the (possibly nested) key is credential-shaped, e.g.
    # ``model.api_key`` (lowercase, so it misses the .env routing above).
    _display_value = value
    if key.rsplit(".", 1)[-1].lower() in _SECRET_CONFIG_KEYS and isinstance(value, str) and value:
        from agent.redact import mask_secret
        _display_value = mask_secret(value)
    print(f"✓ Set {key} = {_display_value} in {config_path}")
    warn_unpinned_cron_jobs_after_model_config_change(key, value, user_config)

    # Post-write unknown-key notice (#34067): value IS saved, but tell the user the runtime may never read
    # it and suggest the likely-intended path.
    if not is_known and not force:
        _print_unknown_key_notice(key, suggestion)


def get_config_value(key: str, *, as_json: bool = False):
    """Print a resolved configuration value."""
    if _is_env_config_key(key):
        env_value = get_env_value(key.upper())
        value = _MISSING if env_value is None else env_value
    else:
        # Mirror set_config_value: read the canonical display.platforms path.
        # See #71047.
        key, _ = _redirect_platform_display_key(key)
        value = _get_nested(load_config(), key)

    if value is _MISSING:
        _exit_invalid(f"Config key not set: {key}")

    print(_format_config_get_value(value, as_json=as_json))


def unset_config_value(key: str):
    """Remove a user-set configuration or .env value."""
    if is_managed():
        managed_error("unset configuration values")
        return
    _exit_if_key_managed(key, "unset")

    if _is_env_config_key(key):
        # Unified lifecycle: also prunes env-seeded credential_pool entries and model-cache rows so
        # the provider is fully removed instead of left resurrectable.
        # See #51071.
        from hermes_cli.credential_lifecycle import remove_provider_env_credential

        if not remove_provider_env_credential(key.upper()).get("found"):
            _exit_invalid(f"Config key not set: {key}")
        print(f"✓ Unset {key} from {get_env_path()}")
        return

    config_path = get_config_path()
    user_config = require_readable_config_before_write(config_path)

    key, _redirect_note = _redirect_platform_display_key(key)
    if _redirect_note:
        # Mirror set_config_value's display.platforms canonicalization (#71047).
        print(_redirect_note.replace("saved as", "resolved as"))
    removed = _unset_nested(user_config, key)

    env_var = terminal_config_env_var_for_key(key)
    if env_var and key != "terminal.cwd":
        removed = remove_env_value(env_var) or removed

    if not removed:
        _exit_invalid(f"Config key not set: {key}")

    _write_user_config(config_path, user_config)
    print(f"✓ Unset {key} from {config_path}")


# ---- Command handler ----

def _usage_exit(usage: str, examples: List[str], extra: Optional[List[str]] = None) -> None:
    print(usage)
    print()
    print("Examples:")
    for line in examples:
        print(f"  {line}")
    for line in extra or ():
        print(line)
    sys.exit(1)


def _run_write_command(fn, *args) -> None:
    """Run a config writer, surfacing the fail-closed write guard's RuntimeError as a clean CLI
    error instead of a traceback."""
    try:
        fn(*args)
    except RuntimeError as exc:
        _exit_invalid(f"✗ {exc}")


_USAGE_GET = ("Usage: hermes config get <key> [--json]", [
    "hermes config get model", "hermes config get terminal.backend",
    "hermes config get skills.config --json"], None)
_USAGE_SET = ("Usage: hermes config set [--force] <key> <value>", [
    "hermes config set model anthropic/claude-sonnet-4", "hermes config set terminal.backend docker",
    "hermes config set OPENROUTER_API_KEY sk-or-..."], [
    "", "  --force: skip the unknown-key notice for unrecognized keys,",
    "           and allow a scalar to replace a whole mapping section"])
_USAGE_UNSET = ("Usage: hermes config unset <key>", [
    "hermes config unset model", "hermes config unset terminal.backend",
    "hermes config unset OPENROUTER_API_KEY"], None)


def _cmd_config_get(args):
    key = getattr(args, 'key', None)
    if not key:
        _usage_exit(*_USAGE_GET)
    get_config_value(key, as_json=getattr(args, 'json', False))


def _cmd_config_set(args):
    key = getattr(args, 'key', None)
    value = getattr(args, 'value', None)
    if not key or value is None:
        _usage_exit(*_USAGE_SET)
    _run_write_command(set_config_value, key, value, bool(getattr(args, 'force', False)))


def _cmd_config_unset(args):
    key = getattr(args, 'key', None)
    if not key:
        _usage_exit(*_USAGE_UNSET)
    _run_write_command(unset_config_value, key)


def _tools_suffix(info: Dict[str, Any], fmt: str) -> str:
    tools = info.get("tools", [])
    return fmt.format(", ".join(tools[:2])) if tools else ""


def _print_banner(text: str) -> None:
    print()
    print(color(text, Colors.CYAN, Colors.BOLD))
    print()


def _cmd_config_migrate(args):
    _print_banner("🔄 Checking configuration for updates...")

    missing_env = get_missing_env_vars(required_only=False)
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version(raise_on_parse_error=True)

    if not missing_env and not missing_config and current_ver >= latest_ver:
        print(color("✓ Configuration is up to date!", Colors.GREEN))
        print()
        return

    if current_ver < latest_ver:
        print(f"  Config version: {current_ver} → {latest_ver}")

    if missing_config:
        print(f"\n  {len(missing_config)} new config option(s) will be added with defaults")

    required_missing = [v for v in missing_env if v.get("is_required")]
    optional_missing = [v for v in missing_env if not v.get("is_required") and not v.get("advanced")]
    for heading, group, suffix in (
        ("⚠️  {} required API key(s) missing:", required_missing, ""),
        ("ℹ️  {} optional API key(s) not configured:", optional_missing, " (enables: {})")):
        if group:
            print(f"\n  {heading.format(len(group))}")
            for var in group:
                print(f"     • {var['name']}{_tools_suffix(var, suffix) if suffix else ''}")

    print()
    results = migrate_config(interactive=True, quiet=False)
    print()
    if results["env_added"] or results["config_added"]:
        print(color("✓ Configuration updated!", Colors.GREEN))
    if results["warnings"]:
        print()
        for warning in results["warnings"]:
            print(color(f"  ⚠️  {warning}", Colors.YELLOW))
    print()


def _cmd_config_check(args):
    """Non-interactive report of what's missing."""
    _print_banner("📋 Configuration Status")

    current_ver, latest_ver = check_config_version(raise_on_parse_error=True)
    if current_ver >= latest_ver:
        print(f"  Config version: {current_ver} ✓")
    else:
        print(color(f"  Config version: {current_ver} → {latest_ver} (update available)", Colors.YELLOW))

    groups = (
        ("Required", REQUIRED_ENV_VARS, lambda n, i: color(f"    ✗ {n} (missing)", Colors.RED)),
        ("Optional", OPTIONAL_ENV_VARS,
         lambda n, i: color(f"    ○ {n}{_tools_suffix(i, ' → {}')}", Colors.DIM)))
    for title, table, missing_line in groups:
        print()
        print(color(f"  {title}:", Colors.BOLD))
        for var_name, info in table.items():
            print(f"    ✓ {var_name}" if get_env_value(var_name) else missing_line(var_name, info))

    missing_config = get_missing_config_fields()
    if missing_config:
        print()
        print(color(f"  {len(missing_config)} new config option(s) available", Colors.YELLOW))
        print("    Run 'hermes config migrate' to add them")

    print()


_CONFIG_SUBCOMMANDS = {
    None: lambda args: show_config(),
    "show": lambda args: show_config(),
    "edit": lambda args: edit_config(),
    "get": _cmd_config_get,
    "set": _cmd_config_set,
    "unset": _cmd_config_unset,
    "path": lambda args: print(get_config_path()),
    "env-path": lambda args: print(get_env_path()),
    "migrate": _cmd_config_migrate,
    "check": _cmd_config_check}

_CONFIG_USAGE = """Available commands:
  hermes config           Show current configuration
  hermes config edit      Open config in editor
  hermes config get <key>          Print a resolved config value
  hermes config set <key> <value>   Set a config value
  hermes config unset <key>        Remove a config value
  hermes config check     Check for missing/outdated config
  hermes config migrate   Update config with new options
  hermes config path      Show config file path
  hermes config env-path  Show .env file path"""


def config_command(args):
    """Handle config subcommands."""
    subcmd = getattr(args, 'config_command', None)
    handler = _CONFIG_SUBCOMMANDS.get(subcmd)
    if handler is not None:
        handler(args)
        return
    print(f"Unknown config command: {subcmd}")
    print()
    print(_CONFIG_USAGE)
    sys.exit(1)


# ---- OPTIONAL_ENV_VARS injection from provider profiles and platform plugins (once, at import) ----

def _inject_profile_env_vars() -> None:
    """Expose env_vars of every ``auth_type="api_key"`` provider in providers/ via OPTIONAL_ENV_VARS
    without editing this file."""
    try:
        from providers import list_providers
        for _pp in list_providers():
            if _pp.auth_type != "api_key":
                continue
            for _var in _pp.env_vars:
                if _var in OPTIONAL_ENV_VARS:
                    continue
                _is_key = not _var.endswith(("_BASE_URL", "_URL"))
                _label = _pp.display_name or _pp.name
                OPTIONAL_ENV_VARS[_var] = {
                    "description": f"{_label} {'API key' if _is_key else 'base URL override'}",
                    "prompt": f"{_label} {'API key' if _is_key else 'base URL (leave empty for default)'}",
                    "url": _pp.signup_url or None,
                    "password": _is_key,
                    "category": "provider",
                    "advanced": True}
    except Exception:
        pass


_inject_profile_env_vars()


def _platform_plugin_manifests():
    """Yield ``(dir_name, manifest_dict)`` for every bundled ``plugins/platforms/*/plugin.y(a)ml``."""
    platforms_dir = get_project_root() / "plugins" / "platforms"
    if not platforms_dir.is_dir():
        return
    for child in platforms_dir.iterdir():
        manifest_path = next(
            (p for p in (child / "plugin.yaml", child / "plugin.yml") if child.is_dir() and p.exists()), None)
        if manifest_path is None:
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = fast_safe_load(f) or {}
        except Exception:
            continue
        yield child.name, manifest


def _inject_platform_plugin_env_vars() -> None:
    """Populate OPTIONAL_ENV_VARS from bundled platform plugin manifests so Teams / IRC / Google
    Chat etc. are configurable in ``hermes config`` UI without the core knowing they exist.

    ``requires_env`` / ``optional_env`` entries are a bare name or a dict with ``name`` plus
    optional ``description``/``url``/``password``/``prompt``/``category``. Failures are swallowed
    so a malformed plugin.yaml can't break CLI import.
    """
    try:
        for dir_name, manifest in _platform_plugin_manifests():
            label = manifest.get("label") or manifest.get("name") or dir_name
            for entry in [*(manifest.get("requires_env") or []), *(manifest.get("optional_env") or [])]:
                meta = {"name": entry} if isinstance(entry, str) else entry if isinstance(entry, dict) else {}
                name = meta.get("name")
                if not name or name in OPTIONAL_ENV_VARS:
                    continue  # hardcoded entry wins (back-compat)
                # *TOKEN / *SECRET / *KEY / *PASSWORD / *JSON are password fields unless overridden.
                is_secret = bool(meta.get("password") or meta.get("secret"))
                if not is_secret and not meta.get("password") is False:
                    is_secret = name.upper().endswith(("_TOKEN", "_SECRET", "_KEY", "_PASSWORD", "_JSON"))
                OPTIONAL_ENV_VARS[name] = {
                    "description": meta.get("description") or f"{label} configuration",
                    "prompt": meta.get("prompt") or name,
                    "url": meta.get("url") or None,
                    "password": is_secret,
                    "category": meta.get("category") or "messaging"}
    except Exception:
        pass


_inject_platform_plugin_env_vars()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def _install_method_project_root(project_root: Optional[Path] = None) -> Path:
    """Resolve the directory that holds the *running code* (the install tree).

    This is the parent of ``hermes_cli/`` — i.e. the git checkout for source
    installs, ``/opt/hermes`` inside the published image. It is a property of
    the running interpreter, NOT of ``$HERMES_HOME``, which is why a
    code-scoped stamp here is immune to two installs sharing one data
    directory.
    """
    if project_root is not None:
        return project_root
    return Path(__file__).parent.parent.resolve()

def stamp_install_method(method: str, project_root: Optional[Path] = None) -> None:
    """Write the install method next to the running code (code-scoped stamp).

    The stamp lives in the install tree (``<install tree>/.install_method``),
    not in ``$HERMES_HOME``, so that two installs sharing one data directory
    do not overwrite each other's marker. See ``detect_install_method`` for
    the full rationale.

    Best-effort: if the install tree is read-only (e.g. the immutable
    ``/opt/hermes`` in the published image, which instead bakes the stamp at
    build time) the write silently no-ops and detection falls back to its
    other signals.
    """
    root = _install_method_project_root(project_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / ".install_method").write_text(method + "\n", encoding="utf-8")
    except OSError:
        pass


_PLUGIN_COMPAT_LAZY = {
    'normalize_route_base_url': ('hermes_cli.route_identity', 'normalize_route_base_url'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
