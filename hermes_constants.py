"""Shared constants for Hermes Agent.

Import-safe module with no dependencies — can be imported from anywhere
without risk of circular imports.
"""

import os
import shutil
import sys
import sysconfig
from contextvars import ContextVar, Token
from pathlib import Path


_profile_fallback_warned: bool = False
_UNSET = object()
_HERMES_HOME_OVERRIDE: ContextVar[str | object] = ContextVar(
    "_HERMES_HOME_OVERRIDE", default=_UNSET
)


def set_hermes_home_override(path: str | Path | None) -> Token:
    """Set a context-local Hermes home override and return its reset token.

    This is for in-process, per-task scoping.  It deliberately does not mutate
    ``os.environ`` because that is shared by every thread in the process.
    """
    value: str | object = _UNSET if path is None else str(path)
    return _HERMES_HOME_OVERRIDE.set(value)


def reset_hermes_home_override(token: Token) -> None:
    """Restore the previous context-local Hermes home override."""
    _HERMES_HOME_OVERRIDE.reset(token)


def get_hermes_home_override() -> str | None:
    """Return the active context-local Hermes home override, if any."""
    override = _HERMES_HOME_OVERRIDE.get()
    if override is _UNSET or not override:
        return None
    return str(override)


def _get_platform_default_hermes_home() -> Path:
    """Return the platform-native default Hermes home path."""
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def get_hermes_home() -> Path:
    """Return the Hermes home directory (default: platform-native path).

    Reads HERMES_HOME env var, falls back to the platform-native default.
    This is the single source of truth — all other copies should import this.

    When ``HERMES_HOME`` is unset but an ``active_profile`` file indicates
    a non-default profile is active, logs a loud one-shot warning to
    ``errors.log`` so cross-profile data corruption is diagnosable instead
    of silent.  Behavior is unchanged otherwise — we still return
    the platform-native default — because raising here would brick 30+ module-level
    callers that import this at load time.  Subprocess spawners are
    expected to propagate ``HERMES_HOME`` explicitly (see the systemd
    template in ``hermes_cli/gateway.py`` and the kanban dispatcher in
    ``hermes_cli/kanban_db.py``).  See https://github.com/NousResearch/hermes-agent/issues/18594.
    """
    override = get_hermes_home_override()
    if override:
        return Path(override)

    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)

    # Guard: if a non-default profile is sticky-active, warn once that
    # the fallback to the default profile is almost certainly wrong.
    global _profile_fallback_warned
    if not _profile_fallback_warned:
        try:
            fallback_home = _get_platform_default_hermes_home()
            active_path = fallback_home / "active_profile"
            active = active_path.read_text().strip() if active_path.exists() else ""
        except (UnicodeDecodeError, OSError):
            active = ""
        if active and active != "default":
            _profile_fallback_warned = True
            # Write directly to stderr.  We intentionally do NOT route this
            # through ``logging`` because (a) this function is called at
            # module-import time from 30+ sites, often before logging is
            # configured, and (b) root-logger propagation would double-emit
            # on consoles where a StreamHandler is already attached.
            msg = (
                f"[HERMES_HOME fallback] HERMES_HOME is unset but active "
                f"profile is {active!r}. Falling back to {fallback_home}, which "
                f"is the DEFAULT profile — not {active!r}. Any data this "
                f"process writes will land in the wrong profile. The "
                f"subprocess spawner should pass HERMES_HOME explicitly "
                f"(see issue #18594)."
            )
            try:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            except Exception:
                pass

    return _get_platform_default_hermes_home()


def get_default_hermes_root() -> Path:
    """Return the root Hermes directory for profile-level operations.

    In standard deployments this is the platform-native Hermes home
    (``~/.hermes`` on POSIX, ``%LOCALAPPDATA%\\hermes`` on native Windows).

    In Docker or custom deployments where ``HERMES_HOME`` points outside
    ``~/.hermes`` (e.g. ``/opt/data``), returns ``HERMES_HOME`` directly
    — that IS the root.

    In profile mode where ``HERMES_HOME`` is ``<root>/profiles/<name>``,
    returns ``<root>`` so that ``profile list`` can see all profiles.
    Works both for standard (``~/.hermes/profiles/coder``) and Docker
    (``/opt/data/profiles/coder``) layouts.

    Import-safe — no dependencies beyond stdlib.
    """
    native_home = _get_platform_default_hermes_home()
    env_home = os.environ.get("HERMES_HOME", "")
    if not env_home:
        return native_home
    env_path = Path(env_home)
    try:
        env_path.resolve().relative_to(native_home.resolve())
        # HERMES_HOME is under ~/.hermes (normal or profile mode)
        return native_home
    except ValueError:
        pass

    # Docker / custom deployment.
    # Check if this is a profile path: <root>/profiles/<name>
    # If the immediate parent dir is named "profiles", the root is
    # the grandparent — this covers Docker profiles correctly.
    if env_path.parent.name == "profiles":
        return env_path.parent.parent

    # Not a profile path — HERMES_HOME itself is the root
    return env_path


def _get_packaged_data_dir(name: str) -> Path | None:
    """Return an installed data-files directory if one exists.

    Used to discover bundled skills/optional-skills when Hermes is installed
    from a wheel that emitted them via setuptools data_files.
    """
    candidates = []
    for scheme in ("data", "purelib", "platlib"):
        raw = sysconfig.get_path(scheme)
        if raw:
            candidates.append(Path(raw) / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """Return the optional-skills directory, honoring package-manager wrappers.

    Packaged installs may ship ``optional-skills`` outside the Python package
    tree and expose it via ``HERMES_OPTIONAL_SKILLS``.
    """
    override = os.getenv("HERMES_OPTIONAL_SKILLS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("optional-skills")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_hermes_home() / "optional-skills"


def get_optional_mcps_dir(default: Path | None = None) -> Path:
    """Return the optional-mcps directory, honoring package-manager wrappers.

    Mirrors :func:`get_optional_skills_dir` for the MCP catalog (Nous-approved
    Model Context Protocol servers shipped with the repo but disabled by
    default). Packaged installs may ship ``optional-mcps`` outside the Python
    package tree and expose it via ``HERMES_OPTIONAL_MCPS``.
    """
    override = os.getenv("HERMES_OPTIONAL_MCPS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("optional-mcps")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_hermes_home() / "optional-mcps"


def get_bundled_skills_dir(default: Path | None = None) -> Path:
    """Return the bundled skills directory for source and packaged installs.

    Resolution order:
        1. ``HERMES_BUNDLED_SKILLS`` env var (Nix wrapper / explicit override)
        2. Wheel-installed ``<sysconfig data>/skills`` (pip install path)
        3. Caller-supplied ``default`` (typically the source-checkout path)
        4. ``<HERMES_HOME>/skills`` last-resort
    """
    override = os.getenv("HERMES_BUNDLED_SKILLS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("skills")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_hermes_home() / "skills"


def get_hermes_dir(new_subpath: str, old_name: str) -> Path:
    """Resolve a Hermes subdirectory with backward compatibility.

    New installs get the consolidated layout (e.g. ``cache/images``).
    Existing installs that already have the old path (e.g. ``image_cache``)
    keep using it — no migration required.

    Args:
        new_subpath: Preferred path relative to HERMES_HOME (e.g. ``"cache/images"``).
        old_name: Legacy path relative to HERMES_HOME (e.g. ``"image_cache"``).

    Returns:
        Absolute ``Path`` — old location if it exists on disk, otherwise the new one.
    """
    home = get_hermes_home()
    old_path = home / old_name
    if old_path.exists():
        return old_path
    return home / new_subpath


def iter_hermes_node_dirs(home: Path | None = None) -> list[Path]:
    """Return Hermes-managed Node.js directories in preferred lookup order.

    Windows installs from ``scripts/install.ps1`` unpack portable Node directly
    into ``%LOCALAPPDATA%\\hermes\\node``. POSIX installs use
    ``$HERMES_HOME/node/bin``. Include both shapes on every platform so mixed
    or migrated installs still work.
    """
    root = home or get_hermes_home()
    dirs = [root / "node"]
    bin_dir = root / "node" / "bin"
    # NOTE: keep this ordering in sync with hermesManagedNodePathEntries() in
    # apps/desktop/electron/main.cjs — the Electron main process is Node and
    # cannot import this module, so the platform-ordering rule is mirrored there.
    if sys.platform == "win32":
        return dirs + [bin_dir]
    return [bin_dir] + dirs


def _candidate_node_command_names(command: str) -> list[str]:
    base = Path(command).name
    if sys.platform != "win32" or "." in base:
        return [base]
    if base.lower() == "npm":
        # Prefer npm.cmd. PowerShell may block npm.ps1 by execution policy, and
        # CreateProcess cannot launch a bare .ps1 the way it can launch .cmd.
        return ["npm.cmd", "npm.exe", "npm"]
    if base.lower() == "npx":
        return ["npx.cmd", "npx.exe", "npx"]
    if base.lower() == "node":
        return ["node.exe", "node"]
    return [f"{base}.cmd", f"{base}.exe", base]


def find_hermes_node_executable(command: str) -> str | None:
    """Return a Hermes-managed Node/npm executable path, if installed."""
    names = _candidate_node_command_names(command)
    for directory in iter_hermes_node_dirs():
        for name in names:
            candidate = directory / name
            if candidate.is_file() and (
                sys.platform == "win32" or os.access(candidate, os.X_OK)
            ):
                return str(candidate)
    return None


def find_node_executable_on_path(command: str) -> str | None:
    """Return a Node/npm executable from PATH with Windows shim ordering.

    ``shutil.which("npm")`` can resolve an extensionless npm shim before the
    ``.cmd`` shim on Windows. Python's CreateProcess cannot execute that shim
    directly, so prefer the launchable variants explicitly for Hermes-owned
    subprocesses.
    """
    if sys.platform != "win32":
        return shutil.which(command)

    command_str = str(command)
    has_path_separator = any(
        sep and sep in command_str for sep in (os.sep, os.altsep, "/", "\\")
    )
    if has_path_separator:
        return command_str if Path(command_str).is_file() else None

    for name in _candidate_node_command_names(command_str):
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if not directory:
                continue
            candidate = Path(directory) / name
            if candidate.is_file():
                return str(candidate)
    return None


def find_node_executable(command: str) -> str | None:
    """Resolve a Node.js command, preferring Hermes-managed installs.

    This is for Hermes-owned subprocesses that should not be broken by a bad,
    missing, or elevation-triggering system Node/npm on PATH.
    """
    return find_hermes_node_executable(command) or find_node_executable_on_path(command)


def with_hermes_node_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return *env* with Hermes-managed Node directories prepended to PATH."""
    merged = dict(os.environ if env is None else env)
    existing = merged.get("PATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    managed = [str(path) for path in iter_hermes_node_dirs() if path.is_dir()]
    for entry in reversed(managed):
        if entry not in parts:
            parts.insert(0, entry)
    merged["PATH"] = os.pathsep.join(parts)
    return merged


def agent_browser_runnable(path: str | None) -> bool:
    """Return True only when *path* is an agent-browser CLI that actually runs.

    A bare presence check (``shutil.which`` / ``Path.exists``) is not enough:
    agent-browser's npm ``postinstall`` re-points a *global* install symlink
    (e.g. ``/opt/homebrew/bin/agent-browser``) at our local
    ``node_modules/agent-browser/bin/...`` binary, which then disappears on the
    next ``hermes update`` — leaving a **dangling symlink** that ``which`` still
    reports but exec fails on with exit 127 (issue #48521). Callers that trust
    such a path silently break every browser tool.

    This validates the candidate by resolving it to a real, executable file and
    running ``--version`` with a short timeout. Returns True only on a clean
    (exit 0) run, so a dead/wrong-arch/hung binary is rejected and the caller
    can fall through to the next resolution candidate.

    Special cases:
      * ``None`` / empty → False.
      * The ``"npx agent-browser"`` fallback form (contains a space, not a real
        file) → True; npx resolves and validates the package at run time, so
        there is nothing to stat here.
    """
    if not path:
        return False
    # The npx fallback is a two-token command string, not a filesystem path.
    if " " in path and path.split()[0].endswith("npx"):
        return True
    # exists() follows symlinks — a dangling link returns False here, so we
    # never even spawn a subprocess for the broken-link case.
    if not os.path.exists(path) or not os.access(path, os.X_OK):
        return False
    import subprocess

    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            timeout=10,
            env=with_hermes_node_path(),
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False
    return result.returncode == 0


def display_hermes_home() -> str:
    """Return a user-friendly display string for the current HERMES_HOME.

    Uses ``~/`` shorthand for readability::

        default:  ``~/.hermes``
        profile:  ``~/.hermes/profiles/coder``
        custom:   ``/opt/hermes-custom``

    Use this in **user-facing** print/log messages instead of hardcoding
    ``~/.hermes``.  For code that needs a real ``Path``, use
    :func:`get_hermes_home` instead.
    """
    home = get_hermes_home()
    try:
        return "~/" + str(home.relative_to(Path.home()))
    except ValueError:
        return str(home)


def secure_parent_dir(path: Path) -> None:
    """Chmod ``0o700`` on the parent directory of *path*, but only if safe.

    Refuses to chmod ``/`` or any top-level directory (resolved parent with
    fewer than 3 parts, i.e. ``/`` or any direct child like ``/usr``) to
    prevent catastrophic host bricking when ``HERMES_HOME`` or other path
    env vars resolve to an unexpected location.

    See https://github.com/NousResearch/hermes-agent/issues/25821.
    """
    parent = path.parent.resolve()
    # Refuse root and its direct children (/usr, /home, /var, /tmp, …).
    if parent == Path("/") or len(parent.parts) < 3:
        return
    try:
        os.chmod(parent, 0o700)
    except OSError:
        pass


def _norm_home_path(path: str | None) -> str:
    """Return a comparable absolute path string, or ``""`` for empty input."""
    raw = (path or "").strip()
    if not raw:
        return ""
    try:
        return os.path.normcase(os.path.abspath(os.path.expanduser(raw)))
    except Exception:
        return os.path.normcase(raw)


def _profile_home_path(env: dict[str, str] | None = None) -> str | None:
    """Return ``{HERMES_HOME}/home`` when the profile-home directory exists."""
    hermes_home = get_hermes_home_override() or (env or {}).get("HERMES_HOME") or os.getenv("HERMES_HOME")
    if not hermes_home:
        return None
    profile_home = os.path.join(hermes_home, "home")
    if os.path.isdir(profile_home):
        return profile_home
    return None


def _is_profile_home(candidate: str | None, profile_home: str | None) -> bool:
    return bool(candidate and profile_home and _norm_home_path(candidate) == _norm_home_path(profile_home))


def _iter_real_home_candidates(env: dict[str, str] | None = None) -> list[str]:
    """Return likely OS-user home candidates in trust order."""
    env = env or {}
    candidates: list[str] = []
    explicit = str(env.get("HERMES_REAL_HOME") or os.getenv("HERMES_REAL_HOME", "")).strip()
    if explicit:
        candidates.append(explicit)
    home = str(env.get("HOME") or os.getenv("HOME", "")).strip()
    if home:
        candidates.append(home)
    try:
        import pwd

        pw_home = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX-only module inside try/except
        if pw_home:
            candidates.append(pw_home)
    except Exception:
        pass
    userprofile = str(env.get("USERPROFILE") or os.getenv("USERPROFILE", "")).strip()
    if userprofile:
        candidates.append(userprofile)
    drive = str(env.get("HOMEDRIVE") or os.getenv("HOMEDRIVE", "")).strip()
    path = str(env.get("HOMEPATH") or os.getenv("HOMEPATH", "")).strip()
    if drive and path:
        candidates.append(f"{drive}{path}" if path.startswith(("\\", "/")) else os.path.join(drive, path))
    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        candidates.append(expanded)
    return candidates


def get_real_home(env: dict[str, str] | None = None) -> str:
    """Return the OS user's real home directory, avoiding Hermes profile HOME.

    ``HERMES_HOME`` scopes Hermes state. ``HOME`` is reserved for the OS/user
    account and the many external CLIs that store credentials under ``~``.
    If a parent process is already running with ``HOME={HERMES_HOME}/home``,
    this helper repairs back to the account home when possible.
    """
    profile_home = _profile_home_path(env)
    seen: set[str] = set()
    for candidate in _iter_real_home_candidates(env):
        key = _norm_home_path(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if not _is_profile_home(candidate, profile_home):
            return candidate
    return "/tmp"


def get_subprocess_home(env: dict[str, str] | None = None) -> str | None:
    """Return a subprocess ``HOME`` override, if one should be applied.

    Policy is controlled by ``terminal.home_mode`` (bridged to
    ``TERMINAL_HOME_MODE``):

    * ``auto`` (default): host installs keep the real user HOME; containers use
      ``{HERMES_HOME}/home`` for persistent state. If a host parent already has
      HOME pointed at the profile home, repair subprocesses back to real HOME.
    * ``real``: always prefer the real OS-user HOME.
    * ``profile``: use ``{HERMES_HOME}/home`` when it exists, preserving the
      older strict per-profile tool-config isolation.
    """
    env = env or {}
    profile_home = _profile_home_path(env)
    mode = str(env.get("TERMINAL_HOME_MODE") or os.getenv("TERMINAL_HOME_MODE", "auto")).strip().lower() or "auto"
    if mode in {"isolated", "profile_home", "profile-home"}:
        mode = "profile"
    if mode in {"host", "user", "real_home", "real-home"}:
        mode = "real"

    if mode == "profile":
        return profile_home

    real_home = get_real_home(env)
    current_home = str(env.get("HOME") or os.getenv("HOME", "")).strip()
    if mode == "real":
        return real_home if _norm_home_path(real_home) != _norm_home_path(current_home) else None

    if profile_home and is_container():
        return profile_home
    if _is_profile_home(current_home, profile_home):
        return real_home if _norm_home_path(real_home) != _norm_home_path(current_home) else None
    return None


def apply_subprocess_home_env(env: dict[str, str]) -> None:
    """Apply Hermes' subprocess HOME contract to *env* in-place."""
    real_home = get_real_home(env)
    if real_home:
        env["HERMES_REAL_HOME"] = real_home
    home = get_subprocess_home(env)
    if home:
        env["HOME"] = home


VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
LEGACY_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
CODEX_GPT55_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
ANTHROPIC_46_REASONING_EFFORTS = ("low", "medium", "high", "max")
ANTHROPIC_MODERN_REASONING_EFFORTS = ("low", "medium", "high", "max")
GEMINI_FLASH_REASONING_EFFORTS = ("minimal", "low", "medium", "high")
GEMINI_STANDARD_REASONING_EFFORTS = ("low", "medium", "high")
GEMINI_PRO_STRICT_REASONING_EFFORTS = ("low", "high")
_REASONING_EFFORT_LABELS = {effort: effort for effort in VALID_REASONING_EFFORTS}
_REASONING_EFFORT_LABELS["none"] = "none"
_REASONING_ALIASES = {
    "minimal": "minimal",
    "minimum": "minimal",
    "low": "low",
    "medium": "medium",
    "med": "medium",
    "high": "high",
    "extra_high": "xhigh",
    "extra-high": "xhigh",
    "extra high": "xhigh",
    "extrahigh": "xhigh",
    "xhigh": "xhigh",
    "x high": "xhigh",
    "x-high": "xhigh",
    "max": "max",
    "maximum": "max",
}


def canonicalize_reasoning_effort(effort: str | None) -> str | None:
    """Return Hermes' canonical provider-style reasoning effort value.

    Hermes stores provider-style tags (``minimal``, ``low``, ``medium``,
    ``high``, ``xhigh``, ``max``).  Older UI/config spellings such as
    ``extra_high`` remain accepted as hidden compatibility aliases.
    """
    if effort is None or not str(effort).strip():
        return None
    raw = str(effort).strip().lower()
    squashed = " ".join(raw.replace("_", " ").replace("-", " ").split())
    return _REASONING_ALIASES.get(raw) or _REASONING_ALIASES.get(squashed)


def _normalize_model_id_for_reasoning(model: str) -> str:
    """Return a provider-prefix/variant-stripped model id for effort lookup.

    Picker rows and runtime configs do not use one canonical spelling:
    OpenRouter uses ``vendor/model``, Bedrock uses dotted inference-profile IDs
    like ``global.anthropic.claude-opus-4-8``, and user/custom providers often
    copy those forms verbatim.  Reasoning effort support belongs to the model
    family, so normalize those transport prefixes before selecting the effort
    surface.  This keeps Desktop, gateway slash commands, and request clamping
    in lock-step for the same model no matter which provider row serves it.
    """
    value = str(model or "").strip().lower().replace("_", "-")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    # Bedrock / routed-provider IDs can be dotted prefix + canonical family,
    # e.g. global.anthropic.claude-sonnet-4-6.  Strip everything before the
    # family marker instead of requiring a particular provider slug.
    for marker in ("claude-", "gpt-5.5", "gemini-"):
        idx = value.find(marker)
        if idx > 0:
            value = value[idx:]
            break
    # Fast/latest/preview are delivery/snapshot variants, not separate effort
    # surfaces.  Keep this narrow so real base ids are not over-normalized.
    for suffix in ("-fast", "-latest", "-preview"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def _provider_for_reasoning(provider: str | None) -> str:
    return str(provider or "").strip().lower().replace("_", "-")


def _is_anthropic_reasoning_model(provider: str | None, model: str | None) -> bool:
    provider_value = _provider_for_reasoning(provider)
    raw_model = str(model or "").strip().lower()
    model_value = _normalize_model_id_for_reasoning(raw_model)
    if not model_value.startswith("claude-"):
        return False
    if provider_value in {
        "anthropic",
        "claude",
        "anthropic-messages",
        "bedrock",
        "aws-bedrock",
        "opencode-zen",
    }:
        return True
    if provider_value == "openrouter" and raw_model.startswith("anthropic/"):
        return True
    # Provider catalogs and custom endpoints frequently return already-normalized
    # Claude IDs (e.g. ``claude-sonnet-4.6``) even when the provider slug is a
    # relay.  The effort surface is still the Claude-family surface; if the
    # model ID itself is unambiguously Claude, keep UI + clamping model-specific
    # rather than falling back to Hermes' stale global enum.
    return raw_model.startswith("claude-")


def _is_gpt55_reasoning_model(provider: str | None, model: str | None) -> bool:
    provider_value = _provider_for_reasoning(provider)
    raw_model = str(model or "").strip().lower()
    model_value = _normalize_model_id_for_reasoning(raw_model)
    if not (model_value == "gpt-5.5" or model_value.startswith("gpt-5.5-")):
        return False
    if provider_value in {"openai-codex", "codex", "openai", "openai-api", "openrouter"}:
        return True
    if raw_model.startswith(("openai/gpt-5.5", "gpt-5.5")):
        return True
    return False


def _is_anthropic_46_reasoning_model(provider: str | None, model: str | None) -> bool:
    if not _is_anthropic_reasoning_model(provider, model):
        return False
    model_value = _normalize_model_id_for_reasoning(str(model or ""))
    return "claude-opus-4.6" in model_value or "claude-opus-4-6" in model_value or (
        "claude-sonnet-4.6" in model_value or "claude-sonnet-4-6" in model_value
    )


def _is_modern_anthropic_reasoning_model(provider: str | None, model: str | None) -> bool:
    if not _is_anthropic_reasoning_model(provider, model):
        return False
    model_value = _normalize_model_id_for_reasoning(str(model or ""))
    legacy_prefixes = (
        "claude-3",
        "claude-opus-4-0", "claude-opus-4.0", "claude-opus-4-1", "claude-opus-4.1",
        "claude-sonnet-4-5", "claude-sonnet-4.5",
        "claude-haiku-4-5", "claude-haiku-4.5",
    )
    return not model_value.startswith(legacy_prefixes)


def _is_gemini_reasoning_model(provider: str | None, model: str | None) -> bool:
    provider_value = _provider_for_reasoning(provider)
    raw_model = str(model or "").strip().lower()
    model_value = _normalize_model_id_for_reasoning(raw_model)
    if provider_value in {"gemini", "google", "google-gemini"}:
        return model_value.startswith("gemini-")
    if provider_value == "openrouter" and raw_model.startswith("google/gemini-"):
        return model_value.startswith("gemini-")
    return False


def _gemini_reasoning_efforts(model: str | None) -> tuple[str, ...]:
    model_value = _normalize_model_id_for_reasoning(str(model or ""))
    if not model_value.startswith("gemini-"):
        return LEGACY_REASONING_EFFORTS
    if model_value.startswith("gemini-2.5-"):
        return GEMINI_STANDARD_REASONING_EFFORTS
    if model_value.startswith("gemini-3.1-"):
        return GEMINI_STANDARD_REASONING_EFFORTS
    if "flash" in model_value and model_value.startswith(("gemini-3-", "gemini-3.0-", "gemini-3.5-")):
        return GEMINI_FLASH_REASONING_EFFORTS
    if "pro" in model_value and model_value.startswith(("gemini-3-", "gemini-3.0-")):
        return GEMINI_PRO_STRICT_REASONING_EFFORTS
    return GEMINI_STANDARD_REASONING_EFFORTS


def is_codex_gpt55_model(provider: str | None, model: str | None) -> bool:
    """True for GPT-5.5 on the ChatGPT Codex/OAuth provider.

    Kept as the narrow Codex-route predicate for callers whose behavior really
    is transport-specific (e.g. context/autoraise rules).  General reasoning
    effort lookup uses ``_is_gpt55_reasoning_model`` so OpenRouter/OpenAI API
    rows for the same GPT-5.5 family do not fall back to Hermes' legacy enum.
    """
    provider_value = _provider_for_reasoning(provider)
    if provider_value not in {"openai-codex", "codex"}:
        return False
    return _is_gpt55_reasoning_model(provider, model)


def reasoning_efforts_for_model(provider: str | None = None, model: str | None = None) -> tuple[str, ...]:
    """Return the selectable reasoning efforts for a provider/model pair.

    The global Hermes config enum remains broad for legacy/provider surfaces.
    UI and gateway pickers should call this helper to avoid advertising levels
    a specific backend does not actually expose.
    """
    if _is_gpt55_reasoning_model(provider, model):
        return CODEX_GPT55_REASONING_EFFORTS
    if _is_anthropic_46_reasoning_model(provider, model):
        return ANTHROPIC_46_REASONING_EFFORTS
    if _is_modern_anthropic_reasoning_model(provider, model):
        return ANTHROPIC_MODERN_REASONING_EFFORTS
    if _is_gemini_reasoning_model(provider, model):
        return _gemini_reasoning_efforts(model)
    return VALID_REASONING_EFFORTS


def normalize_reasoning_effort_for_model(
    effort: str,
    provider: str | None = None,
    model: str | None = None,
) -> str | None:
    """Normalize and validate an effort for a provider/model pair.

    Returns the canonical stored/wire value, or ``None`` when unavailable.
    ``xhigh`` is the canonical value for the user-facing "Extra High" tier;
    ``max`` is a distinct strongest tier on providers that expose it.
    """
    value = canonicalize_reasoning_effort(effort)
    if value in reasoning_efforts_for_model(provider, model):
        return value
    return None


def resolve_reasoning_effort_for_request(
    effort: str | None,
    provider: str | None = None,
    model: str | None = None,
    *,
    default: str | None = None,
) -> str | None:
    """Return a safe wire effort for a provider/model request.

    This is slightly more permissive than ``normalize_reasoning_effort_for_model``
    because request builders also receive stale persisted configs.  In particular,
    Claude 4.6 rejects ``xhigh`` but accepts ``max``; map that one stale value to
    the strongest supported level instead of forwarding a guaranteed API 400.
    """
    normalized = normalize_reasoning_effort_for_model(effort or "", provider, model)
    if normalized is not None:
        return normalized
    value = canonicalize_reasoning_effort(effort) or str(effort or "").strip().lower()
    if value == "minimal" and _is_anthropic_reasoning_model(provider, model):
        # ``minimal`` is a legacy Hermes UI/config tier.  Anthropic adaptive
        # thinking exposes ``low`` as its smallest wire effort; the direct
        # Anthropic adapter has long mapped minimal→low, so keep OpenRouter and
        # other request builders on the same safe path instead of omitting the
        # field or forwarding an unsupported value.
        return "low"
    if _is_gpt55_reasoning_model(provider, model):
        # GPT-5.5 exposes Low / Medium / High / Extra High, with Extra High
        # carried as Hermes' canonical ``xhigh``.  Persisted configs are global,
        # so a session can legitimately switch from an Opus preset with ``max``
        # to GPT-5.5.  Clamp that stale-but-semantically-strongest request to
        # GPT-5.5's strongest accepted level instead of falling through to a
        # low-effort default.
        if value == "max":
            return "xhigh"
        if value == "minimal":
            return "low"
    if value == "xhigh" and _is_anthropic_reasoning_model(provider, model):
        return "max"
    if default is not None:
        return normalize_reasoning_effort_for_model(default, provider, model)
    return None


def codex_reasoning_effort_wire_value(effort: str | None) -> str | None:
    """Return the OpenAI Codex/GPT-5.5 ``reasoning.effort`` wire value."""
    canonical = canonicalize_reasoning_effort(effort)
    if canonical == "minimal":
        return "low"
    if canonical in {"low", "medium", "high", "xhigh"}:
        return canonical
    if canonical == "max":
        return "xhigh"
    return None


def anthropic_reasoning_effort_wire_value(effort: str | None) -> str | None:
    """Return the Anthropic/Claude adaptive-thinking wire effort."""
    canonical = canonicalize_reasoning_effort(effort)
    if canonical == "minimal":
        return "low"
    if canonical in {"xhigh", "max"}:
        return "max"
    if canonical in {"low", "medium", "high"}:
        return canonical
    return None


def xai_reasoning_effort_wire_value(effort: str | None) -> str | None:
    """Return the current xAI Responses reasoning-effort wire value."""
    canonical = canonicalize_reasoning_effort(effort)
    if canonical == "minimal":
        return "low"
    if canonical in {"xhigh", "max"}:
        return "high"
    if canonical in {"low", "medium", "high"}:
        return canonical
    return None


def gemini_thinking_config_for_reasoning(
    model: str,
    reasoning_config: dict | None,
) -> dict | None:
    """Translate Hermes reasoning config to Gemini native thinkingConfig."""
    if reasoning_config is None or not isinstance(reasoning_config, dict):
        return None

    normalized_model = (model or "").strip().lower()
    if normalized_model.startswith("google/"):
        normalized_model = normalized_model.split("/", 1)[1]

    if not normalized_model.startswith("gemini"):
        return None

    if reasoning_config.get("enabled") is False:
        return {"includeThoughts": False}

    raw_effort = str(reasoning_config.get("effort", "medium") or "medium").strip().lower()
    if raw_effort == "none":
        return {"includeThoughts": False}

    canonical = canonicalize_reasoning_effort(raw_effort) or "medium"
    thinking_config: dict = {"includeThoughts": True}

    # Gemini 2.5 exposes budget controls, not the 3.x named thinking levels.
    # ``includeThoughts`` alone requests visible thought parts without guessing
    # a numeric budget.
    if normalized_model.startswith("gemini-2.5-"):
        return thinking_config

    if normalized_model.startswith(("gemini-3", "gemini-3.1", "gemini-3.5")):
        if "flash" in normalized_model:
            if canonical == "minimal":
                thinking_config["thinkingLevel"] = "minimal"
            elif canonical == "low":
                thinking_config["thinkingLevel"] = "low"
            elif canonical in {"high", "xhigh", "max"}:
                thinking_config["thinkingLevel"] = "high"
            else:
                thinking_config["thinkingLevel"] = "medium"
        elif "pro" in normalized_model:
            if canonical == "low":
                thinking_config["thinkingLevel"] = "low"
            elif canonical in {"high", "xhigh", "max"}:
                thinking_config["thinkingLevel"] = "high"
            else:
                thinking_config["thinkingLevel"] = "medium"

    return thinking_config


def gemini_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    """Backward-compatible alias for Gemini reasoning config translation."""
    return gemini_thinking_config_for_reasoning(model, reasoning_config)


def reasoning_effort_display_label(effort: str | None) -> str:
    """Return the provider-style display tag for a reasoning effort."""
    raw = str(effort or "").strip()
    if not raw:
        return ""
    normalized = canonicalize_reasoning_effort(raw)
    if normalized:
        return _REASONING_EFFORT_LABELS.get(normalized, normalized)
    return raw


def format_reasoning_effort_labels(efforts: tuple[str, ...] | list[str]) -> str:
    """Return a comma-separated human-facing label list for reasoning efforts."""
    return ", ".join(reasoning_effort_display_label(effort) for effort in efforts)


def parse_reasoning_effort(effort: str) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    Valid global levels: "none", "minimal", "low", "medium", "high", "xhigh", "max".
    Provider/model-specific UIs may expose a narrower set via
    ``reasoning_efforts_for_model``.
    Returns None when the input is empty or unrecognized (caller uses default).
    Returns {"enabled": False} for "none".
    Returns {"enabled": True, "effort": <level>} for valid effort levels.
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    canonical = canonicalize_reasoning_effort(effort)
    if canonical in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": canonical}
    return None


def is_termux() -> bool:
    """Return True when running inside a Termux (Android) environment.

    Checks ``TERMUX_VERSION`` (set by Termux) or the Termux-specific
    ``PREFIX`` path.  Import-safe — no heavy deps.
    """
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_wsl_detected: bool | None = None


def is_wsl() -> bool:
    """Return True when running inside WSL (Windows Subsystem for Linux).

    Checks ``/proc/version`` for the ``microsoft`` marker that both WSL1
    and WSL2 inject.  Result is cached for the process lifetime.
    Import-safe — no heavy deps.
    """
    global _wsl_detected
    if _wsl_detected is not None:
        return _wsl_detected
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            _wsl_detected = "microsoft" in f.read().lower()
    except Exception:
        _wsl_detected = False
    return _wsl_detected


_container_detected: bool | None = None


def is_container() -> bool:
    """Return True when running inside a container.

    Recognizes Docker (``/.dockerenv``), Podman (``/run/.containerenv``),
    and — via ``/proc/1/cgroup`` — the docker/podman/lxc cgroup-v1 markers.

    cgroup v2 collapses ``/proc/1/cgroup`` to a single ``0::/`` line with no
    runtime marker, so containerd/CRI-O runtimes (the common case on
    Kubernetes/k3s) were previously missed. To cover those, also check:
      * ``KUBERNETES_SERVICE_HOST`` env var — set in every Kubernetes pod.
      * ``kubepods`` / ``containerd`` / ``crio`` markers in ``/proc/1/cgroup``.
      * the same markers in ``/proc/self/mountinfo`` (cgroup-v2 fallback).

    Result is cached for the process lifetime.  Import-safe — no heavy deps.

    See: NousResearch/hermes-agent#47111
    """
    global _container_detected
    if _container_detected is not None:
        return _container_detected
    if os.path.exists("/.dockerenv"):
        _container_detected = True
        return True
    if os.path.exists("/run/.containerenv"):
        _container_detected = True
        return True
    # Kubernetes always injects this into pod containers; absent on hosts.
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        _container_detected = True
        return True
    _CGROUP_MARKERS = ("docker", "podman", "/lxc/", "kubepods", "containerd", "crio")
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8") as f:
            cgroup = f.read()
            if any(marker in cgroup for marker in _CGROUP_MARKERS):
                _container_detected = True
                return True
    except OSError:
        pass
    # cgroup v2: /proc/1/cgroup is just "0::/" with no marker. The container
    # runtime still shows up in the mount table (overlay rootfs, runtime mount
    # paths), so scan mountinfo as a last resort.
    try:
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as f:
            mountinfo = f.read()
            if any(marker in mountinfo for marker in ("kubepods", "containerd", "crio")):
                _container_detected = True
                return True
    except OSError:
        pass
    _container_detected = False
    return False


# ─── Well-Known Paths ─────────────────────────────────────────────────────────


def get_config_path() -> Path:
    """Return the path to ``config.yaml`` under HERMES_HOME.

    Replaces the ``get_hermes_home() / "config.yaml"`` pattern repeated
    in 7+ files (skill_utils.py, hermes_logging.py, hermes_time.py, etc.).
    """
    return get_hermes_home() / "config.yaml"


def get_skills_dir() -> Path:
    """Return the path to the skills directory under HERMES_HOME."""
    return get_hermes_home() / "skills"



def get_env_path() -> Path:
    """Return the path to the ``.env`` file under HERMES_HOME."""
    return get_hermes_home() / ".env"


# ─── Network Preferences ─────────────────────────────────────────────────────


def apply_ipv4_preference(force: bool = False) -> None:
    """Monkey-patch ``socket.getaddrinfo`` to prefer IPv4 connections.

    On servers with broken or unreachable IPv6, Python tries AAAA records
    first and hangs for the full TCP timeout before falling back to IPv4.
    This affects httpx, requests, urllib, the OpenAI SDK — everything that
    uses ``socket.getaddrinfo``.

    When *force* is True, patches ``getaddrinfo`` so that calls with
    ``family=AF_UNSPEC`` (the default) resolve as ``AF_INET`` instead,
    skipping IPv6 entirely.  If no A record exists, falls back to the
    original unfiltered resolution so pure-IPv6 hosts still work.

    Safe to call multiple times — only patches once.
    Set ``network.force_ipv4: true`` in ``config.yaml`` to enable.
    """
    if not force:
        return

    import socket

    # Guard against double-patching
    if getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False):
        return

    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family == 0:  # AF_UNSPEC — caller didn't request a specific family
            try:
                return _original_getaddrinfo(
                    host, port, socket.AF_INET, type, proto, flags
                )
            except socket.gaierror:
                # No A record — fall back to full resolution (pure-IPv6 hosts)
                return _original_getaddrinfo(host, port, family, type, proto, flags)
        return _original_getaddrinfo(host, port, family, type, proto, flags)

    _ipv4_getaddrinfo._hermes_ipv4_patched = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _ipv4_getaddrinfo  # type: ignore[assignment]


# ─── Streaming Response Constants ────────────────────────────────────────────

# Response ID for partial stream stubs used during error recovery
PARTIAL_STREAM_STUB_ID = "partial-stream-stub"

FINISH_REASON_LENGTH = "length"


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
