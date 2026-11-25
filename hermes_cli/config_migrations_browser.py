"""Config step 51 -> 52 (registered in ``hermes_cli.config_migrations.MIGRATIONS``): agent-browser and the
built-in browser_* tools left Hermes, so ``browser.backend`` / ``browser.extension_control`` go and the
AGENT_BROWSER_* settings a home's .env gave agent-browser move to the ``browser.*`` keys the launcher reads."""

from __future__ import annotations

import os
import re
from typing import Any


# AGENT_BROWSER_* (read by agent-browser, which left Hermes) -> the browser.* key Hermes reads now.
_AGENT_BROWSER_ENV_KEYS = (
    ("AGENT_BROWSER_EXECUTABLE_PATH", "executable_path"),
    ("AGENT_BROWSER_PROFILE", "user_data_dir"),
    ("AGENT_BROWSER_ARGS", "chromium_args"),
    ("AGENT_BROWSER_HEADED", "headed"),
    ("AGENT_BROWSER_ENGINE", "engine"),
)


def _agent_browser_env(home) -> dict[str, str]:
    """AGENT_BROWSER_* as *home* set them: its dotenv layers, then the process env for the launch home
    only (a served profile's env is the launch profile's, not its own)."""
    from hermes_cli.env_loader import resolve_home_dotenv
    from hermes_constants import get_routing_process_hermes_home, hermes_home_key

    dotenv = resolve_home_dotenv(home)
    launch = hermes_home_key(home) == hermes_home_key(get_routing_process_hermes_home())
    found = {}
    for name, _key in _AGENT_BROWSER_ENV_KEYS:
        value = (dotenv.get(name) or (os.environ.get(name) if launch else None) or "").strip()
        if value:
            found[name] = value
    return found


def _inside_pm_store(value: str) -> bool:
    """pm (and the Docker image, from pm) exported its own Chromium as AGENT_BROWSER_EXECUTABLE_PATH:
    that is the default now, and a copied versioned store path would break at the next Chromium bump."""
    from pathlib import Path

    from pm import paths

    try:
        resolved = Path(value).expanduser().resolve()
        roots = {paths.store_root().resolve(), paths.writable_store_root().resolve()}
    except (OSError, RuntimeError):
        return False
    return any(resolved.is_relative_to(root) for root in roots)


def _browser_value_from_env(key: str, value: str) -> Any:
    """The ``browser.<key>`` value an AGENT_BROWSER_* value meant, or None when it carries nothing."""
    if key == "chromium_args":
        return [arg.strip() for arg in re.split(r"[,\n]", value) if arg.strip()] or None
    if key == "headed":
        return True if value.lower() in ("1", "true", "yes", "on") else None
    if key == "engine":
        return value.lower() if value.lower() in ("auto", "lightpanda", "chrome") else None
    if key == "executable_path" and _inside_pm_store(value):
        return None
    if key == "user_data_dir" and not os.path.isabs(os.path.expanduser(value)):
        # agent-browser inherited Hermes's cwd and resolved a relative profile there; browser.user_data_dir
        # anchors relative paths at HERMES_HOME, so pin the directory the logins are in now.
        return os.path.abspath(os.path.expanduser(value))
    return value


def _config_decided(key: str, current: Any) -> bool:
    """``browser.<key>`` already won over its AGENT_BROWSER_* var, as core resolved them: ``engine: auto``
    deferred to AGENT_BROWSER_ENGINE and ``headed`` was config OR env, so those defaults (core's `hermes
    tools` Local Browser row wrote ``engine: auto``; old saves materialised both) do not block the move."""
    if key == "engine":
        return str(current or "").strip().lower() not in ("", "auto")
    if key == "headed":
        return current is not None and str(current).strip().lower() in ("true", "1", "yes")
    return current not in (None, "", [])


# browser.backend chose Browser Use mode (browser_exec) or the built-in browser_* tools; any value but
# ``browser-use`` meant the built-in tools, on whatever browser the value named.
_BACKEND_REASONS = {
    "browser-use": "browser_exec (Browser Use mode) is the only browser now",
    "agent-browser": "agent-browser left Hermes; browser_exec is the browser now",
    "camofox": "Camofox is now the camofox plugin (`hermes plugins install camofox`)",
    "local": "browser_exec launches Hermes's packaged Chromium itself (browser.cloud_provider: local)",
    **dict.fromkeys(("browserbase", "firecrawl", "nous"),
                    "a cloud browser is chosen with browser.cloud_provider (`hermes tools` → Browser Automation)"),
}


def _backend_removed_note(backend: Any) -> str:
    value = "" if backend is None or isinstance(backend, bool) else str(backend).strip()
    if not value:
        return "removed browser.backend (unset)"
    reason = _BACKEND_REASONS.get(value.lower(), "browser_exec is the only browser now")
    return f"removed browser.backend: {value} — {reason}"


def move_agent_browser_env(home, browser: dict) -> tuple[list[str], list[str]]:
    """Move *home*'s AGENT_BROWSER_* settings into *browser* (its raw ``browser:`` section), never over a
    value that already won over them (:func:`_config_decided`). Returns the ``NAME → browser.key`` moves and
    the notes for a relative profile pinned to an absolute directory, which the user must read."""
    moved, pinned = [], []
    for name, value in _agent_browser_env(home).items():
        key = dict(_AGENT_BROWSER_ENV_KEYS)[name]
        if _config_decided(key, browser.get(key)):
            continue
        new = _browser_value_from_env(key, value)
        if new is None:
            continue
        browser[key] = new
        moved.append(f"{name} → browser.{key}")
        if key == "user_data_dir" and new != value:
            # Resolved against this process's cwd, which need not be where Hermes ran before.
            pinned.append(f"pinned browser.user_data_dir to {new}: {name}={value} was relative to the directory "
                          f"Hermes ran in; check that the browser logins are there")
    return moved, pinned


def agent_browser_env_pending() -> bool:
    """The active home's config.yaml would still take AGENT_BROWSER_* settings from its .env (the v52 move)."""
    from pathlib import Path

    from hermes_cli.config import read_raw_config
    from hermes_constants import get_hermes_home

    config = read_raw_config()
    browser = config.get("browser") if isinstance(config.get("browser"), dict) else {}
    return bool(move_agent_browser_env(Path(get_hermes_home()), dict(browser))[0])


def adopt_agent_browser_env(home, config: dict) -> list[str]:
    """*config* is about to give *home* its first ``_config_version`` outside a config step, so the v52 step
    will never move this home's AGENT_BROWSER_* settings: move them into *config* now. Once only, because
    every later write carries the stamp. Returns the moves; a pinned profile becomes a startup notice."""
    browser = config.get("browser") if isinstance(config.get("browser"), dict) else {}
    moved, pinned = move_agent_browser_env(home, browser)
    if moved:
        config["browser"] = browser
    if pinned:
        from hermes_cli.left_core_migration import defer_startup_notice
        for note in pinned:
            defer_startup_notice(f"{note[0].upper()}{note[1:]}.")
    return moved


def migrate_to_52(results: dict[str, Any], quiet: bool) -> None:
    # 51 -> 52: agent-browser and the built-in browser_* tools left Hermes; browser_exec is the only
    # browser. `browser.backend` chose between the two, so it is dropped (an `off`, the opt-out of
    # browser_exec, gets a notice saying how to turn browsing off now). The AGENT_BROWSER_* settings
    # this home's .env gave agent-browser move to the browser.* keys the launcher reads, never over
    # a value that already won over them (:func:`_config_decided`). .env itself is left alone. Camofox left core in the same release: a home
    # running it is recorded now, from its pre-v52 files, so its left-core row converts it later
    # while a newer home that installs the camofox plugin itself keeps browser_exec. The
    # browser-extension control lane dispatched to the removed tools, so its key goes too.
    from pathlib import Path

    from hermes_cli.config_migrations import _persist_migration, read_raw_config
    from hermes_cli.left_core_migration import record_core_use
    from hermes_constants import get_hermes_home

    config = read_raw_config()
    browser = config.get("browser") if isinstance(config.get("browser"), dict) else {}
    # notices: what the user must read even when nothing prints this migration (boot runs it quietly).
    added, messages, notices = [], [], []
    if "backend" in browser:
        backend = browser.pop("backend")
        # YAML 1.1 reads an unquoted `off` as False; core treated that as off too.
        if backend is False or (isinstance(backend, str) and backend.strip().lower() == "off"):
            # config_added is what `hermes update` re-surfaces on its quiet path, so the notice rides in it.
            note = ("removed browser.backend: off — the built-in browser tools are gone and browser_exec is the "
                    "browser now; to turn browsing off, disable the `browser` toolset in `hermes tools`")
            added.append(note)
            messages.append(f"  ✓ {note[0].upper()}{note[1:]}.")
            notices.append(f"{note[0].upper()}{note[1:]}.")
        else:
            added.append(_backend_removed_note(backend))
    if "extension_control" in browser:
        browser.pop("extension_control")
        added.append("removed browser.extension_control")
    moved, pinned = move_agent_browser_env(Path(get_hermes_home()), browser)
    for note in pinned:
        added.append(note)
        messages.append(f"  ✓ {note[0].upper()}{note[1:]}.")
        notices.append(f"{note[0].upper()}{note[1:]}.")
    if moved:
        added.append(f"browser: {', '.join(moved)}")
        messages.append(f"  ✓ Moved agent-browser settings from .env to config.yaml: {', '.join(moved)}")
    used = record_core_use(Path(get_hermes_home()), config, 52)
    if not added and not used:
        return
    if added:
        config["browser"] = browser
    _persist_migration(config)
    results["config_added"].extend(added)
    results.setdefault("notices", []).extend(notices)
    if not quiet:
        for message in messages:
            print(message)
