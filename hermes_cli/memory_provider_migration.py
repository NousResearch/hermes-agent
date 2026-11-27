"""Move a user from a memory provider that left core onto its catalog plugin.

A bundled ``plugins/memory/<name>`` that becomes a standalone catalog plugin keeps the same provider
name, the settings it already read, its data directory and tool names, so the migration is only
"the code now lives under ``HERMES_HOME/plugins/<name>``". Two hooks call :func:`migrate_home`:

* ``hermes update`` — for every profile home that shares the venv (primary; runs where the venv was
  just rebuilt anyway).
* agent init — when the configured provider cannot be found at all, once per process (Desktop
  users update through the app and never run ``hermes update`` by hand). After a failed attempt,
  starts skip it for :data:`STARTUP_RETRY_SECONDS`, doubling with each further failure up to
  :data:`STARTUP_RETRY_MAX_SECONDS` (``hermes update`` always retries): every ``hermes chat``, cron
  run and Desktop restart otherwise paid the network round trip again and showed the same failure.

Both install the catalog entry at its reviewed pin through the normal plugin install path (kill
list, dependency constraints, enable), never a custom source, holding the lock every unattended plugin
installer holds (``default_plugins.installer_lock``) and re-checking the home after taking it, so a
provider another Hermes process placed meanwhile is neither cloned again nor reported as failed. Every
outcome — installed, refused, failed, absent from the catalog — reaches the user (terminal, Desktop,
chat platform), never only ``agent.log``: a provider that silently stays missing is lost memory.
"""

from __future__ import annotations

import enum
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_attempted: set[tuple[str, str]] = set()
_attempt_claim = threading.Lock()  # check-and-add on _attempted: parallel starts in one process claim once
# When a start last found the installer lock held, by home and provider: a later start in this process
# retries after BUSY_RETRY_SECONDS without the long wait, so a long-held lock costs one wait per process
# and at most a one-second try per interval, not a wait per gateway message.
_lock_busy: dict[tuple[str, str], tuple[float, bool, str]] = {}  # -> (when, a watched start was told, the line)
BUSY_RETRY_SECONDS = 60.0

# Providers that shipped bundled in core. Unattended consent covers only these: their users already
# accepted the deps when they picked the built-in. Any other catalog plugin named in memory.provider
# (a synced config, a cloned profile) still needs an explicit install.
_LEFT_CORE = frozenset({"hindsight", "honcho", "mem0", "supermemory", "openviking", "retaindb", "byterover", "holographic"})

# Providers that left core for the Hermes Labs repo rather than a maintained home: the one install
# that moves a home onto the plugin also says, once, that nobody supports it any more.
_LABS_NOTICES = {
    name: (f" {label} is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not "
           f"maintained going forward. Fixes from the community are welcome.")
    for name, label in (("byterover", "ByteRover"), ("retaindb", "RetainDB"))
}

# The startup retry window doubles with each consecutive failure, up to a week: a flat window had a
# home whose install can never succeed (a version gate, a root-owned plugins dir, no route to GitHub)
# fail once an hour for good, while an offline laptop still gets its plugin within the hour.
STARTUP_RETRY_SECONDS = 3600.0
STARTUP_RETRY_MAX_SECONDS = 7 * 24 * 3600.0


def _failure_stamp(home: Path, plugin: str) -> Path:
    """Written on every failed automatic install: its mtime and failure count gate the next startup attempt."""
    return home / "cache" / f"left-core-{plugin}.failed"


def _failure_count(stamp: Path) -> int:
    """Consecutive failures recorded in *stamp*; the empty stamp an older Hermes touched counts once."""
    try:
        text = stamp.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return 0
    return int(text) if text.isdigit() else 1


def _note_failure(home: Path, plugin: str) -> None:
    stamp = _failure_stamp(home, plugin)
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(_failure_count(stamp) + 1), encoding="utf-8")
    except OSError as exc:
        logger.debug("left-core retry stamp not written for %s: %s", home, exc)


def _failed_recently(home: Path, plugin: str) -> bool:
    stamp = _failure_stamp(home, plugin)
    try:
        age = time.time() - stamp.stat().st_mtime
    except OSError:
        return False
    failures = min(_failure_count(stamp), 32)
    return age < min(STARTUP_RETRY_SECONDS * 2 ** (failures - 1), STARTUP_RETRY_MAX_SECONDS)


def _note_success(home: Path, plugin: str) -> None:
    """A later failure (the plugin removed, then missed again) starts the backoff from an hour."""
    _failure_stamp(home, plugin).unlink(missing_ok=True)


def _plugin_choice(home: Path):
    """*home*'s ``memory.provider`` and plugin lists as the agent sees them, in one side-effect-free
    read: the loader's YAML layers (user config, ``${VAR}`` expansion, managed overlay; no defaults).
    The active home (an agent start) reads in the scope it has; another home (the update scan) reads its
    own files only, and a ``${VAR}`` provider there (from either file) is left to that profile's agent
    start. This scans every profile on ``hermes update``; the CLI config loader would
    set a profile up (directories, ``SOUL.md``) and the cached effective reader keeps a config backup,
    just to be read. A config that does not parse selects,
    enables and disables nothing, as in the memory loader."""
    from contextlib import nullcontext

    from agent.runtime_scope import bind_profile_scope
    from gateway.config_loader import read_yaml_layers
    from hermes_constants import get_hermes_home, hermes_home_key
    from plugins.plugin_loader import PluginChoice
    from hermes_cli.managed_scope import is_key_managed, load_managed_config
    from pm.plugins_state import read_home_selection
    # The active home (agent start) keeps the scope it already has; any other (the update scan) gets its
    # own files only: no vault or helper is asked for secrets just to learn there is nothing to migrate.
    bound = hermes_home_key(get_hermes_home()) == hermes_home_key(home)
    try:
        with nullcontext() if bound else bind_profile_scope(home, hydrate_secrets=False):
            config = read_yaml_layers(home)
        # The winning value before expansion: a managed pin, else the user's own.
        raw = "" if bound else str((((load_managed_config() if is_key_managed("memory.provider")
                                      else read_home_selection(home)) or {}).get("memory") or {}).get("provider") or "")
    except Exception:  # health: allow BLE001 -- malformed config.yaml: the loader turns nothing on from it either
        logger.debug("memory provider migration: unreadable config for %s", home, exc_info=True)
        return PluginChoice()
    memory, plugins = (config.get(key) if isinstance(config.get(key), dict) else {} for key in ("memory", "plugins"))
    selected = memory.get("provider")
    if "${" in raw:
        # A reference may resolve through a secret source this scan does not load: which provider the agent
        # uses is unknown here, so its own start (full scope) migrates it, never a guess from the scan.
        logger.info("memory provider migration: %s selects memory.provider %r; left to its agent start", home, raw)
        selected = ""
    enabled, disabled = (frozenset(v for v in plugins.get(key) if isinstance(v, str))
                         if isinstance(plugins.get(key), list) else frozenset() for key in ("enabled", "disabled"))
    return PluginChoice(selected.strip() if isinstance(selected, str) else "", enabled, disabled)


def _still_wanted(name: str, home: Path) -> bool:
    """*home* still selects *name* and has not disabled it, from one read taken now (under the
    installer lock), not when the attempt began. A catalog plugin not installed yet has one spelling in
    ``plugins.disabled``, its name, which is also the directory and manifest name the loader matches."""
    choice = _plugin_choice(home)
    return choice.selected == name and name not in choice.disabled


class _Outcome(enum.Enum):
    """What one install attempt left in the home."""
    PRESENT = "present"          # installed now, or by another Hermes process meanwhile
    FAILED = "failed"            # the install ran and failed (reported, stamped)
    BUSY = "busy"                # the installer lock was not had: nothing ran (reported, never stamped)
    NOT_WANTED = "not_wanted"    # the user changed the selection or disabled it meanwhile: nothing owed
    BACKED_OFF = "backed_off"    # an install failed while this start waited: not run again


def provider_present(name: str, home: Path) -> bool:
    """True when the provider resolves anywhere Hermes looks for *home* (bundled, that home's user
    plugins, entry point). The lookup reads the active home, so it is bound explicitly: the update
    hook walks several profile homes from one process."""
    from plugins.memory import find_provider_dir
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        return find_provider_dir(name) is not None
    finally:
        reset_hermes_home_override(token)


def catalog_source(name: str) -> Optional[str]:
    """The catalog entry that ships provider *name*, or None when the catalog has no such plugin."""
    from hermes_cli.plugin_catalog import get_live_catalog_entry
    entry = get_live_catalog_entry(name)
    return entry.name if entry is not None else None


def catalog_install_hint(name: str, *, category: Optional[str] = None) -> Optional[str]:
    """``hermes [-p <profile>] plugins install <name>`` when this checkout's catalog ships plugin
    *name* (of *category*, when given), else None.

    Recovery copy for a provider that left core and is not installed (offline, lazy installs off, a
    failed migration): doctor, ``memory status``, ``memory setup <name>`` and the CLI's unknown-command
    error point at the one command that fixes it. In-tree catalog only, never the network — these run
    offline and on error paths."""
    try:
        from hermes_cli.plugin_catalog import get_catalog_entry
        entry = get_catalog_entry(name) if name else None
    except Exception:
        return None
    if entry is None or (category is not None and entry.category != category):
        return None
    from hermes_constants import get_hermes_home
    return _install_command(name, get_hermes_home())


def _pending_provider(home: Path, *, say: Callable[[str], None]) -> Optional[str]:
    """The provider *home* needs from the catalog, or None (nothing to do, or a catalog miss already
    reported through *say*). Read-only."""
    choice = _plugin_choice(home)  # one read: the provider and the off switch agree
    name = choice.selected
    from agent.memory_provider import is_core_memory_provider
    if is_core_memory_provider(name) or provider_present(name, home):
        return None
    if name in choice.disabled:
        say(f"  ⚠ Memory provider '{name}' is turned off: plugins.disabled lists it, so Hermes did not install "
            f"its plugin and external memory is off (built-in memory still works). To use it, run "
            f"`{_install_command(name, home)}`." + _LABS_NOTICES.get(name, ""))
        return None
    if catalog_source(name) is None:
        say(f"  ⚠ Memory provider '{name}' is configured but not installed and not in the plugin catalog. "
            f"Install it with `{_profile_command('plugins install <source>', home)}` or change memory.provider.")
        return None
    return name


def _install_command(name: str, home: Path) -> str:
    return _profile_command(f"plugins install {name}", home)


def _profile_command(command: str, home: Path) -> str:
    """``hermes <command>`` aimed at *home*. ``-p`` is dropped only for the default home while no
    sticky profile is set: a bare command run from a shell targets the sticky profile, never the
    home of the agent (Desktop, gateway, ``hermes -p``) that printed it."""
    from hermes_cli.profiles import get_active_profile
    from hermes_constants import profile_name_for_home
    profile = profile_name_for_home(home)
    if profile is None or (profile == "default" and get_active_profile() == "default"):
        return f"hermes {command}"
    return f"hermes -p {profile} {command}"


def kept_off_sentence(name: str, home: Path) -> str:
    """Why an automatic install left *name* off in *home* (its ``plugins.disabled`` named it when the
    enable ran, :func:`_install_into`), and the command that turns it on."""
    enable = _install_command(name, home).replace(" plugins install ", " plugins enable ")
    return f"plugins.disabled lists '{name}', so it is installed but off: run `{enable}` to turn it on."


def _install_pending(home: Path, name: str, *, install: Callable[[str], dict],
                     say: Callable[[str], None], wait: float = 900.0, backoff: bool = False,
                     aside: Optional[Callable[[str], None]] = None) -> _Outcome:
    """Install *name* into *home* under the installer lock every unattended plugin installer holds
    (:func:`default_plugins.installer_lock`). The home is read again under the lock: another Hermes
    process may have placed the provider while this one waited (nothing reported: that one reports its
    own), the user may have switched to another provider or disabled this one (nothing owed), and with
    *backoff* (agent start) an install that failed while this one waited is not run again. *say* takes
    the install's own outcome; *aside* (default *say*) the lines for an install that did not run (a busy
    lock, the backoff), which an unattended start only logs."""
    from hermes_cli.default_plugins import installer_lock
    aside = aside or say
    try:
        with installer_lock(wait):
            if provider_present(name, home):
                logger.info("memory provider '%s' was installed for %s by another Hermes process meanwhile", name, home)
                return _Outcome.PRESENT
            if not _still_wanted(name, home):
                logger.info("memory provider '%s' is no longer selected (or is disabled) for %s; not installing it",
                            name, home)
                return _Outcome.NOT_WANTED
            if backoff and _failed_recently(home, name):
                aside(_failed_recently_line(name, home))
                return _Outcome.BACKED_OFF
            return _Outcome.PRESENT if _install_locked(home, name, install=install, say=say) else _Outcome.FAILED
    except TimeoutError:  # nothing was tried, so no failure is stamped
        aside(_busy_line(name, home))
        return _Outcome.BUSY
    except OSError as exc:  # the lock file cannot be opened or locked: report it, never end the update loop
        aside(f"  ⚠ Memory provider '{name}': the plugin installer lock could not be taken ({exc}), so its "
              f"plugin was not installed. Run `{_install_command(name, home)}`." + _LABS_NOTICES.get(name, ""))
        return _Outcome.BUSY


def _install_locked(home: Path, name: str, *, install: Callable[[str], dict],
                    say: Callable[[str], None]) -> bool:
    try:
        result = install(name)
    except Exception as exc:  # health: allow BLE001 -- network, uv, kill list: reported to the user, never raised
        result = {"ok": False, "error": str(exc)}
    if result.get("kept_disabled"):
        say(f"  ⚠ Memory provider '{name}' moved out of core — installed its plugin from the catalog, but "
            f"{kept_off_sentence(name, home)}")
        return True
    if result.get("ok"):
        _note_success(home, name)
        say(f"  ✓ Memory provider '{name}' moved out of core — installed its plugin from the catalog "
            f"(memory.provider and your stored memories are unchanged; check its settings with "
            f"`{_profile_command('memory status', home)}`)." + _LABS_NOTICES.get(name, ""))
        return True
    if not result.get("raced"):  # an enable that only lost to other selection writers waits out no backoff
        _note_failure(home, name)
    error = str(result.get("error") or "unknown error").rstrip(". ")
    say(f"  ⚠ Memory provider '{name}' moved out of core and could not be installed automatically: "
        f"{error}. Run `{_install_command(name, home)}`." + _LABS_NOTICES.get(name, ""))
    return False


def migrate_home(home: Path, *, install: Callable[[str], dict], say: Callable[[str], None] = print,
                 wait: float = 900.0) -> Optional[str]:
    """Install the configured provider's catalog plugin into *home* when the provider is gone.

    Returns the installed plugin name, or None when nothing needed doing or the install could not
    happen (already reported through *say*). Never raises: memory being down must not take the
    update or the agent down with it.
    """
    name = _pending_provider(home, say=say)
    if name is None:
        return None
    return name if _install_pending(home, name, install=install, say=say, wait=wait) is _Outcome.PRESENT else None


def _interactive() -> bool:
    return sys.stdin is not None and sys.stdout is not None and sys.stdin.isatty() and sys.stdout.isatty()


def _unattended_consent() -> bool:
    """Without a terminal (Desktop, gateway, ``hermes update`` from a script) nobody can answer the
    dependency prompt, so every provider that declares Python deps would fail to migrate. The
    configured ``memory.provider`` plus ``security.allow_lazy_installs`` (read for the active home,
    i.e. the one being migrated) is the same consent that let the bundled provider install its deps
    on demand; with a terminal the user is still asked."""
    from pm.install import lazy_installs_allowed

    return not _interactive() and lazy_installs_allowed()


def _in_home(home: Path, read: Callable[[], bool]) -> bool:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        return read()
    finally:
        reset_hermes_home_override(token)


def _home_consent(home: Path) -> bool:
    return _in_home(home, _unattended_consent)


def _update_may_install(home: Path, name: str, *, say: Callable[[str], None]) -> bool:
    """``hermes update`` installs plugin code for *home* only when its provider shipped in core, and
    without a terminal only when the home allows lazy installs (in a terminal the user ran the update
    and is asked about dependencies). The dependency-consent flag is not this gate: a plugin that
    declares no dependencies (ByteRover) is placed and enabled whatever it says, so without this a
    stale ``memory.provider`` alone would fetch and activate code."""
    from pm.install import lazy_installs_allowed

    if name not in _LEFT_CORE:
        say(f"  ⚠ Memory provider '{name}' is configured but not installed. It never shipped with Hermes, so "
            f"`hermes update` does not install it: run `{_install_command(name, home)}`.")
        return False
    if not _interactive() and not _in_home(home, lazy_installs_allowed):
        say(f"  ⚠ Memory provider '{name}' moved out of core and is not installed. security.allow_lazy_installs "
            f"is off, so `hermes update` did not fetch it: run `{_install_command(name, home)}`."
            + _LABS_NOTICES.get(name, ""))
        return False
    return True


def _install_into(home: Path, *, consent: Optional[bool] = None) -> Callable[[str], dict]:
    """The unattended install into *home*: a ``plugins.disabled`` entry naming the plugin when its enable
    runs keeps it off (``kept_disabled``), never lifted."""
    def _install(name: str) -> dict:
        from hermes_cli.plugins_cmd import dashboard_install_plugin
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        token = set_hermes_home_override(home)
        try:
            return dashboard_install_plugin("", force=False, enable=True, catalog_name=name, unattended=True,
                                            assume_deps_consent=(name in _LEFT_CORE and _unattended_consent())
                                            if consent is None else consent)
        finally:
            reset_hermes_home_override(token)
    return _install


def _home_label(home: Path) -> str:
    from hermes_constants import profile_name_for_home
    profile = profile_name_for_home(home)
    return f"profile '{profile}'" if profile else str(home)


def migrate_all_homes(*, say: Callable[[str], None] = print) -> list[str]:
    """``hermes update`` hook: every profile home sharing this venv. Returns installed plugin names.

    Each line names the profile it is about. Homes missing the same provider share one environment,
    so they share one set of dependency answers (#125794): the first install asks, the rest reuse
    its answers. Homes are grouped by provider AND by their unattended consent (each home's own
    ``security.allow_lazy_installs``), so a refusal only speaks for homes that would be refused the
    same way: when an install in a group fails, the rest of that group are named in one line
    instead of failing (or re-prompting) one by one, and the other groups still migrate. Ctrl-C
    ends the migration with a message, not a traceback into the updater.
    """
    from hermes_cli.plugins_cmd_install import shared_dependency_answers
    from pm.plugins_state import dependency_homes

    def labelled(home: Path) -> Callable[[str], None]:
        return lambda message: say(f"  [{_home_label(home)}] {message.lstrip()}")

    pending: dict[tuple[str, bool], list[Path]] = {}
    for home in dependency_homes():
        try:
            name = _pending_provider(home, say=labelled(home))
            if name and not _update_may_install(home, name, say=labelled(home)):
                continue
            consent = bool(name) and _home_consent(home)
        except Exception as exc:
            logger.debug("memory provider migration skipped for %s: %s", home, exc)
            continue
        if name:
            pending.setdefault((name, consent), []).append(home)

    installed: list[str] = []
    try:
        for (name, _consent), homes in pending.items():
            if len(homes) > 1 and _interactive():
                say(f"  Memory provider '{name}' is configured in {len(homes)} profiles "
                    f"({', '.join(_home_label(h) for h in homes)}); your answers to its dependency "
                    f"questions apply to all of them.")
            with shared_dependency_answers():
                for index, home in enumerate(homes):
                    outcome = _install_pending(home, name, install=_install_into(home), say=labelled(home))
                    if outcome is _Outcome.PRESENT:
                        installed.append(name)
                    if outcome in (_Outcome.PRESENT, _Outcome.NOT_WANTED):
                        continue
                    rest = homes[index + 1:]
                    if rest:
                        say(f"  ⚠ Memory provider '{name}' was not installed for "
                            f"{', '.join(_home_label(h) for h in rest)} either. Run "
                            + ", ".join(f"`{_install_command(name, h)}`" for h in rest) + ".")
                    break
    except KeyboardInterrupt:
        say("  ⚠ Memory provider migration cancelled. Profiles already migrated keep their plugin; "
            "run `hermes plugins install <name>` (with `-p <profile>`) for the rest.")
    return installed


def _busy_line(name: str, home: Path) -> str:
    return (f"  ⚠ Memory provider '{name}': another Hermes process is still installing plugins, so its "
            f"plugin was not installed now. Run `{_install_command(name, home)}`." + _LABS_NOTICES.get(name, ""))


def _failed_recently_line(name: str, home: Path) -> str:
    return (f"⚠ Memory provider '{name}' is not installed, so external memory is off for this session. "
            f"Its automatic install failed recently: run `{_install_command(name, home)}`.")


def _claim_attempt(key: tuple[str, str]) -> bool:
    """Take this process's one attempt for *key*; False when another start already took it."""
    with _attempt_claim:
        if key in _attempted:
            return False
        _attempted.add(key)
        return True


def recover_at_startup(name: str, *, say: Optional[Callable[[str], None]] = None) -> bool:
    """Agent-init hook for a configured provider that resolved nowhere. One attempt per process per
    home and name; honours ``security.allow_lazy_installs`` because it installs code. True when installed.

    Without *say* nobody watches this start (a cron run): the outcome of an install attempt (with the
    one-time Labs notice) is kept in the home for its next turn someone sees
    (``left_core_migration.claim_kept_notices``) instead of only being logged; repeated per-start
    lines (lazy installs off, a recent failure) are logged only, the watched start says them itself."""
    from hermes_cli.left_core_migration import defer_startup_notice
    from hermes_constants import get_hermes_home, hermes_home_key

    home = get_hermes_home()
    key = (hermes_home_key(home), name)
    # A watched start claims the attempt now. An unattended one (say None) claims it only when it is about
    # to install, so the lines it can only log (lazy installs off, a disabled provider) are left for a
    # watched start in this process to report.
    if key in _attempted or (say is not None and not _claim_attempt(key)):
        return False

    def report(message: str) -> None:
        logger.warning(message)
        if say is not None:
            try:
                say(message)
            except Exception:
                logger.debug("Memory migration notification failed", exc_info=True)

    from pm.install import lazy_installs_allowed
    if not lazy_installs_allowed():
        report(f"⚠ Memory provider '{name}' is not installed, so external memory is off for this session. "
               f"security.allow_lazy_installs is off, so Hermes did not fetch it: "
               f"run `{_install_command(name, home)}`." + _LABS_NOTICES.get(name, ""))
        return False
    # Agent init cannot answer a dependency prompt: under the CLI the prompt_toolkit input owns the
    # terminal (the question hangs the turn), elsewhere there is no terminal (the install is refused,
    # every process). So it installs with consent or not at all: a provider that shipped in core
    # carries the consent its built-in had; any other one needs the user's own install.
    if name not in _LEFT_CORE:
        if _pending_provider(home, say=report) == name:
            report(f"⚠ Memory provider '{name}' is not installed, so external memory is off for this session. "
                   f"It never shipped with Hermes, so Hermes installs it only when you ask: "
                   f"run `{_install_command(name, home)}`.")
        return False
    if _failed_recently(home, name):
        report(_failed_recently_line(name, home))
        return False
    from hermes_cli.left_core_migration import startup_lock_wait
    if _pending_provider(home, say=report) != name:
        return False
    if say is None and not _claim_attempt(key):
        return False
    busy_since, told, busy_line = _lock_busy.get(key, (None, False, ""))
    if busy_since is not None and time.monotonic() - busy_since < BUSY_RETRY_SECONDS:
        logger.info("memory provider '%s': the installer lock was unavailable %.0fs ago; retrying later", name,
                    time.monotonic() - busy_since)
        if say is not None and not told:  # only a cron run met the lock: the first watched start is told why
            report(busy_line.lstrip())
            _lock_busy[key] = (busy_since, True, busy_line)
        _attempted.discard(key)
        return False
    asides: list[str] = []

    def aside(message: str) -> None:
        asides.append(message)
        (logger.info if say is None or told else report)(message)

    # Only an install's own outcome is kept for an unattended start; lines about an install that did not run
    # are logged there, and on a retry after a busy lock (whose line was given once). A retry also skips the
    # long wait.
    outcome = _install_pending(home, name, install=_install_into(home, consent=True),
                               say=report if say is not None else defer_startup_notice,
                               wait=0.0 if busy_since is not None else startup_lock_wait(gateway_start=False),
                               backoff=True, aside=aside)
    if outcome is _Outcome.BUSY:  # nothing ran: a later start in this process (a gateway, Desktop) tries again
        # Recorded (with the real reason: held, or unusable) before the claim is released, so no start
        # slips into the long wait.
        _lock_busy[key] = (time.monotonic(), told or say is not None, asides[-1])
        _attempted.discard(key)
    else:
        _lock_busy.pop(key, None)
    return outcome is _Outcome.PRESENT
