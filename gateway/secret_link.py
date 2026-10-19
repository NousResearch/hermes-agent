"""One-time HTTPS secret entry links for Telegram (SPEC: secret entry link).

A Telegram user runs ``/secret`` (or a skill turn hits a missing key) and Hermes sends a link to
``secret_links.public_base_url``; the page posts the key, the value and the target profiles, and
the value goes straight into each profile's ``.env`` — never through Telegram, the agent, logs or
the process environment. This module holds the submit rules (profiles, key, value, conflicts,
writes, restart) shared by the routes in ``gateway/secret_link_routes.py``; records live in
``gateway/secret_link_store.py``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit

from gateway.secret_link_store import TRIGGER_COMMAND, TRIGGER_MISSING_KEY

logger = logging.getLogger(__name__)

PATH_PREFIX = "/secret-link/"
MAX_VALUE_BYTES = 8192
_FORBIDDEN_VALUE_CHARS = ("\r", "\n", "\0")
# Strong references for post-response work (confirmation, restart) until it finishes.
_FOLLOWUPS: set = set()


_TOKEN_PATH_RE = re.compile(re.escape(PATH_PREFIX) + r"([^\s\"'?#]*)")


def redact_token_paths(text: str) -> str:
    """``text`` with the token of every ``/secret-link/<token>`` path replaced by ``<token>``."""
    return _TOKEN_PATH_RE.sub(PATH_PREFIX + "<token>", text)


def redact_secret_links(text: str, original: Optional[str] = None) -> str:
    """``text`` (replied-to text) without the link token. ``text`` may be a native partial quote of
    ``original``, the replied-to message (default: ``text`` itself): a bare token the quote selected
    from a link in ``original`` goes too. Everything else in the text stays."""
    for token in set(_TOKEN_PATH_RE.findall(text if original is None else original)) - {""}:
        text = text.replace(token, "<token>")
    return redact_token_paths(text)


# ── public base URL ────────────────────────────────────────────────────────────────────────────
def _https_base(raw: object) -> Optional[str]:
    """``raw`` without a trailing slash when it is an absolute ``https://`` URL with a host."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        parts = urlsplit(text)
        _ = parts.port  # raises ValueError for a bad port
    except ValueError:
        return None
    if parts.scheme != "https" or not parts.hostname or parts.query or parts.fragment or parts.username:
        return None
    return text.rstrip("/")


def public_base_url() -> Optional[str]:
    """``secret_links.public_base_url`` from the gateway process's own config.yaml (the listener
    is a process asset, so a routed profile's config never changes it), or None."""
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import get_process_hermes_home

    cfg = load_user_config_effective(get_process_hermes_home() / "config.yaml")
    section = cfg.get("secret_links") if isinstance(cfg, dict) else None
    return _https_base(section.get("public_base_url") if isinstance(section, dict) else None)


def origin_of(url: str) -> Optional[str]:
    """Browser-serialized origin (``https://host[:port]``, default port omitted) of ``url``."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    host = parts.hostname
    host = f"[{host}]" if ":" in host else host
    default_port = {"https": 443, "http": 80}.get(parts.scheme)
    return f"{parts.scheme}://{host}" + (f":{port}" if port not in (None, default_port) else "")


# ── submit rules ───────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SubmitForm:
    """Every submitted field except the secret value (that one stays a local of the handler)."""
    nonce: str
    key: str
    profiles: tuple[str, ...]
    all_profiles: bool
    replace: bool


@dataclass
class WriteReport:
    written: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    not_tried: list[str] = field(default_factory=list)


def resolve_profiles(form: SubmitForm) -> tuple[list[str], Optional[str]]:
    """``(sorted profile names, None)`` or ``([], fixed error)``. "All profiles" is the live list
    at submit time; a named profile must pass ``validate_profile_name``, be on that list and have
    a directory whose ``.env`` stays inside it."""
    from hermes_cli.config_env_write import profile_env_path
    from hermes_cli.profiles import get_profile_dir, list_profile_names, validate_profile_name

    live = list_profile_names()
    names = list(live) if form.all_profiles else list(dict.fromkeys(form.profiles))
    if not names:
        return [], "Select at least one profile."
    for name in names:
        try:
            validate_profile_name(name)
            profile_dir = get_profile_dir(name)
        except ValueError:
            return [], "Profile not found."
        if name not in live or not profile_dir.is_dir():
            return [], "Profile not found."
        try:
            profile_env_path(profile_dir)
        except (OSError, ValueError):
            return [], f"Profile {name} cannot be written."
    return sorted(names), None


def key_error(key: str, profiles: Iterable[str]) -> Optional[str]:
    """Fixed error for a key name the ``.env`` writer would refuse for any selected profile."""
    from hermes_cli.config import env_write_refusal, validate_env_var_name_for_write

    try:
        validate_env_var_name_for_write(key)
    except ValueError:
        return "Invalid key name."
    if any(env_write_refusal(key, "set") for _name in profiles):
        return "This key cannot be changed on this host."
    return None


def value_error(value: str) -> Optional[str]:
    """Fixed error for a value the link refuses; the value is never changed to fit."""
    if not value:
        return "Enter a secret value."
    if any(char in value for char in _FORBIDDEN_VALUE_CHARS):
        return "The secret value has a line break or a NUL character. Remove it and try again."
    if len(value.encode("utf-8")) > MAX_VALUE_BYTES:
        return f"The secret value is longer than {MAX_VALUE_BYTES} bytes."
    return None


def find_conflicts(key: str, profiles: Iterable[str]) -> list[str]:
    """Selected profiles whose ``.env`` already defines ``key``."""
    from hermes_cli.config_env_write import env_file_defines_key, profile_env_path
    from hermes_cli.profiles import get_profile_dir

    return [name for name in profiles if env_file_defines_key(profile_env_path(get_profile_dir(name)), key)]


def write_profiles(key: str, value: str, profiles: list[str]) -> WriteReport:
    """Write ``profiles`` in order; stop at the first failure (no rollback of earlier writes)."""
    from hermes_cli.config_env_write import save_env_value_to_profile
    from hermes_cli.profiles import get_profile_dir

    report = WriteReport()
    for index, name in enumerate(profiles):
        try:
            save_env_value_to_profile(get_profile_dir(name), key, value)
        except Exception as exc:  # health: allow BLE001 -- SR-35: log type + profile only, never the text
            logger.warning("Secret link write failed for profile %s (%s)", name, type(exc).__name__)
            report.failed.append(name)
            report.not_tried.extend(profiles[index + 1:])
            return report
        report.written.append(name)
    return report


# ── after a use: confirmation, then restart ────────────────────────────────────────────────────
def _same_home(name: str, home: Optional[Path]) -> bool:
    from hermes_cli.profiles import get_profile_dir

    return home is not None and get_profile_dir(name).resolve() == Path(home).resolve()


def restart_targets(written: list[str], owner_home: Optional[Path], trigger: str = TRIGGER_COMMAND) -> list[str]:
    """Written profiles whose gateway runs, the owner profile last. No authoritative list of
    startup-only keys exists, so every such profile restarts (SPEC §12.2) — except the owner of a
    missing-key request: a skill reads its secrets per turn, and that turn is still running in the
    owner's gateway, waiting for this very save."""
    from hermes_cli.profiles import list_profiles

    running = {info.name for info in list_profiles(lazy_skill_count=True) if info.gateway_running}
    targets = [name for name in written if name in running]
    if trigger == TRIGGER_MISSING_KEY:
        targets = [name for name in targets if not _same_home(name, owner_home)]
    return sorted(targets, key=lambda name: _same_home(name, owner_home))


def restart_profiles(targets: list[str]) -> None:
    """Restart each target with the dashboard's restart helper, one at a time."""
    import subprocess

    from hermes_cli import web_server_gateway

    for index, name in enumerate(targets):
        result = web_server_gateway._restart_gateway_after(name, what="a secret link save", label="Secret link")
        proc = web_server_gateway._ACTION_PROCS.get("gateway-restart")
        if index < len(targets) - 1 and result.get("restart_started") and proc is not None:
            try:  # the next spawn would be refused while this one is still running
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                logger.warning("Secret link: gateway restart for profile %s still running", name)


def confirmation_text(key: str, report: WriteReport) -> str:
    if not report.failed:
        return f"Saved {key}."
    return (f"{key} was not saved to all profiles. Written: {', '.join(report.written) or 'none'}. "
            f"Failed: {', '.join(report.failed)}. Not tried: {', '.join(report.not_tried) or 'none'}.")


async def after_use(record, key: str, report: WriteReport, targets: list[str]) -> None:
    """Send the confirmation to the request chat, then restart (the owner profile last)."""
    try:
        await record.notify(confirmation_text(key, report))
    except Exception:  # health: allow BLE001 -- a failed send must not cancel the restart
        logger.warning("Secret link: confirmation send failed", exc_info=True)
    if targets:
        await asyncio.to_thread(restart_profiles, targets)


def spawn_followup(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _FOLLOWUPS.add(task)
    task.add_done_callback(_FOLLOWUPS.discard)


# ── link creation (Telegram) ───────────────────────────────────────────────────────────────────
CLEANUP_INTERVAL_SECONDS = 30.0
NOT_CONFIGURED_MESSAGE = (
    "Secure secret entry is not set up: the public HTTPS URL is not configured. Set "
    "secret_links.public_base_url in config.yaml to an https:// address that reaches the gateway listener.")
TOO_MANY_LINKS_MESSAGE = "This chat already has 5 open secret links. Use one of them or wait until they expire."
TELEGRAM_ONLY_MESSAGE = "Secret links are available on Telegram only."


def is_telegram(source) -> bool:
    from gateway.config import Platform

    return getattr(source, "platform", None) == Platform.TELEGRAM


def link_message(url: str, expires: str, suggested_key: str) -> str:
    key_line = f"Key name: {suggested_key} (you can change it on the page)\n" if suggested_key else ""
    return (f"Open this link to enter a secret. Do not paste secrets in this chat.\n{url}\n{key_line}"
            f"The link works once and expires at {expires} (10 minutes).")


def _chat_notifier(runner, source):
    """``notify(text)``: send to the request chat through the bot that answers it, under the
    owning profile's scope, with link previews off (a preview bot must not open the link)."""
    metadata = {**(runner._thread_metadata_for_source(source) or {}), "disable_link_preview": True}

    async def notify(text: str):
        async with runner._async_profile_scope_for_source(source):
            adapter = runner._delivery_adapter_for(source)
            if adapter is None:
                logger.warning("Secret link: no adapter can answer the request chat")
                return None
            return await adapter.send(str(source.chat_id), text, metadata=metadata)
    return notify


async def create_link(runner, source, *, suggested_key: str = "", on_done=None,
                      trigger: str = TRIGGER_COMMAND) -> bool:
    """Create one link for ``source``'s Telegram chat and send it there. False (after a plain
    notice) when no public HTTPS URL is configured or the chat has too many open links."""
    from datetime import UTC, datetime, timedelta

    from gateway import secret_link_store

    notify = _chat_notifier(runner, source)
    base = await asyncio.to_thread(public_base_url)
    if base is None:
        await notify(NOT_CONFIGURED_MESSAGE)
        return False
    owner_home = runner._resolve_profile_home_for_source(source)
    created = secret_link_store.STORE.create(
        chat_key=f"telegram:{source.chat_id}", owner_home=owner_home, suggested_key=suggested_key,
        notify=notify, on_done=on_done, trigger=trigger)
    if created is None:
        await notify(TOO_MANY_LINKS_MESSAGE)
        return False
    expires = datetime.now(UTC) + timedelta(seconds=secret_link_store.LINK_TTL_SECONDS)
    text = link_message(f"{base}{PATH_PREFIX}{created[0]}", expires.strftime("%H:%M UTC"), suggested_key)
    del created  # the raw token now lives only in the outgoing message
    result = await notify(text)
    return bool(getattr(result, "success", False))


# ── missing-key requests from a skill turn (trigger 2) ─────────────────────────────────────────
_TURN_REQUESTERS: dict = {}  # session_key -> TurnSecretRequests (Telegram turns only)
_TURN_LOCK = threading.Lock()


class TurnSecretRequests:
    """The per-turn bridge the skill-setup path (``tools/skills_tool_setup.py``) uses from the
    agent thread: send a link for a missing key, block until the link is used or dies, and return
    ``{"key", "success"}`` only. ``close()`` (turn end) unblocks anything still waiting."""

    def __init__(self, runner, source, session_key: str, loop, interrupted) -> None:
        self._runner, self._source, self._session_key = runner, source, session_key
        self._loop, self._interrupted = loop, interrupted
        self._waits: list = []
        self._closed = False

    def request(self, key_name: str) -> dict:
        import time

        from gateway.secret_link_store import LINK_TTL_SECONDS
        from tools.environments.base import touch_activity_if_due

        done, outcome = threading.Event(), {"success": False}

        def on_done(success: bool) -> None:
            outcome["success"] = success
            done.set()

        self._waits.append(done)
        coro = create_link(self._runner, self._source, suggested_key=key_name, on_done=on_done,
                           trigger=TRIGGER_MISSING_KEY)
        try:
            sent = asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=60)
        except Exception:  # health: allow BLE001 -- a closed loop or failed send means "not captured"
            coro.close()
            logger.warning("Secret link for a missing key could not be sent", exc_info=True)
            sent = False
        deadline = time.monotonic() + LINK_TTL_SECONDS + 5
        activity = {"last_touch": time.monotonic(), "start": time.monotonic()}
        while sent and not done.wait(1.0):
            if self._closed or self._interrupted() or time.monotonic() > deadline:
                break
            touch_activity_if_due(activity, "waiting for the secret link")
        return {"key": key_name, "success": bool(sent and done.is_set() and outcome["success"])}

    def close(self) -> None:
        self._closed = True
        with _TURN_LOCK:
            if _TURN_REQUESTERS.get(self._session_key) is self:
                del _TURN_REQUESTERS[self._session_key]
        for done in self._waits:
            done.set()


def open_turn_secret_requests(runner, source, session_key: str, loop, interrupted) -> TurnSecretRequests:
    """Register the turn's requester when the turn is a Telegram conversation (else inert)."""
    requests = TurnSecretRequests(runner, source, session_key, loop, interrupted)
    if is_telegram(source) and session_key:
        with _TURN_LOCK:
            _TURN_REQUESTERS[session_key] = requests
    return requests


def current_turn_secret_requester():
    """``request(key_name)`` for the Telegram turn running on this thread, else None."""
    from gateway.session_context import get_session_env

    session_key = get_session_env("HERMES_SESSION_KEY") or ""
    with _TURN_LOCK:
        requests = _TURN_REQUESTERS.get(session_key)
    return requests.request if requests is not None else None
