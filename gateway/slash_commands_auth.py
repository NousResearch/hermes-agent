"""Native, private Telegram access to existing credential-pool enrollment and order."""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import html
import logging
import re
import shlex
import threading
import unicodedata
from types import SimpleNamespace

from gateway.config import Platform
from gateway.slash_access import policy_for_source, policy_from_extra

logger = logging.getLogger("gateway.run")

_USAGE = (
    'Usage (private Telegram chat):\n'
    '/auth list [provider]\n'
    '/auth add openai-codex --label "Work Pro" [--priority 0]\n'
    '/auth priority <provider> <entry-id|label|index> <priority>\n'
    '/auth cancel\n\n'
    'Priority 0 is first under fill_first; omitting --priority appends a new account. '
    'Never send API keys or tokens in chat.'
)
_PRIVATE_ONLY = "Account management is available only in your private Telegram chat with this bot."
_DENIED = "Only a gateway admin can manage inference accounts."


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's normal error echoes unknown arguments, which may contain a pasted key.
        raise ValueError(_USAGE)


def _label(value: str) -> str:
    value = value.strip()
    if (not value or len(value) > 60
            or any(unicodedata.category(char).startswith("C") for char in value)
            or re.match(r"^(sk-|gh[pousr]_|github_pat_|eyJ)", value)):
        raise argparse.ArgumentTypeError("Use a non-empty display name of at most 60 characters, not a key.")
    return value


def _parse(raw: str):
    parser = _Parser(prog="/auth", add_help=False, allow_abbrev=False)
    sub = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    listing = sub.add_parser("list", add_help=False, allow_abbrev=False)
    listing.add_argument("provider", nargs="?", default="openai-codex")
    add = sub.add_parser("add", add_help=False, allow_abbrev=False)
    add.add_argument("provider", choices=["openai-codex"])
    add.add_argument("--label", required=True, type=_label)
    add.add_argument("--priority", type=int)
    priority = sub.add_parser("priority", add_help=False, allow_abbrev=False)
    priority.add_argument("provider")
    priority.add_argument("target")
    priority.add_argument("priority", type=int)
    sub.add_parser("cancel", add_help=False)
    args = parser.parse_args(shlex.split(raw))
    provider = getattr(args, "provider", "")
    if provider and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}", provider):
        raise ValueError(_USAGE)
    return args


def _display_label(value: str) -> str:
    try:
        return html.escape(_label(value))
    except argparse.ArgumentTypeError:
        return "Account (select by entry ID)"


def _pool_listing(provider: str) -> str:
    from agent.credential_pool import get_pool_strategy, load_pool
    from hermes_cli.auth_commands import _normalize_provider

    provider = _normalize_provider(provider)
    entries = load_pool(provider).entries()
    strategy = get_pool_strategy(provider)
    lines = [f"{provider} account order — strategy: {strategy}"]
    for position, entry in enumerate(entries, 1):
        entry_id = entry.id if re.fullmatch(r"[a-zA-Z0-9_-]{1,16}", entry.id) else "use label/index"
        health = entry.last_status if entry.last_status in {"exhausted", "dead"} else "ready"
        lines.append(f"{position}. {_display_label(entry.label)} · id={entry_id} · priority={entry.priority} · {health}")
    if not entries:
        lines.append("No configured accounts for this provider.")
    lines.append("Priority orders fill_first; other strategies may select differently. Existing turns keep their account.")
    return "\n".join(lines)


def _move_account(args) -> str:
    from agent.credential_pool import load_pool
    from hermes_cli.auth_commands import _normalize_provider

    provider = _normalize_provider(args.provider)
    pool = load_pool(provider)
    _, entry, error = pool.resolve_target(args.target)
    if entry is None:
        return "Account not found or label is ambiguous. Use the entry ID from /auth list."
    moved = pool.move_entry(entry.id, args.priority)
    if moved is None:
        return "Account changed during selection. Run /auth list again."
    return (f"Set {_display_label(moved.label)} to priority {moved.priority}. "
            "Account health and rotation strategy were preserved.\n" + _pool_listing(provider))


def _enroll(args, verification, cancelled) -> str:
    from agent.credential_pool import AUTH_TYPE_OAUTH, load_pool
    from hermes_cli.auth_commands import _add_credential

    if cancelled.is_set():
        return "Sign-in cancelled."
    options = SimpleNamespace(label=args.label, on_verification=verification, cancel_event=cancelled)
    pool = load_pool(args.provider)
    entry = _add_credential(options, args.provider, pool, AUTH_TYPE_OAUTH)
    # Enrollment and placement are separate native operations. Never call a saved grant a failed login.
    try:
        if args.priority is not None:
            entry = load_pool(args.provider).move_entry(entry.id, args.priority)
            if entry is None:
                raise ValueError("Account no longer exists")
    except Exception:
        return "Account was saved, but its order could not be set. Use /auth list and /auth priority; do not sign in again."
    return (f"Added {_display_label(entry.label)} at priority {entry.priority}.\n"
            "This sets account order, not a model/provider override. Existing sessions may keep their account; "
            "use /new when you are ready to start a fresh session.\n" + _pool_listing(args.provider))


def _scoped(home, action):
    from gateway.run import _profile_runtime_scope

    with _profile_runtime_scope(home):
        return action()


class GatewayAuthCommandsMixin:
    def _auth_admission(self, source):
        if (source.platform != Platform.TELEGRAM or source.chat_type not in {"dm", "private"}
                or not source.chat_id or not source.user_id):
            return _PRIVATE_ONLY
        if not policy_for_source(self.config, source).is_admin(source.user_id):
            return _DENIED
        adapter = self._delivery_adapter_for(source)
        if adapter is None:
            return "The receiving Telegram bot is unavailable; no account operation was started."
        extra = getattr(getattr(adapter, "config", None), "extra", {})
        if isinstance(extra, dict) and not policy_from_extra(extra, "dm").is_admin(source.user_id):
            return _DENIED
        return None

    async def _handle_auth_command(self, event) -> str:
        source = event.source
        denied = self._auth_admission(source)
        if denied:
            return denied
        raw = event.get_command_args().strip()
        if not raw:
            return _USAGE
        try:
            args = _parse(raw)
        except (ValueError, argparse.ArgumentError):
            return _USAGE
        home = self._resolve_profile_home_for_source(source)
        key = (str(home), str(source.chat_id), str(source.user_id))
        pending = getattr(self, "_auth_enrollment", None)
        if args.action == "cancel":
            if pending is None:
                return "No account sign-in is pending."
            if pending[0] != key:
                return "That sign-in belongs to another private session."
            pending[1].set()
            return "Cancelling your account sign-in."
        if args.action == "add":
            if pending is not None:
                return "An account sign-in is already pending. Finish it or use /auth cancel in its original chat."
            cancelled = threading.Event()
            self._auth_enrollment = (key, cancelled)
            self._retain_background_task(asyncio.create_task(
                self._complete_auth_enrollment(source, home, args, cancelled)))
            return "OpenAI account sign-in started. The verification link and code will arrive in this private chat."
        actions = {"list": functools.partial(_pool_listing, args.provider),
                   "priority": functools.partial(_move_account, args)}
        try:
            return await asyncio.to_thread(_scoped, home, actions[args.action])
        except (Exception, SystemExit) as exc:
            logger.warning("Native account operation failed (%s)", type(exc).__name__)
            return "Account operation failed. No credential details were returned. Run /auth list to check the saved state."

    async def _auth_send(self, source, message):
        denied = self._auth_admission(source)
        if denied:
            raise RuntimeError("Private account delivery is no longer authorized")
        adapter = self._delivery_adapter_for(source)
        result = await adapter.send(str(source.chat_id), message,
                                    metadata=self._thread_metadata_for_source(source))
        if not result or not getattr(result, "success", False):
            raise RuntimeError("Private account delivery failed")

    async def _complete_auth_enrollment(self, source, home, args, cancelled):
        loop = asyncio.get_running_loop()
        # OAuth can wait for minutes: do not occupy the gateway's shared inference/shutdown executor.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes-auth")

        def verification(url, code):
            if cancelled.is_set():
                raise RuntimeError("Sign-in cancelled")
            future = asyncio.run_coroutine_threadsafe(self._auth_send(
                source, f"OpenAI account verification\n{url}\nCode: {code}\n"
                "Sign in with the intended account; never send your password or tokens here."), loop)
            try:
                future.result(timeout=15)
            except BaseException:
                future.cancel()
                raise

        try:
            result = await loop.run_in_executor(executor, _scoped, home,
                functools.partial(_enroll, args, verification, cancelled))
            await self._auth_send(source, result)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except (Exception, SystemExit) as exc:
            logger.warning("Native account sign-in failed (%s)", type(exc).__name__)
            message = "Sign-in cancelled." if cancelled.is_set() else (
                "Sign-in did not complete. Run /auth list to check the saved accounts before retrying.")
            try:
                await self._auth_send(source, message)
            except Exception:
                logger.warning("Could not deliver native account sign-in result")
        finally:
            cancelled.set()
            executor.shutdown(wait=False, cancel_futures=True)
            pending = getattr(self, "_auth_enrollment", None)
            if pending is not None and pending[1] is cancelled:
                self._auth_enrollment = None
