"""``/group`` when a Group Chat's host goes offline: the host line, ``continue`` and ``keep``.

The gateway decides everything through the canonical ``groups.succession.*`` methods, which
are called by name through the same dispatch as every other ``/group`` command, and only
when ``groups.capabilities`` advertises them. Messaging shows the state and offers the one
action the gateway allows:

* ``/group N`` gains the host line and the computers that can continue the group;
* ``/group N continue`` shows what continuing on this computer involves (``prepare``). Only
  ``/group N continue confirm``, or the button where the chat has one, then calls
  ``promote`` with that summary's ``preview_id``;
* ``/group N keep <computer>`` chooses the computer that keeps a group continued on two.

Continuing and keeping are the owner's: only a private chat, whose grant names the person,
may ask for them. A shared chat sees the state and never calls ``prepare``, ``promote`` or
``keep``. Replies use plain words (host, backup copy, continue on, offline, shown separately,
unknown), never internal ones such as authority, epoch or custodian.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
import time
import unicodedata

from gateway.group_chat_slash import PAUSED, Refused, _labels, safe
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)
STATUS = 'groups.succession.status'
PREPARE = 'groups.succession.prepare'
PROMOTE = 'groups.succession.promote'
KEEP = 'groups.succession.keep'
WAIT_SECONDS = 15.0  # how long a confirmed continue waits for the move to finish before replying
POLL_SECONDS = 1.0
SUMMARY_SECONDS = 600  # a confirmation answers the summary shown at most this long ago
MAX_SUMMARIES = 256

UNAVAILABLE = 'Continuing groups on another computer isn’t available on this gateway.'
OWNER_ONLY = 'Only the group’s owner can do that.'
PRIVATE_ONLY = 'Only the group’s owner can do that, in a private chat with this Bot.'
CANCELLED = 'Cancelled. Nothing changed.'
_STEPS = {'fencing': 'Stopping work from {host}', 'catching_up': 'Catching up history',
          'reconciling': 'Checking work in progress', 'finishing': 'Finishing'}
_GENERIC = frozenset({'this computer', 'the host', 'another computer', 'the group’s owner'})
_MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


@dataclass(frozen=True)
class Summary:
    """The confirmation a chat was shown: what ``promote`` will be asked to do, and its words."""
    preview_id: str
    target_id: str
    target: str
    host: str
    group: str
    expires: float


# ---- words: names, counts and moments, as the gateway reports them ---------------------------

def _obj(value) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value) -> int:
    return value if type(value) is int and value > 0 else 0


def _this(status) -> str | None:
    install = _obj(status.get('this_install')).get('install_id')
    return install if isinstance(install, str) and install else None


def computer(entry, status, fallback: str = 'another computer') -> str:
    """A computer's display label, made inert; plain words when the gateway has no name for it."""
    entry = _obj(entry)
    name = safe(entry.get('name'), 48) if isinstance(entry.get('name'), str) else ''
    if name:
        return name
    if entry.get('install_id') and entry.get('install_id') == _this(status):
        return 'this computer'
    return fallback


def cap(text: str) -> str:
    """Plain words start a sentence capitalized; a computer's own label stays as its owner wrote it."""
    return text[:1].upper() + text[1:] if text in _GENERIC else text


def join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f'{", ".join(names[:-1])} and {names[-1]}'


def _owner(status) -> str:
    return safe(_obj(status.get('owner')).get('name'), 64) or 'the group’s owner'


def when(value) -> str | None:
    """A moment in this computer's local time, with its zone: "14:05 CEST", "1 Oct 14:05 CEST"."""
    try:
        if type(value) in (int, float) and value > 0:
            moment = datetime.fromtimestamp(value).astimezone()
        elif isinstance(value, str):
            moment = datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone()
        else:
            return None
    except (ValueError, OverflowError, OSError):
        return None
    now = datetime.now(moment.tzinfo)
    clock = f'{moment:%H:%M}'
    if moment.date() != now.date():
        year = '' if moment.year == now.year else f' {moment.year}'
        clock = f'{moment.day} {_MONTHS[moment.month - 1]}{year} {clock}'
    return f'{clock} {moment:%Z}'.strip()


def _key(text) -> str:
    """How a typed computer name is compared: case, width and spacing don't matter."""
    return ' '.join(unicodedata.normalize('NFKC', str(text or '')).casefold().split())


# ---- the gateway's state, in plain words -------------------------------------------------------

def _targets(status, action) -> list[str]:
    targets = []
    for entry in status.get('actions') or ():
        if isinstance(entry, dict) and entry.get('action') == action:
            targets.extend(t for t in entry.get('targets') or () if isinstance(t, str))
    return targets


def continues_here(status) -> bool:
    """The gateway offers this caller continuing the group on this computer."""
    return (status.get('unavailable_reason') != 'not_owner' and _this(status) is not None
            and _this(status) in _targets(status, 'continue'))


def _eligible(status, *, besides=None) -> list[str]:
    """The computers the gateway would offer to continue the group: those the owner allowed
    (``backups[].successor``) that answered lately with a verified copy, so not offline, not on a Hermes
    too old to keep one (``unsupported``) and not waiting to be reconnected (``needs_reauthorization``)."""
    return [computer(b, status) for b in status.get('backups') or ()
            if isinstance(b, dict) and b.get('successor') is True
            and b.get('readiness') not in {'offline', 'unsupported', 'needs_reauthorization'}
            and (besides is None or b.get('install_id') != besides)]


def eligibility(status) -> str:
    names = _eligible(status)
    if not names:
        return 'No computer can continue this group yet; choose one in Hermes Desktop.'
    return f'Can continue on: {", ".join(names)}.'


def reconnect(status) -> list[str]:
    """Backup computers whose permission to keep a copy of the group expired
    (``backups[].readiness`` ``needs_reauthorization``): they have to be reconnected, they aren't offline."""
    return [f'{cap(computer(b, status))} needs to be reconnected: its permission to keep a copy of this group '
            'expired.' for b in status.get('backups') or ()
            if isinstance(b, dict) and b.get('readiness') == 'needs_reauthorization']


def moving(status) -> str:
    move = _obj(status.get('moving'))
    target = computer(move.get('to'), status)
    if move.get('step') == 'waiting_for_turns':  # a move on purpose lets the replies in progress finish first
        running = _number(move.get('running'))
        return f'Moving to {target} after the replies in progress finish{f" ({running})" if running else ""}.'
    step = _STEPS.get(move.get('step'))
    host = computer(status.get('host'), status, 'the host')
    return f'Continuing on {target}… {step.format(host=host)}.' if step else f'Continuing on {target}…'


def moved(status) -> list[str]:
    target = computer(_obj(status.get('moved')).get('to'), status)
    return [f'This group moved to {target}.',
            f'This computer now keeps a backup copy. Open the group on {target} to keep chatting.']


def _conflict_hosts(status) -> tuple[list[dict], str]:
    """The two hosts of a group continued on two, the one still running it first, and a sentence
    saying which one that is (empty when the gateway doesn't say)."""
    info = _obj(status.get('conflict'))
    running = _obj(info.get('running_on')).get('install_id')
    hosts = sorted((h for h in info.get('hosts') or () if isinstance(h, dict)),
                   key=lambda h: not running or h.get('install_id') != running)
    if not running or len(hosts) != 2 or hosts[0].get('install_id') != running:
        return hosts, ''
    return hosts, f'{computer(hosts[0], status)} is running the group; {computer(hosts[1], status)} stopped.'


def conflict(status, group: str, g: str, *, may_act: bool) -> str:
    hosts, running = _conflict_hosts(status)
    names = [computer(h, status) for h in hosts]
    text = f'{group} was continued on two computers' + (f': {join(names)}.' if names else '.')
    text += f' {running}' if running else ''
    if may_act and names and _targets(status, 'keep'):
        return f'{text} Reply ' + ' or '.join(f'{g} keep {name}' for name in names) + '.'
    return f'{text} Waiting for {_owner(status)} to choose which computer keeps the group.'


def host_lines(status, *, group: str, g: str, may_act: bool) -> list[str]:
    """What ``/group N`` says about the group's host; nothing for a state it doesn't know."""
    state = status.get('state')
    host = computer(status.get('host'), status, 'the host')
    if state == 'moving':
        return [moving(status)]
    if state == 'continued_on_two':
        return [conflict(status, group, g, may_act=may_act)]
    if state == 'moved_away':
        return moved(status)
    if state == 'host_unreachable':
        since = when(_obj(status.get('host')).get('since'))
        lines = [f'Host: {host}, offline since {since}. Paused.' if since else f'Host: {host}, offline. Paused.']
    elif state == 'host_restarting':
        lines = [f'Host: {host}, restarting.']
    elif state == 'ok':
        lines = [f'Host: {host}.']
    else:
        return []
    lines.append(eligibility(status))
    lines.extend(reconnect(status))
    if state == 'host_unreachable' and may_act and continues_here(status):
        here = computer(status.get('this_install'), status, 'this computer')
        lines.append(f'Reply {g} continue to continue it on {here}.')
    elif state == 'host_unreachable' and _eligible(status) and (
            not may_act or status.get('unavailable_reason') == 'not_owner'):
        lines.append(f'Only {_owner(status)} can continue this group on another computer.')
    return lines


def _cannot_continue(status, group: str, g: str) -> str | None:
    """Why this computer can't continue the group now, or None when the gateway offers it."""
    state = status.get('state')
    host = computer(status.get('host'), status, 'the host')
    if status.get('unavailable_reason') == 'not_owner':
        return OWNER_ONLY
    if state == 'host_restarting':
        return f'{cap(host)} is restarting; the group will continue in a moment.'
    if state == 'moving':
        return f'{moving(status)} Send {g} to check.'
    if state == 'continued_on_two':
        return conflict(status, group, g, may_act=True)
    if state == 'moved_away':
        return ' '.join(moved(status))
    if _this(status) is not None and _obj(status.get('host')).get('install_id') == _this(status):
        return f'{group} is already hosted on {computer(status.get("this_install"), status, "this computer")}.'
    if state == 'ok' or status.get('unavailable_reason') == 'host_reachable':
        return f'{group} doesn’t need to move: {host} is online.'
    if state != 'host_unreachable':
        return f'{group} can’t be continued right now. Send {g} to check.'
    if continues_here(status):
        return None
    names = _eligible(status, besides=_this(status))
    if names:
        return f'This computer can’t continue {group}. It can continue on: {", ".join(names)}.'
    return (f'This computer can’t continue {group}. '
            'No computer can continue this group yet; choose one in Hermes Desktop.')


def _counted(caution) -> tuple[str, int]:
    """The computers a caution is about: their names, or how many when some are unnamed."""
    names = [safe(n, 48) for n in caution.get('names') or () if isinstance(n, str) and safe(n, 48)]
    count = max(_number(caution.get('count')), len(names))
    return (join(names) if names and count == len(names) else f'{count} computer{"" if count == 1 else "s"}'), count


def summary_text(preview, status, *, g: str, labels: dict) -> str:
    """The summary to confirm: what continuing on the target involves, then how to confirm it."""
    target = computer(preview.get('target'), status, 'this computer')
    host = computer(status.get('host'), status, 'the host')
    lines = [f'Continue this group on {target}?',
             f'{cap(target)} becomes the group’s host. The conversation, members and history stay the same.']
    # Display only: someone other than the group's owner runs the target computer.
    operator = _obj(preview.get('target')).get('operator_name')
    owner = _obj(preview.get('owner')).get('name') or _obj(status.get('owner')).get('name')
    if (isinstance(operator, str) and isinstance(owner, str) and safe(operator, 64) and owner.strip()
            and operator != owner):
        lines.append(f'{safe(operator, 64)} will manage this group from {target}.')
    bots = [safe(b.get('name'), 48) or labels.get(b.get('member_id')) or 'a Bot'
            for b in preview.get('unavailable_bots') or () if isinstance(b, dict)]
    if bots:
        one = len(bots) == 1
        lines.append(f'{len(bots)} Bot{"" if one else "s"} {"runs" if one else "run"} on {host} and '
                     f'{"stays" if one else "stay"} unavailable until the group moves back to {host}: '
                     f'{", ".join(bots)}.')
    work = _obj(preview.get('work'))
    done, elsewhere, unknown = (_number(work.get(k)) for k in ('completed', 'elsewhere', 'unknown'))
    if done or elsewhere or unknown:
        lines.append(f'Work in progress: {done} finished, {elsewhere} still running on other computers, '
                     f'{unknown} unknown. Unknown work won’t run again automatically.')
    # Messages no computer that answered holds: only the host had them (``at_risk``).
    missing = _number(_obj(preview.get('at_risk')).get('count'))
    if missing:
        lines.append(f'{cap(target)} is missing {missing} recent message{"" if missing == 1 else "s"}. '
                     f'{"It’ll" if missing == 1 else "They’ll"} appear if {host} comes back.')
    behind = _number(preview.get('behind_by'))  # held by another computer: fetched while continuing
    if behind:
        lines.append(f'{cap(target)} is catching up {behind} message{"" if behind == 1 else "s"} from another '
                     'computer.')
    lines.append(f'If {host} comes back, it rejoins as a member. Anything it did while offline is shown '
                 'separately, not mixed into the conversation.')
    cautions = []
    for caution in preview.get('cautions') or ():
        caution = _obj(caution)
        who, count = _counted(caution)
        if caution.get('code') == 'host_may_be_running':
            cautions.append(f'Only continue if {host} is really offline. If it’s still running somewhere you can’t '
                            'reach, both computers may keep working until they reconnect, and you’ll be asked to '
                            'choose one.')
        elif caution.get('code') == 'participant_not_fenced' and count:
            cautions.append(f'{who} {"runs" if count == 1 else "run"} an older Hermes and may still accept '
                            f'work from {host} if it is still running.')
        elif caution.get('code') == 'voters_unreachable' and count:
            cautions.append(f'{who} can’t be reached, so this computer can’t confirm {host} has stopped. '
                            f'Continue only if {host} is really offline.')
    if cautions:
        lines.extend(['', *cautions])
    lines.extend(['', f'Reply {g} continue confirm to proceed.'])
    return '\n'.join(lines)


def refusal(exc: RuntimeStoreError, status, *, target: str, host: str, group: str, g: str) -> str:
    """A refused ``prepare`` or ``promote``, in plain words: "Couldn't continue on {target}: …"."""
    detail = getattr(exc, 'detail', None)
    detail = detail if isinstance(detail, dict) else {}
    if exc.reason in {'not_owner', 'permission_denied'}:
        return OWNER_ONLY
    if exc.reason == 'runtime_coordination_required':
        return PAUSED
    if isinstance(detail.get('target'), dict):
        target = computer(detail['target'], status, target)
    if exc.reason == 'host_reachable':
        return f'Couldn’t continue on {target}: {host} is online again.'
    if exc.reason == 'room_authority_promised':
        other = computer(detail.get('other'), status)
        return f'Couldn’t continue on {target}: {other} is already continuing {group}.'
    if exc.reason == 'preview_stale':
        return (f'Couldn’t continue on {target}: the group changed after that summary. '
                f'Reply {g} continue to see what changed.')
    if exc.reason == 'target_not_ready':
        return f'Couldn’t continue on {target}: it isn’t ready to continue this group yet.'
    return f'Couldn’t continue on {target}. Reply {g} continue to try again.'


def work_counts(status, driver) -> tuple[int, int]:
    """Unknown and waiting work after a move: the status's ``work``, else the driver's task states."""
    work = status.get('work')
    if isinstance(work, dict):
        return _number(work.get('unknown')), _number(work.get('waiting_for_host'))
    tasks = [t for t in _obj(driver).get('tasks') or () if isinstance(t, dict)]
    return (sum(t.get('state') in {'unknown', 'indeterminate'} for t in tasks),
            sum(t.get('state') == 'waiting_for_host' for t in tasks))


# ---- the summary a chat may confirm: kept per chat and room, in this process only ------------

def _summaries(runner, now) -> dict:
    store = getattr(runner, '_group_chat_summaries', None)
    if store is None:
        store = runner._group_chat_summaries = {}
    for key in [key for key, shown in store.items() if shown.expires <= now]:
        del store[key]
    return store


def remember(runner, key, shown: Summary) -> None:
    store = _summaries(runner, time.monotonic())
    store.pop(key, None)
    while len(store) >= MAX_SUMMARIES:
        del store[next(iter(store))]  # the oldest
    store[key] = shown


def take(runner, key, preview_id: str | None = None) -> Summary | None:
    """The chat's current summary for a room, once; a button only takes the summary it came with."""
    store = _summaries(runner, time.monotonic())
    shown = store.get(key)
    if shown is None or (preview_id is not None and shown.preview_id != preview_id):
        return None
    return store.pop(key)


# ---- the commands ----------------------------------------------------------------------------

async def advertised(cmd, *methods) -> bool:
    """Whether the gateway lists these methods in ``groups.capabilities`` (asked once per command)."""
    known = getattr(cmd, '_advertised', None)
    if known is None:
        try:
            listed = _obj(await cmd._call('groups.capabilities', {})).get('methods')
        except (RuntimeStoreError, Refused):
            listed = None
        names = listed if isinstance(listed, list) else ()
        known = cmd._advertised = frozenset(m for m in names if isinstance(m, str))
    return all(method in known for method in methods)


async def host_status(cmd, room_id) -> dict | None:
    """The room's host status for ``/group N``, or None when the gateway can't say (nothing is shown)."""
    if not await advertised(cmd, STATUS):
        return None
    try:
        result = await cmd._call(STATUS, {'room_id': room_id})
    except Exception:  # the room's own view stays available without it
        logger.debug('Group Chat host status unavailable for messaging', exc_info=True)
        return None
    return result if isinstance(result, dict) else None


async def _status(cmd, ref, room_id) -> dict:
    unavailable = f'Group {ref} isn’t available right now. Send {cmd.prefix}group list to check.'
    try:
        result = await cmd._call(STATUS, {'room_id': room_id})
    except RuntimeStoreError as exc:
        raise Refused(OWNER_ONLY if exc.reason in {'not_owner', 'permission_denied'} else unavailable) from exc
    if not isinstance(result, dict):
        raise Refused(unavailable)
    return result


async def _room(cmd, ref, room_id):
    """The group's quoted name, its Bots' labels and driver status; plainer words when unreadable here."""
    try:
        state = await cmd._call('groups.state', {'room_id': room_id})
        room = state['room']
        return f'“{safe(room["name"], 72)}”', _labels(room), state.get('driver_status')
    except (RuntimeStoreError, Refused, KeyError, TypeError):
        return f'Group {ref}', {}, None


async def _allowed(cmd, *methods):
    if not await advertised(cmd, *methods):
        raise Refused(UNAVAILABLE)
    # A shared chat's grant names the chat, not the owner: it never reaches these methods.
    if cmd.chat.kind != 'private':
        raise Refused(PRIVATE_ONLY)


async def continue_command(cmd, command):
    room_id = cmd._room_id(command.ref)
    await _allowed(cmd, STATUS, PREPARE, PROMOTE)
    if command.choice == 'confirm':
        shown = take(cmd.runner, (cmd.chat.key, room_id))
        if shown is not None:
            return await _promote(cmd, command.ref, room_id, shown)
    # Nothing confirmed yet, or the summary is gone: show what continuing involves now.
    return await _offer(cmd, command.ref, room_id)


async def _offer(cmd, ref, room_id):
    g = f'{cmd.prefix}group {ref}'
    current = await _status(cmd, ref, room_id)
    group, labels, _ = await _room(cmd, ref, room_id)
    reason = _cannot_continue(current, group, g)
    if reason:
        raise Refused(reason)
    target = computer(current.get('this_install'), current, 'this computer')
    host = computer(current.get('host'), current, 'the host')
    try:
        preview = await cmd._call(PREPARE, {'room_id': room_id, 'target_install_id': _this(current)})
    except RuntimeStoreError as exc:
        raise Refused(refusal(exc, current, target=target, host=host, group=group, g=g)) from exc
    preview_id = _obj(preview).get('preview_id')
    if not isinstance(preview_id, str) or not preview_id:
        raise Refused(f'Couldn’t continue on {target}. Reply {g} continue to try again.')
    shown = Summary(preview_id, _this(current), computer(preview.get('target'), current, target), host, group,
                    time.monotonic() + SUMMARY_SECONDS)
    remember(cmd.runner, (cmd.chat.key, room_id), shown)
    text = summary_text(preview, current, g=g, labels=labels)
    return None if await _buttons(cmd, ref, room_id, shown, text) else text


async def _buttons(cmd, ref, room_id, shown: Summary, text: str) -> bool:
    """Continue and Cancel buttons where the chat's adapter has them; the typed reply always works."""
    picker = getattr(cmd.runner, '_try_send_choice_picker', None)
    session_key = getattr(cmd.runner, '_session_key_for_source', None)
    if picker is None or session_key is None:
        return False

    async def chosen(_chat_id, value):
        current = take(cmd.runner, (cmd.chat.key, room_id), shown.preview_id)
        if value != 'confirm':
            return CANCELLED
        if current is None:
            return f'That summary has expired. Reply {cmd.prefix}group {ref} continue to see a new one.'
        try:
            return await _promote(cmd, ref, room_id, current)
        except Refused as exc:
            return str(exc)
    try:
        return await picker(cmd.event, session_key(cmd.event.source), title=text, choices=[
            {'value': 'confirm', 'label': f'Continue on {shown.target}', 'is_current': False},
            {'value': 'cancel', 'label': 'Cancel', 'is_current': False}], on_choice_selected=chosen)
    except Exception:
        logger.debug('Group Chat continue buttons unavailable; the typed reply stays', exc_info=True)
        return False


async def _promote(cmd, ref, room_id, shown: Summary):
    """Ask the gateway to continue the group here, for the summary the person confirmed."""
    from gateway.session_group_controls import dispatch_group_control
    g = f'{cmd.prefix}group {ref}'
    await cmd._recheck()
    params = {'room_id': room_id, 'target_install_id': shown.target_id, 'preview_id': shown.preview_id,
              'confirm': True}
    try:
        result = await dispatch_group_control(cmd.connection, PROMOTE, params)
    except RuntimeStoreError as exc:
        raise Refused(refusal(exc, {}, target=shown.target, host=shown.host, group=shown.group, g=g)) from exc
    except Exception as exc:
        raise cmd._uncertain(PROMOTE, ref) from exc
    current = _obj(result)
    deadline = time.monotonic() + WAIT_SECONDS
    while current.get('state') == 'moving' and time.monotonic() < deadline:
        await asyncio.sleep(POLL_SECONDS)
        try:
            latest = await cmd._call(STATUS, {'room_id': room_id})
        except (RuntimeStoreError, Refused):
            break
        current = _obj(latest) or current
    state = current.get('state')
    here = _this(current)
    if state == 'ok' and here is not None and (_obj(current.get('this_install')).get('role') == 'host'
                                               or _obj(current.get('host')).get('install_id') == here):
        driver = None if isinstance(current.get('work'), dict) else (await _room(cmd, ref, room_id))[2]
        unknown, waiting = work_counts(current, driver)
        return (f'Done. {shown.group} now continues on {shown.target}. {unknown} task{"" if unknown == 1 else "s"} '
                f'unknown, {waiting} waiting for {shown.host}.')
    if state == 'moving':
        return f'{moving(current)} Send {g} to check.'
    if state == 'continued_on_two':
        return conflict(current, shown.group, g, may_act=True)
    if state in {'ok', 'host_unreachable', 'host_restarting', 'moved_away'}:
        return f'Couldn’t continue on {shown.target}. Reply {g} continue to try again.'
    return str(cmd._uncertain(PROMOTE, ref))  # no state we can read: never guess, never retry


async def keep_command(cmd, command):
    from gateway.session_group_controls import dispatch_group_control
    room_id = cmd._room_id(command.ref)
    await _allowed(cmd, STATUS, KEEP)
    g = f'{cmd.prefix}group {command.ref}'
    current = await _status(cmd, command.ref, room_id)
    group, _, _ = await _room(cmd, command.ref, room_id)
    if current.get('unavailable_reason') == 'not_owner':
        raise Refused(OWNER_ONLY)
    if current.get('state') != 'continued_on_two':
        raise Refused(f'{group} wasn’t continued on two computers, so there’s nothing to choose.')
    hosts = [h for h in _obj(current.get('conflict')).get('hosts') or ()
             if isinstance(h, dict) and isinstance(h.get('install_id'), str) and h['install_id']]
    wanted = _key(command.text)
    chosen = [h for h in hosts if wanted and wanted in {
        _key(h.get('name')), _key(safe(h.get('name'))), _key(computer(h, current))}]
    if len(chosen) > 1:
        raise Refused('Two computers have that name. Choose which one keeps the group in Hermes Desktop.')
    if not chosen:
        raise Refused(conflict(current, group, g, may_act=True))
    name = computer(chosen[0], current)
    await cmd._recheck()
    try:
        await dispatch_group_control(cmd.connection, KEEP, {'room_id': room_id, 'install_id': chosen[0]['install_id']})
    except RuntimeStoreError as exc:
        if exc.reason in {'not_owner', 'permission_denied'}:
            raise Refused(OWNER_ONLY) from exc
        raise Refused(PAUSED if exc.reason == 'runtime_coordination_required' else
                      f'Couldn’t keep {name}. Send {g} to see where things stand.') from exc
    except Exception as exc:
        raise cmd._uncertain(KEEP, command.ref) from exc
    others = [computer(h, current) for h in hosts if h is not chosen[0]]
    kept = f' Messages from {join(others)} are kept and shown separately.' if others else ''
    return f'Done. {group} now continues on {name}.{kept}'

