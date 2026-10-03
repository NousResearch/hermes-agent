"""``/group`` when a Group Chat's host goes offline: the host line, ``continue`` and ``keep``.

The gateway decides everything through the canonical ``groups.succession.*`` methods, which
are called by name through the same dispatch as every other ``/group`` command, and only
when ``groups.capabilities`` advertises them. Messaging shows the state and offers the one
action the gateway allows:

* ``/group N`` gains the host line, the computers that can continue the group and whether
  it moves by itself (automatic takeover), or why its host paused it to stay safe;
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
AUTOMATIC = 'groups.custody.automatic'
WAIT_SECONDS = 15.0  # how long a confirmed continue waits for the move to finish before replying
POLL_SECONDS = 1.0
SUMMARY_SECONDS = 600  # a confirmation answers the summary shown at most this long ago
MAX_SUMMARIES = 256

UNAVAILABLE = 'Continuing groups on another computer isn’t available on this gateway.'
OWNER_ONLY = 'Only the group’s owner can do that.'
PRIVATE_ONLY = 'Only the group’s owner can do that, in a private chat with this Bot.'
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


def _moment(value) -> datetime | None:
    try:
        if type(value) in (int, float) and value > 0:
            return datetime.fromtimestamp(value).astimezone()
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone()
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _clock(moment: datetime) -> str:
    now = datetime.now(moment.tzinfo)
    if moment.date() == now.date():
        return f'{moment:%H:%M}'
    year = '' if moment.year == now.year else f' {moment.year}'
    return f'{moment.day} {_MONTHS[moment.month - 1]}{year} {moment:%H:%M}'


def when(value) -> str | None:
    """A moment in this computer's local time, with its zone: "14:05 CEST", "1 Oct 14:05 CEST"."""
    moment = _moment(value)
    return None if moment is None else f'{_clock(moment)} {moment:%Z}'.strip()


def span(start, end) -> str | None:
    """Two moments: "14:05–14:20 CEST" within a day, else each in full."""
    first, last = _moment(start), _moment(end)
    if first is None or last is None:
        return None
    if first.date() == last.date() and first.tzname() == last.tzname():
        return f'{_clock(first)}–{last:%H:%M} {last:%Z}'.strip()
    return f'{when(start)}–{when(end)}'


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


def _names(entries, status) -> list[str]:
    """Computer names given as labels or as ``{install_id, name}`` entries."""
    names = [safe(e, 48) if isinstance(e, str) else computer(e, status) if isinstance(e, dict) else ''
             for e in entries or ()]
    return [name for name in names if name]


def readiness(status) -> str | None:
    """Whether the group moves by itself when a computer goes offline (``automatic``), in one line."""
    automatic = _obj(status.get('automatic'))
    # The owner's change waits until the other computers have taken it on (``pending``, requested).
    if automatic.get('enabled') is True and automatic.get('pending') is False:
        return 'Moving by itself: turning off… (waiting for the other computers)'
    if automatic.get('enabled') is False and automatic.get('pending') is True:
        return 'Moving by itself: turning on… (waiting for the other computers)'
    state = automatic.get('state')
    if state == 'ready':
        # Careful (two computers): the standby waits out about three minutes of silence first.
        later = ' after about 3 minutes' if automatic.get('mode') == 'careful' else ''
        return (f'Keeps running if a computer goes offline: ready '
                f'({computer(automatic.get("standby"), status)} takes over{later}).')
    if state == 'not_ready':
        offline = _names(automatic.get('offline'), status)
        return f'Not automatic right now: {join(offline)} offline.' if offline else 'Not automatic right now.'
    if state == 'unavailable':
        needed = _number(automatic.get('needed'))
        more = 'one more always-on computer' if needed <= 1 else f'{needed} more always-on computers'
        return f'Not automatic yet: add {more} in Hermes Desktop.'
    if state == 'off':
        return 'Moves only when you choose.'
    return None


def _unreachable(status) -> tuple[str, str]:
    """For a host paused to stay safe: the computers it can't reach, and when it resumes."""
    names = _names(_obj(status.get('paused')).get('waiting_for'), status)
    if not names:
        return 'the other computers', 'one of them is back'
    return join(names), (f'{names[0]} is back' if len(names) == 1 else 'one of them is back')


def _paused(status) -> tuple[str, str, str]:
    """Why a host paused its group to stay safe: what follows "{host}" in a notice, the line that
    explains it in ``/group N``, and when it resumes. All empty for a reason this version doesn't
    know, which shows the pause alone."""
    host = cap(computer(status.get('host'), status, 'the host'))
    reason = _obj(status.get('paused')).get('reason')
    if reason == 'lost_majority':
        who, back = _unreachable(status)
        return (f'can’t reach {who}', f'{host} can’t reach {who}, so it can’t be sure another computer hasn’t '
                f'taken over. It resumes as soon as {back}.', f'It resumes as soon as {back}.')
    if reason == 'no_lease_layer':
        cause = 'can’t take part in automatic moves right now. Its connection to the other computers isn’t ready'
        return cause, f'{host} {cause}.', ''
    return '', '', ''


def paused_notice(status, group: str) -> str:
    """The proactive notice, and the reply to continue, while the host has paused the group."""
    cause = _paused(status)[0]
    host = computer(status.get('host'), status, 'the host')
    return f'{group} is paused to stay safe: {host} {cause}.' if cause else f'{group} is paused to stay safe.'


def moving(status) -> str:
    move = _obj(status.get('moving'))
    target = computer(move.get('to'), status)
    if move.get('step') == 'waiting_for_turns':  # a move on purpose lets the replies in progress finish first
        running = _number(move.get('running'))
        return f'Moving to {target} after the replies in progress finish{f" ({running})" if running else ""}.'
    step = _STEPS.get(move.get('step'))
    host = computer(status.get('host'), status, 'the host')
    text = f'{cap(host)} went offline. Moving to {target}…' if move.get('reason') == 'automatic' \
        else f'Continuing on {target}…'
    return f'{text} {step.format(host=host)}.' if step else text


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


def ran_on_two(status, group: str) -> str:
    """The notice after a careful move, once both computers have run the group."""
    hosts, running = _conflict_hosts(status)
    names = [computer(h, status) for h in hosts]
    when_ = span(_obj(status.get('conflict')).get('start'), _obj(status.get('conflict')).get('end'))
    where = f'both {join(names)}' if len(names) == 2 else join(names) if names else 'two computers'
    return (f'{group} ran on {where} while they couldn’t reach each other{f" ({when_})" if when_ else ""}. '
            f'{f"{running} " if running else ""}Choose which one to keep. The other’s messages are kept '
            'separately.')


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
    if state == 'paused':
        explained = _paused(status)[1]
        return [f'Host: {host}, paused to stay safe.', *([explained] if explained else [])]
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
    if readiness(status):
        lines.append(readiness(status))
    lines.extend(back_again(status))
    if state == 'host_unreachable' and may_act and continues_here(status):
        here = computer(status.get('this_install'), status, 'this computer')
        lines.append(f'Reply {g} continue to continue it on {here}.')
    elif state == 'host_unreachable' and status.get('unavailable_reason') == 'takeover_waiting':
        lines.append(deciding(status))
    elif state == 'host_unreachable' and _eligible(status) and (
            not may_act or status.get('unavailable_reason') == 'not_owner'):
        lines.append(f'Only {_owner(status)} can continue this group on another computer.')
    return lines


def back_again(status) -> list[str]:
    """Bots left on the computer the group moved away from, now reachable again: moving back (in
    Desktop or the CLI) would bring them back."""
    left: dict[str, tuple[str, list[str]]] = {}
    for bot in status.get('unavailable_bots') or ():
        on = _obj(_obj(bot).get('on'))
        if on.get('reachable') is True and isinstance(on.get('install_id'), str):
            name = safe(bot.get('name'), 48) or safe(bot.get('member_id'), 48) or 'a Bot'
            left.setdefault(on['install_id'], (computer(on, status), []))[1].append(name)
    return [f'{join(names)} can take part again if the group moves back to {computer_name} '
            '(in Hermes Desktop or `hermes groups move`).' for computer_name, names in left.values()]


def deciding(status) -> str:
    """While the other computers decide by themselves which one takes over (``takeover_waiting``)."""
    host = cap(computer(status.get('host'), status, 'the host'))
    return f'{host} went offline. The other computers are deciding which one takes over; this can take a few minutes.'


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
    if state == 'paused':
        resumes = _paused(status)[2]
        return f'{paused_notice(status, group)} {resumes}' if resumes else paused_notice(status, group)
    if _this(status) is not None and _obj(status.get('host')).get('install_id') == _this(status):
        return f'{group} is already hosted on {computer(status.get("this_install"), status, "this computer")}.'
    if state == 'ok' or status.get('unavailable_reason') == 'host_reachable':
        return f'{group} doesn’t need to move: {host} is online.'
    if state != 'host_unreachable':
        return f'{group} can’t be continued right now. Send {g} to check.'
    if continues_here(status):
        return None
    if status.get('unavailable_reason') == 'takeover_waiting':
        return deciding(status)
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


def take(runner, key) -> Summary | None:
    """The chat's current summary for a room, once."""
    return _summaries(runner, time.monotonic()).pop(key, None)


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
        # A summary shown with buttons: typing confirm is the same as its Continue button.
        from gateway.group_chat_actions import confirm_typed
        typed = await confirm_typed(cmd.runner, cmd.authority, cmd.chat, room_id, 'continue')
        if typed is not None:
            return typed
    # Nothing confirmed yet, or the summary is gone: show what continuing involves now.
    return await _offer(cmd, command.ref, room_id)


async def prepare_summary(cmd, ref, room_id) -> tuple[Summary, str]:
    """What continuing on this computer involves (``prepare``), as the summary to confirm."""
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
    return shown, summary_text(preview, current, g=g, labels=labels)


async def _offer(cmd, ref, room_id):
    shown, text = await prepare_summary(cmd, ref, room_id)
    # Where the chat has buttons: Continue and Cancel that still work later; typing works everywhere.
    from gateway.group_chat_access import chat_target
    target = chat_target(cmd.runner, cmd.grant)
    if target is None or getattr(type(target[0]), 'send_group_actions', None) is None:
        remember(cmd.runner, (cmd.chat.key, room_id), shown)
    else:
        from gateway.group_chat_actions import offer
        preview = {'preview_id': shown.preview_id, 'target_id': shown.target_id, 'target': shown.target,
                   'host': shown.host, 'group': shown.group}
        if await offer(cmd.runner, cmd.authority, cmd.grant, room_id=room_id, group=shown.group, kind='continue',
                       data={'here': shown.target, 'preview': preview, 'command': True}, text=text.rsplit('\n\n', 1)[0],
                       view='cont', confirm=text.rsplit('\n\n', 1)[0]):
            return None
        remember(cmd.runner, (cmd.chat.key, room_id), shown)
    return text


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


def go_back_prompt(status, group: str, g: str) -> str:
    """The confirmation before going back to the computer a careful move left."""
    moved_in = _obj(status.get('moved_in'))
    back = computer(moved_in.get('from'), status)
    here = computer(status.get('this_install'), status, 'this computer')
    since = when(moved_in.get('at'))
    return '\n'.join([
        f'Go back to {back}? {cap(here)} pauses now, and the group continues on {back} as soon as it’s '
        f'reachable. Messages sent on {here} since {since or "the move"} are kept separately.',
        f'Reply {g} keep {back} confirm to go back.'])


def go_back_target(status) -> dict | None:
    """The computer a careful move came from, while the gateway still offers going back to it."""
    back = _obj(_obj(status.get('moved_in')).get('from'))
    return back if back.get('install_id') and back['install_id'] in _targets(status, 'keep') else None


async def keep_on(connection, room_id, install_id, *, g: str, failed: str) -> None:
    """``keep`` one computer, through the chat's own principal; refusals in plain words, never retried."""
    from gateway.session_group_controls import dispatch_group_control
    try:
        await dispatch_group_control(connection, KEEP, {'room_id': room_id, 'install_id': install_id})
    except RuntimeStoreError as exc:
        if exc.reason in {'not_owner', 'permission_denied'}:
            raise Refused(OWNER_ONLY) from exc
        raise Refused(PAUSED if exc.reason == 'runtime_coordination_required' else
                      f'{failed} Send {g} to see where things stand.') from exc
    except Exception as exc:
        logger.warning('Group Chat %s from messaging ended without a confirmed outcome', KEEP)
        raise Refused(f'Hermes couldn’t confirm whether that worked. Send {g} before trying again.') from exc


async def ask_first(connection, room_id, *, group: str, g: str) -> str:
    """Turn automatic moves off for the group: it asks the owner next time."""
    from gateway.session_group_controls import dispatch_group_control
    try:
        result = await dispatch_group_control(connection, AUTOMATIC, {'room_id': room_id, 'enabled': False})
    except RuntimeStoreError as exc:
        if exc.reason in {'not_owner', 'permission_denied'}:
            raise Refused(OWNER_ONLY) from exc
        raise Refused(PAUSED if exc.reason == 'runtime_coordination_required' else
                      f'Couldn’t change that for {group}. Send {g} to see where things stand.') from exc
    except Exception as exc:
        logger.warning('Group Chat %s from messaging ended without a confirmed outcome', AUTOMATIC)
        raise Refused(f'Hermes couldn’t confirm whether that worked. Send {g} before trying again.') from exc
    if _obj(result).get('pending') is True:  # asked for, not yet taken on by the other computers
        return 'Turning off… (waiting for the other computers)'
    return f'Done. {group} will ask you before moving.'


async def keep_command(cmd, command):
    """Choose the computer that keeps a group continued on two, or go back after a careful move."""
    room_id = cmd._room_id(command.ref)
    await _allowed(cmd, STATUS, KEEP)
    g = f'{cmd.prefix}group {command.ref}'
    current = await _status(cmd, command.ref, room_id)
    group, _, _ = await _room(cmd, command.ref, room_id)
    if current.get('unavailable_reason') == 'not_owner':
        raise Refused(OWNER_ONLY)
    wanted = _key(command.text)
    if current.get('state') != 'continued_on_two':
        back = go_back_target(current)
        if back is None:
            raise Refused(f'{group} wasn’t continued on two computers, so there’s nothing to choose.')
        name = computer(back, current)
        if command.choice != 'confirm' or wanted not in {_key(back.get('name')), _key(safe(back.get('name'))),
                                                         _key(name)}:
            return go_back_prompt(current, group, g)
        await cmd._recheck()
        await keep_on(cmd.connection, room_id, back['install_id'], g=g, failed=f'Couldn’t go back to {name}.')
        here = computer(current.get('this_install'), current, 'this computer')
        return f'Done. {cap(here)} paused {group}; it continues on {name} as soon as it’s reachable.'
    hosts = [h for h in _obj(current.get('conflict')).get('hosts') or ()
             if isinstance(h, dict) and isinstance(h.get('install_id'), str) and h['install_id']]
    chosen = [h for h in hosts if wanted and wanted in {
        _key(h.get('name')), _key(safe(h.get('name'))), _key(computer(h, current))}]
    if len(chosen) > 1:
        raise Refused('Two computers have that name. Choose which one keeps the group in Hermes Desktop.')
    if not chosen:
        raise Refused(conflict(current, group, g, may_act=True))
    name = computer(chosen[0], current)
    await cmd._recheck()
    await keep_on(cmd.connection, room_id, chosen[0]['install_id'], g=g, failed=f'Couldn’t keep {name}.')
    others = [computer(h, current) for h in hosts if h is not chosen[0]]
    kept = f' Messages from {join(others)} are kept and shown separately.' if others else ''
    return f'Done. {group} now continues on {name}.{kept}'


async def ask_command(cmd, command):
    """``/group N ask first``: the group asks the owner before it moves again."""
    room_id = cmd._room_id(command.ref)
    await _allowed(cmd, AUTOMATIC)
    group, _, _ = await _room(cmd, command.ref, room_id)
    await cmd._recheck()
    return await ask_first(cmd.connection, room_id, group=group, g=f'{cmd.prefix}group {command.ref}')


# ---- the notice's numbers ------------------------------------------------------------------------

async def continue_refs(runner, room_id) -> list[tuple]:
    """``[(adapter, chat_id, metadata, n)]``: the room owner's main channel among their private chats
    here (the home channel when it is one of them, else all of them), each with the number
    ``/group n continue`` reaches the room by there (given now if it had none)."""
    from gateway.group_chat_access import chat_target, ensure_ref, grants
    from gateway.group_chat_slash import connection_for
    from gateway.session_authorities import all_authorities
    from gateway.session_group_controls import dispatch_group_control
    found = []
    for authority in all_authorities(runner):
        if getattr(authority, 'hosted_room_service', None) is None:
            continue
        with authority.db._read_ctx() as conn:
            private = [grant for grant in grants(conn) if grant['kind'] == 'private']
        if private:  # each owner's main channel: the home channel when it is one of their chats
            from gateway.group_chat_notices import homes_for, main_chats
            homes, owners = homes_for(runner, authority), {}
            for grant in private:
                owners.setdefault(grant['owner'], []).append(grant)
            private = [grant for owned in owners.values() for grant in main_chats(runner, owned, homes)]
        for grant in private:
            connection = connection_for(authority, grant)
            try:
                listed = _obj(await dispatch_group_control(connection, 'groups.capabilities', {})).get('methods')
                if not isinstance(listed, list) or STATUS not in listed:
                    break  # this profile's gateway can't continue groups: none of its chats can
                current = await dispatch_group_control(connection, STATUS, {'room_id': room_id})
                target = chat_target(runner, grant)
                if not isinstance(current, dict) or current.get('unavailable_reason') == 'not_owner' or target is None:
                    continue  # not the owner here, or that Bot isn't connected right now
                n = await asyncio.to_thread(ensure_ref, authority, grant, room_id)
            except Exception:
                logger.debug('A Group Chat continue reference was skipped', exc_info=True)
                continue
            found.append((target[0], grant['chat_id'], target[1], n))
    return found
