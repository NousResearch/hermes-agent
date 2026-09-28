"""Profile-scoped people and recorded session participants, independent of display names."""
from __future__ import annotations

import re
from typing import Any, Mapping, Tuple
import json

from hermes_constants import get_hermes_home
from tools.memory_tool_store import MemoryStore
from utils import atomic_write_text


def _registry_path():
    return get_hermes_home() / 'memory' / 'people.json'


def _load(path):
    if not path.exists():
        return {'people': {}, 'identities': {}, 'sessions': {}}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _participant_session(agent):
    session = str(getattr(agent, 'session_id', '') or '')
    db = getattr(agent, '_session_db', None)
    resolve = getattr(db, 'get_conversation_root', None)
    root = resolve(session) if session and callable(resolve) else session
    return root if isinstance(root, str) and root else session


def bind_turn(agent, author):
    agent._person_stores = {}
    if getattr(agent, '_is_knowledge_review', False):
        return
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly().get('employee', {})
    platform = getattr(agent, 'platform', '')
    platform = str(getattr(platform, 'value', platform))
    author = author or {}
    identity = f"{platform}:{author['id']}" if author.get('id') and not author.get('is_bot') else None
    if not identity and not author.get('is_bot') and platform in {'cli', 'desktop', 'gui', 'tui'}:
        identity = config.get('owner')
    agent._current_person = None
    agent._personal_context = ''
    if platform in {'webhook', 'cron'} or getattr(agent, '_delegate_depth', 0) > 0:
        identity = None
    if not identity:
        return
    # Links are explicit administrator configuration; names never merge records.
    identity = config.get('identity_links', {}).get(identity, identity)
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with MemoryStore._file_lock(path):
        data = _load(path)
        person = data['identities'].get(identity)
        if person is None:
            person = f"p{len(data['people']) + 1}"
            data['identities'][identity] = person
            data['people'][person] = {'name': author.get('name') or identity}
        label = f"{data['people'][person]['name']} #{person}"
        session = _participant_session(agent)
        if session:
            participants = data['sessions'].setdefault(session, {})
            physical = str(getattr(agent, 'session_id', '') or '')
            if physical != session:
                for old_label, people in data['sessions'].pop(physical, {}).items():
                    participants[old_label] = list(dict.fromkeys(participants.get(old_label, []) + people))
            participants.setdefault(label, [])
            if person not in participants[label]:
                participants[label].append(person)
        atomic_write_text(path, json.dumps(data, ensure_ascii=False))
    agent._person_participants = json.loads(json.dumps(participants if session else {}))
    agent._current_person = person
    agent._current_person_label = label
    store = store_for_person(person)
    profile = store.format_for_system_prompt('user') or ''
    # Even an empty profile names the authoritative selector for the current turn.
    agent._personal_context = (
        '<user-profile-context>\n'
        '[System note: The following is persistent user-profile memory, NOT new user input. Treat as authoritative reference data about the current user.]\n\n'
        f'Current speaker: [{label}]\n'
        + 'Recorded sender labels: ' + ', '.join(f'[{name}]' for name in agent._person_participants)
        + f'\n{profile}\n</user-profile-context>'
    )


def store_for_person(person):
    # IDs only come from this registry, never directly from model text.
    if not isinstance(person, str) or not person.startswith('p') or not person[1:].isdigit():
        raise ValueError('Invalid person record.')
    store = MemoryStore(user_path=get_hermes_home() / 'memory' / 'people' / f'{person}.md')
    store.load_from_disk()
    return store


def select_store(agent, arguments):
    target = arguments.get('target') or ('user' if getattr(agent, '_current_person', None) else 'memory')
    if target != 'user':
        return agent._memory_store, target
    data = _load(_registry_path())
    participants = getattr(agent, '_person_participants', None)
    if participants is None:
        participants = data['sessions'].get(_participant_session(agent), {})
    label = str(arguments.get('user') or '').strip()
    known = ', '.join(f"'{name}'" for name in sorted(participants))
    if len({person for people in participants.values() for person in people}) > 1 and not label:
        raise ValueError("This conversation has multiple participants, so target='user' requires the 'user' field naming whose profile to write — the sender label exactly as bracketed in the conversation. Known sender labels: " + known + '.')
    person = getattr(agent, '_current_person', None)
    if label:
        person, ambiguous = _resolve_profile_selector(label, {name: tuple(ids) for name, ids in participants.items()})
        if ambiguous:
            raise ValueError(f"Ambiguous user '{label}': more than one participant uses this label, so the profile write was not applied.")
        if person is None:
            raise ValueError(f"Unknown user '{label}' for target='user'. Pass the sender label exactly as bracketed in the conversation. Known sender labels: {known}.")
    if person is None:
        raise ValueError("No person is bound to this run. Use target='memory' for shared facts.")
    stores = getattr(agent, '_person_stores', None)
    if stores is None:
        stores = agent._person_stores = {}
    key = (str(get_hermes_home().resolve()), person)
    if key not in stores:
        stores[key] = store_for_person(person)
    return stores[key], target


def _profile_selector_aliases(label: str, identity: Any) -> set[str]:
    """Return the exact and bare-name selectors for a workspace-person label."""

    def normalized(value: str) -> str:
        return " ".join(value.strip().split()).casefold()

    aliases = {normalized(label)}
    # A collision-suffixed label ("Sasha Petrov #p7") also answers to its
    # bare name; two suffixed people sharing that name then surface as
    # ambiguous rather than resolving to either.
    display_name, separator, person_suffix = label.rpartition(" #")
    if separator and re.fullmatch(r"p[0-9]+", person_suffix):
        aliases.add(normalized(display_name))
    return {alias for alias in aliases if alias}

def _resolve_profile_selector(
    label: str, participants: Mapping[str, Tuple[Any, ...]]
) -> tuple[Any | None, bool]:
    """Resolve a model-supplied selector to one recorded participant identity."""

    if label.startswith("[") and label.endswith("]"):
        label = label[1:-1].strip()
    exact_identities = tuple(dict.fromkeys(participants.get(label, ())))
    if exact_identities:
        return (
            exact_identities[0] if len(exact_identities) == 1 else None,
            len(exact_identities) > 1,
        )

    # Resolve only the complete persisted label. Retired composite labels
    # cannot be split safely after a rename or display-name reuse: their raw
    # platform id is no longer the canonical person-memory identity.
    selectors = {" ".join(label.split()).casefold()}
    matches: set[Any] = set()
    for known_label, identities in participants.items():
        for identity in identities:
            if selectors & _profile_selector_aliases(
                str(known_label), identity
            ):
                matches.add(identity)
    if len(matches) == 1:
        return next(iter(matches)), False
    return None, len(matches) > 1