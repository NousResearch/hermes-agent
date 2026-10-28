"""Regression: async-delegation wakes were refused with eternal ``admission_conflict``
whenever the wake's reconstructed source differed from the committed human source in ANY
field of the full source dict.

Proven incident: NAMED Telegram forum topics. The persisted routing origin of a
delegation does not carry ``chat_topic``, so the wake source could never byte-match the
committed human source of a named topic; unnamed topics (``chat_topic`` absent on BOTH
sides) delivered normally — same code, same chat, opposite outcomes decided solely by
whether the forum topic had a name. (Exactly one field drifted; user/display fields match
via the persisted origin and stay in the identity — user drift must keep refusing.)

``_route_identity`` now compares the source fields normalized exactly the way
``build_session_key`` normalizes them, so the wake is admitted iff both sides map to the
same session route — and still refuses genuine route/user drift:

- user binding (participant + ``user_name``) drift must refuse
  (``group_sessions_per_user`` isolation; upstream parametrized contract from
  ``871d05520a``);
- Discord auto-thread twins must match (channel-initiating source with
  ``prospective_thread_id`` vs in-thread follow-up with ``thread_id``);
- participant identity follows ``_canonical_participant`` (``user_id_alt or user_id``,
  WhatsApp canonicalization); profile ``None``/``'default'`` normalize; scope only
  matters for Slack.
"""
from dataclasses import replace

from gateway.session import Platform, SessionSource
from gateway.session_automation import _route_identity


def _topic_source(chat_topic):
    """A live source in the incident shape: Telegram group forum topic 9952."""
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id='-1004308125450', chat_type='group',
        thread_id='9952', chat_name='Hermes VPS Team', chat_topic=chat_topic,
        user_id='58509012', user_name='Gold', is_bot=False,
    )


def test_named_topic_wake_matches_committed_source():
    """The live incident: the wake's origin source (chat_topic lost in routing
    persistence) vs the committed human source (named topic). Routing identity must be
    equal so the wake is admitted."""
    origin = _topic_source(None)                         # routing origin: chat_topic lost
    committed = _topic_source('virgo-investor-portal')   # last human message: named topic
    assert _route_identity(origin) == _route_identity(committed)


def test_discord_auto_thread_twin_matches():
    """The Discord twin of the incident: a process started by the channel-initiating
    message (chat_type='channel', thread_id=None, prospective_thread_id=T) vs a human
    follow-up inside the auto-thread (chat_type='thread', thread_id=T).
    build_session_key maps both to ONE session; route identity must agree."""
    initiating = SessionSource(
        platform=Platform.DISCORD, chat_id='999', chat_type='channel',
        prospective_thread_id='42', user_id='u1',
    )
    follow_up = SessionSource(
        platform=Platform.DISCORD, chat_id='999', chat_type='thread',
        thread_id='42', user_id='u1',
    )
    assert _route_identity(initiating) == _route_identity(follow_up)


def test_user_drift_still_refused():
    """Upstream contract (871d05520a, parametrized drift tests): a wake whose source
    carries a different user must NOT match — ``group_sessions_per_user`` isolation
    depends on the user binding staying inside the identity."""
    source = _topic_source('virgo-investor-portal')
    assert _route_identity(replace(source, user_id='OTHER')) != _route_identity(source)
    assert _route_identity(replace(source, user_name='NotGold')) != _route_identity(source)


def test_participant_identity_is_canonical():
    """user_id_alt/user_id are the same participant in different slots (Signal
    re-register, WhatsApp JID/LID): key parity requires identity parity."""
    base = SessionSource(platform=Platform.SIGNAL, chat_id='c1', chat_type='dm',
                         user_id='uuid-A')
    alt = replace(base, user_id=None, user_id_alt='uuid-A')
    assert _route_identity(base) == _route_identity(alt)
    assert _route_identity(replace(base, user_id='uuid-B')) != _route_identity(base)


def test_profile_none_and_default_normalize():
    """Key parity: ``None`` and ``'default'`` share the ``agent:main`` namespace, so a
    profile-slot drift between them must not refuse a wake."""
    unnamed = replace(_topic_source('virgo-investor-portal'), profile=None)
    default = replace(unnamed, profile='default')
    assert _route_identity(unnamed) == _route_identity(default)
    multiplex = replace(unnamed, profile='socio')
    assert _route_identity(multiplex) != _route_identity(unnamed)


def test_scope_only_discriminates_slack():
    """Only Slack scopes session keys (session.py): a Discord scope drift must not refuse
    what the key admits; a Slack scope drift must."""
    discord_a = SessionSource(platform=Platform.DISCORD, chat_id='1', chat_type='dm',
                              scope_id='guild-a', user_id='u1')
    discord_b = replace(discord_a, scope_id='guild-b')
    assert _route_identity(discord_a) == _route_identity(discord_b)
    slack_a = SessionSource(platform=Platform.SLACK, chat_id='1', chat_type='channel',
                            scope_id='team-a', user_id='u1')
    assert _route_identity(slack_a) != _route_identity(replace(slack_a, scope_id='team-b'))


def test_whatsapp_dm_identifier_drift_matches():
    """The WhatsApp twin: the bridge can surface one human as a LID or a phone JID,
    with device suffixes or a + prefix, WITHIN one conversation (whatsapp_identity.py).
    build_session_key canonicalizes the DM chat_id; route identity must agree."""
    def _wa(chat_id):
        return SessionSource(platform=Platform.WHATSAPP, chat_id=chat_id, chat_type='dm',
                             user_name='Gold')
    canonical = _wa('6012@s.whatsapp.net')
    assert _route_identity(_wa('6012:47@s.whatsapp.net')) == _route_identity(canonical)
    assert _route_identity(_wa('+6012')) == _route_identity(canonical)
    assert _route_identity(_wa('6012@lid.whatsapp.net')) == _route_identity(canonical)


def test_slack_empty_scope_equals_missing():
    """A falsy Slack scope is no scope (the key drops falsy scope_id): '' must not
    differ from None."""
    empty = SessionSource(platform=Platform.SLACK, chat_id='1', chat_type='channel',
                          scope_id='', user_id='u1')
    missing = SessionSource(platform=Platform.SLACK, chat_id='1', chat_type='channel',
                            user_id='u1')
    assert _route_identity(empty) == _route_identity(missing)


def test_route_discrimination_is_preserved():
    """The identity keeps its teeth: different chats, threads and platforms are still
    different routes."""
    base = _topic_source('virgo-investor-portal')
    assert _route_identity(replace(base, thread_id='342')) != _route_identity(base)
    assert _route_identity(replace(base, chat_id='-999')) != _route_identity(base)
    assert _route_identity(replace(base, chat_type='dm')) != _route_identity(base)
    assert _route_identity(replace(base, platform=Platform.DISCORD)) != _route_identity(base)


def test_message_id_stays_out_of_route_identity():
    """The wake clears ``message_id`` (it is not a reply to the stale trigger); identity
    must ignore it (#52694)."""
    source = _topic_source('virgo-investor-portal')
    assert _route_identity(source) == _route_identity(replace(source, message_id='18768'))


def test_duck_source_missing_optional_fields_still_compares():
    """A duck-typed source missing the optional attrs must not raise ``AttributeError``
    mid-admission: missing fields degrade to None and still compare deterministically."""
    class DuckSource:
        platform = Platform.TELEGRAM
        chat_id = '-1004308125450'
        chat_type = 'group'
        thread_id = '9952'
        user_id = None
        user_id_alt = None
        user_name = None

    identity = _route_identity(DuckSource())
    assert identity['chat_id'] == '-1004308125450'
    assert identity['scope_id'] is None
    assert identity['profile'] is None
    assert identity['participant'] is None
