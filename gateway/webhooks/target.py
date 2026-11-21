"""Canonical local webhook Bot Chat targets and conservative store-incarnation pins."""
from pathlib import Path

from persistence.core.errors import AuthorityRefusal


def is_bot_chat(target):
    return isinstance(target, str) and (target == 'bot-chat' or target.startswith('bot-chat:'))


def target_profile(target):
    """None means the effective producer, never the receiving adapter's launch profile."""
    from hermes_constants import PROFILE_ID_RE
    if target == 'bot-chat':
        return None
    if not is_bot_chat(target):
        raise ValueError('not a bot-chat target')
    profile = target[len('bot-chat:'):]
    if not PROFILE_ID_RE.fullmatch(profile):
        raise ValueError('bot-chat requires a canonical local profile name')
    return profile


def target_home(target, producer_home):
    from hermes_cli.profiles import get_profile_dir, profile_exists
    profile = target_profile(target)
    if profile is None:
        return Path(producer_home).resolve()
    path = get_profile_dir(profile)
    if not profile_exists(profile) or path.is_symlink() or not path.is_dir():
        raise ValueError('bot-chat target profile unavailable')
    return path.resolve()


def capture_target(home):
    """The assembly's read-only capture never initializes a recipient store."""
    from persistence_assembly import capture_store_incarnation, profile_scope
    return capture_store_incarnation(profile_scope(home=Path(home))).to_wire()


def refresh_target(pin):
    from persistence.authority.webhooks import StoreIncarnation
    expected = StoreIncarnation.from_wire(pin)
    current = capture_target(expected.home)
    if current['directory'] != pin['directory'] or (pin['store'] is not None and current['store'] != pin['store']):
        raise ValueError('target incarnation changed')
    return current


def require_incarnation(authority, pin):
    """Recipient-side OPEN writer check; admission repeats it in its own write."""
    if pin is None:
        return
    from gateway.session_persistence import authority_persistence
    from persistence.authority.webhooks import StoreIncarnation
    try:
        expected = StoreIncarnation.from_wire(pin)
    except (ValueError, KeyError, TypeError):
        raise AuthorityRefusal('admission_conflict') from None
    authority_persistence(authority).webhook_deliveries.require_incarnation(expected)
