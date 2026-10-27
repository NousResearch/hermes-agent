"""Who may write the owner's memory files from a group chat."""

from gateway.channel_memory import channel_memory_binding
from gateway.config import Platform
from gateway.session import SessionSource


class _Runner:
    """Admits DMs only from the owner, as a TELEGRAM_ALLOWED_USERS=owner install does."""

    def __init__(self):
        self.checked = []

    def _under_authorization_profile(self, source, check):
        return check()

    def _principal_authorized(self, source, *, allow_adapter_delegation):
        self.checked.append((source.chat_type, source.chat_id))
        return source.chat_type == "dm" and source.user_id == "owner"


def _group(user_id, **extra):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id=user_id, **extra)


def test_only_senders_who_could_dm_the_agent_write_owner_memory_from_a_group():
    runner = _Runner()
    stranger = channel_memory_binding(runner, _group("stranger"), {})
    owner = channel_memory_binding(runner, _group("owner"), {})

    assert stranger.scope == owner.scope == "telegram/-1001"
    assert stranger.owner_writes is False
    assert owner.owner_writes is True
    assert runner.checked == [("dm", "stranger"), ("dm", "owner")]

    # Opting owners out of group writes, a DM, and channel memory switched off.
    opted_out = {"memory": {"channel_memory": {"owner_writes_from_groups": False}}}
    assert channel_memory_binding(runner, _group("owner"), opted_out).owner_writes is False
    dm = SessionSource(platform=Platform.TELEGRAM, chat_id="owner", chat_type="dm", user_id="owner")
    assert channel_memory_binding(runner, dm, {}) is None
    disabled = {"memory": {"channel_memory": {"enabled": False}}}
    assert channel_memory_binding(runner, _group("stranger"), disabled) is None
