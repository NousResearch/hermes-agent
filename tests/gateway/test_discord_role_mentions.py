from types import SimpleNamespace

from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter(*, user_id=7, role_ids=()):
    adapter = object.__new__(DiscordAdapter)
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=user_id, roles=[SimpleNamespace(id=role_id) for role_id in role_ids])
    )
    return adapter


def _message(*, content="", mention_roles=(), guild=None):
    return SimpleNamespace(content=content, mentions=[], mention_roles=list(mention_roles), guild=guild)


def test_managed_role_mention_counts_as_an_explicit_bot_mention():
    adapter = _adapter(role_ids=(42,))

    assert adapter._self_is_explicitly_mentioned(
        _message(content="<@&42> hello", mention_roles=(42,))
    )


def test_unrelated_role_mention_does_not_count_as_a_bot_mention():
    adapter = _adapter(role_ids=(42,))

    assert not adapter._self_is_explicitly_mentioned(
        _message(content="<@&99> hello", mention_roles=(99,))
    )
