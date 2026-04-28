"""Tests for ``_setup_weixin()`` in ``hermes_cli/gateway.py``.

Locks in the iLink-bot DM-only clarification surfaced by issue #17094 — the
group-chat prompt must be preceded by a note that QR-login bots may not
receive ordinary WeChat group events, so users do not waste time debugging
``WEIXIN_GROUP_POLICY`` / ``WEIXIN_GROUP_ALLOWED_USERS`` for messages iLink
never delivers in the first place.
"""

from unittest.mock import MagicMock, patch


def _run_setup_weixin(*, group_idx: int):
    """Drive ``_setup_weixin`` to the group-chat prompt and capture I/O.

    Returns ``(saved_env, info_calls, warning_calls, success_calls)``.
    """
    saved_env: dict[str, str] = {}
    existing_env: dict[str, str] = {}

    def mock_save(name, value):
        saved_env[name] = value

    def mock_get(name):
        return existing_env.get(name, "")

    info_mock = MagicMock()
    warning_mock = MagicMock()
    success_mock = MagicMock()

    qr_credentials = {
        "account_id": "test-account",
        "token": "test-token",
        "base_url": "https://ilinkai.weixin.qq.com",
        "user_id": "test-user",
    }

    # access_idx=0 (DM pairing), then group_idx; finally home-channel yes/no.
    prompt_choice_responses = [0, group_idx]
    prompt_yes_no_responses = [True, False]  # start QR; skip home channel
    prompt_responses = [""]  # group allowlist input (only consumed if group_idx==2)

    with patch("hermes_cli.gateway.save_env_value", side_effect=mock_save), \
         patch("hermes_cli.gateway.get_env_value", side_effect=mock_get), \
         patch("hermes_cli.gateway.prompt_yes_no", side_effect=prompt_yes_no_responses), \
         patch("hermes_cli.gateway.prompt_choice", side_effect=prompt_choice_responses), \
         patch("hermes_cli.gateway.prompt", side_effect=prompt_responses), \
         patch("hermes_cli.gateway.print_info", info_mock), \
         patch("hermes_cli.gateway.print_success", success_mock), \
         patch("hermes_cli.gateway.print_warning", warning_mock), \
         patch("hermes_cli.gateway.print_error"), \
         patch("hermes_cli.gateway.color", side_effect=lambda t, c: t), \
         patch("gateway.platforms.weixin.check_weixin_requirements", return_value=True), \
         patch("gateway.platforms.weixin.qr_login", new=lambda *_a, **_k: qr_credentials), \
         patch("asyncio.run", return_value=qr_credentials):

        from hermes_cli.gateway import _setup_weixin
        _setup_weixin()

    info_calls = [str(c.args[0]) if c.args else "" for c in info_mock.call_args_list]
    warning_calls = [str(c.args[0]) if c.args else "" for c in warning_mock.call_args_list]
    success_calls = [str(c.args[0]) if c.args else "" for c in success_mock.call_args_list]
    return saved_env, info_calls, warning_calls, success_calls


class TestSetupWeixinIlinkBotClarification:
    """#17094: setup must warn that QR-login iLink bots may not deliver group events."""

    def test_disabled_path_prints_ilink_bot_dm_only_note(self):
        _, info_calls, _, _ = _run_setup_weixin(group_idx=0)
        joined = "\n".join(info_calls)
        # Three independent assertions so a future trim that drops one of the
        # three load-bearing facts (bot identity, group delivery, group@bot
        # mention semantics) still fails this test.
        assert "iLink bot" in joined, info_calls
        assert "ordinary WeChat groups" in joined or "ordinary WeChat group" in joined, info_calls
        assert "@im.bot" in joined, info_calls

    def test_open_path_still_prints_note_and_qualifies_warning(self):
        saved_env, info_calls, warning_calls, _ = _run_setup_weixin(group_idx=1)
        joined_info = "\n".join(info_calls)
        joined_warning = "\n".join(warning_calls)

        assert saved_env["WEIXIN_GROUP_POLICY"] == "open"
        assert "iLink bot" in joined_info, info_calls
        # The "All group chats enabled" warning is now qualified so the user
        # knows the policy is necessary-but-not-sufficient.
        assert "iLink" in joined_warning, warning_calls

    def test_allowlist_path_still_prints_note_and_qualifies_success(self):
        saved_env, info_calls, _, success_calls = _run_setup_weixin(group_idx=2)
        joined_info = "\n".join(info_calls)
        joined_success = "\n".join(success_calls)
        assert saved_env["WEIXIN_GROUP_POLICY"] == "allowlist"
        assert "iLink bot" in joined_info, info_calls
        # The allowlist success line must be qualified with the iLink-delivery
        # caveat so users don't assume the policy alone guarantees delivery.
        assert any(
            "Group allowlist saved" in line and "iLink" in line
            for line in success_calls
        ), success_calls
