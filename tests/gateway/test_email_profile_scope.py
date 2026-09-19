import asyncio

from gateway.config import PlatformConfig
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from plugins.platforms.email.adapter import EmailAdapter


def test_check_inbox_executor_keeps_profile_context(tmp_path, monkeypatch):
    """The blocking IMAP fetch must run in the profile that owns the adapter."""
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "profiles" / "secondary"
    launch_home.mkdir(parents=True)
    served_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    observed_homes = []
    token = set_hermes_home_override(served_home)
    try:
        adapter = EmailAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "address": "bot@example.com",
                    "imap_host": "imap.example.com",
                    "smtp_host": "smtp.example.com",
                },
            )
        )

        def _fake_fetch_new_messages():
            observed_homes.append(get_hermes_home())
            return []

        adapter._fetch_new_messages = _fake_fetch_new_messages
        asyncio.run(adapter._check_inbox())
    finally:
        reset_hermes_home_override(token)

    assert observed_homes == [served_home]
