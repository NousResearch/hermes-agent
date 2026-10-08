"""Regression for #129827 part 4: warn about shared resources, not shared passwords."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway, profile_cmd, profiles


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    worker = root / "profiles" / "worker"
    worker.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gateway, "_SYSTEM_UNIT_DIR", tmp_path / "system-units")
    return root, worker


@pytest.mark.parametrize("case,expected", [
    ("email-tools", []),
    ("email-disabled", []),
    ("email-distinct-mailboxes", []),
    ("email-unused-token", []),
    ("email-distinct-hosts", []),
    ("email-same-inbox", ["email"]),
    ("telegram-distinct", []),
    ("telegram-shared", ["telegram"]),
    ("telegram-disabled", []),
])
def test_profile_list_warns_only_for_shared_active_resources(homes, capsys, case, expected):
    from hermes_cli.profile_channels import shared_channel_credentials

    root, worker = homes
    for index, home in enumerate(homes):
        config = "gateway:\n  multiplex_profiles: false\n"
        if case.startswith("email"):
            address = f"bot{index}@example.test" if case in {"email-distinct-mailboxes", "email-unused-token"} else "bot@example.test"
            password = f"app-password-{index}" if case == "email-same-inbox" else "shared-password"
            env = f"EMAIL_ADDRESS={address}\nEMAIL_PASSWORD={password}\nEMAIL_SMTP_HOST=smtp.example.test\n"
            if case != "email-tools":
                host = f"imap{index}.example.test" if case == "email-distinct-hosts" else "imap.example.test"
                env += f"EMAIL_IMAP_HOST={host}\n"
            if case == "email-unused-token":
                config += "platforms:\n  email:\n    token: unused-shared-token\n"
            if case in {"email-disabled", "email-tools"}:
                config += "platforms:\n  email:\n    enabled: false\n"
        else:
            token = f"bot-token-{index}" if case == "telegram-distinct" else "shared-bot-token"
            env = f"TELEGRAM_BOT_TOKEN={token}\n"
            config += "platforms:\n  telegram:\n    token: ${TELEGRAM_BOT_TOKEN}\n"
            if case == "telegram-disabled":
                config += "    enabled: false\n"
        (home / "config.yaml").write_text(config, encoding="utf-8")
        (home / ".env").write_text(env, encoding="utf-8")

    assert shared_channel_credentials(worker, root) == expected
    profile_cmd.cmd_profile(SimpleNamespace(profile_action="list"))
    output = capsys.readouterr().out
    warnings = [line for line in output.splitlines() if line.startswith("⚠")]
    assert bool(warnings) == bool(expected)
    if expected:
        assert "worker" in warnings[0] and expected[0] in warnings[0]
        assert "multiplex" not in warnings[0]
        if expected == ["email"]:
            assert "INBOX" in warnings[0]
            assert "bot can only belong" not in warnings[0]
            # The mailbox, not the app password, owns unread delivery. Exercise both real adapters
            # against one IMAP transport: RFC822 marks the message seen for the second poller.
            from contextlib import contextmanager
            from unittest.mock import patch
            from hermes_cli.gateway_migrate import _multiplex_read_mode, _profile_gateway_config
            from gateway.run import _profile_runtime_scope
            from plugins.platforms.email.adapter import EmailAdapter

            class Inbox:
                seen = False

                def uid(self, operation, *args):
                    if operation == "search":
                        assert args == (None, "UNSEEN")
                        return "OK", [b"" if self.seen else b"1"]
                    assert operation == "fetch" and args == (b"1", "(RFC822)")
                    self.seen = True
                    return "OK", [(b"1", b"From: user@example.test\r\nSubject: hello\r\n\r\nhello")]

            inbox = Inbox()

            @contextmanager
            def transport():
                yield inbox

            adapters = []
            with _multiplex_read_mode():
                for home in homes:
                    with _profile_runtime_scope(home):
                        config = _profile_gateway_config(home)
                        settings = next(v for k, v in config.platforms.items() if k.value == "email")
                        adapters.append(EmailAdapter(settings))
            assert adapters[0]._password != adapters[1]._password
            with patch.object(adapters[0], "_inbox", transport), patch.object(adapters[1], "_inbox", transport):
                assert len(adapters[0]._fetch_new_messages()) == 1
                assert adapters[1]._fetch_new_messages() == []
    assert "shared-password" not in output and "shared-bot-token" not in output
    assert "app-password-" not in output and "bot@example.test" not in output


@pytest.mark.parametrize("served", [[], ["default", "worker"]], ids=["standalone", "multiplexed"])
def test_profile_list_warning_uses_live_gateway_mode(homes, monkeypatch, capsys, served):
    from hermes_cli import gateway_multiplex_served

    root, worker = homes
    for home in homes:
        (home / "config.yaml").write_text("gateway:\n  multiplex_profiles: false\n", encoding="utf-8")
        (home / ".env").write_text("TELEGRAM_BOT_TOKEN=shared-bot-token\n", encoding="utf-8")
    # Only external liveness is simulated; read the real persisted runtime record.
    monkeypatch.setattr(gateway_multiplex_served, "live_default_gateway_pid", lambda: 99999999)
    import json
    (root / "gateway_state.json").write_text(json.dumps({"served_profiles": served}), encoding="utf-8")
    profile_cmd.cmd_profile(SimpleNamespace(profile_action="list"))
    warnings = [line for line in capsys.readouterr().out.splitlines() if line.startswith("⚠")]
    assert len(warnings) == 1
    assert ("multiplexed gateway" in warnings[0]) == bool(served)
    assert "shared-bot-token" not in warnings[0]
    assert {p.name for p in profiles.list_profiles()} == {"default", "worker"}
