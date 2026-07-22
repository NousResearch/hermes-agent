"""Regression tests for the Email gateway plugin setup flow."""

from types import SimpleNamespace

from plugins.platforms.email import adapter as email_adapter


class _RegistrationContext:
    def __init__(self):
        self.kwargs = None

    def register_platform(self, **kwargs):
        self.kwargs = kwargs


def test_email_plugin_registers_complete_interactive_setup():
    ctx = _RegistrationContext()

    email_adapter.register(ctx)

    assert ctx.kwargs["setup_fn"] is email_adapter.interactive_setup
    assert ctx.kwargs["required_env"] == [
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
    ]


def test_email_connection_status_requires_complete_configuration(monkeypatch):
    import hermes_cli.gateway as gateway

    values = {"EMAIL_ADDRESS": "agent@example.com"}
    monkeypatch.setattr(gateway, "get_env_value", lambda name: values.get(name, ""))

    assert email_adapter._is_connected(object()) is False

    values.update(
        {
            "EMAIL_PASSWORD": "app-password",
            "EMAIL_IMAP_HOST": "imap.example.com",
            "EMAIL_SMTP_HOST": "smtp.example.com",
        }
    )
    assert email_adapter._is_connected(object()) is True


def test_email_connection_status_honors_platform_config_extra(monkeypatch):
    import hermes_cli.gateway as gateway

    values = {"EMAIL_PASSWORD": "app-password"}
    monkeypatch.setattr(gateway, "get_env_value", lambda name: values.get(name, ""))
    platform_config = SimpleNamespace(
        extra={
            "address": "agent@example.com",
            "imap_host": "imap.example.com",
            "smtp_host": "smtp.example.com",
        }
    )

    assert email_adapter._is_connected(platform_config) is True


def test_email_interactive_setup_collects_and_saves_gateway_configuration(monkeypatch):
    import hermes_cli.cli_output as cli_output
    import hermes_cli.config as config

    answers = {
        "Email address": "agent@gmail.com",
        "Email password / app password": "app-password",
        "IMAP host": "imap.gmail.com",
        "SMTP host": "smtp.gmail.com",
        "IMAP port": "993",
        "SMTP port": "587",
        "Allowed sender addresses (comma-separated)": " owner@example.com, , team@example.com ",
        "Home address for cron/notification delivery (optional)": "owner@example.com",
    }
    saved = {}

    monkeypatch.setattr(config, "get_env_value", lambda name: "")
    monkeypatch.setattr(config, "save_env_value", saved.__setitem__)
    monkeypatch.setattr(
        cli_output,
        "prompt",
        lambda question, default=None, password=False: answers.get(question, default or ""),
    )
    monkeypatch.setattr(cli_output, "print_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_warning", lambda *_args, **_kwargs: None)

    email_adapter.interactive_setup()

    assert saved == {
        "EMAIL_ADDRESS": "agent@gmail.com",
        "EMAIL_PASSWORD": "app-password",
        "EMAIL_IMAP_HOST": "imap.gmail.com",
        "EMAIL_SMTP_HOST": "smtp.gmail.com",
        "EMAIL_IMAP_PORT": "993",
        "EMAIL_SMTP_PORT": "587",
        "EMAIL_ALLOWED_USERS": "owner@example.com,team@example.com",
        "EMAIL_HOME_ADDRESS": "owner@example.com",
    }


def test_email_interactive_setup_disables_unconfirmed_open_access(monkeypatch):
    import hermes_cli.cli_output as cli_output
    import hermes_cli.config as config

    current = {"EMAIL_ALLOW_ALL_USERS": "true"}
    answers = {
        "Email address": "agent@example.com",
        "Email password / app password": "app-password",
        "IMAP host": "imap.example.com",
        "SMTP host": "smtp.example.com",
        "IMAP port": "993",
        "SMTP port": "587",
        "Allowed sender addresses (comma-separated)": "",
        "Home address for cron/notification delivery (optional)": "",
    }
    removed = []
    confirmations = []

    monkeypatch.setattr(config, "get_env_value", lambda name: current.get(name, ""))
    monkeypatch.setattr(config, "save_env_value", lambda *_args: None)
    monkeypatch.setattr(config, "remove_env_value", removed.append)
    monkeypatch.setattr(
        cli_output,
        "prompt",
        lambda question, default=None, password=False: answers.get(question, default or ""),
    )
    monkeypatch.setattr(
        cli_output,
        "prompt_yes_no",
        lambda question, default=True: confirmations.append((question, default)) or False,
    )
    monkeypatch.setattr(cli_output, "print_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_warning", lambda *_args, **_kwargs: None)

    email_adapter.interactive_setup()

    assert confirmations == [("Open access is enabled. Keep accepting email from any sender?", False)]
    assert removed == ["EMAIL_ALLOW_ALL_USERS"]


def test_email_interactive_setup_suggests_common_provider_hosts(monkeypatch):
    import hermes_cli.cli_output as cli_output
    import hermes_cli.config as config

    monkeypatch.setattr(config, "get_env_value", lambda _name: "")
    monkeypatch.setattr(config, "save_env_value", lambda *_args: None)
    monkeypatch.setattr(cli_output, "print_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_output, "print_warning", lambda *_args, **_kwargs: None)

    cases = {
        "agent@gmail.com": ("imap.gmail.com", "smtp.gmail.com"),
        "agent@outlook.com": ("outlook.office365.com", "smtp.office365.com"),
    }
    for email_address, expected_hosts in cases.items():
        defaults = {}

        def fake_prompt(question, default=None, password=False):
            if question == "Email address":
                return email_address
            if question == "Email password / app password":
                return "app-password"
            if question in {"IMAP host", "SMTP host"}:
                defaults[question] = default
            return default or ""

        monkeypatch.setattr(cli_output, "prompt", fake_prompt)
        email_adapter.interactive_setup()

        assert (defaults["IMAP host"], defaults["SMTP host"]) == expected_hosts
