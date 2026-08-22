"""Focused tests for the webhook setup flow and gateway menu wiring."""

from hermes_cli import gateway as gateway_cli
from hermes_cli import setup as setup_mod


def test_setup_webhooks_reports_effective_host_and_port(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("WEBHOOK_ENABLED", "WEBHOOK_HOST", "WEBHOOK_PORT", "WEBHOOK_SECRET"):
        monkeypatch.delenv(name, raising=False)

    def answer(question, *args, **kwargs):
        if question.startswith("Webhook bind host"):
            return "hooks.example"
        if question.startswith("Webhook port"):
            return "9777"
        if question.startswith("Global HMAC secret"):
            return "test-secret"
        raise AssertionError(f"unexpected prompt: {question}")

    monkeypatch.setattr(setup_mod, "prompt", answer)
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *args, **kwargs: False)

    setup_mod._setup_webhooks()

    output = capsys.readouterr().out
    assert "hooks.example:9777/webhooks/<route-name>" in output
    assert output.count("Open config in your editor") == 1


def test_webhooks_menu_entry_dispatches_to_setup(monkeypatch):
    platforms = gateway_cli._all_platforms()
    entry = next(platform for platform in platforms if platform["key"] == "webhook")
    assert gateway_cli._builtin_setup_fn(entry["key"]) is setup_mod._setup_webhooks
    assert gateway_cli._builtin_setup_fn("webhook") is setup_mod._setup_webhooks
