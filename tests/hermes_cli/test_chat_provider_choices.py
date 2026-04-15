import hermes_cli.main as main_mod
from hermes_cli.models import CANONICAL_PROVIDERS


def test_chat_provider_choices_track_canonical_registry():
    assert main_mod.CLI_PROVIDER_CHOICES == [
        "auto",
        *(provider.slug for provider in CANONICAL_PROVIDERS),
    ]


def test_chat_accepts_ai_gateway_provider(monkeypatch):
    seen = {}

    def fake_cmd_chat(args):
        seen["provider"] = args.provider
        seen["query"] = args.query

    monkeypatch.setattr(main_mod, "cmd_chat", fake_cmd_chat)
    monkeypatch.setattr(
        "sys.argv",
        ["hermes", "chat", "--provider", "ai-gateway", "-q", "ping"],
    )

    main_mod.main()

    assert seen == {"provider": "ai-gateway", "query": "ping"}
