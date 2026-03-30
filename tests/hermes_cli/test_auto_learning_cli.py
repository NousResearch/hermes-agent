import sys

from tools.auto_learning_store import AutoLearningStore



def test_cli_autolearning_enable_routes_to_handler(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_auto_learning(args):
        captured["action"] = args.auto_learning_action

    monkeypatch.setattr(main_mod, "cmd_auto_learning", fake_cmd_auto_learning)
    monkeypatch.setattr(sys, "argv", ["hermes", "autolearning", "enable"])

    main_mod.main()

    assert captured == {"action": "enable"}



def test_cli_autolearning_list_routes_status_filter(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_auto_learning(args):
        captured["action"] = args.auto_learning_action
        captured["status"] = args.status

    monkeypatch.setattr(main_mod, "cmd_auto_learning", fake_cmd_auto_learning)
    monkeypatch.setattr(sys, "argv", ["hermes", "autolearning", "list", "--status", "candidate"])

    main_mod.main()

    assert captured == {"action": "list", "status": "candidate"}



def test_auto_learning_command_enable_updates_config(monkeypatch):
    from hermes_cli.auto_learning import auto_learning_command

    calls = []

    monkeypatch.setattr("hermes_cli.auto_learning.set_config_value", lambda key, value: calls.append((key, value)))

    class Args:
        auto_learning_action = "enable"

    auto_learning_command(Args())

    assert calls == [("auto_learning.enabled", "true")]



def test_auto_learning_command_disable_updates_config(monkeypatch):
    from hermes_cli.auto_learning import auto_learning_command

    calls = []

    monkeypatch.setattr("hermes_cli.auto_learning.set_config_value", lambda key, value: calls.append((key, value)))

    class Args:
        auto_learning_action = "disable"

    auto_learning_command(Args())

    assert calls == [("auto_learning.enabled", "false")]



def test_auto_learning_command_promote_and_reject_update_store(tmp_path, monkeypatch):
    from hermes_cli.auto_learning import auto_learning_command

    store = AutoLearningStore(path=tmp_path / "candidates.jsonl", max_entries=10)
    entry = store.add_candidate(
        category="memory",
        summary="User prefers concise responses",
        confidence=0.8,
        evidence={"source": "test"},
    )

    monkeypatch.setattr("hermes_cli.auto_learning._load_store", lambda: store)
    monkeypatch.setattr("hermes_cli.auto_learning.load_config", lambda: {"auto_learning": {"enabled": True}})

    class PromoteArgs:
        auto_learning_action = "promote"
        id = entry["id"]

    auto_learning_command(PromoteArgs())
    assert store.list_candidates(status="promoted")[0]["id"] == entry["id"]

    second = store.add_candidate(
        category="memory",
        summary="Another candidate",
        confidence=0.7,
        evidence={"source": "test"},
    )

    class RejectArgs:
        auto_learning_action = "reject"
        id = second["id"]

    auto_learning_command(RejectArgs())
    assert store.list_candidates(status="rejected")[0]["id"] == second["id"]
