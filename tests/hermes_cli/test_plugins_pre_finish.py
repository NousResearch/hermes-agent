"""``pre_finish`` uses the same continue/block-stop directive as ``pre_verify``."""

from hermes_cli.plugins_dispatch import get_pre_finish_continue_message


class TestGetPreFinishContinueMessage:
    def test_none_when_no_hooks(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kwargs: [])
        assert get_pre_finish_continue_message() is None

    def test_invokes_pre_finish_and_allows_empty_changed_paths(self, monkeypatch):
        seen = {}

        def capture(hook_name, **kwargs):
            seen["hook"] = hook_name
            seen.update(kwargs)
            return [{"action": "continue", "message": "confirm the remote write"}]

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", capture)
        assert get_pre_finish_continue_message(changed_paths=[]) == "confirm the remote write"
        assert seen["hook"] == "pre_finish"
        assert seen["changed_paths"] == []

    def test_first_valid_continue_wins(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [
                {"action": "continue", "message": "first"},
                {"action": "continue", "message": "second"},
            ],
        )
        assert get_pre_finish_continue_message() == "first"
