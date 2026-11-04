"""Regression tests for the gateway setup platform menu (#43598).

A plain Enter used to confirm the last item ("Done"), print "Skipped (keeping current)"
and exit the wizard without configuring anything — the "run setup, setup says run setup"
loop. The menu now starts on the first platform while a cancel still maps back to Done.
"""

from hermes_cli import gateway_setup_wizard as wizard


PLATFORMS = [
    {"emoji": "💬", "label": "Mattermost"},
    {"emoji": "💬", "label": "Weixin / WeChat"},
]


class _FakeGateway:
    def __init__(self, choices):
        self._choices = iter(choices)
        self.prompts = []
        self.configured = []

    def print_header(self, *args, **kwargs):
        pass

    def _all_platforms(self):
        return list(PLATFORMS)

    def prompt_choice(self, question, menu_items, default, **kwargs):
        self.prompts.append((default, kwargs.get("cancel_returns"), len(menu_items)))
        return next(self._choices)

    def _configure_platform(self, platform):
        self.configured.append(platform)


def _run_loop(monkeypatch, choices):
    fake = _FakeGateway(choices)
    monkeypatch.setattr(wizard, "_gw", lambda: fake)
    monkeypatch.setattr(wizard, "_platform_status", lambda platform: "not configured")
    wizard._wizard_platform_loop()
    return fake


def test_platform_menu_defaults_to_first_platform_and_cancel_maps_to_done(monkeypatch):
    fake = _run_loop(monkeypatch, [len(PLATFORMS)])

    assert fake.prompts == [(0, len(PLATFORMS), len(PLATFORMS) + 1)]
    assert fake.configured == []


def test_plain_enter_configures_the_first_platform(monkeypatch):
    fake = _run_loop(monkeypatch, [0, len(PLATFORMS)])

    assert [p["label"] for p in fake.configured] == ["Mattermost"]
    assert len(fake.prompts) == 2
