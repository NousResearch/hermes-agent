"""`hermes memory status` should explain *why* a provider is unavailable.

Regression coverage for NousResearch/hermes-agent#2765: when the selected
provider reports unavailable, status lists the missing env vars and surfaces
the systemd/gateway ``.env``-inheritance gotcha that most often causes it.
"""

import hermes_cli.memory_setup as memory_setup


class _UnavailableProvider:
    def is_available(self):
        return False

    def get_config_schema(self):
        return [
            {
                "key": "api_key",
                "env_var": "HINDSIGHT_API_KEY",
                "secret": True,
                "url": "https://ui.hindsight.vectorize.io",
            }
        ]


def test_status_surfaces_env_inheritance_hint_when_unavailable(monkeypatch, capsys):
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("hindsight", "cloud", _UnavailableProvider())],
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "hindsight", "hindsight": {}}},
    )

    memory_setup.cmd_status(object())
    out = capsys.readouterr().out

    assert "not available" in out
    assert "HINDSIGHT_API_KEY" in out  # names the missing var
    assert ".env" in out               # systemd/gateway root-cause hint
