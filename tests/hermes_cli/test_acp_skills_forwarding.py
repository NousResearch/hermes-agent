from __future__ import annotations

import sys


def test_hermes_acp_command_forwards_global_skills(monkeypatch):
    from hermes_cli import main as main_mod
    from acp_adapter import entry as entry_mod

    calls = {}

    def fake_acp_main(argv=None, **kwargs):
        calls["argv"] = list(argv or [])
        calls.update(kwargs)

    monkeypatch.setattr(entry_mod, "main", fake_acp_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "--skills", "alpha,beta", "-s", "gamma", "acp", "--check"],
    )

    main_mod.main()

    assert calls["argv"] == ["--check"]
    assert calls["skills"] == ["alpha,beta", "gamma"]
