"""Pin/unpin messaging on unmanaged skills (#92993).

`hermes curator pin <name>` on an unmanaged skill (curation-eligible but no
`created_by` marker — the pre-marker population `list-unmanaged` shows) used
to print "will bypass auto-transitions" and exit 0. But `curated_report()`
only walks marker-carrying skills, so auto-transitions never consider an
unmanaged skill at all: the pin is recorded yet inert, and the message
claimed an effect that does not exist. The fix keeps the write (the flag
becomes meaningful after `hermes curator adopt`) and makes the message say
what actually happened.
"""

from __future__ import annotations

from types import SimpleNamespace


def _ns(skill: str) -> SimpleNamespace:
    return SimpleNamespace(skill=skill)


def _stub(monkeypatch, *, managed: bool):
    """Point the CLI at stubbed skill_usage surfaces.

    ``managed`` drives ``is_curator_managed`` — the policy flag the new
    branch reads. ``is_agent_created`` stays True so the existing bundled/
    hub refusal guard passes and the code reaches the managed check.
    """
    import tools.skill_usage as skill_usage

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(skill_usage, "is_agent_created", lambda name: True)
    monkeypatch.setattr(skill_usage, "is_curator_managed", lambda name: managed)
    monkeypatch.setattr(
        skill_usage,
        "set_pinned",
        lambda name, pinned: calls.append((name, pinned)),
    )
    return calls


def test_pin_unmanaged_records_flag_and_prints_adopt_hint(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    calls = _stub(monkeypatch, managed=False)

    rc = curator_cli._cmd_pin(_ns("legacy-skill"))

    # The write still happens — the flag becomes meaningful after adopt.
    assert calls == [("legacy-skill", True)]
    assert rc == 0
    out = capsys.readouterr().out
    assert "unmanaged" in out
    assert "hermes curator adopt legacy-skill" in out
    # The old lie must be gone: the pin does NOT bypass anything here.
    assert "will bypass auto-transitions" not in out


def test_pin_managed_keeps_bypass_message(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    calls = _stub(monkeypatch, managed=True)

    rc = curator_cli._cmd_pin(_ns("agent-skill"))

    assert calls == [("agent-skill", True)]
    assert rc == 0
    out = capsys.readouterr().out
    assert "will bypass auto-transitions" in out
    assert "unmanaged" not in out


def test_unpin_unmanaged_says_it_was_never_managed(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    calls = _stub(monkeypatch, managed=False)

    rc = curator_cli._cmd_unpin(_ns("legacy-skill"))

    assert calls == [("legacy-skill", False)]
    assert rc == 0
    out = capsys.readouterr().out
    assert "unmanaged" in out
    assert "never under auto-transitions" in out


def test_pin_still_refuses_bundled_skills(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    calls = _stub(monkeypatch, managed=True)
    # _stub leaves is_agent_created True; override AFTER so the refusal
    # guard fires before the managed check.
    monkeypatch.setattr(skill_usage, "is_agent_created", lambda name: False)

    rc = curator_cli._cmd_pin(_ns("bundled-skill"))

    assert rc == 1
    assert calls == []
    assert "cannot pin" in capsys.readouterr().out
