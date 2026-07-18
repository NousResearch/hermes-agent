"""Tests for the `hermes status` Skill Curator staleness section.

The section surfaces the curator's cheap staleness count as a count plus a
command pointer (never a skill list) and must stay silent — no output, no
errors — when the curator is disabled or broken.
"""

from types import SimpleNamespace

from hermes_cli.status import show_status


def _curator_section(out: str) -> str:
    return out.split("◆ Skill Curator", 1)[1].split("◆", 1)[0]


def test_staleness_line_when_stale_skills_exist(monkeypatch, capsys):
    import agent.curator as curator

    monkeypatch.setattr(curator, "is_enabled", lambda: True)
    monkeypatch.setattr(curator, "stale_skill_count", lambda: 3)
    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 30)

    show_status(SimpleNamespace(all=False, deep=False))

    out = capsys.readouterr().out
    assert "◆ Skill Curator" in out
    assert "Staleness:    3 skills unused >30d — see hermes curator usage" in out


def test_staleness_line_singular_and_configured_window(monkeypatch, capsys):
    import agent.curator as curator

    monkeypatch.setattr(curator, "is_enabled", lambda: True)
    monkeypatch.setattr(curator, "stale_skill_count", lambda: 1)
    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 45)

    show_status(SimpleNamespace(all=False, deep=False))

    out = capsys.readouterr().out
    assert "Staleness:    1 skill unused >45d — see hermes curator usage" in out


def test_zero_stale_prints_quiet_line_without_pointer(monkeypatch, capsys):
    import agent.curator as curator

    monkeypatch.setattr(curator, "is_enabled", lambda: True)
    monkeypatch.setattr(curator, "stale_skill_count", lambda: 0)
    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 30)

    show_status(SimpleNamespace(all=False, deep=False))

    out = capsys.readouterr().out
    section = _curator_section(out)
    assert "Staleness:    no skills unused >30d" in section
    assert "hermes curator usage" not in section


def test_disabled_curator_prints_nothing_and_never_counts(monkeypatch, capsys):
    import agent.curator as curator

    calls = []
    monkeypatch.setattr(curator, "is_enabled", lambda: False)
    monkeypatch.setattr(curator, "stale_skill_count", lambda: calls.append(1) or 0)

    show_status(SimpleNamespace(all=False, deep=False))

    out = capsys.readouterr().out
    assert "◆ Skill Curator" not in out
    assert "Staleness:" not in out
    assert not calls
    # The rest of the status page still renders.
    assert "◆ Sessions" in out


def test_curator_exception_does_not_crash_show_status(monkeypatch, capsys):
    import agent.curator as curator

    def _boom():
        raise RuntimeError("usage file unreadable")

    monkeypatch.setattr(curator, "is_enabled", lambda: True)
    monkeypatch.setattr(curator, "stale_skill_count", _boom)

    show_status(SimpleNamespace(all=False, deep=False))

    out = capsys.readouterr().out
    # Section is skipped whole — no partial header — and the page completes.
    assert "◆ Skill Curator" not in out
    assert "◆ Sessions" in out
