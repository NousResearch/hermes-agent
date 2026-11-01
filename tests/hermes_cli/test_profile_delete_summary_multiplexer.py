"""The delete summary distinguishes a multiplexed profile from one with its own gateway (#144703).

A profile served by the running multiplexer has no `gateway.pid`, so `_stop_gateway_process` is a
no-op for it — the summary still promised "Gateway is running — it will be stopped".
"""

from hermes_cli import profiles


def _summary(monkeypatch, profile_dir, gw_running: bool) -> str:
    monkeypatch.setattr(profiles, "_read_config_model", lambda _d: (None, None))
    monkeypatch.setattr(profiles, "_count_skills", lambda _d: 0)
    monkeypatch.setattr(profiles, "_read_distribution_meta", lambda _d: (None, None, None))
    profiles._print_delete_summary("beta", profile_dir, gw_running, None)
    return ""


def test_a_multiplexed_profile_does_not_claim_the_gateway_stops(monkeypatch, tmp_path, capsys):
    home = tmp_path / "beta"
    home.mkdir()

    _summary(monkeypatch, home, gw_running=True)
    out = capsys.readouterr().out

    assert "it will be stopped" not in out
    assert "multiplexer" in out


def test_a_profile_with_its_own_gateway_still_warns_about_the_stop(monkeypatch, tmp_path, capsys):
    home = tmp_path / "beta"
    home.mkdir()
    (home / "gateway.pid").write_text("12345", encoding="utf-8")

    _summary(monkeypatch, home, gw_running=True)
    out = capsys.readouterr().out

    assert "it will be stopped" in out


def test_no_warning_when_nothing_runs(monkeypatch, tmp_path, capsys):
    home = tmp_path / "beta"
    home.mkdir()

    _summary(monkeypatch, home, gw_running=False)

    assert "Gateway is running" not in capsys.readouterr().out
