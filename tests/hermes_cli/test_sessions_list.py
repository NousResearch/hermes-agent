import sys
import time


def test_sessions_list_titled_layout_shows_source(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def list_sessions_rich(self, **kwargs):
            assert kwargs["source"] is None
            assert kwargs["exclude_sources"] == ["tool"]
            return [
                {
                    "id": "20260401_201329_d85961",
                    "source": "telegram",
                    "title": "Named chat",
                    "preview": "hello from telegram",
                    "last_active": time.time(),
                }
            ]

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "list"])

    main_mod.main()

    output = capsys.readouterr().out
    header = output.splitlines()[0]
    assert "Title" in header
    assert "Src" in header
    assert "Named chat" in output
    assert "telegram" in output
    assert "20260401_201329_d85961" in output
