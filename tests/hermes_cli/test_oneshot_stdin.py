from __future__ import annotations

import io


class _NonTtyStdin(io.StringIO):
    def isatty(self):
        return False


class _TtyStdin(io.StringIO):
    def isatty(self):
        return True


class _BrokenStdin(io.StringIO):
    def isatty(self):
        return False

    def read(self, *args, **kwargs):
        raise OSError("broken stdin")


def test_oneshot_dash_reads_prompt_from_stdin(monkeypatch, capsys):
    import hermes_cli.oneshot as oneshot_mod

    captured = {}

    def fake_run_agent(prompt, **_kwargs):
        captured["prompt"] = prompt
        return ("done", {"final_response": "done"})

    monkeypatch.setattr(oneshot_mod, "_run_agent", fake_run_agent)
    monkeypatch.setattr(oneshot_mod.sys, "stdin", _NonTtyStdin("file body\n"))

    assert oneshot_mod.run_oneshot("-") == 0
    assert captured["prompt"] == "file body\n"
    assert capsys.readouterr().out == "done\n"


def test_oneshot_literal_prompt_does_not_drain_non_tty_stdin(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    captured = {}
    stdin = _NonTtyStdin("loop input\n")

    def fake_run_agent(prompt, **_kwargs):
        captured["prompt"] = prompt
        return ("done", {"final_response": "done"})

    monkeypatch.setattr(oneshot_mod, "_run_agent", fake_run_agent)
    monkeypatch.setattr(oneshot_mod.sys, "stdin", stdin)

    assert oneshot_mod.run_oneshot("literal prompt") == 0
    assert captured["prompt"] == "literal prompt"
    assert stdin.read() == "loop input\n"


def test_oneshot_dash_rejects_empty_stdin(monkeypatch, capsys):
    import hermes_cli.oneshot as oneshot_mod

    monkeypatch.setattr(oneshot_mod.sys, "stdin", _NonTtyStdin(""))

    assert oneshot_mod.run_oneshot("-") == 2
    assert "requires non-empty stdin" in capsys.readouterr().err


def test_oneshot_dash_rejects_tty_stdin(monkeypatch, capsys):
    import hermes_cli.oneshot as oneshot_mod

    monkeypatch.setattr(oneshot_mod.sys, "stdin", _TtyStdin("ignored"))

    assert oneshot_mod.run_oneshot("-") == 2
    assert "requires stdin" in capsys.readouterr().err


def test_oneshot_dash_reports_stdin_read_failure(monkeypatch, capsys):
    import hermes_cli.oneshot as oneshot_mod

    monkeypatch.setattr(oneshot_mod.sys, "stdin", _BrokenStdin("ignored"))

    assert oneshot_mod.run_oneshot("-") == 2
    assert "failed to read prompt from stdin" in capsys.readouterr().err
