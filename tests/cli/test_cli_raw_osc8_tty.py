def test_cprint_raw_terminal_writes_osc8_bytes_to_controlling_tty(monkeypatch):
    import cli

    writes = []
    closed = []

    monkeypatch.setattr(cli.os, "open", lambda path, flags: 42)
    monkeypatch.setattr(cli.os, "write", lambda fd, payload: writes.append((fd, payload)) or len(payload))
    monkeypatch.setattr(cli.os, "close", lambda fd: closed.append(fd))

    text = "\x1b]8;;https://example.com\x1b\\label\x1b]8;;\x1b\\"
    cli._cprint_raw_terminal(text)

    assert writes == [(42, (text + "\n").encode("utf-8"))]
    assert closed == [42]


def test_cprint_raw_terminal_fallback_strips_osc_for_prompt_toolkit(monkeypatch):
    import cli

    printed = []
    monkeypatch.setattr(cli.os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no tty")))
    monkeypatch.setattr(cli, "_cprint", printed.append)

    class NonTty:
        def isatty(self):
            return False

    monkeypatch.setattr(cli.sys, "stdout", NonTty())
    monkeypatch.setattr(cli.sys, "__stdout__", NonTty())

    cli._cprint_raw_terminal("\x1b]8;;https://example.com\x1b\\label\x1b]8;;\x1b\\")

    assert printed == ["label"]
