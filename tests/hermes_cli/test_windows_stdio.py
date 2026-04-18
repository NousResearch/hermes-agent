from hermes_cli import stdio


class _EncodingStream:
    def __init__(self, encoding: str = "cp1252"):
        self.encoding = encoding
        self.reconfigure_calls = []
        self.writes = []

    def write(self, data):
        if isinstance(data, str):
            data.encode(self.encoding)
        self.writes.append(data)
        return len(data)

    def flush(self):
        return None

    def fileno(self):
        return 1

    def isatty(self):
        return True

    def reconfigure(self, **kwargs):
        self.reconfigure_calls.append(kwargs)
        self.encoding = kwargs.get("encoding", self.encoding)


def test_windows_safe_writer_falls_back_to_ascii_escape():
    inner = _EncodingStream()
    writer = stdio._WindowsSafeWriter(inner)

    result = writer.write("status -> ready -> \u2192")

    assert result == len("status -> ready -> \u2192")
    assert inner.writes == [r"status -> ready -> \u2192"]


def test_install_windows_stdio_reconfigures_and_wraps(monkeypatch):
    fake_out = _EncodingStream()
    fake_err = _EncodingStream()
    codepage_calls = []

    monkeypatch.setattr(stdio.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(stdio.sys, "stdout", fake_out)
    monkeypatch.setattr(stdio.sys, "stderr", fake_err)
    monkeypatch.setattr(
        stdio,
        "_set_console_utf8_codepage",
        lambda: codepage_calls.append("called"),
    )

    stdio.install_windows_stdio()

    assert codepage_calls == ["called"]
    assert isinstance(stdio.sys.stdout, stdio._WindowsSafeWriter)
    assert isinstance(stdio.sys.stderr, stdio._WindowsSafeWriter)
    assert fake_out.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert fake_err.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]

    stdio.sys.stdout.write("\u2713 done")
    assert fake_out.writes == ["\u2713 done"]


def test_install_windows_stdio_noops_off_windows(monkeypatch):
    fake_out = _EncodingStream()
    fake_err = _EncodingStream()

    monkeypatch.setattr(stdio.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(stdio.sys, "stdout", fake_out)
    monkeypatch.setattr(stdio.sys, "stderr", fake_err)

    stdio.install_windows_stdio()

    assert stdio.sys.stdout is fake_out
    assert stdio.sys.stderr is fake_err
    assert fake_out.reconfigure_calls == []
    assert fake_err.reconfigure_calls == []
