from types import SimpleNamespace


def test_process_start_time_uses_psutil_monotonic_value(monkeypatch):
    import gateway.status as status

    calls = []

    class RawProcess:
        def create_time(self, **kwargs):
            calls.append(kwargs)
            return 12.34

    class Process:
        def __init__(self, pid):
            self._proc = RawProcess()

        def create_time(self):
            raise AssertionError("wall-clock fallback should not be used")

    monkeypatch.setitem(__import__("sys").modules, "psutil", SimpleNamespace(Process=Process))
    monkeypatch.setattr(status.Path, "read_text", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError))

    assert status._get_process_start_time(123) == 1234
    assert calls == [{"monotonic": True}]


def test_process_start_time_matches_pre_monotonic_persisted_record(monkeypatch):
    import gateway.status as status

    class RawProcess:
        def create_time(self, **kwargs):
            assert kwargs == {"monotonic": True}
            return 86400.25

    class Process:
        _proc = RawProcess()

        def __init__(self, pid):
            assert pid == 123

        def create_time(self):
            return 1700000000.50

    monkeypatch.setitem(__import__("sys").modules, "psutil", SimpleNamespace(Process=Process))
    monkeypatch.setattr(status, "_get_proc_start_time", lambda pid: None)

    # Older records contain epoch centiseconds; the live value is now monotonic.
    assert status._get_process_start_time(123) == 8640025
    assert status._process_start_time_matches(123, 170000000050)


def test_process_start_time_does_not_accept_recycled_legacy_record(monkeypatch):
    import gateway.status as status

    class RawProcess:
        def create_time(self, **kwargs):
            return 86400.25

    class Process:
        _proc = RawProcess()

        def __init__(self, pid):
            pass

        def create_time(self):
            return 1700000000.50

    monkeypatch.setitem(__import__("sys").modules, "psutil", SimpleNamespace(Process=Process))
    monkeypatch.setattr(status, "_get_proc_start_time", lambda pid: None)

    assert not status._process_start_time_matches(123, 170000000150)
