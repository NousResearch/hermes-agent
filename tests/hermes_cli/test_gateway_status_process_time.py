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