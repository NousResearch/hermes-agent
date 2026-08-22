from __future__ import annotations

import pytest

from hermes_cli import windows_process_authority as authority


class FakeThread:
    def __init__(self, *, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


class FakeApi:
    def __init__(self, *, parent_started_at_ms=1_700_000_000_000):
        self.parent_started_at = parent_started_at_ms
        self.calls = []
        self.closed = []
        self.job = object()
        self.parent = object()
        self.current = object()
        self.fail_at = None

    def _call(self, name, *args):
        self.calls.append((name, *args))
        if self.fail_at == name:
            raise authority.ProcessAuthorityError(f"{name} failed")

    def create_job(self):
        self._call("create_job")
        return self.job

    def enable_kill_on_close(self, job_handle):
        self._call("enable_kill_on_close", job_handle)

    def current_process_handle(self):
        self._call("current_process_handle")
        return self.current

    def assign_current_process(self, job_handle, process_handle):
        self._call("assign_current_process", job_handle, process_handle)

    def open_parent(self, pid):
        self._call("open_parent", pid)
        return self.parent

    def process_started_at_ms(self, process_handle):
        self._call("process_started_at_ms", process_handle)
        return self.parent_started_at

    def wait_for_process_exit(self, process_handle):
        self._call("wait_for_process_exit", process_handle)

    def close_handle(self, handle):
        self.calls.append(("close_handle", handle))
        self.closed.append(handle)


@pytest.fixture(autouse=True)
def reset_authority():
    authority._reset_process_authority_for_tests()
    yield
    authority._reset_process_authority_for_tests()


def env(**overrides):
    values = {
        authority.AUTHORITY_MODE_ENV: authority.AUTHORITY_MODE,
        authority.GENERATION_ENV: "e2f531e4-14b1-47ff-9f87-bc278cfa816d",
        authority.PARENT_PID_ENV: "4242",
        authority.PARENT_STARTED_AT_ENV: "1700000000000",
    }
    values.update(overrides)
    return values


def test_unmarked_process_is_untouched():
    api = FakeApi()
    assert authority.install_windows_process_authority(
        environ={}, platform="win32", api=api, thread_factory=FakeThread
    ) is None
    assert api.calls == []


def test_off_windows_is_noop_even_with_envelope():
    api = FakeApi()
    assert authority.install_windows_process_authority(
        environ=env(), platform="linux", api=api, thread_factory=FakeThread
    ) is None
    assert api.calls == []


@pytest.mark.parametrize(
    "key,value,match",
    [
        (authority.AUTHORITY_MODE_ENV, "pid-v1", "unsupported"),
        (authority.GENERATION_ENV, "short", "generation"),
        (authority.PARENT_PID_ENV, "not-a-pid", "identity"),
        (authority.PARENT_PID_ENV, "0", "positive"),
        (authority.PARENT_STARTED_AT_ENV, "-1", "positive"),
    ],
)
def test_malformed_authority_envelope_fails_before_os_calls(key, value, match):
    api = FakeApi()
    with pytest.raises(authority.ProcessAuthorityError, match=match):
        authority.install_windows_process_authority(
            environ=env(**{key: value}),
            platform="win32",
            api=api,
            thread_factory=FakeThread,
        )
    assert api.calls == []


def test_installs_job_and_binds_parent_generation_before_assignment():
    api = FakeApi()
    installed = authority.install_windows_process_authority(
        environ=env(), platform="win32", api=api, thread_factory=FakeThread
    )

    assert installed is not None
    assert installed.spec.parent_pid == 4242
    assert installed.spec.parent_started_at_ms == 1_700_000_000_000
    assert installed.watcher.started is True
    assert [call[0] for call in api.calls] == [
        "create_job",
        "enable_kill_on_close",
        "open_parent",
        "process_started_at_ms",
        "current_process_handle",
        "assign_current_process",
    ]
    assert api.calls[-1][1:] == (api.job, api.current)

    installed.watcher.target()
    assert [call[0] for call in api.calls[-3:]] == [
        "wait_for_process_exit",
        "close_handle",
        "close_handle",
    ]
    assert api.closed[-2:] == [api.parent, api.job]


def test_recycled_parent_pid_is_rejected_and_handles_are_closed():
    api = FakeApi(parent_started_at_ms=1_700_000_020_000)
    with pytest.raises(authority.ProcessAuthorityError, match="generation mismatch"):
        authority.install_windows_process_authority(
            environ=env(),
            platform="win32",
            api=api,
            thread_factory=FakeThread,
        )
    assert "assign_current_process" not in [call[0] for call in api.calls]
    assert api.closed == [api.parent, api.job]


def test_assignment_failure_closes_parent_and_job_without_fallback():
    api = FakeApi()
    api.fail_at = "assign_current_process"
    with pytest.raises(authority.ProcessAuthorityError, match="assign_current_process"):
        authority.install_windows_process_authority(
            environ=env(), platform="win32", api=api, thread_factory=FakeThread
        )
    assert api.closed == [api.parent, api.job]
    assert all(call[0] != "taskkill" for call in api.calls)


def test_same_generation_is_idempotent_but_different_generation_is_rejected():
    api = FakeApi()
    first = authority.install_windows_process_authority(
        environ=env(), platform="win32", api=api, thread_factory=FakeThread
    )
    second = authority.install_windows_process_authority(
        environ=env(), platform="win32", api=api, thread_factory=FakeThread
    )
    assert second is first
    assert [call[0] for call in api.calls].count("create_job") == 1

    with pytest.raises(
        authority.ProcessAuthorityError,
        match="different desktop process generation",
    ):
        authority.install_windows_process_authority(
            environ=env(
                **{
                    authority.GENERATION_ENV:
                        "3b371df1-dbb7-4b0c-b87c-44f4ec1e2293"
                }
            ),
            platform="win32",
            api=api,
            thread_factory=FakeThread,
        )


def test_negative_tolerance_is_configuration_error_before_os_calls():
    api = FakeApi()
    with pytest.raises(ValueError, match="non-negative"):
        authority.install_windows_process_authority(
            environ=env(),
            platform="win32",
            api=api,
            thread_factory=FakeThread,
            parent_start_tolerance_ms=-1,
        )
    assert api.calls == []
