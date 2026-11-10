"""A registry gap is refused in the caller, never by a PM worker round trip.

Git on POSIX is a declared gap ("uses system git by choice"). ``selected_git_env()`` asks PM for it
before every checkpoint git child, so a worker spawn per ask made each gateway /rollback or /diff
pay seconds and timed those RPCs out on a loaded CI runner.
"""
import pytest

from hermes_cli._subprocess_compat import selected_git_env
import pm
from pm import client, install
from pm.package import InstallError


@pytest.fixture
def no_worker(monkeypatch):
    monkeypatch.setattr(client, "is_runtime", lambda: False)
    monkeypatch.setattr(install, "is_installed", lambda name: False)
    monkeypatch.setattr(install, "current_target", lambda: "linux-x64")
    # The product default; the harness disables lazy installs, which refuses before any worker.
    monkeypatch.setattr(install, "lazy_installs_allowed", lambda: True)

    def worker(operation, arguments, **kwargs):
        pytest.fail(f"a PM worker was asked to {operation} {arguments}")

    monkeypatch.setattr(client, "_request", worker)


def test_a_registry_gap_is_refused_without_a_worker(no_worker):
    with pytest.raises(InstallError) as refused:
        pm.ensure("git", base_env={})
    assert refused.value.remedy == "none"
    assert "POSIX uses system git by choice" in refused.value.cause


def test_selected_git_env_keeps_the_system_git_without_a_worker(no_worker):
    assert selected_git_env({"PATH": "/usr/bin"}) == {"PATH": "/usr/bin"}
