"""selected_git_env remembers a registry refusal instead of re-paying PM per call."""

import pm
from hermes_cli import _subprocess_compat
from hermes_cli._subprocess_compat import selected_git_env


def test_registry_refusal_is_consulted_once(monkeypatch):
    """POSIX's deliberate git gap fails identically every call; consult PM once."""
    monkeypatch.setattr(_subprocess_compat, "_pm_git_declined", False)
    calls = []

    def refusing(name, *, base_env):
        calls.append(base_env)
        raise pm.InstallError(
            name, "unavailable on darwin-arm64: POSIX uses system git by choice", "none"
        )

    monkeypatch.setattr(pm, "ensure", refusing)
    first = selected_git_env({"PATH": "/bin"})
    second = selected_git_env({"PATH": "/usr/bin"})
    assert first == {"PATH": "/bin"}
    assert second == {"PATH": "/usr/bin"}
    assert len(calls) == 1


def test_transient_failures_keep_retrying(monkeypatch):
    """Download-shaped errors and non-PM failures must not poison later attempts."""
    monkeypatch.setattr(_subprocess_compat, "_pm_git_declined", False)

    def unavailable(name, *, base_env):
        raise pm.InstallError(name, "download failed")

    monkeypatch.setattr(pm, "ensure", unavailable)
    assert selected_git_env({"PATH": "/bin"}) == {"PATH": "/bin"}
    assert selected_git_env({"PATH": "/bin"}) == {"PATH": "/bin"}

    def crashing(name, *, base_env):
        raise RuntimeError("worker exited unsuccessfully")

    monkeypatch.setattr(pm, "ensure", crashing)
    assert selected_git_env({"PATH": "/bin"}) == {"PATH": "/bin"}

    provided = pm.Runner("git", {"PATH": "/pm/bin"})
    monkeypatch.setattr(pm, "ensure", lambda name, *, base_env: provided)
    assert selected_git_env({"PATH": "/bin"}) == {"PATH": "/pm/bin"}


def test_successful_selection_is_not_cached(monkeypatch):
    """A changed PM selection later in this process still applies."""
    monkeypatch.setattr(_subprocess_compat, "_pm_git_declined", False)
    first = pm.Runner("git", {"PATH": "/first"})
    second = pm.Runner("git", {"PATH": "/second"})
    monkeypatch.setattr(pm, "ensure", lambda name, *, base_env: first)
    assert selected_git_env()["PATH"] == "/first"
    monkeypatch.setattr(pm, "ensure", lambda name, *, base_env: second)
    assert selected_git_env()["PATH"] == "/second"
