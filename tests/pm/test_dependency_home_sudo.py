"""``dependency_home_root`` follows the invoking user's home when this process is root under sudo.

sudo strips HERMES_HOME and sets HOME=/root, so without this the package manager scopes its
dependency state (``<root>/installs/<key>``) to root's home and a sudo run builds a root-owned
duplicate install. The tool-store fallbacks (``store_root`` and ``_unstamped_store``) resolve through
the same function. ``hermes_constants.sudo_invoker_default_home`` already names that home for the
profile and gateway code paths. The effective uid, SUDO_USER and the password database are faked
with monkeypatch; nothing here runs as root or touches a path outside ``tmp_path``.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from pm import environments

pytestmark = pytest.mark.platforms("posix")


def _fake_machine(tmp_path, monkeypatch, *, euid, sudo_user):
    """Root's HOME is ``tmp_path/root``; the invoking user's home is ``tmp_path/alice``."""
    import pwd

    root_home = tmp_path / "root"
    user_home = tmp_path / "alice"
    root_home.mkdir()
    user_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: root_home)
    monkeypatch.setattr(os, "geteuid", lambda: euid, raising=False)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_dir=str(user_home)))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    if sudo_user is None:
        monkeypatch.delenv("SUDO_USER", raising=False)
    else:
        monkeypatch.setenv("SUDO_USER", sudo_user)
    return root_home / ".hermes", user_home / ".hermes"


def test_root_under_sudo_uses_the_invoking_users_home(tmp_path, monkeypatch):
    root_default, user_default = _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="alice")

    assert environments.dependency_home_root() == user_default
    assert environments.installs_root() == user_default / "installs"
    assert environments.dependency_home_root() != root_default


def test_non_root_process_ignores_sudo_user(tmp_path, monkeypatch):
    root_default, _ = _fake_machine(tmp_path, monkeypatch, euid=1000, sudo_user="alice")

    assert environments.dependency_home_root() == root_default


def test_sudo_user_root_keeps_the_process_default(tmp_path, monkeypatch):
    root_default, _ = _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="root")

    assert environments.dependency_home_root() == root_default


def test_root_without_sudo_keeps_the_process_default(tmp_path, monkeypatch):
    root_default, _ = _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user=None)

    assert environments.dependency_home_root() == root_default


def test_unknown_sudo_user_keeps_the_process_default(tmp_path, monkeypatch):
    import pwd

    root_default, _ = _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="alice")

    def unknown_user(name):
        raise KeyError(name)

    monkeypatch.setattr(pwd, "getpwnam", unknown_user)

    assert environments.dependency_home_root() == root_default


def test_explicit_hermes_home_wins_over_sudo_user(tmp_path, monkeypatch):
    _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="alice")
    explicit = tmp_path / "explicit-root"
    explicit.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(explicit))

    assert environments.dependency_home_root() == explicit


def test_blank_hermes_home_counts_as_unset_under_sudo(tmp_path, monkeypatch):
    _, user_default = _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="alice")
    monkeypatch.setenv("HERMES_HOME", "  ")

    assert environments.dependency_home_root() == user_default


def test_context_override_wins_over_sudo_user(tmp_path, monkeypatch):
    _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="alice")
    context_home = tmp_path / "context-home"
    context_home.mkdir()
    token = set_hermes_home_override(context_home)
    try:
        assert environments.dependency_home_root() == context_home
    finally:
        reset_hermes_home_override(token)


def _unstamped_checkout(tmp_path, monkeypatch):
    """A source tree with no stamp, no payload manifest and no owning root."""
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    return checkout


def test_store_root_fallback_under_sudo_uses_the_invoking_users_home(tmp_path, monkeypatch):
    _, user_default = _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="alice")
    checkout = _unstamped_checkout(tmp_path, monkeypatch)

    assert environments.store_root(checkout) == user_default / "tools"


def test_unstamped_store_under_sudo_uses_the_invoking_users_home(tmp_path, monkeypatch):
    _, user_default = _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="alice")
    checkout = _unstamped_checkout(tmp_path, monkeypatch)

    assert environments._unstamped_store(checkout) == user_default / "tools"


@pytest.mark.parametrize(
    ("euid", "sudo_user"),
    [(0, None), (0, "root"), (1000, "alice")],
    ids=["root-without-sudo", "sudo-user-root", "non-root"],
)
def test_tool_store_fallback_without_sudo_keeps_the_process_default(tmp_path, monkeypatch, euid, sudo_user):
    root_default, _ = _fake_machine(tmp_path, monkeypatch, euid=euid, sudo_user=sudo_user)
    checkout = _unstamped_checkout(tmp_path, monkeypatch)

    assert environments.store_root(checkout) == root_default / "tools"
    assert environments._unstamped_store(checkout) == root_default / "tools"


def test_tool_store_fallback_follows_an_explicit_hermes_home(tmp_path, monkeypatch):
    _fake_machine(tmp_path, monkeypatch, euid=0, sudo_user="alice")
    checkout = _unstamped_checkout(tmp_path, monkeypatch)
    explicit = tmp_path / "explicit-root"
    explicit.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(explicit))

    assert environments.store_root(checkout) == explicit / "tools"
    assert environments._unstamped_store(checkout) == explicit / "tools"


def test_tool_store_fallback_follows_a_context_override(tmp_path, monkeypatch):
    """Without sudo, a context-only home resolves the tool store like a process launched in it."""
    _fake_machine(tmp_path, monkeypatch, euid=1000, sudo_user=None)
    checkout = _unstamped_checkout(tmp_path, monkeypatch)
    process_root = tmp_path / "process-root"
    context_root = tmp_path / "context-root"
    process_root.mkdir()
    context_root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_root))
    token = set_hermes_home_override(context_root)
    try:
        assert environments.store_root(checkout) == context_root / "tools"
        assert environments._unstamped_store(checkout) == context_root / "tools"
    finally:
        reset_hermes_home_override(token)
    assert environments.store_root(checkout) == process_root / "tools"
