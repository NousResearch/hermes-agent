"""Tests for hermes_constants._hermes_home_from_installed_unit and
sudo_invoker_default_home / sudo_aware_default_hermes_root.

Covers:
 - The two must-fix parser bugs Ronan identified and independently reproduced
   (multi-assignment quoting corruption, wrong-section Environment= accepted).
 - Normal happy-path cases that were already passing.
 - sudo_invoker_default_home: non-root fast-path, root-with-SUDO_USER path,
   systemd-unit-pin takes priority over pw_dir fallback.
 - sudo_aware_default_hermes_root: delegates correctly.

No real filesystem I/O against the live hermes home — all unit files are
written to tmp_path fixtures.
"""

from __future__ import annotations

import importlib
import os
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fn(monkeypatch, tmp_unit_dir: Path):
    """Return _hermes_home_from_installed_unit with SYSTEMD_SYSTEM_UNIT_DIR
    redirected to tmp_unit_dir, without polluting the global module state."""
    import hermes_constants as hc
    monkeypatch.setattr(hc, "_SYSTEMD_SYSTEM_UNIT_DIR", tmp_unit_dir)
    return hc._hermes_home_from_installed_unit


def _write_unit(unit_dir: Path, name: str, content: str) -> Path:
    p = unit_dir / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _hermes_home_from_installed_unit — parser correctness
# ---------------------------------------------------------------------------

class TestUnitParser:
    def test_single_quoted_assignment_happy_path(self, monkeypatch, tmp_path):
        """Standard single-assignment, fully-quoted line — the common production case."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Unit]
            Description=Hermes gateway
            [Service]
            User=hermes
            Environment="HERMES_HOME=/opt/data"
            ExecStart=/usr/bin/hermes gateway run
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        assert fn("hermes") == Path("/opt/data")

    def test_unquoted_assignment(self, monkeypatch, tmp_path):
        """Environment= without outer quotes — also valid systemd syntax."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Unit]
            Description=Hermes
            [Service]
            User=hermes
            Environment=HERMES_HOME=/opt/data
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        assert fn("hermes") == Path("/opt/data")

    # --- MUST-FIX 1: multi-assignment quoting ---
    def test_multi_assignment_returns_correct_path_not_garbage(self, monkeypatch, tmp_path):
        """Bug: old code returned PosixPath('/opt/data\" \"OTHER=x').
        Fixed: shlex.split picks HERMES_HOME= token out of the list correctly."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Unit]
            Description=Hermes
            [Service]
            User=hermes
            Environment="HERMES_HOME=/opt/data" "OTHER=x"
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        result = fn("hermes")
        assert result == Path("/opt/data"), f"got {result!r}"
        # Specifically must NOT contain trailing garbage
        assert '"' not in str(result)
        assert "OTHER" not in str(result)

    def test_hermes_home_not_first_token_is_found(self, monkeypatch, tmp_path):
        """HERMES_HOME= appears after another assignment on the same line."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Unit]
            Description=Hermes
            [Service]
            User=hermes
            Environment="OTHER=x" "HERMES_HOME=/opt/data"
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        assert fn("hermes") == Path("/opt/data")

    # --- MUST-FIX 2: wrong-section Environment= ---
    def test_environment_in_unit_section_is_ignored(self, monkeypatch, tmp_path):
        """Bug: old code accepted Environment= from [Unit] section.
        Fixed: only Environment= lines inside [Service] are honoured."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Unit]
            Description=Hermes
            Environment="HERMES_HOME=/wrong/section"
            [Service]
            User=hermes
            ExecStart=/usr/bin/hermes gateway run
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        assert fn("hermes") is None

    def test_environment_before_any_section_is_ignored(self, monkeypatch, tmp_path):
        """Environment= before the first section header must also be ignored."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            Environment="HERMES_HOME=/too/early"
            [Service]
            User=hermes
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        assert fn("hermes") is None

    def test_environment_in_install_section_is_ignored(self, monkeypatch, tmp_path):
        """[Install] is not [Service]; Environment= there must be ignored."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Service]
            User=hermes
            ExecStart=/usr/bin/hermes gateway run
            [Install]
            Environment="HERMES_HOME=/install/section"
            WantedBy=multi-user.target
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        # No Environment= in [Service] → None
        assert fn("hermes") is None

    def test_service_section_after_unit_section_works(self, monkeypatch, tmp_path):
        """Standard [Unit] then [Service] layout — the real-world happy path."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Unit]
            Description=Hermes gateway
            After=network.target
            [Service]
            User=hermes
            Environment="HERMES_HOME=/opt/data"
            ExecStart=/usr/bin/hermes gateway run
            [Install]
            WantedBy=multi-user.target
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        assert fn("hermes") == Path("/opt/data")

    def test_wrong_user_returns_none(self, monkeypatch, tmp_path):
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Service]
            User=otheruser
            Environment="HERMES_HOME=/opt/data"
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        assert fn("hermes") is None

    def test_no_matching_unit_returns_none(self, monkeypatch, tmp_path):
        fn = _load_fn(monkeypatch, tmp_path)
        assert fn("hermes") is None

    def test_malformed_quoting_does_not_raise(self, monkeypatch, tmp_path):
        """shlex.split raises ValueError on unterminated quotes — we must catch it."""
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Service]
            User=hermes
            Environment="HERMES_HOME=/opt/data
        """)
        fn = _load_fn(monkeypatch, tmp_path)
        # Must not raise; may return None or a value depending on shlex behaviour
        result = fn("hermes")
        # The malformed line should be skipped gracefully → None
        assert result is None

    def test_unreadable_unit_skipped(self, monkeypatch, tmp_path):
        p = _write_unit(tmp_path, "hermes-gateway.service", """\
            [Service]
            User=hermes
            Environment="HERMES_HOME=/opt/data"
        """)
        p.chmod(0o000)
        fn = _load_fn(monkeypatch, tmp_path)
        try:
            result = fn("hermes")
            # If we're root the file is still readable; just check no exception raised.
        except Exception as exc:
            pytest.fail(f"should not raise: {exc}")
        finally:
            p.chmod(0o644)


# ---------------------------------------------------------------------------
# sudo_invoker_default_home
# ---------------------------------------------------------------------------

class TestSudoInvokerDefaultHome:
    def test_returns_none_when_not_root(self, monkeypatch):
        import hermes_constants as hc
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        assert hc.sudo_invoker_default_home() is None

    def test_returns_none_when_no_sudo_user(self, monkeypatch):
        import hermes_constants as hc
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        monkeypatch.setenv("SUDO_USER", "")
        assert hc.sudo_invoker_default_home() is None

    def test_returns_none_when_sudo_user_is_root(self, monkeypatch):
        import hermes_constants as hc
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        monkeypatch.setenv("SUDO_USER", "root")
        assert hc.sudo_invoker_default_home() is None

    def test_unit_pin_takes_priority_over_pwd_fallback(self, monkeypatch, tmp_path):
        """When a systemd unit pins HERMES_HOME, that wins over pw_dir/.hermes."""
        import hermes_constants as hc
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Service]
            User=hermes
            Environment="HERMES_HOME=/opt/data"
        """)
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        monkeypatch.setenv("SUDO_USER", "hermes")
        monkeypatch.setattr(hc, "_SYSTEMD_SYSTEM_UNIT_DIR", tmp_path)
        result = hc.sudo_invoker_default_home()
        assert result == Path("/opt/data")

    def test_falls_back_to_pwd_when_no_unit(self, monkeypatch, tmp_path):
        """No matching unit → fall back to pw_dir/.hermes via pwd module."""
        import hermes_constants as hc
        import pwd as _pwd

        monkeypatch.setattr(os, "geteuid", lambda: 0)
        monkeypatch.setenv("SUDO_USER", "hermes")
        monkeypatch.setattr(hc, "_SYSTEMD_SYSTEM_UNIT_DIR", tmp_path)  # empty dir

        fake_entry = _pwd.getpwuid(os.getuid())  # borrow a real struct, patch pw_dir
        class _FakePwEntry:
            pw_dir = str(tmp_path / "fake_home")

        import pwd as pwd_mod
        monkeypatch.setattr(pwd_mod, "getpwnam", lambda name: _FakePwEntry())
        result = hc.sudo_invoker_default_home()
        assert result == Path(tmp_path / "fake_home") / ".hermes"


# ---------------------------------------------------------------------------
# sudo_aware_default_hermes_root
# ---------------------------------------------------------------------------

class TestSudoAwareDefaultHermesRoot:
    def test_uses_sudo_home_when_root(self, monkeypatch, tmp_path):
        import hermes_constants as hc
        _write_unit(tmp_path, "hermes-gateway.service", """\
            [Service]
            User=hermes
            Environment="HERMES_HOME=/opt/data"
        """)
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        monkeypatch.setenv("SUDO_USER", "hermes")
        monkeypatch.setattr(hc, "_SYSTEMD_SYSTEM_UNIT_DIR", tmp_path)
        # Should return get_default_hermes_root(home=Path("/opt/data"))
        result = hc.sudo_aware_default_hermes_root()
        assert str(result).startswith("/opt/data")

    def test_falls_back_to_normal_default_when_not_root(self, monkeypatch):
        import hermes_constants as hc
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        # Should equal plain get_default_hermes_root() when not sudo
        assert hc.sudo_aware_default_hermes_root() == hc.get_default_hermes_root()
