"""Tests for the Windows ACL guard on the managed tool store (#126860).

An elevated ``hermes update`` can publish tool entries whose DACL grants the
ordinary user nothing at all, permanently poisoning the store: every later
non-elevated update dies with a raw ``[WinError 5]`` on the verify/replace
rename and the tools are unusable at runtime for that user. The guard probes
what the publisher is about to record as installed (elevated sessions only)
and repairs it; an access-denied OSError from a later install names the entry
and the recovery instead of the bare errno.

The kernel probe and icacls are Windows-only, so the tests fix the probe
boundary (``standard_user_denied`` / ``_icacls`` / ``_account_sid``) and
exercise the guard logic on any host OS.
"""

from pathlib import Path

import pytest

from pm import windows_acl as acl
from pm.package import InstallError


class TestIsAccessDenied:
    def test_winerror_5(self):
        err = OSError(5, "Access is denied")
        err.winerror = 5
        assert acl.is_access_denied(err) is True

    def test_errno_13(self):
        assert acl.is_access_denied(OSError(13, "Permission denied")) is True

    def test_other_errors(self):
        assert acl.is_access_denied(OSError(2, "No such file")) is False


class TestAccessDeniedError:
    def test_error_names_entry_and_recovery(self):
        err = OSError(5, "Access is denied")
        err.winerror = 5
        raised = acl.access_denied_error("git", "installing", r"C:\store\git-2.53", err)
        assert isinstance(raised, InstallError)
        assert raised.package == "git"
        assert "access denied" in raised.cause
        assert "elevated" in raised.cause
        assert "icacls" in raised.remedy
        assert "hermes update" in raised.remedy

    def test_wrap_passes_unrelated_errors_through(self):
        with pytest.raises(OSError) as excinfo:
            acl.wrap_access_denied("git", "installing", "x", OSError(2, "No such file"))
        assert excinfo.value.errno == 2

    def test_wrap_converts_access_denied(self):
        err = OSError(13, "Permission denied")
        raised = acl.wrap_access_denied("node", "replacing", "entry", err)
        assert isinstance(raised, InstallError)
        assert raised.package == "node"


class TestEnsureUserReadable:
    def test_noop_off_windows(self, monkeypatch, tmp_path):
        probed = []
        monkeypatch.setattr(acl, "_is_windows", lambda: False)
        monkeypatch.setattr(acl, "standard_user_denied",
                            lambda p: probed.append(p) or True)
        acl.ensure_user_readable("git", tmp_path / "entry", tmp_path)
        assert probed == []

    def test_noop_when_not_elevated(self, monkeypatch, tmp_path):
        probed = []
        monkeypatch.setattr(acl, "_is_windows", lambda: True)
        monkeypatch.setattr(acl, "is_elevated", lambda: False)
        monkeypatch.setattr(acl, "standard_user_denied",
                            lambda p: probed.append(p) or True)
        acl.ensure_user_readable("git", tmp_path / "entry", tmp_path)
        assert probed == []

    def test_healthy_elevated_store_is_untouched(self, monkeypatch, tmp_path):
        """A readable entry and store root: one probe each, no repair, no raise."""
        calls = []
        monkeypatch.setattr(acl, "_is_windows", lambda: True)
        monkeypatch.setattr(acl, "is_elevated", lambda: True)
        monkeypatch.setattr(acl, "standard_user_denied", lambda p: calls.append(p) or False)
        monkeypatch.setattr(acl, "_icacls",
                            lambda args: (_ for _ in ()).throw(AssertionError("icacls ran")))
        acl.ensure_user_readable("git", tmp_path / "entry", tmp_path)
        assert sorted(calls) == sorted([tmp_path / "entry", tmp_path])

    def test_repairs_poisoned_entry(self, monkeypatch, tmp_path):
        entry, root = tmp_path / "entry", tmp_path
        icacls_args = []
        state = {"denied_paths": {entry}}

        def denied(path):
            return Path(path) in state["denied_paths"]

        def icacls(args):
            icacls_args.append(args)
            state["denied_paths"].clear()  # the repair healed the tree
            return True

        monkeypatch.setattr(acl, "_is_windows", lambda: True)
        monkeypatch.setattr(acl, "is_elevated", lambda: True)
        monkeypatch.setattr(acl, "standard_user_denied", denied)
        monkeypatch.setattr(acl, "_icacls", icacls)
        acl.ensure_user_readable("git", entry, root)
        assert any("/reset" in a for a in icacls_args), "the poisoned entry was not repaired"

    def test_raises_when_repair_fails(self, monkeypatch, tmp_path):
        entry, root = tmp_path / "entry", tmp_path
        monkeypatch.setattr(acl, "_is_windows", lambda: True)
        monkeypatch.setattr(acl, "is_elevated", lambda: True)
        monkeypatch.setattr(acl, "standard_user_denied", lambda p: True)
        monkeypatch.setattr(acl, "_icacls", lambda args: True)  # icacls "succeeds", probe still denies
        with pytest.raises(InstallError) as excinfo:
            acl.ensure_user_readable("git", entry, root)
        assert "standard user" in excinfo.value.cause
        assert "icacls" in excinfo.value.remedy


class TestRepairUserAcl:
    def test_reset_alone_heals(self, monkeypatch, tmp_path):
        args = []
        state = {"denied": True}

        def probe(path):
            return state["denied"]

        def icacls(call_args):
            args.append(call_args)
            state["denied"] = False  # /reset restored inheritance
            return True

        monkeypatch.setattr(acl, "_is_windows", lambda: True)
        monkeypatch.setattr(acl, "standard_user_denied", probe)
        monkeypatch.setattr(acl, "_icacls", icacls)
        assert acl.repair_user_acl(tmp_path) is True
        assert any("/reset" in a for a in args)
        assert not any("/grant" in a for a in args), "grant ran though /reset healed the tree"

    def test_grant_fallback_when_reset_insufficient(self, monkeypatch, tmp_path):
        args = []
        monkeypatch.setattr(acl, "_is_windows", lambda: True)
        monkeypatch.setattr(acl, "standard_user_denied", lambda p: True)
        monkeypatch.setattr(acl, "_icacls", lambda a: args.append(a) or True)
        monkeypatch.setattr(acl, "_account_sid", lambda: "S-1-5-21-1234")
        assert acl.repair_user_acl(tmp_path) is False  # probe still denies: honest failure
        grants = [a for a in args if "/grant" in a]
        assert grants and any("S-1-5-21-1234" in " ".join(a) for a in grants)

    def test_no_sid_no_grant(self, monkeypatch, tmp_path):
        args = []
        monkeypatch.setattr(acl, "_is_windows", lambda: True)
        monkeypatch.setattr(acl, "standard_user_denied", lambda p: True)
        monkeypatch.setattr(acl, "_icacls", lambda a: args.append(a) or True)
        monkeypatch.setattr(acl, "_account_sid", lambda: None)
        acl.repair_user_acl(tmp_path)
        assert not any("/grant" in a for a in args)


class TestStandardUserDenied:
    def test_false_off_windows(self, monkeypatch, tmp_path):
        monkeypatch.setattr(acl, "_is_windows", lambda: False)
        assert acl.standard_user_denied(tmp_path) is False

    def test_probe_failure_is_not_denial(self, monkeypatch, tmp_path):
        """A broken probe must never trigger an ACL rewrite."""
        monkeypatch.setattr(acl, "_is_windows", lambda: True)
        monkeypatch.setattr(acl, "_access_check_denied",
                            lambda p: (_ for _ in ()).throw(RuntimeError("probe broke")))
        assert acl.standard_user_denied(tmp_path) is False
