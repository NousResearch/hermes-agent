"""Tests for gateway code-skew detection (stale-checkout guard).

Companion to ``tests/test_stale_utils_module_import.py``: that test proves the
crash; these prove the guard that turns it into a clear "restart the gateway"
message before a model switch can hit it.
"""

import pytest

from gateway import code_skew


@pytest.fixture(autouse=True)
def _reset_boot_fingerprint(monkeypatch):
    """Each test starts with no recorded boot fingerprint."""
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)


class TestDetectCodeSkew:
    def test_no_boot_fingerprint_means_no_skew(self, monkeypatch):
        # Nothing recorded (e.g. non-git install) -> never a false positive.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def456")
        assert code_skew.detect_code_skew() is None

    def test_unchanged_checkout_is_not_skew(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()
        assert code_skew.detect_code_skew() is None

    def test_drift_is_detected_with_short_revs(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()

        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def4567890123")
        # A runtime .py changed between the revs -> real skew.
        monkeypatch.setattr(code_skew, "_runtime_python_changed", lambda b, d: True)
        skew = code_skew.detect_code_skew()
        assert skew == ("abc1234567", "def4567890")

    def test_docs_only_drift_is_suppressed(self, monkeypatch):
        # SHA advanced but no runtime module changed (docs/skill/locale/test/YAML
        # only) -> must NOT refuse a model switch.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def4567890123")
        monkeypatch.setattr(code_skew, "_runtime_python_changed", lambda b, d: False)
        assert code_skew.detect_code_skew() is None

    def test_uncomputable_diff_falls_back_to_refuse(self, monkeypatch):
        # If we can't tell WHAT changed (git error, missing object), stay
        # conservative and report the skew -> refuse.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def4567890123")
        monkeypatch.setattr(code_skew, "_runtime_python_changed", lambda b, d: None)
        assert code_skew.detect_code_skew() == ("abc1234567", "def4567890")

    def test_unresolved_sha_falls_back_to_refuse(self, monkeypatch):
        # A ref whose object couldn't be read (unresolved) can't seed a diff;
        # since it drifted, refuse conservatively.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:unresolved")
        # _runtime_python_changed must NOT be consulted (no sha to diff); refuse.
        called = {"n": 0}
        def _boom(b, d):
            called["n"] += 1
            return False
        monkeypatch.setattr(code_skew, "_runtime_python_changed", _boom)
        skew = code_skew.detect_code_skew()
        assert skew == ("abc1234567", "unresolved")
        assert called["n"] == 0

    def test_same_sha_different_ref_is_not_skew(self, monkeypatch):
        # Same commit SHA, only the branch/ref label differs -> no code changed,
        # must NOT refuse (and must not even attempt a diff).
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/deploy:abc1234567890")
        called = {"n": 0}
        def _boom(b, d):
            called["n"] += 1
            return True
        monkeypatch.setattr(code_skew, "_runtime_python_changed", _boom)
        assert code_skew.detect_code_skew() is None
        assert called["n"] == 0

    def test_unreadable_current_rev_does_not_false_positive(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()

        monkeypatch.setattr(code_skew, "_fingerprint", lambda: None)
        assert code_skew.detect_code_skew() is None

    def test_record_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:first")
        code_skew.record_boot_fingerprint()
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:second")
        code_skew.record_boot_fingerprint()  # must not overwrite the boot snapshot
        assert code_skew._boot_fingerprint == "git:refs/heads/main:first"


class TestShort:
    def test_shortens_long_sha(self):
        assert code_skew._short("git:refs/heads/main:abcdef0123456789") == "abcdef0123"

    def test_keeps_unresolved_marker(self):
        assert code_skew._short("git:refs/heads/main:unresolved") == "unresolved"

    def test_passes_short_sha_through_untruncated(self):
        assert code_skew._short("git:HEAD:abc1234") == "abc1234"


class TestModelSwitchSkewGuard:
    def test_guard_returns_none_without_skew(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
        assert slash_commands._model_switch_skew_guard() is None

    def test_guard_message_names_revs_and_restart(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = slash_commands._model_switch_skew_guard()
        assert msg is not None
        assert "abc1234567" in msg
        assert "def4567890" in msg
        assert "hermes gateway restart" in msg


class TestIsRuntimePython:
    def test_plain_module_is_runtime(self):
        assert code_skew._is_runtime_python("gateway/code_skew.py") is True

    def test_nested_module_is_runtime(self):
        assert code_skew._is_runtime_python("hermes_cli/main.py") is True

    def test_test_file_is_not_runtime(self):
        assert code_skew._is_runtime_python("tests/test_code_skew.py") is False

    def test_docs_file_is_not_runtime(self):
        assert code_skew._is_runtime_python("docs/sync/whatever.py") is False

    def test_non_python_is_not_runtime(self):
        assert code_skew._is_runtime_python("README.md") is False
        assert code_skew._is_runtime_python("locale/en.json") is False
        assert code_skew._is_runtime_python("config.example.yaml") is False
        assert code_skew._is_runtime_python("skills/foo/SKILL.md") is False


class TestRuntimePythonChanged:
    """End-to-end against a throwaway git repo — proves the real diff logic,
    not just a monkeypatched stand-in."""

    def _git(self, cwd, *args):
        import subprocess
        subprocess.run(["git", "-C", str(cwd), *args], check=True,
                       capture_output=True, text=True)

    def _repo(self, tmp_path):
        import subprocess
        self._git(tmp_path, "init", "-q")
        self._git(tmp_path, "config", "user.email", "t@t.t")
        self._git(tmp_path, "config", "user.name", "t")
        return tmp_path

    def _commit(self, tmp_path, rel, content):
        import subprocess
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self._git(tmp_path, "add", "-A")
        self._git(tmp_path, "commit", "-q", "-m", f"touch {rel}")
        return subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()

    def test_runtime_py_change_detected(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path)
        a = self._commit(repo, "gateway/x.py", "x = 1\n")
        b = self._commit(repo, "gateway/x.py", "x = 2\n")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", repo)
        assert code_skew._runtime_python_changed(a, b) is True

    def test_docs_only_change_not_detected(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path)
        a = self._commit(repo, "README.md", "hi\n")
        b = self._commit(repo, "docs/guide.md", "more\n")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", repo)
        assert code_skew._runtime_python_changed(a, b) is False

    def test_test_only_change_not_detected(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path)
        a = self._commit(repo, "tests/test_a.py", "def test_a(): pass\n")
        b = self._commit(repo, "tests/test_a.py", "def test_a(): assert True\n")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", repo)
        assert code_skew._runtime_python_changed(a, b) is False

    def test_mixed_change_detected(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path)
        a = self._commit(repo, "docs/g.md", "a\n")
        # second commit touches BOTH a doc and a runtime module
        (repo / "docs" / "g.md").write_text("b\n")
        (repo / "gateway").mkdir(exist_ok=True)
        (repo / "gateway" / "y.py").write_text("y = 1\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-q", "-m", "mixed")
        import subprocess
        b = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", repo)
        assert code_skew._runtime_python_changed(a, b) is True

    def test_bad_sha_returns_none(self, tmp_path, monkeypatch):
        repo = self._repo(tmp_path)
        self._commit(repo, "gateway/x.py", "x = 1\n")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", repo)
        assert code_skew._runtime_python_changed("deadbeef", "cafebabe") is None
