"""Profile-mode ``HOME`` must not disarm the live ``state.db`` guard.

A dispatched worker has ``HOME`` remapped to ``<root>/profiles/<name>/home``,
so a deny-list built from ``expanduser("~")`` misses the real root. These
tests drive the real guard against paths under a fake root inside
``tmp_path``; nothing here touches a real database. Background in PR #101995.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_state


def _make_profile_layout(tmp_path: Path) -> tuple[Path, Path]:
    """Build ``<root>/profiles/scotty/home`` and return ``(root, home)``.

    Mirrors the real on-disk shape the terminal tool creates, so the
    profile-mode branch is exercised against a directory that actually
    exists rather than a synthetic string.
    """
    root = tmp_path / "fake-hermes-root" / ".hermes"
    profile_home = root / "profiles" / "scotty" / "home"
    profile_home.mkdir(parents=True, exist_ok=True)
    return root, profile_home


class TestProfileModeHomeRootResolution:
    """AC1: the true root is resolved when HOME is profile-remapped."""

    def test_profile_mode_home_resolves_shared_root(self, tmp_path, monkeypatch):
        """``<root>/profiles/<name>/home`` must yield ``<root>``.

        This is the whole defect in one assertion: before the fix the only
        candidate was ``<root>/profiles/<name>/home/.hermes``.
        """
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(profile_home))

        roots = hermes_state._real_platform_state_roots()

        assert root.resolve() in roots, (
            f"profile-mode HOME {profile_home} must resolve the shared root "
            f"{root}; got {roots}"
        )

    def test_profile_mode_keeps_expanduser_candidate_too(
        self, tmp_path, monkeypatch
    ):
        """Swapping expanduser for the profile branch would reopen plain HOME."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(profile_home))

        roots = hermes_state._real_platform_state_roots()

        assert hermes_state._hermes_root_for_home(profile_home).resolve() in roots

    def test_hermes_real_home_is_consulted(self, tmp_path, monkeypatch):
        """``HERMES_REAL_HOME`` is the most reliable in-worker signal."""
        operator_home = tmp_path / "operator"
        operator_home.mkdir()
        monkeypatch.setenv("HERMES_REAL_HOME", str(operator_home))
        monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))

        roots = hermes_state._real_platform_state_roots()

        expected = hermes_state._hermes_root_for_home(operator_home).resolve()
        assert expected in roots

    def test_real_home_ranks_ahead_of_expanduser(self, tmp_path, monkeypatch):
        """Trust order: ``HERMES_REAL_HOME`` first.

        ``_real_platform_state_root()`` (singular, still used by three test
        modules as "the root the guard denies") returns ``roots[0]``, so the
        ordering is load-bearing, not cosmetic.
        """
        operator_home = tmp_path / "operator"
        operator_home.mkdir()
        monkeypatch.setenv("HERMES_REAL_HOME", str(operator_home))
        monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))

        expected = hermes_state._hermes_root_for_home(operator_home).resolve()
        assert hermes_state._real_platform_state_root() == expected

    def test_singular_shim_still_returns_a_root_on_a_plain_home(
        self, tmp_path, monkeypatch
    ):
        """No behaviour change on an ordinary developer machine."""
        plain_home = tmp_path / "devhome"
        plain_home.mkdir()
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(plain_home))

        expected = hermes_state._hermes_root_for_home(plain_home).resolve()
        assert hermes_state._real_platform_state_root() == expected

    def test_roots_are_deduplicated(self, tmp_path, monkeypatch):
        """Same home via both signals must not produce a duplicate entry."""
        plain_home = tmp_path / "devhome"
        plain_home.mkdir()
        monkeypatch.setenv("HERMES_REAL_HOME", str(plain_home))
        monkeypatch.setenv("HOME", str(plain_home))

        roots = hermes_state._real_platform_state_roots()
        assert len(roots) == len(set(roots))


class TestProfileModeHomeGuardRefuses:
    """AC2: the guard must refuse live DBs under a profile-mode HOME."""

    def test_guard_refuses_root_state_db(self, tmp_path, monkeypatch):
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv(hermes_state._STATE_DB_GUARD_BYPASS_ENV, raising=False)
        monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_BYPASS", False)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")

        with pytest.raises(RuntimeError, match="live-system guard"):
            hermes_state._ensure_test_isolation(root / "state.db")

    def test_guard_refuses_profile_state_db(self, tmp_path, monkeypatch):
        """A sibling of the remapped HOME, not an ancestor."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv(hermes_state._STATE_DB_GUARD_BYPASS_ENV, raising=False)
        monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_BYPASS", False)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")

        with pytest.raises(RuntimeError, match="live-system guard"):
            hermes_state._ensure_test_isolation(
                root / "profiles" / "scotty" / "state.db"
            )

    def test_guard_refuses_another_profiles_state_db(self, tmp_path, monkeypatch):
        """Profile isolation is not a licence to clobber a SIBLING profile."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv(hermes_state._STATE_DB_GUARD_BYPASS_ENV, raising=False)
        monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_BYPASS", False)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")

        with pytest.raises(RuntimeError, match="live-system guard"):
            hermes_state._ensure_test_isolation(
                root / "profiles" / "hemmer" / "state.db"
            )

    def test_guard_refuses_via_hermes_real_home_alone(self, tmp_path, monkeypatch):
        """Even with HOME pointing somewhere harmless, HERMES_REAL_HOME denies.

        This is the shape of a child process that rebuilt its environment
        and lost the ``HOME`` remap but kept the exported real home.
        """
        operator_home = tmp_path / "operator"
        operator_home.mkdir()
        real_root = hermes_state._hermes_root_for_home(operator_home)
        monkeypatch.setenv("HERMES_REAL_HOME", str(operator_home))
        monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
        monkeypatch.delenv(hermes_state._STATE_DB_GUARD_BYPASS_ENV, raising=False)
        monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_BYPASS", False)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")

        with pytest.raises(RuntimeError, match="live-system guard"):
            hermes_state._ensure_test_isolation(real_root / "state.db")

    def test_sessiondb_construction_is_refused_under_profile_home(
        self, tmp_path, monkeypatch
    ):
        """End-to-end at the real chokepoint, not just the helper.

        ``SessionDB.__init__`` is what every caller goes through; a pin on
        the helper alone would not catch the guard being unwired from it.
        """
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv(hermes_state._STATE_DB_GUARD_BYPASS_ENV, raising=False)
        monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_BYPASS", False)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")

        with pytest.raises(RuntimeError, match="live-system guard"):
            hermes_state.SessionDB(db_path=root / "state.db")

        assert not (root / "state.db").exists(), (
            "guard must refuse BEFORE any file is created"
        )


class TestNoFalsePositives:
    """AC3: hermetic tests and real runs must be unaffected."""

    def test_hermetic_tmp_db_still_allowed_under_profile_home(
        self, tmp_path, monkeypatch
    ):
        """A tmp DB outside the fake root must still open."""
        _root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")

        db = hermes_state.SessionDB(db_path=tmp_path / "hermetic" / "state.db")
        try:
            db.create_session("t70246eb8-hermetic", "cli")
            assert db.get_session("t70246eb8-hermetic") is not None
        finally:
            db.close()

    def test_workspace_db_under_root_is_not_production(self, tmp_path, monkeypatch):
        """Worker workspaces under the root must stay writable."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")

        scratch = root / "kanban" / "workspaces" / "t_deadbeef" / "state.db"
        hermes_state._ensure_test_isolation(scratch)  # must not raise

    def test_naive_expanduser_root_is_still_denied(self, tmp_path, monkeypatch):
        """The old resolver's root stays denied, since the fix only adds roots."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv(hermes_state._STATE_DB_GUARD_BYPASS_ENV, raising=False)
        monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_BYPASS", False)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")

        with pytest.raises(RuntimeError, match="live-system guard"):
            hermes_state._ensure_test_isolation(
                root / "profiles" / "scotty" / "home" / ".hermes" / "state.db"
            )

    def test_guard_is_dormant_outside_a_test_context(self, tmp_path, monkeypatch):
        """Widening the deny-list must not arm the guard in production."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setattr(hermes_state, "_in_test_context", lambda: False)

        hermes_state._ensure_test_isolation(root / "state.db")  # must not raise

    def test_bypass_env_still_wins(self, tmp_path, monkeypatch):
        """The sanctioned escape hatch must survive the widened deny-list."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "profile_mode::test (call)")
        monkeypatch.setenv(hermes_state._STATE_DB_GUARD_BYPASS_ENV, "1")

        hermes_state._ensure_test_isolation(root / "state.db")  # must not raise

    def test_roots_never_raise_when_expanduser_explodes(self, monkeypatch):
        """Resolution must degrade to a list, never propagate.

        The guard must not be the reason a real run fails. Drives the seam
        that can actually throw (``os.path.expanduser``) rather than trying
        to plant an unsettable value in ``os.environ``.
        """

        def _boom(_p):
            raise OSError("no passwd entry")

        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setattr(hermes_state.os.path, "expanduser", _boom)

        assert hermes_state._real_platform_state_roots() == []
        assert hermes_state._real_platform_state_root() is None

    def test_unresolvable_real_home_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        """A junk ``HERMES_REAL_HOME`` must not sink the other candidates."""
        plain_home = tmp_path / "devhome"
        plain_home.mkdir()
        monkeypatch.setenv("HERMES_REAL_HOME", "   ")
        monkeypatch.setenv("HOME", str(plain_home))

        roots = hermes_state._real_platform_state_roots()
        assert hermes_state._hermes_root_for_home(plain_home).resolve() in roots


class TestConftestProductionHomeGate:
    """The same defect one layer up, in ``tests/conftest.py``.

    ``_hermes_home_points_at_production`` decides whether pytest replaces a
    pre-set ``HERMES_HOME`` with a sandbox. Resolving the real root through
    ``Path.home()`` let a production ``HERMES_HOME`` count as custom, so the
    sandbox never applied and collection-time imports froze production
    paths. That is the #82770 escape vector.

    Measured on the unfixed tree with ``HOME`` profile-remapped: all three
    of ``<root>``, ``<root>/profiles/scotty`` and ``<root>/profiles/hemmer``
    returned ``False`` (= not production = do not sandbox).

    The function is read out of ``conftest.py`` by AST rather than imported:
    importing the module would re-run its import-time sandbox setup and
    mutate this process's environment.
    """

    @staticmethod
    def _load_gate():
        import ast

        conftest = Path(__file__).resolve().parents[1] / "conftest.py"
        src = conftest.read_text()
        wanted = (
            "_real_hermes_roots_for_gate",
            "_hermes_home_points_at_production",
        )
        found = {}
        for n in ast.parse(src).body:
            if isinstance(n, ast.FunctionDef) and n.name in wanted:
                seg = ast.get_source_segment(src, n)
                if seg:
                    found[n.name] = seg
        assert "_hermes_home_points_at_production" in found, (
            "conftest no longer defines the production-HOME gate"
        )
        ns = {"Path": Path, "os": os, "sys": sys}
        for name in wanted:
            if name in found:
                exec(compile(found[name], "<conftest>", "exec"), ns)
        return ns["_hermes_home_points_at_production"]

    def test_production_root_is_sandboxed_under_profile_home(
        self, tmp_path, monkeypatch
    ):
        """A pre-set HERMES_HOME of ``<root>`` must be replaced, not honored."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(profile_home))

        gate = self._load_gate()
        assert gate(str(root)) is True

    def test_production_profile_root_is_sandboxed_under_profile_home(
        self, tmp_path, monkeypatch
    ):
        """``<root>/profiles/<name>`` is production too."""
        root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(profile_home))

        gate = self._load_gate()
        assert gate(str(root / "profiles" / "scotty")) is True
        assert gate(str(root / "profiles" / "hemmer")) is True

    def test_gate_honours_hermes_real_home(self, tmp_path, monkeypatch):
        """``HERMES_REAL_HOME`` must reach the gate as well as the guard."""
        operator_home = tmp_path / "operator"
        operator_home.mkdir()
        monkeypatch.setenv("HERMES_REAL_HOME", str(operator_home))
        monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))

        gate = self._load_gate()
        real_root = hermes_state._hermes_root_for_home(operator_home)
        assert gate(str(real_root)) is True

    def test_genuinely_custom_home_is_still_honoured(self, tmp_path, monkeypatch):
        """A custom HERMES_HOME must survive, or Docker installs break."""
        _root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(profile_home))

        gate = self._load_gate()
        assert gate(str(tmp_path / "custom-hermes-home")) is False

    def test_empty_home_fails_safe(self, tmp_path, monkeypatch):
        """Unset HERMES_HOME must still be treated as production."""
        _root, profile_home = _make_profile_layout(tmp_path)
        monkeypatch.setenv("HOME", str(profile_home))

        gate = self._load_gate()
        assert gate("") is True


class TestRealWorkerProcess:
    """AC1+AC2 end-to-end in a child process with a genuine profile HOME."""

    def test_child_with_profile_mode_home_refuses_root_state_db(self, tmp_path):
        """Spawn a child shaped exactly like a dispatched worker.

        In-process ``monkeypatch.setenv("HOME", ...)`` proves the logic; a
        real child proves the logic survives a genuine process boundary,
        which is where the original #82770 leak actually happened.
        """
        root, profile_home = _make_profile_layout(tmp_path)
        repo_root = Path(hermes_state.__file__).resolve().parent

        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("PYTEST_")
            and k
            not in (
                "HERMES_HOME",
                "HERMES_REAL_HOME",
                "HERMES_TEST_ISOLATION",
                "PYTHONPATH",
                hermes_state._STATE_DB_GUARD_BYPASS_ENV,
            )
        }
        env["HOME"] = str(profile_home)
        env["PYTHONPATH"] = str(repo_root)
        env["PYTEST_CURRENT_TEST"] = "tests/fake.py::test_child (call)"

        code = (
            "import sys\n"
            "import hermes_state as hs\n"
            f"root = {str(root)!r}\n"
            "from pathlib import Path\n"
            "verdicts = []\n"
            "for p in (Path(root) / 'state.db',\n"
            "          Path(root) / 'profiles' / 'scotty' / 'state.db'):\n"
            "    try:\n"
            "        hs._ensure_test_isolation(p)\n"
            "        verdicts.append('ALLOWED')\n"
            "    except RuntimeError:\n"
            "        verdicts.append('REFUSED')\n"
            "print(','.join(verdicts))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        verdict = proc.stdout.strip().splitlines()[-1]
        assert verdict == "REFUSED,REFUSED", (
            "a profile-mode-HOME child must refuse BOTH the root state.db "
            f"and the profile state.db; got {verdict!r}"
        )
