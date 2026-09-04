"""The composable ``platforms`` marker: spec evaluation + collection gating.

Behavior-contract tests for the gate introduced alongside the fixed
platforms("linux")/platforms("macos")/platforms("windows") trio: any-of semantics, negation,
POSIX grouping, arch filters, and the hard errors on unknown specs and
stray keyword arguments.
"""

from __future__ import annotations

import pytest

from tests.conftest import _host_matches_platforms


class TestSpecEvaluation:
    """Pure evaluation: takes the host as data, no host faking."""

    # (specs, host_platform, expected_ok)
    CASES = [
        (("linux",), "linux", True),
        (("linux",), "win32", False),
        (("macos",), "darwin", True),
        (("windows",), "win32", True),
        (("windows",), "linux", False),
        (("posix",), "linux", True),
        (("posix",), "darwin", True),
        (("posix",), "win32", False),
        (("not macos",), "linux", True),
        (("not macos",), "darwin", False),
        (("not windows",), "win32", False),
        (("not windows",), "linux", True),
        (("linux", "win32host-mismatch"), "win32", False),  # unknown spec never matches
        (("any",), "linux", True),
        ((), "linux", True),  # no specs = documentation form, matches all
    ]

    @pytest.mark.parametrize(("specs", "host", "expected"), CASES)
    def test_spec_matrix(self, specs, host, expected, monkeypatch):
        monkeypatch.setattr("tests.conftest.sys.platform", host)
        ok, _reason = _host_matches_platforms(specs)
        assert ok is expected

    def test_unknown_spec_is_reported_not_matched(self, monkeypatch):
        monkeypatch.setattr("tests.conftest.sys.platform", "linux")
        ok, reason = _host_matches_platforms(("amiga",))
        assert ok is False
        assert "unknown spec" in reason

    def test_negation_of_unknown_spec_is_rejected(self, monkeypatch):
        monkeypatch.setattr("tests.conftest.sys.platform", "linux")
        ok, reason = _host_matches_platforms(("not amiga",))
        assert ok is False
        assert "unknown spec" in reason

    def test_case_insensitive_specs(self, monkeypatch):
        monkeypatch.setattr("tests.conftest.sys.platform", "win32")
        ok, _ = _host_matches_platforms(("WINDOWS",))
        assert ok is True


class TestArchFilter:
    @pytest.mark.parametrize(
        ("arch", "machine", "negate", "expected"),
        [
            ("arm64", "arm64", False, True),
            ("arm64", "x86_64", False, False),
            ("aarch64", "arm64", False, True),  # alias
            ("arm64", "arm64", True, False),
            ("arm64", "x86_64", True, True),
        ],
    )
    def test_arch_matrix(self, arch, machine, negate, expected, monkeypatch):
        monkeypatch.setattr("tests.conftest.sys.platform", "win32")
        monkeypatch.setattr("tests.conftest._platform_machine", lambda: machine)
        ok, reason = _host_matches_platforms(("windows",), arch=arch, arch_negate=negate)
        assert ok is expected, reason

    def test_arch_reason_names_the_machine(self, monkeypatch):
        monkeypatch.setattr("tests.conftest.sys.platform", "win32")
        monkeypatch.setattr("tests.conftest._platform_machine", lambda: "x86_64")
        ok, reason = _host_matches_platforms(("windows",), arch="arm64")
        assert ok is False
        assert "x86_64" in reason


class TestAnyOfSemantics:
    def test_multiple_specs_are_any_of(self, monkeypatch):
        monkeypatch.setattr("tests.conftest.sys.platform", "darwin")
        ok, _ = _host_matches_platforms(("linux", "macos"))
        assert ok is True

    def test_first_matching_spec_wins_over_later_unknown(self, monkeypatch):
        # any-of: a matching spec satisfies the gate even if a later spec
        # is garbage — unknown specs only matter when nothing matched.
        monkeypatch.setattr("tests.conftest.sys.platform", "linux")
        ok, _ = _host_matches_platforms(("linux", "amiga"))
        assert ok is True


class TestCollectionGating:
    """The marker must actually skip/gate collected items on this host."""

    @pytest.mark.platforms("not " + __import__("sys").platform.split("_")[0])
    def test_never_runs_on_this_host_shape(self):
        # The spec is built to exclude whatever this host is (linux → "not
        # linux", win32 → "not windows"); if it RUNS the gate is broken.
        raise AssertionError("platforms() gate failed to skip this host")

    @pytest.mark.platforms("any")
    def test_any_spec_runs_everywhere(self):
        assert True

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="linux-host assertion; inverted on the linux lane below",
    )
    @pytest.mark.platforms("linux")
    def test_runs_on_linux(self):
        assert True


class TestHardErrors:
    def test_stray_kwarg_is_a_usage_error(self):
        # The gate raises UsageError (surfaced by pytest as a collection
        # error) for keyword arguments it does not understand — evaluated
        # directly because the raise happens inside the project conftest's
        # collection hook.
        import pytest as _pytest

        from tests.conftest import _platforms_gate_reason

        class _Item:
            nodeid = "tests/x.py::test_x"

            @staticmethod
            def iter_markers(name):
                yield _pytest.mark.platforms("linux", bogus=True).mark

        with _pytest.raises(_pytest.UsageError, match="unexpected keyword"):
            _platforms_gate_reason(_Item)


class TestMachineAliases:
    """_platform_machine normalizes the raw platform.machine() spellings."""

    @pytest.mark.parametrize(
        ("raw", "normalized"),
        [
            ("AMD64", "x86_64"),
            ("x86", "x86_64"),
            ("aarch64", "arm64"),
            ("arm64", "arm64"),
            ("x86_64", "x86_64"),
        ],
    )
    def test_alias_matrix(self, raw, normalized, monkeypatch):
        import platform as _platform

        monkeypatch.setattr(_platform, "machine", lambda: raw)
        from tests.conftest import _platform_machine

        assert _platform_machine() == normalized
