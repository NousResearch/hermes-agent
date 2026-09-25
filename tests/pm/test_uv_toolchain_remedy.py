"""A source build that cannot spawn its compiler names the tool and the fix (#122402).

A historical-takeover `uv sync` carrying `matrix` builds `python-olm` from
source; the managed interpreter's build configuration asked for `clang++` on a
host without one, and the surfaced error was only "uv sync exited 1" plus an
output tail. The classifier must recognize the missing-toolchain shape and
attach a remedy naming the missing executable and the CC/CXX override.
"""
from __future__ import annotations

from pm.environment import _missing_build_tool, classify_uv_failure
from pm.package import InstallError
from pm.workspace import ResolutionConflict, classify_uv_failure as workspace_classifier


# The reporter's real tail (issue #122402), trimmed to its load-bearing lines.
_REPORTED_OUTPUT = """\
building '_libolm' extension
clang++ -pthread -Wsign-compare -WUNreachable-code -fno-common -dynamic -DNDEBUG ...
error: [Errno 2] No such file or directory: 'clang++'
hint: `python-olm` (v3.2.16) was included because `hermes-agent[matrix]` (v0.0.0)
"""


def test_reported_missing_compiler_names_the_tool_and_the_override():
    from pm.environment import BuildFailure

    error = classify_uv_failure("sync", 1, _REPORTED_OUTPUT)
    assert isinstance(error, BuildFailure)
    assert error.cause.startswith("uv sync exited 1")
    assert "clang++" in error.remedy
    assert "CC=gcc CXX=g++" in error.remedy
    assert "clang++" in str(error)


def test_missing_toolchain_wins_over_the_generic_build_marker():
    from pm.environment import BuildFailure

    output = "the build backend returned an error\nerror: [Errno 2] No such file or directory: 'clang++'"
    error = classify_uv_failure("sync", 1, output)
    assert isinstance(error, BuildFailure)
    assert "clang++" in error.remedy


def test_build_failure_without_a_toolchain_gap_keeps_the_default_remedy():
    from pm.environment import BuildFailure

    error = classify_uv_failure("sync", 1, "the build backend returned an error\nFailed to build wheel")
    assert isinstance(error, BuildFailure)
    assert error.remedy == "retry, or run `hermes pm doctor`"


def test_missing_data_file_is_not_reported_as_a_missing_compiler():
    output = "error: [Errno 2] No such file or directory: 'libolm/olm.h'"
    error = classify_uv_failure("sync", 1, output)
    assert type(error) is InstallError
    assert error.remedy == "retry, or run `hermes pm doctor`"


def test_resolver_conflict_still_classifies_as_resolution_conflict():
    error = classify_uv_failure("sync", 1, "No solution found for python-olm")
    assert isinstance(error, ResolutionConflict)


def test_workspace_re_export_is_the_same_classifier():
    assert workspace_classifier is classify_uv_failure


def test_spawn_shapes_across_build_frontends():
    cases = {
        "error: command 'gcc' failed: No such file or directory": "gcc",
        "unable to execute 'clang++': No such file or directory": "clang++",
        "sh: 1: clang++: command not found": "clang++",
        "clang++: command not found": "clang++",
        "bash: line 1: cmake: not found": "cmake",
        "error: linker `cc` not found": "cc",
        "x86_64-linux-gnu-gcc: command not found": "x86_64-linux-gnu-gcc",
        r"error: [Errno 2] No such file or directory: 'C:\Build\clang.EXE'": "clang.EXE",
    }
    for output, expected in cases.items():
        assert _missing_build_tool(output) == expected, output


def test_non_tool_names_in_spawn_shapes_are_ignored():
    assert _missing_build_tool("error: [Errno 2] No such file or directory: 'libolm/olm.py'") is None
    assert _missing_build_tool("sh: 1: activate: not found") is None
    assert _missing_build_tool("nothing happened") is None
