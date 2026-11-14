"""Source builds under PM's pinned interpreter use compilers the host actually has."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

# What python-build-standalone 3.14 records on x86_64 Linux (sysconfig.get_config_var).
_LDFLAGS = " -shared  -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -Wl,--build-id=sha1   -Wl,--exclude-libs,ALL"
_PBS_LINUX = {
    "CC": "clang -pthread", "CXX": "clang++ -pthread",
    "LDSHARED": "clang -pthread" + _LDFLAGS, "LDCXXSHARED": "clang++ -pthread" + _LDFLAGS,
    "AR": "/tools/llvm/bin/llvm-ar",
}


def _executables(directory: Path, *names: str) -> str:
    directory.mkdir()
    for name in names:
        tool = directory / name
        tool.write_text("#!/bin/sh\n", encoding="utf-8")
        tool.chmod(0o755)
    return str(directory)


@pytest.mark.platforms("linux")
def test_pinned_interpreter_builds_with_gcc_when_its_clang_is_missing(tmp_path, monkeypatch):
    from pm.environment import managed_environment

    pinned = tmp_path / "pinned-python"
    pinned.write_text(f"#!/bin/sh\necho '{json.dumps(_PBS_LINUX)}'\n", encoding="utf-8")
    pinned.chmod(0o755)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (tmp_path / "uv", pinned))
    gcc_only = _executables(tmp_path / "gcc-only", "gcc", "g++", "ar")

    built = managed_environment(tmp_path / "gcc", env={"PATH": gcc_only}, cache=tmp_path).env
    assert {key: built.get(key) for key in _PBS_LINUX} == {
        "CC": "gcc -pthread", "CXX": "g++ -pthread",
        "LDSHARED": "gcc -pthread" + _LDFLAGS, "LDCXXSHARED": "g++ -pthread" + _LDFLAGS,
        "AR": "ar",
    }
    # A compiler the user chose wins, and so does the linker setuptools derives from it.
    chosen = managed_environment(tmp_path / "chosen", env={"PATH": gcc_only, "CXX": "clang++"},
                                 cache=tmp_path).env
    assert (chosen["CXX"], chosen["CC"]) == ("clang++", "gcc -pthread")
    assert "LDCXXSHARED" not in chosen
    # A host with the recorded compilers keeps the interpreter's own toolchain.
    with_clang = _executables(tmp_path / "with-clang", "clang", "clang++", "gcc", "g++")
    kept = managed_environment(tmp_path / "kept", env={"PATH": with_clang}, cache=tmp_path).env
    assert not {"CC", "CXX", "LDSHARED", "LDCXXSHARED"} & kept.keys()
    # Any other interpreter (Nix, a user's own) keeps its sysconfig untouched.
    other = managed_environment(tmp_path / "other", python=Path(sys.executable),
                                env={"PATH": gcc_only}, cache=tmp_path).env
    assert other == {"PATH": gcc_only}


def test_build_failure_without_any_compiler_names_the_install(tmp_path):
    from pm.environment import BuildFailure, classify_uv_failure

    # uv's real tail when neither clang++ nor g++ exists (python-olm 3.2.16, Ubuntu 24.04).
    missing = ("error: Failed to build `python-olm==3.2.16`\n  ├─▶ The build backend returned an error\n"
               "  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)\n"
               "      error: [Errno 2] No such file or directory: 'clang++'\n")
    failure = classify_uv_failure("pip", 1, missing)
    assert isinstance(failure, BuildFailure)
    assert "build-essential" in failure.remedy
    other = classify_uv_failure("pip", 1, missing.replace("'clang++'", "'libolm/include/olm.h'"))
    assert "build-essential" not in other.remedy
    absent = missing.replace("[Errno 2] No such file or directory: 'clang++'",
                             "command 'clang++' failed: No such file or directory")
    assert "build-essential" in classify_uv_failure("pip", 1, absent).remedy
    ran = missing.replace("[Errno 2] No such file or directory: 'clang++'",
                          "command '/usr/bin/gcc' failed with exit code 1")
    assert "build-essential" not in classify_uv_failure("pip", 1, ran).remedy
