"""Source builds under PM's pinned interpreter use compilers the host actually has."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
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


@pytest.mark.platforms("linux")
def test_staged_copy_of_pinned_interpreter_gets_gcc_defaults(tmp_path, monkeypatch):
    from pm.environment import managed_environment

    pinned = tmp_path / "cache-tools/python/bin/python3.14"
    pinned.parent.mkdir(parents=True)
    script = f"#!/bin/sh\necho '{json.dumps(_PBS_LINUX)}'\n"
    pinned.write_text(script, encoding="utf-8")
    pinned.chmod(0o755)
    staged_real = tmp_path / "payload/tools/python/bin/python3.14"
    staged_real.parent.mkdir(parents=True)
    shutil.copy2(pinned, staged_real)
    staged = staged_real.with_name("python3")
    staged.symlink_to("python3.14")
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (tmp_path / "uv", pinned))
    gcc_only = _executables(tmp_path / "gcc-only", "gcc", "g++", "ar")

    built = managed_environment(tmp_path / "staged", python=staged, env={"PATH": gcc_only},
                                cache=tmp_path).env
    assert {key: built.get(key) for key in _PBS_LINUX} == {
        "CC": "gcc -pthread", "CXX": "g++ -pthread",
        "LDSHARED": "gcc -pthread" + _LDFLAGS, "LDCXXSHARED": "g++ -pthread" + _LDFLAGS,
        "AR": "ar",
    }

    chosen = managed_environment(tmp_path / "chosen", python=staged,
                                 env={"PATH": gcc_only, "CC": "custom-cc"}, cache=tmp_path).env
    assert (chosen["CC"], chosen["CXX"]) == ("custom-cc", "g++ -pthread")
    assert "LDSHARED" not in chosen

    foreign = tmp_path / "foreign/bin/python3.14"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#" * len(script), encoding="utf-8")
    foreign.chmod(0o755)
    assert managed_environment(tmp_path / "foreign-env", python=foreign,
                               env={"PATH": gcc_only}, cache=tmp_path).env == {"PATH": gcc_only}


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


def test_build_failure_on_missing_re2_headers_names_the_packages():
    from pm.environment import BuildFailure, classify_uv_failure

    # The compiler ran; the header google-re2's sdist includes is not installed (Alpine without re2-dev).
    missing = ("error: Failed to build `google-re2==1.1.20251105`\n  ├─▶ The build backend returned an error\n"
               "  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)\n"
               "      _re2.cc:21:10: fatal error: re2/filtered_re2.h: No such file or directory\n"
               "      error: command '/usr/bin/g++' failed with exit code 1\n")
    failure = classify_uv_failure("pip", 1, missing)
    assert isinstance(failure, BuildFailure)
    assert all(package in failure.remedy for package in ("build-base", "re2-dev", "abseil-cpp-dev"))
    abseil = missing.replace("re2/filtered_re2.h", "absl/strings/string_view.h")
    assert "abseil-cpp-dev" in classify_uv_failure("pip", 1, abseil).remedy
    # Any other missing header is a source error this hint does not explain.
    other = missing.replace("re2/filtered_re2.h", "pybind11/buffer_info.h")
    assert "re2-dev" not in classify_uv_failure("pip", 1, other).remedy


def test_missing_re2_headers_under_clang_get_the_same_hint():
    from pm.environment import classify_uv_failure

    # PM keeps the recorded clang++ whenever it is on PATH; clang quotes the missing header.
    clang = ("error: Failed to build `google-re2==1.1.20251105`\n  ├─▶ The build backend returned an error\n"
             "  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)\n"
             "      _re2.cc:21:10: fatal error: 're2/filtered_re2.h' file not found\n"
             "      error: command '/usr/bin/clang++' failed with exit code 1\n")
    assert "re2-dev" in classify_uv_failure("pip", 1, clang).remedy
    abseil = clang.replace("re2/filtered_re2.h", "absl/strings/string_view.h")
    assert "abseil-cpp-dev" in classify_uv_failure("pip", 1, abseil).remedy


def test_re2_header_hint_names_the_packages_of_every_host_that_builds_the_sdist():
    from pm.environment import classify_uv_failure

    # The sdist is built on Alpine, and also on macOS < 13, old glibc and other CPUs: the hint
    # must not send those hosts to apk.
    missing = ("error: Failed to build `google-re2==1.1.20251105`\n  ├─▶ The build backend returned an error\n"
               "  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)\n"
               "      _re2.cc:21:10: fatal error: re2/filtered_re2.h: No such file or directory\n")
    remedy = classify_uv_failure("pip", 1, missing).remedy
    for host, command in (("Alpine", "apk add build-base re2-dev abseil-cpp-dev"),
                          ("Debian/Ubuntu", "sudo apt install build-essential libre2-dev libabsl-dev"),
                          ("Fedora", "sudo dnf install gcc-c++ re2-devel abseil-cpp-devel"),
                          ("Arch", "sudo pacman -S re2 abseil-cpp"),
                          ("macOS", "brew install re2 abseil")):
        assert f"{host}: `{command}`" in remedy


def test_google_re2_build_without_a_compiler_names_compiler_and_headers_at_once():
    from pm.environment import classify_uv_failure

    # A fresh Alpine host has no compiler and no headers: one failure must give the whole fix,
    # not a compiler hint that leads to a second failure on the headers.
    no_compiler = ("error: Failed to build `google-re2==1.1.20251105`\n  ├─▶ The build backend returned an error\n"
                   "  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)\n"
                   "      error: command 'clang++' failed: No such file or directory\n")
    remedy = classify_uv_failure("pip", 1, no_compiler).remedy
    assert "build-base re2-dev abseil-cpp-dev" in remedy
    # Any other package keeps the compiler-only hint, which now names Alpine as well.
    other = classify_uv_failure("pip", 1, no_compiler.replace("google-re2==1.1.20251105", "python-olm==3.2.16")).remedy
    assert "re2-dev" not in other and "Alpine: `apk add build-base`" in other
