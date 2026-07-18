from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL = REPO_ROOT / "scripts" / "install-termux.sh"
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _source_installer(
    body: str,
    *,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", "-c", f'source "$1"\n{body}', "bash", str(INSTALL)],
        env=merged_env,
        capture_output=True,
        text=True,
        check=check,
    )


def _python_version_shim(path: Path, version: str) -> None:
    _write_executable(
        path,
        "#!/bin/bash\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        '  code="$2"\n'
        "  exec python3 -c 'import sys; "
        'sys.version_info=tuple(map(int, sys.argv[2].split("."))); '
        'exec(sys.argv[1])\' "$code" "' + version + '"\n'
        "fi\n"
        f"printf 'Python {version}\\n'\n",
    )


def test_termux_options_are_parsed_as_explicit_installer_inputs(tmp_path: Path) -> None:
    requested_python = tmp_path / "python3.13"
    requested_python.touch()
    result = _source_installer(
        'parse_args --python "$REQUESTED_PYTHON" --android-api-level 28 '
        "--skip-setup --skip-browser\n"
        'printf \'%s|%s|%s|%s\\n\' "$PYTHON_BIN" "$ANDROID_BUILD_API" '
        '"$RUN_SETUP" "$SKIP_BROWSER"',
        env={"REQUESTED_PYTHON": str(requested_python)},
        check=True,
    )
    assert result.stdout.strip() == f"{requested_python}|28|false|true"


def test_python_support_check_rejects_314_and_accepts_313(tmp_path: Path) -> None:
    python_313 = tmp_path / "python3.13"
    python_314 = tmp_path / "python3.14"
    _python_version_shim(python_313, "3.13.9")
    _python_version_shim(python_314, "3.14.0")

    result = _source_installer(
        'if python_is_supported "$PYTHON_314"; then exit 90; fi\n'
        'python_is_supported "$PYTHON_313"',
        env={"PYTHON_313": str(python_313), "PYTHON_314": str(python_314)},
    )
    assert result.returncode == 0, result.stderr


def test_automatic_python_selection_requires_cp313_for_immutable_wheels(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python_312 = fake_bin / "python3.12"
    python_313 = fake_bin / "python3.13"
    _python_version_shim(python_312, "3.12.11")
    _python_version_shim(python_313, "3.13.14")

    result = _source_installer(
        "find_immutable_python",
        env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=True,
    )
    assert result.stdout.strip() == str(python_313)


def test_automatic_python_selection_does_not_use_312_as_immutable_target(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python_312 = fake_bin / "python3.12"
    _python_version_shim(python_312, "3.12.11")

    result = _source_installer(
        "find_immutable_python",
        env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )
    assert result.returncode != 0
    assert result.stdout == ""


def _pinned_python_toolchain(
    tmp_path: Path,
    *,
    checksum_status: int = 0,
    package_version: str = "3.13.14",
) -> tuple[Path, dict[str, str]]:
    fake_bin = tmp_path / "pinned-bin"
    fake_bin.mkdir()
    prefix = tmp_path / "prefix"
    prefix.mkdir()

    _write_executable(
        fake_bin / "curl",
        "#!/bin/bash\n"
        "set -e\n"
        "url=''\nout=''\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    -o) out="$2"; shift 2 ;;\n'
        "    --retry) shift 2 ;;\n"
        '    http*) url="$1"; shift ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'printf \'%s\\n\' "$url" > "$PINNED_URL_LOG"\n'
        ': > "$out"\n',
    )
    _write_executable(
        fake_bin / "sha256sum",
        '#!/bin/bash\ncat > "$PINNED_SHA_LOG"\nexit "$PINNED_CHECKSUM_STATUS"\n',
    )
    _write_executable(
        fake_bin / "dpkg",
        "#!/bin/bash\n"
        'if [ "${1:-}" = "--print-architecture" ]; then\n'
        "  printf 'aarch64\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
    )
    _write_executable(
        fake_bin / "dpkg-deb",
        "#!/bin/bash\n"
        "set -e\n"
        'case "${1:-}" in\n'
        "  -f)\n"
        '    case "${3:-}" in\n'
        "      Package) printf 'python\\n' ;;\n"
        "      Version) printf '%s\\n' \"$PINNED_PACKAGE_VERSION\" ;;\n"
        "      Architecture) printf 'aarch64\\n' ;;\n"
        "      *) exit 3 ;;\n"
        "    esac\n"
        "    ;;\n"
        "  -x)\n"
        '    root="$3$PREFIX"\n'
        '    mkdir -p "$root/bin" "$root/lib/python3.13" "$root/lib/pkgconfig" \
'
        '      "$root/share/man/man1"\n'
        "    cat > \"$root/bin/python3.13\" <<'PYTHON'\n"
        "#!/bin/bash\n"
        'if [ "${1:-}" = "-c" ]; then exit 0; fi\n'
        'if [ "${1:-}" = "--version" ]; then printf \'Python 3.13.14\\n\'; exit 0; fi\n'
        "exit 0\n"
        "PYTHON\n"
        '    chmod 755 "$root/bin/python3.13"\n'
        '    touch "$root/lib/python3.13/STDLIB"\n'
        '    touch "$root/lib/pkgconfig/python-3.13.pc" \
'
        '      "$root/lib/pkgconfig/python-3.13-embed.pc" \
'
        '      "$root/share/man/man1/python3.13.1.gz"\n'
        '    ln -s python-3.13.pc "$root/lib/pkgconfig/python3.pc"\n'
        '    ln -s python-3.13-embed.pc "$root/lib/pkgconfig/python3-embed.pc"\n'
        '    ln -s python3.13.1.gz "$root/share/man/man1/python.1.gz"\n'
        '    ln -s python3.13.1.gz "$root/share/man/man1/python3.1.gz"\n'
        "    for alias in python python3 python-config python3-config pip pip3 "
        "idle idle3 pydoc pydoc3 2to3; do\n"
        '      ln -s python3.13 "$root/bin/$alias"\n'
        "    done\n"
        "    ;;\n"
        "  *) exit 4 ;;\n"
        "esac\n",
    )

    env = {
        "PREFIX": str(prefix),
        "TMPDIR": str(tmp_path),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PINNED_URL_LOG": str(tmp_path / "pinned-url.log"),
        "PINNED_SHA_LOG": str(tmp_path / "pinned-sha.log"),
        "PINNED_CHECKSUM_STATUS": str(checksum_status),
        "PINNED_PACKAGE_VERSION": package_version,
    }
    return prefix, env


def test_pinned_python_fallback_downloads_and_verifies_exact_release(
    tmp_path: Path,
) -> None:
    prefix, env = _pinned_python_toolchain(tmp_path)
    result = _source_installer("install_pinned_python313", env=env)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pinned-url.log").read_text(encoding="utf-8").strip() == (
        "https://github.com/adybag14-cyber/termux-python/releases/download/"
        "termux-aarch64-20260719.9.1/python_3.13.14_aarch64.deb"
    )
    checksum_parts = (
        (tmp_path / "pinned-sha.log").read_text(encoding="utf-8").strip().split()
    )
    assert checksum_parts[0] == (
        "42376a2a47e50048cb7eca2d0f442fc1895fbca2aee2dee3d2fd82728ea1bd80"
    )
    assert Path(checksum_parts[1]).name == "python_3.13.14_aarch64.deb"
    assert (prefix / "bin" / "python3.13").is_file()
    assert (prefix / "lib" / "python3.13" / "STDLIB").is_file()
    assert (prefix / "lib" / "pkgconfig" / "python-3.13.pc").is_file()
    assert (prefix / "share" / "man" / "man1" / "python3.13.1.gz").is_file()
    for alias in (
        "python",
        "python3",
        "python-config",
        "python3-config",
        "pip",
        "pip3",
        "idle",
        "idle3",
        "pydoc",
        "pydoc3",
        "2to3",
    ):
        assert not (prefix / "bin" / alias).exists()
    for alias in (
        prefix / "lib" / "pkgconfig" / "python3.pc",
        prefix / "lib" / "pkgconfig" / "python3-embed.pc",
        prefix / "share" / "man" / "man1" / "python.1.gz",
        prefix / "share" / "man" / "man1" / "python3.1.gz",
    ):
        assert not alias.exists()


def test_pinned_python_fallback_rejects_checksum_mismatch(tmp_path: Path) -> None:
    prefix, env = _pinned_python_toolchain(tmp_path, checksum_status=1)
    result = _source_installer("install_pinned_python313", env=env)

    assert result.returncode != 0
    assert "checksum verification failed" in result.stderr
    assert not (prefix / "bin" / "python3.13").exists()


def test_pinned_python_fallback_rejects_different_package_version(
    tmp_path: Path,
) -> None:
    prefix, env = _pinned_python_toolchain(tmp_path, package_version="3.13.15")
    result = _source_installer("install_pinned_python313", env=env)

    assert result.returncode != 0
    assert "metadata did not match the locked release" in result.stderr
    assert not (prefix / "bin" / "python3.13").exists()


def test_failed_venv_creation_restores_previous_environment(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    old_venv = install_dir / "venv"
    old_venv.mkdir(parents=True)
    (old_venv / "keep.txt").write_text("previous environment", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "uv", "#!/bin/bash\nexit 42\n")

    result = _source_installer(
        'create_venv "$FAKE_PYTHON"',
        env={
            "HERMES_INSTALL_DIR": str(install_dir),
            "HERMES_HOME": str(tmp_path / "home"),
            "FAKE_PYTHON": str(tmp_path / "python3.13"),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )
    assert result.returncode != 0
    assert (install_dir / "venv" / "keep.txt").read_text(encoding="utf-8") == (
        "previous environment"
    )
    assert not list(install_dir.glob("venv.pre-native-termux-*"))


def test_launcher_clears_python_environment_and_routes_version(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix"
    install_dir = tmp_path / "hermes-agent"
    bin_dir = prefix / "bin"
    venv_bin = install_dir / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    venv_bin.mkdir(parents=True)
    (bin_dir / "bash").symlink_to("/bin/bash")
    launch_log = tmp_path / "launcher.log"

    _write_executable(
        venv_bin / "python",
        "#!/bin/bash\n"
        "printf 'python|PYTHONPATH=%s|PYTHONHOME=%s|UV_LINK_MODE=%s\\n' "
        '"${PYTHONPATH-<unset>}" "${PYTHONHOME-<unset>}" '
        '"${UV_LINK_MODE-<unset>}" >> "$LAUNCH_LOG"\n'
        "printf 'version-ok\\n'\n",
    )
    _write_executable(
        venv_bin / "hermes",
        "#!/bin/bash\n"
        "printf 'hermes|PYTHONPATH=%s|PYTHONHOME=%s|UV_LINK_MODE=%s|args=%s\\n' "
        '"${PYTHONPATH-<unset>}" "${PYTHONHOME-<unset>}" '
        '"${UV_LINK_MODE-<unset>}" "$*" >> "$LAUNCH_LOG"\n',
    )

    _source_installer(
        "install_launcher",
        env={
            "PREFIX": str(prefix),
            "HERMES_INSTALL_DIR": str(install_dir),
            "HERMES_HOME": str(tmp_path / "home"),
            "LAUNCH_LOG": str(launch_log),
        },
        check=True,
    )

    launcher_env = os.environ.copy()
    launcher_env.update({
        "PYTHONPATH": "/wrong/python/path",
        "PYTHONHOME": "/wrong/python/home",
        "LAUNCH_LOG": str(launch_log),
    })
    version = subprocess.run(
        [str(bin_dir / "hermes"), "version"],
        env=launcher_env,
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [str(bin_dir / "hermes"), "chat", "--resume"],
        env=launcher_env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert version.stdout.strip() == "version-ok"
    lines = launch_log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "python|PYTHONPATH=<unset>|PYTHONHOME=<unset>|UV_LINK_MODE=copy"
    )
    assert lines[1] == (
        "hermes|PYTHONPATH=<unset>|PYTHONHOME=<unset>|UV_LINK_MODE=copy|"
        "args=chat --resume"
    )


def test_termux_dispatches_to_local_installer_without_network(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(INSTALL_SH, scripts / "install.sh")
    child = scripts / "install-termux.sh"
    _write_executable(child, "#!/bin/bash\nprintf 'termux:%s\\n' \"$*\"\n")

    env = os.environ.copy()
    env.update({"TERMUX_VERSION": "0.118", "PREFIX": "/data/data/com.termux/files/usr"})
    result = subprocess.run(
        ["bash", str(scripts / "install.sh"), "--skip-setup"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "termux:--skip-setup"


def test_streamed_termux_dispatch_honors_requested_ref(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(INSTALL_SH, scripts / "install.sh")

    stub = tmp_path / "install-termux.stub.sh"
    _write_executable(stub, "#!/bin/bash\nprintf 'remote:%s\\n' \"$*\"\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    _write_executable(
        fake_curl,
        "#!/bin/bash\n"
        "url=''\nout=''\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    -o) out="$2"; shift 2 ;;\n'
        "    -*) shift ;;\n"
        '    *) url="$1"; shift ;;\n'
        "  esac\n"
        "done\n"
        'printf \'%s\\n\' "$url" > "$CURL_LOG"\n'
        'cp "$TERMUX_STUB_SOURCE" "$out"\n',
    )

    env = os.environ.copy()
    env.update({
        "TERMUX_VERSION": "0.118",
        "PREFIX": "/data/data/com.termux/files/usr",
        "PATH": f"{fake_bin}:{env['PATH']}",
        "TERMUX_STUB_SOURCE": str(stub),
        "CURL_LOG": str(tmp_path / "curl.log"),
    })
    result = subprocess.run(
        [
            "bash",
            str(scripts / "install.sh"),
            "--branch",
            "feature/native",
            "--skip-setup",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "remote:--branch feature/native --skip-setup"
    assert (
        (tmp_path / "curl.log")
        .read_text(encoding="utf-8")
        .strip()
        .endswith("/feature/native/scripts/install-termux.sh")
    )
