import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"

_BASH = shutil.which("bash")
# ``posix`` keeps the Linux main lane and the macOS lane running this file (it
# always has on Linux); ``windows`` is what makes the Windows OS lane import it
# — without a ``platforms`` marker the lane's ``-m "platforms and not
# integration"`` deselects every row, so the win32 fixture/dispatch branches
# this file carries were exercised on no host at all.
pytestmark = [
    pytest.mark.skipif(_BASH is None, reason="running setup-hermes.sh needs bash"),
    pytest.mark.platforms("posix", "windows"),
]


def test_setup_hermes_script_is_valid_shell():
    # as_posix: on Windows the argument reaches bash through its own command
    # line, where backslashes are escape characters — Git Bash wants forward
    # slashes.
    result = subprocess.run(["bash", "-n", SETUP_SCRIPT.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "home_kind",
    [
        "custom",
        "profile",
        "profile_trailing_slash",
        "custom_profile_root",
        "custom_profiles_segment",
        "profiles_ancestor_profile",
        "nested_under_default",
        "raw_alias",
        "raw_alias_dot",
        "home_var",
        "dotdot",
        "runtime_override",
    ],
)
def test_setup_stages_uv_into_pms_store_root(tmp_path, monkeypatch, home_kind):
    """setup-hermes.sh stages uv where pm.paths.store_root() resolves it (#101269).

    The script hardcoded ~/.hermes/tools while pm's own store follows
    HERMES_HOME, so with HERMES_HOME set the two never met: the script staged a
    sha256-verified uv PM could not see, and PM re-downloaded it.
    """
    bash = shutil.which("bash")
    assert bash, "setup-hermes.sh is a bash script"

    # Deliberately different from $HOME/.hermes, so a hardcoded default in the
    # script stages somewhere pm will never look.
    home = tmp_path / "custom-home" / ".hermes"
    if home_kind == "custom":
        hermes_home = home
    elif home_kind in {"profile", "profile_trailing_slash"}:
        hermes_home = home / "profiles" / "coder"
    elif home_kind == "custom_profile_root":
        hermes_home = tmp_path / "custom-root" / "profiles" / "coder"
    elif home_kind == "profiles_ancestor_profile":
        # A named profile under a custom home that itself has an unrelated ``profiles``
        # ancestor: only the immediate pair may fold, not the ancestor segment.
        hermes_home = tmp_path / "profiles" / "alice" / ".hermes" / "profiles" / "coder"
    elif home_kind == "nested_under_default":
        # Under the native home but not a named profile: pm folds anything under
        # the default home, so a rule that only looks for a ``profiles`` parent
        # stages uv somewhere pm never looks.
        hermes_home = tmp_path / "native-home" / ".hermes" / "foo" / "bar"
    elif home_kind in {"raw_alias", "raw_alias_dot", "home_var"}:
        # Raw environment strings a user can actually export, which building
        # through Path would normalize away before the shell ever sees them:
        # repeated separators, a trailing "." segment, an unexpanded variable.
        # The resolver must expand and normalize exactly like Path does BEFORE
        # applying the fold rule, or the profile-local slot is selected instead
        # of pm's machine root.
        hermes_home = f"{tmp_path}/custom-root/profiles//coder"
        if home_kind == "raw_alias_dot":
            hermes_home = f"{tmp_path}/custom-root/profiles/coder/."
        elif home_kind == "home_var":
            hermes_home = "${HOME}/.hermes/profiles/coder"
    elif home_kind == "dotdot":
        # Codex P1: ".." must not pass a lexical "under the default home"
        # check — pm decides containment on the RESOLVED path, so the shell
        # must too, or bootstrap state lands inside ~/.hermes while pm reads
        # the store under <native-home>/custom.
        hermes_home = tmp_path / "native-home" / ".hermes" / ".." / "custom"
    else:
        hermes_home = tmp_path / "profiles" / "alice" / ".hermes"
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    hermes_home_env = str(hermes_home)
    if home_kind == "profile_trailing_slash":
        hermes_home_env += os.sep
    monkeypatch.setenv("HERMES_HOME", hermes_home_env)
    if home_kind == "runtime_override":
        runtime_store = tmp_path / "runtime-store"
        monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime_store))
    else:
        runtime_store = None
        monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)

    from pm.paths import store_root

    store = Path(store_root())
    if home_kind in {"profile", "profile_trailing_slash"}:
        expected_root = home
    elif home_kind in {"custom_profile_root", "profiles_ancestor_profile"}:
        expected_root = hermes_home.parent.parent
    elif home_kind == "nested_under_default":
        expected_root = tmp_path / "native-home" / ".hermes"
    elif home_kind in {"raw_alias", "raw_alias_dot"}:
        expected_root = tmp_path / "custom-root"
    elif home_kind == "home_var":
        # ${HOME} expands to the native home, whose default folds to itself.
        expected_root = tmp_path / "native-home" / ".hermes"
    elif home_kind == "dotdot":
        # Verified against pm: containment is decided on the RESOLVED path
        # (../custom escapes, so nothing folds), but the returned root is the
        # LEXICAL form — ".." components stay in the store path verbatim. The
        # shell must match BOTH halves, which is what hermes_root_of does.
        expected_root = hermes_home
    else:
        expected_root = hermes_home
    # HERMES_RUNTIME_DIR overrides the store slot only; the uv cache and python
    # dir still follow the machine root.
    expected_store = runtime_store if runtime_store is not None else expected_root / "tools"
    assert store == expected_store, "pm must resolve its store under the machine-scoped Hermes root"

    lock = json.loads((REPO_ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))
    uv_version = lock["packages"]["uv"]["version"]
    # The store-entry name follows the target the script derives from uname:
    # Git Bash/MSYS on Windows reports win32, where the pinned artifact is
    # uv.exe — seeding the POSIX name would send the script to the (stubbed)
    # network and fail every row on a Windows host.
    if sys.platform == "win32":
        system, uv_filename = "win32", "uv.exe"
    elif sys.platform == "darwin":
        system, uv_filename = "darwin", "uv"
    else:
        system, uv_filename = "linux", "uv"
    machine = platform.machine()
    if system == "win32":
        # Same precedence as the script's own detection: the registry's machine
        # arch first, because an x64-emulated process on Windows-on-ARM reports
        # AMD64 while the staged artifact is ARM64.
        import winreg
        try:
            machine = winreg.QueryValueEx(
                winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                ),
                "PROCESSOR_ARCHITECTURE",
            )[0]
        except OSError:
            machine = os.environ.get("PROCESSOR_ARCHITECTURE", "") or machine
    arch = "arm64" if machine.lower() in {"arm64", "aarch64"} else "x64"
    uv = store / f"uv-{uv_version}-{system}-{arch}" / uv_filename
    uv.parent.mkdir(parents=True)
    record = tmp_path / f"uv-state-{home_kind}"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        '  --version) echo "uv 0.0.0-test"; exit 0 ;;\n'
        f'  python) printf "%s|%s|%s\\n" "$0" "$UV_CACHE_DIR" '
        f'"$UV_PYTHON_INSTALL_DIR" >> {shlex.quote(str(record))}; exit 1 ;;\n'
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    # A staging attempt must fail locally instead of reaching the network.
    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir()
    curl = stub_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 6\n", encoding="utf-8")
    curl.chmod(0o755)

    env = {**os.environ, "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(
        [bash, SETUP_SCRIPT.as_posix()], env=env, cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    assert "Staging pinned uv" not in output, output
    assert "pinned uv found" in output, output
    assert record.is_file(), "setup-hermes.sh never executed the pinned store uv"
    for line in record.read_text(encoding="utf-8").splitlines():
        invoked_uv, cache_dir, python_dir = line.split("|")
        assert Path(invoked_uv) == uv, line
        assert cache_dir == str(expected_root / "cache" / "uv-bootstrap"), line
        assert python_dir == str(expected_root / "cache" / "uv-python"), line
