"""Child-scoped desktop build tools, independent of the invoking Hermes install.

This module is stdlib-only until preparation runs inside the isolated child.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid


_INSTALL_ENV = {
    "HERMES_INSTALL_ROOT", "HERMES_PAYLOAD_ROOT", "HERMES_PAYLOAD_TAG", "HERMES_BUILD_COMMIT",
    "HERMES_SITE", "HERMES_PYTHON", "HERMES_NODE", "HERMES_PROFILE", "HERMES_REAL_HOME",
    "HERMES_DATA_DIR_SUFFIX", "HERMES_BUNDLED_SKILLS", "HERMES_OPTIONAL_SKILLS",
    "HERMES_BUNDLED_PLUGINS", "HERMES_BUNDLED_LOCALES", "HERMES_OPTIONAL_MCPS",
    "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX",
}


def bootstrap_environment(source: Path, work: Path, cache: Path,
                          env: Mapping[str, str]) -> dict[str, str]:
    """Return launch state; never import PM, mutate the process, or write files."""
    source, work, cache = source.resolve(), work.resolve(), cache.resolve()
    owned = {"HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "HERMES_HOME", "HERMES_RUNTIME_DIR",
             "HERMES_PYTHON_SRC_ROOT", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
             "UV_CACHE_DIR", "NPM_CONFIG_CACHE", "NPM_EXECPATH", "NPM_NODE_EXECPATH",
             "NPM_CONFIG_USERCONFIG", "NPM_CONFIG_GLOBALCONFIG", "NPM_CONFIG_PREFIX",
             "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE", "CARGO_HOME", "RUSTUP_HOME"}
    result = {key: value for key, value in env.items() if key.upper() not in _INSTALL_ENV | owned}
    canonical = {key.upper(): value for key, value in env.items()}
    # Rustup must still find the runner's installed toolchain after HOME moves.
    home_key = "USERPROFILE" if os.name == "nt" else "HOME"
    original_home = Path(canonical.get("HERMES_REAL_HOME") or canonical.get(home_key) or Path.home())
    for key, directory in (("CARGO_HOME", ".cargo"), ("RUSTUP_HOME", ".rustup")):
        result[key] = canonical.get(key) or str(original_home / directory)
    home = work / "home"
    result.update({
        "HOME": str(home), "USERPROFILE": str(home),
        "LOCALAPPDATA": str(home / "AppData/Local"),
        "APPDATA": str(home / "AppData/Roaming"),
        "HERMES_HOME": str(work / "hermes-home"),
        "HERMES_RUNTIME_DIR": str(cache / "tools"),
        "HERMES_PYTHON_SRC_ROOT": str(source),
        "XDG_CONFIG_HOME": str(home / "config"), "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_DATA_HOME": str(home / "data"), "XDG_STATE_HOME": str(home / "state"),
        "UV_CACHE_DIR": str(cache / "python/runtime"),
        "npm_config_cache": str(cache / "npm"),
        "npm_config_userconfig": os.devnull,
        "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
    })
    return result


def run_preparation(source: Path, work: Path, cache: Path, request_file: Path,
                    *, worker: Path | None = None) -> int:
    """Bootstrap PM after process isolation, then run the preparation worker.

    The request contains paths/selection only; credentials stay in the inherited
    child environment. Return the worker's exit status without hiding failures.
    """
    source, work, cache = source.resolve(), work.resolve(), cache.resolve()
    environment = bootstrap_environment(source, work, cache, os.environ)
    Path(environment["HOME"]).mkdir(parents=True, exist_ok=True)
    launcher = (
        "import subprocess, sys; from pathlib import Path; "
        "source, cache, request, entry, worker = map(Path, sys.argv[1:]); "
        "sys.path.insert(0, str(source)); from pm.runtime import runtime_command; "
        "command = runtime_command(entry, "
        "[str(worker), '--request', str(request), '--worker'], "
        "cache=cache / 'python/runtime'); "
        "sys.exit(subprocess.run(command, stdin=subprocess.DEVNULL).returncode)"
    )
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", launcher,
         str(source), str(cache), str(request_file.resolve()), str(Path(__file__).resolve()),
         str(worker or source / "scripts/bundles/desktop_prepare.py")],
        cwd=source, env=environment, stdin=subprocess.DEVNULL,
    ).returncode


def prepare_tools(source: Path, work: Path, cache: Path,
                  env: Mapping[str, str]) -> tuple[Path, Path, dict[str, str]]:
    """Run only in the bootstrapped worker; return native Python, Node and env.

    PM owns pin selection and verification. The caller passes the resulting env
    to every build child, including its explicit native-wheel UV_CACHE_DIR.
    """
    import pm
    from pm.paths import store_root

    source, work, cache = source.resolve(), work.resolve(), cache.resolve()
    if store_root() != cache / "tools":
        raise ValueError("desktop tools require the isolated preparation worker")
    prepared = dict(env)
    if pm.current_target() == "win32-arm64":
        from scripts.build.windows_deps import prepare_windows_environment

        prepared = prepare_windows_environment(source=source, state=cache / "native/prerequisites", env=prepared)
    if pm.current_target().startswith("darwin"):
        # python-build-standalone's sysconfig names its absent build-host tools.
        prepared.setdefault("AR", "/usr/bin/ar")
        prepared.setdefault("CC", "clang")
    for name in ("uv", "npm"):
        pm.ensure(name, explicit=True)
    binaries = {}
    for name in ("python", "node"):
        installed = pm.installed_package(name)
        if installed is None or installed.binary is None or not installed.binary.is_file():
            raise FileNotFoundError(f"PM did not prepare a native {name} executable")
        binaries[name] = installed.binary
    prepared = pm.env_for("python", "npm", base_env=prepared)
    prepared.update({"HERMES_PYTHON": str(binaries["python"]), "HERMES_NODE": str(binaries["node"]),
                     "UV_CACHE_DIR": str(native_cache_path(cache, prepared))})
    return binaries["python"], binaries["node"], prepared


def native_cache_path(cache: Path, env: Mapping[str, str]) -> Path:
    """Partition built wheels by observed native inputs, not just the OS label.

    Missing tool/SDK identity deliberately gets a fresh partition. No inherited
    environment or credential-bearing Cargo configuration is written to cache.
    """
    from pm.store import current_target

    target = current_target()
    keys = {"CC", "CXX", "AR", "CFLAGS", "CXXFLAGS", "LDFLAGS", "RUSTFLAGS", "RUSTUP_TOOLCHAIN",
            "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "WindowsSDKVersion", "VCToolsVersion",
            "WindowsSdkDir", "INCLUDE", "LIB", "LIBPATH", "OPENSSL_DIR", "OPENSSL_LIB_DIR",
            "OPENSSL_INCLUDE_DIR", "OPENSSL_STATIC"}
    selected = {key: value for key, value in env.items() if key in keys or key.startswith("CC_")}
    identity = {"recipe": 1, "target": target, "host": platform.platform(),
                "libc": platform.libc_ver(), "inputs": selected, "probes": [], "openssl": {}}
    commands = [["rustc", "-vV"]]
    if target.startswith("win32"):
        commands.append(["cl", "/Bv"])
        complete = bool(env.get("WindowsSDKVersion") and env.get("VCToolsVersion"))
    else:
        commands.extend([[env.get("CC") or "cc", "--version"], [env.get("CXX") or "c++", "--version"]])
        complete = True
    commands.extend([[value, "--version"] for key, value in selected.items() if key.startswith("CC_")])
    if target.startswith("darwin"):
        commands.extend([["xcrun", "--show-sdk-path"], ["xcrun", "--show-sdk-build-version"]])
    openssl = env.get("OPENSSL_DIR")
    if openssl or env.get("OPENSSL_LIB_DIR") or env.get("OPENSSL_INCLUDE_DIR"):
        roots = {"include": Path(env["OPENSSL_INCLUDE_DIR"]) if env.get("OPENSSL_INCLUDE_DIR")
                 else Path(openssl or "") / "include",
                 "lib": Path(env["OPENSSL_LIB_DIR"]) if env.get("OPENSSL_LIB_DIR")
                 else Path(openssl or "") / "lib"}
        for kind, root in roots.items():
            files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
            complete = complete and bool(files)
            for path in files:
                with path.open("rb") as stream:
                    identity["openssl"][f"{kind}/{path.relative_to(root).as_posix()}"] = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        commands.append(["openssl", "version", "-a"])
        if target == "win32-arm64":
            complete = False  # The provider must identify the linked static libraries.
    for command in commands:
        try:
            result = subprocess.run(command, env=dict(env), capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=30, stdin=subprocess.DEVNULL)
            output = result.stdout + result.stderr
            # cl /Bv reports its identity and exits 2 when no source is supplied.
            valid = result.returncode in ((0, 2) if command == ["cl", "/Bv"] else (0,)) and bool(output.strip())
            identity["probes"].append([command, output])
            complete = complete and valid
        except (OSError, subprocess.TimeoutExpired):
            complete = False
    # Build-owned locations move across runners; their bytes and relative layout
    # identify them. Keep external compiler/SDK paths significant.
    encoded = json.dumps(identity, sort_keys=True)
    owned = str(cache.resolve())
    encoded = encoded.replace(json.dumps(owned)[1:-1], "<build-cache>")
    # uv builds sdists below this root. Keep the partition compact so compiler
    # outputs fit Windows MAX_PATH; the target is already part of the identity.
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:32] if complete else uuid.uuid4().hex
    return cache.resolve() / "python/runtime" / digest


if __name__ == "__main__":
    import runpy
    import truststore

    # Like pm/launch.py: initialize platform trust before PM creates HTTPS clients.
    truststore.inject_into_ssl()
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
