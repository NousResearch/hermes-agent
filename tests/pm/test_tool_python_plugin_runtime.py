"""Solstice provider must load in the plugin-host runtime (Task 1 probe).

Runtime identity (evidence 2026-10-15, failing run):
- Plugin-host interpreter: the PM store tool python, read from the
  checkout's own published launcher (``.hermes/bin/hermes`` embeds exactly
  what production execs):
  ``/home/lorenzo/.hermes/tools/python-3.14.7+20260901-linux-x64/bin/python3``
  (CPython 3.14.7, run with ``-I`` isolated mode, matching
  ``runtime_command``'s ``[python, "-I", "-c", ...]`` shape; identical to
  ``hermes_cli._launchers.resolve_store_python`` for this checkout).
- Its ``sys.path`` under ``-I`` is the bare stdlib only
  (``lib/python314.zip``, ``lib/python3.14``, ``lib-dynload``,
  ``lib/python3.14/site-packages``); ``import httpx`` raises
  ``ModuleNotFoundError: No module named 'httpx'`` there.
- Verdict: ``PLUGIN_HOST = "tool-python"`` (the store tool python, NOT a
  venv python). The probe below spawns exactly that interpreter with ``-I``
  and loads the bundled solstice plugin through the production discovery
  path (``providers._import_plugin_dir(..., "bundled")``); before the fix it
  fails with ``Failed to load bundled provider plugin solstice: No module
  named 'httpx'`` (the production ``errors.log`` signature), because the
  import chain ``transport.py -> agent/gemini_native_adapter.py:20
  (import httpx)`` cannot resolve under ``-I``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Probe executed inside the plugin-host interpreter. It loads the bundled
# solstice plugin exactly the way production discovery does (file-location
# import under the stable ``plugins.model_providers.<name>`` module name,
# which pulls ``transport.py -> gemini_native_adapter (import httpx)``) and
# reports success only when the provider registers with its transport class.
_PROBE = (
    "import sys; "
    "repo_root = sys.argv[1]; "
    "sys.path.insert(0, repo_root); "
    "from pathlib import Path; "
    "from providers import _REGISTRY, _import_plugin_dir; "
    "_import_plugin_dir(Path(repo_root) / 'plugins' / 'model-providers' / 'solstice', 'bundled'); "
    "profile = _REGISTRY.get('solstice'); "
    "assert profile is not None, 'solstice provider did not register'; "
    "module = sys.modules.get('plugins.model_providers.solstice'); "
    "assert module is not None and getattr(module, 'SolsticeClient', None) is not None, "
    "'solstice transport class missing'; "
    "print('SOLSTICE_PLUGIN_OK')"
)


def _resolve_plugin_host_interpreter() -> tuple[Path, bool]:
    """The plugin-host interpreter, resolved the same way the launcher does.

    Returns ``(interpreter, is_store_tool_python)``. Primary source is the
    checkout's own published launcher (``.hermes/bin/hermes``): its embedded
    interpreter is exactly what production execs (``exec <python> -I -c
    ...``), and reading it is a pure checkout-file read, so the test-session
    ``HERMES_HOME`` sandbox cannot divert it the way a home-derived store
    lookup would. Falls back to the PM store resolution and finally to the
    current interpreter (the launcher's own fallback when no store exists).
    """
    from hermes_cli._launchers import _launcher_python, resolve_store_python

    launcher = REPO_ROOT / ".hermes" / "bin" / ("hermes.exe" if os.name == "nt" else "hermes")
    python = _launcher_python(launcher)
    if python is not None:
        try:
            if python.is_file():
                return (python, True)
        except OSError:
            pass
    store_python = resolve_store_python(REPO_ROOT)
    if store_python is not None:
        return (store_python, True)
    return (Path(sys.executable), False)


def test_solstice_loads_in_plugin_host_runtime(tmp_path):
    interpreter, is_store = _resolve_plugin_host_interpreter()
    home = tmp_path / "home"
    home.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    env["HERMES_HOME"] = str(home)
    completed = subprocess.run(
        [str(interpreter), "-I", "-c", _PROBE, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert "SOLSTICE_PLUGIN_OK" in completed.stdout, (
        f"solstice plugin failed to load in the plugin-host runtime "
        f"(interpreter={interpreter}, store_tool_python={is_store}): "
        f"rc={completed.returncode} stdout={completed.stdout[-1500:]!r} "
        f"stderr={completed.stderr[-2000:]!r}"
    )


# --- Task 3A: the tool-python payload ships httpx (stage-level contracts) ---
#
# The Task 1 probe above pins the production symptom (bare store `-I` must
# import the solstice chain). The tests below pin the mechanism that fixes
# it persistently: `Python.stage()` installs the httpx closure into the
# staged tree from versioned inputs, so every republish carries the fix
# (the property manual store copies lacked — wiped 2026-10-15 17:36:42).


def _fake_staged_python(root: Path, target: str) -> Path:
    """A scratch staged tree with the target's interpreter layout."""
    staged = root / "tree"
    if target.startswith("win32"):
        site = staged / "Lib" / "site-packages"
        site.mkdir(parents=True)
        (staged / "python.exe").write_bytes(b"fake-interpreter")
    elif target == "linux-arm64-bionic":
        prefix = staged / "data" / "data" / "com.termux" / "files" / "usr"
        site = prefix / "lib" / "python3.14" / "site-packages"
        site.mkdir(parents=True)
        (prefix / "bin").mkdir(parents=True, exist_ok=True)
        (prefix / "bin" / "python3.14").write_bytes(b"fake-interpreter")
    else:
        site = staged / "lib" / "python3.14" / "site-packages"
        site.mkdir(parents=True)
        (staged / "bin").mkdir(parents=True, exist_ok=True)
        (staged / "bin" / "python3").write_bytes(b"fake-interpreter")
    return staged


@pytest.fixture
def canned_payload_install(tmp_path, monkeypatch):
    """Stub the uv runner with a deterministic canned payload; capture calls."""
    import pm.packages as packages

    calls = []

    def fake_run(uv, requirements, *, python_version, platform, dest, cache):
        calls.append({
            "uv": uv, "requirements": tuple(requirements),
            "python_version": python_version, "platform": platform,
            "dest": dest, "cache": cache,
        })
        pkg = dest / "httpx"
        pkg.mkdir()
        (pkg / "__init__.py").write_text('__version__ = "0.28.1"\n', encoding="utf-8")
        dist = dest / "httpx-0.28.1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Name: httpx\nVersion: 0.28.1\n", encoding="utf-8")
        (dist / "RECORD").write_text(
            "bin/httpx,sha256=aaa,10\n"
            "httpx/__init__.py,sha256=bbb,24\n"
            "httpx-0.28.1.dist-info/METADATA,sha256=ccc,26\n"
            "httpx-0.28.1.dist-info/RECORD,,\n",
            encoding="utf-8",
        )
        scripts = dest / "bin"
        scripts.mkdir()
        (scripts / "httpx").write_text("#!/wrong/resolution-interpreter\n", encoding="utf-8")

    monkeypatch.setattr(packages, "_run_payload_install", fake_run)
    monkeypatch.setattr(packages, "_payload_uv_bin", lambda: Path("/fake/uv"))
    monkeypatch.setattr(packages, "uv_cache_dir", lambda: tmp_path / "uv-cache")
    return calls


@pytest.mark.parametrize("target,platform", [
    ("linux-x64", "linux"),
    ("darwin-arm64", "macos"),
    ("win32-arm64", "windows"),
    ("linux-arm64-bionic", "linux"),
])
def test_tool_python_stage_installs_importable_payload(tmp_path, monkeypatch,
                                                       canned_payload_install,
                                                       target, platform):
    import types
    import pm.packages as packages
    from pm.store import Store

    # Stage never signs on this host (sys.platform gate), but a mac lane
    # would codesign the fake interpreter: stub the seam, not the behavior.
    monkeypatch.setitem(
        sys.modules, "hermes_cli.macos_signing",
        types.SimpleNamespace(sign_managed_python=lambda *args: None),
    )

    staged = _fake_staged_python(tmp_path / "first", target)
    packages.Python().stage(Store(tmp_path / "store"), staged, "3.14.7+20260901", target)

    call = canned_payload_install[0]
    assert call["platform"] == platform
    assert call["python_version"] == "3.14"
    assert call["requirements"] == packages._TOOL_PYTHON_PLUGIN_PINS

    site = packages._tool_site_packages(staged, target, packages.Python.prefix_rel)
    assert (site / "httpx" / "__init__.py").is_file()
    assert not (site / "bin").exists()
    record = (site / "httpx-0.28.1.dist-info" / "RECORD").read_text(encoding="utf-8")
    assert "bin/httpx" not in record

    probe = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); "
         "import httpx; assert httpx.__version__ == '0.28.1'; print('PAYLOAD_IMPORT_OK')",
         str(site)],
        capture_output=True, text=True, timeout=60,
    )
    assert "PAYLOAD_IMPORT_OK" in probe.stdout, probe.stderr[-1000:]


def test_tool_python_restage_is_digest_stable(tmp_path, canned_payload_install):
    import pm.packages as packages
    from pm.store import Store, tree_digest

    digests = []
    for index in ("first", "second"):
        staged = _fake_staged_python(tmp_path / index, "linux-x64")
        packages.Python().stage(Store(tmp_path / "store"), staged, "3.14.7+20260901", "linux-x64")
        digests.append(tree_digest(staged))
    assert digests[0] == digests[1]


def test_tool_python_payload_install_is_offline_and_targeted(tmp_path, monkeypatch):
    import subprocess as stdlib_subprocess
    import pm.packages as packages

    seen = []

    def fake_run(command, **kwargs):
        seen.append((list(command), kwargs))
        if "--offline" in command:
            return stdlib_subprocess.CompletedProcess(command, 1, "", "index unreachable")
        return stdlib_subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(stdlib_subprocess, "run", fake_run)
    dest = tmp_path / "payload"
    dest.mkdir()
    packages._run_payload_install(
        Path("/fake/uv"), ("httpx[socks]==0.28.1",),
        python_version="3.14", platform="windows", dest=dest, cache=tmp_path / "cache",
    )

    assert len(seen) == 2  # offline first, online retry on a cold cache
    offline_argv, offline_kwargs = seen[0]
    online_argv, _ = seen[1]
    assert "--offline" in offline_argv
    assert "--offline" not in online_argv
    for argv in (offline_argv, online_argv):
        assert "--only-binary" in argv and ":all:" in argv
        assert "--link-mode=copy" in argv
        assert "--no-build" not in argv  # conflicts with --only-binary on uv 0.12
        assert "--python-version" in argv and "3.14" in argv
        assert "--python-platform" in argv and "windows" in argv
        assert "--target" in argv and str(dest) in argv
        assert "httpx[socks]==0.28.1" in argv
    env = offline_kwargs["env"]
    assert env["UV_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["UV_PYTHON_DOWNLOADS"] == "never"
    assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env


def test_tool_python_payload_install_fails_closed(tmp_path, monkeypatch):
    import subprocess as stdlib_subprocess
    import pm.packages as packages
    from pm.package import InstallError

    monkeypatch.setattr(
        stdlib_subprocess, "run",
        lambda *args, **kwargs: stdlib_subprocess.CompletedProcess(args[0], 1, "", "no network"),
    )
    dest = tmp_path / "payload"
    dest.mkdir()
    with pytest.raises(InstallError):
        packages._run_payload_install(
            Path("/fake/uv"), ("httpx[socks]==0.28.1",),
            python_version="3.14", platform="linux", dest=dest, cache=tmp_path / "cache",
        )


def test_tool_python_stage_rejects_compiled_payload(tmp_path, monkeypatch):
    import pm.packages as packages
    from pm.package import InstallError
    from pm.store import Store

    def fake_run(uv, requirements, *, python_version, platform, dest, cache):
        (dest / "_native").mkdir()
        (dest / "_native" / "fast.so").write_bytes(b"elf")

    monkeypatch.setattr(packages, "_run_payload_install", fake_run)
    monkeypatch.setattr(packages, "_payload_uv_bin", lambda: Path("/fake/uv"))
    monkeypatch.setattr(packages, "uv_cache_dir", lambda: tmp_path / "uv-cache")
    staged = _fake_staged_python(tmp_path, "linux-x64")
    with pytest.raises(InstallError, match="pure-python"):
        packages.Python().stage(Store(tmp_path / "store"), staged, "3.14.7+20260901", "linux-x64")


def test_tool_python_stage_fails_closed_without_site_packages(tmp_path, canned_payload_install):
    import pm.packages as packages
    from pm.package import InstallError
    from pm.store import Store

    staged = tmp_path / "tree"
    staged.mkdir()
    with pytest.raises(InstallError, match="no site-packages"):
        packages.Python().stage(Store(tmp_path / "store"), staged, "3.14.7+20260901", "linux-x64")


def test_tool_python_payload_pins_track_declared_dependencies():
    """Stage pins mirror [project].dependencies (httpx[socks]) and exist in uv.lock."""
    import pm.packages as packages

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = " ".join(project["project"]["dependencies"])
    match = re.search(r"httpx(\[[^\]]*\])?==([^,;\s]+)", declared)
    assert match is not None, "pyproject must declare the httpx pin this payload ships"
    assert match.group(1) == "[socks]"
    assert f"httpx[socks]=={match.group(2)}" in packages._TOOL_PYTHON_PLUGIN_PINS

    locked: dict[str, set[str]] = {}
    name, in_package = "", False
    for raw in (REPO_ROOT / "uv.lock").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "[[package]]":
            in_package, name = True, ""
            continue
        if line.startswith("["):
            in_package, name = False, ""
            continue
        if not in_package:
            continue
        entry = re.fullmatch(r'name = "([^"]+)"', line)
        if entry:
            name = entry.group(1).lower().replace("-", "_")
            continue
        release = re.fullmatch(r'version = "([^"]+)"', line)
        if release and name:
            locked.setdefault(name, set()).add(release.group(1))
            name = ""
    for pin in packages._TOOL_PYTHON_PLUGIN_PINS:
        pinned = re.match(r"([A-Za-z0-9_.-]+)(\[[^\]]*\])?==([^,;\s]+)", pin)
        assert pinned is not None, f"payload pin must stay exact: {pin}"
        key = pinned.group(1).lower().replace("-", "_")
        assert pinned.group(3) in locked.get(key, set()), \
            f"payload pin {pin} drifted from uv.lock"
