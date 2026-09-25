"""pm.extras: anchor availability, ensure_import, ensure_and_bind, and the
spec→extra install shim. Network-free — sync_venv is stubbed at the client
seam used by extras; the engine and worker have separate transaction tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import pm
import pm.client as client
import pm.extras as extras


# ---- per-extra platform gates ([tool.hermes.extras-platforms]) ----


def test_extra_supported_ungated_extra_is_true():
    extras._PLATFORM_GATES = None
    assert extras.extra_supported("no-such-gate-for-this-one") is True


def test_ensure_import_raises_on_gated_off_extra(monkeypatch, synced):
    from packaging.markers import default_environment

    version = default_environment()["python_full_version"]
    monkeypatch.setattr(extras, "_PLATFORM_GATES", {
        "gated-extra": f"python_full_version < '{version}'",
    })
    monkeypatch.setattr(extras, "available", lambda e: False)
    with pytest.raises(pm.InstallError, match="not supported on this platform"):
        extras.ensure_import("gated-extra")
    assert synced == []


def test_sync_refuses_python_gated_extra_before_touching_environment(monkeypatch, tmp_path):
    import importlib
    from pathlib import Path
    from packaging.markers import default_environment

    engine = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    version = default_environment()["python_full_version"]
    monkeypatch.setattr(extras, "_PLATFORM_GATES", {
        "unavailable-engine": f"python_full_version < '{version}'",
    })
    # Installed caller anchors cannot make a new managed graph compatible.
    monkeypatch.setitem(sys.modules, "unavailable_engine", SimpleNamespace())
    assert extras.extra_supported("unavailable-engine")
    assert not extras.extra_supported("unavailable-engine", importable=lambda _: False)

    def refuse_environment_access(*args, **kwargs):
        pytest.fail("unsupported request reached dependency environment machinery")

    monkeypatch.setattr(engine, "get_package", refuse_environment_access)
    with pytest.raises(pm.InstallError, match="not supported by this Python/platform"):
        engine.sync_venv(["unavailable-engine"], explicit=True)


def test_declared_extra_gates_match_dependency_selection():
    import tomllib
    from pathlib import Path
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    optional = metadata["project"]["optional-dependencies"]
    targets = [
        ("linux", "Linux", "x86_64"), ("linux", "Linux", "aarch64"),
        ("darwin", "Darwin", "x86_64"), ("darwin", "Darwin", "arm64"),
        ("win32", "Windows", "AMD64"), ("win32", "Windows", "ARM64"),
    ]
    for system, platform_system, machine in targets:
        for python in ("3.12", "3.13", "3.14"):
            environment = {**default_environment(), "sys_platform": system,
                           "platform_system": platform_system, "platform_machine": machine,
                           "python_version": python, "python_full_version": python + ".0"}
            for extra in metadata["tool"]["hermes"]["extras-platforms"]:
                selected = any(req.marker is None or req.marker.evaluate(environment)
                               for req in map(Requirement, optional[extra]))
                assert extras.extra_supported(extra, environment=environment,
                                              importable=lambda _: False) == selected, (
                    extra, system, machine, python,
                )


def test_faster_whisper_targets_are_gated(monkeypatch):
    """The local-STT extra's anchor is faster-whisper, which has no win_arm64 or darwin-x64 build.

    An extra whose anchor can never import there must be refused up front: without the gate,
    ensure_import rebuilt the whole dependency environment and still failed the anchor — on every
    status probe, forever. ``voice`` is deliberately NOT gated (its sounddevice/numpy are
    installable on those targets; see the selection test above), so the lazy STT path asks for
    ``stt-whisper``, the extra that carries only faster-whisper.
    """
    monkeypatch.setattr(extras, "_PLATFORM_GATES", None)
    targets = {
        "win32-arm64": {"sys_platform": "win32", "platform_system": "Windows",
                        "platform_machine": "ARM64", "os_name": "nt"},
        "darwin-x64": {"sys_platform": "darwin", "platform_system": "Darwin",
                       "platform_machine": "x86_64", "os_name": "posix"},
        "linux-x64": {"sys_platform": "linux", "platform_system": "Linux",
                      "platform_machine": "x86_64", "os_name": "posix"},
    }
    supported = {
        target: extras.extra_supported("stt-whisper", environment=environment,
                                       importable=lambda _: False)
        for target, environment in targets.items()
    }
    assert supported == {"win32-arm64": False, "darwin-x64": False, "linux-x64": True}


@pytest.fixture
def synced(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(client, "sync_venv", lambda x=None: calls.append(list(x or [])))
    return calls



def test_available_missing_module():
    assert extras.available("no-such-extra-anywhere") is False


@pytest.mark.parametrize(("extra", "module"), [
    ("teams", "microsoft_teams.apps"),
])
def test_available_counts_sys_modules_fakes(monkeypatch, extra, module):
    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    assert extras.available(extra) is True


def test_google_readiness_requires_its_oauth_imports(monkeypatch):
    present = {"googleapiclient", "google.auth", "google_auth_httplib2"}
    monkeypatch.setattr(extras, "_importable", lambda name: name in present)
    assert not extras.available("google")
    present.add("google_auth_oauthlib.flow")
    assert extras.available("google")


def test_available_unknown_extra_uses_underscore_guess(monkeypatch):
    monkeypatch.setitem(sys.modules, "some_new_thing", SimpleNamespace())
    assert extras.available("some-new-thing") is True


def test_ensure_import_noop_when_available(monkeypatch, synced):
    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace())
    extras.ensure_import("fal")
    assert synced == []


def test_ensure_import_syncs_when_missing(monkeypatch, synced, tmp_path):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(extras, "available", lambda e: False)
    extras.ensure_import("fal")
    assert synced == [["fal"]]


def test_ensure_import_respects_terminal_decline_without_installing(monkeypatch, synced):
    import builtins

    monkeypatch.setattr(extras, "available", lambda _: False)
    monkeypatch.setattr(extras, "missing", lambda _: ["fal_client"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda text: prompts.append(text) or "n")
    with pytest.raises(pm.InstallError, match="declined"):
        extras.ensure_import("fal")
    # Users know the feature, not the Python modules behind it.
    assert "'fal' feature" in prompts[0] and "fal_client" not in prompts[0]
    assert synced == []


def test_ensure_import_propagates_install_error(monkeypatch):
    def boom(x=None):
        raise pm.InstallError("venv", "lazy installs are disabled")

    monkeypatch.setattr(client, "sync_venv", boom)
    monkeypatch.setattr(extras, "available", lambda e: False)
    with pytest.raises(pm.InstallError):
        extras.ensure_import("fal")


@pytest.mark.parametrize("failure", [None, "install", "import"])
def test_ensure_and_bind_preserves_target_on_failure(monkeypatch, failure):
    monkeypatch.setattr(extras, "available", lambda _: failure != "install")
    def sync(*args):
        raise pm.InstallError("venv", "install refused")
    monkeypatch.setattr(client, "sync_venv", sync)
    def importer():
        if failure == "import":
            raise ImportError("still broken")
        return {"NAME": 42}
    target = {"existing": "kept"}
    assert extras.ensure_and_bind("fal", importer, target) is (failure is None)
    assert target == ({"existing": "kept", "NAME": 42} if failure is None else {"existing": "kept"})




def test_every_anchor_extra_exists_in_pyproject():
    """Contract: ANCHORS maps real pyproject extras (no orphaned names)."""
    import tomllib
    from pathlib import Path

    py = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    declared = set(py["project"]["optional-dependencies"])
    orphans = set(extras.ANCHORS) - declared
    assert not orphans, f"ANCHORS names extras pyproject does not declare: {sorted(orphans)}"


def test_legacy_selection_carries_extras_the_main_era_venv_lazily_installed(monkeypatch, tmp_path):
    """Migrating a main-era venv must keep opt-in extras it already had (FAL
    image generation, a messaging SDK), or the first PM launch prompts to
    reinstall them. Umbrella and gated-off extras are never carried."""
    monkeypatch.setattr(extras, "_PLATFORM_GATES", {"piper": "python_version < '0'"})
    site = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    (site / "fal_client").mkdir(parents=True)
    (site / "telegram").mkdir()
    (site / "piper").mkdir()
    (site / "google").mkdir()
    (site / "google" / "auth").mkdir()
    (site / "exa_py.cpython-311-x86_64-linux-gnu.so").write_bytes(b"")
    (site / "hindsight_client").mkdir()

    selection = extras.legacy_selection(tmp_path)

    assert selection[0] == "all"
    assert {"fal", "telegram", "vertex", "exa"} <= set(selection)
    assert "messaging" not in selection
    assert "piper" not in selection
    assert "hindsight" not in selection  # Catalog plugin owns this dependency, not a core extra.
    assert extras.legacy_selection(tmp_path / "no-venv") == ["all"]


def test_runtime_marker_evaluation_answers_for_the_given_environment():
    """The delegate really evaluates the marker (in PM's runtime interpreter)."""
    import subprocess
    from pathlib import Path

    helper = Path(extras.__file__).with_name("_marker_eval.py")
    env = '{"sys_platform": "linux"}'
    out = [subprocess.run([sys.executable, str(helper), marker, env], capture_output=True,
                          text=True, timeout=60, check=True).stdout.strip()
           for marker in ("sys_platform == 'linux'", "sys_platform == 'win32'")]
    assert out == ["1", "0"]
