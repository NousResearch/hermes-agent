"""Setup must own startup, not merely select lazy routing or trust durable aliases."""
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sqlite3
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
load = runpy.run_path(str(ROOT / "plugins/hermes-realms/realms/_binding.py"))["load_runtime"]
pytestmark = pytest.mark.linux_only


def wait_job(flow, service, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = flow.status(service, "owner", job["id"])
        if result["state"] != "running":
            return result
        time.sleep(.01)
    pytest.fail("setup did not finish")


@pytest.fixture
def setup_service(tmp_path, monkeypatch):
    """Real proposal, job, integration, ownership and managers; fake host probes."""
    integration, flow = load("integration"), load("setup_flow")
    service = integration.RealmIntegration(tmp_path / "profile")
    service.bind(session_id="owner", runtime_session_id="runtime")
    service.owners.set_mode("owner", "ask")
    plan, installer, manager = load("setup_plan"), load("install_driver"), load("manager")
    monkeypatch.setattr(plan.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(plan.os, "access", lambda *args: True)
    monkeypatch.setattr(installer, "execution_verified", lambda path: True)
    monkeypatch.setattr(installer, "_execution_identity", lambda path: {"fixture": "pinned-driver"})
    monkeypatch.setattr(manager, "scope_info", lambda unit: {"ActiveState": "active"})
    monkeypatch.setattr(load("setup_worker"), "run_child", lambda *args, **kwargs: None)
    source = plan.setup_module()
    target = load("config").driver_path(service.home)
    target.parent.mkdir(parents=True)
    target.with_name(".realms-setup.json").write_text(json.dumps(source["_receipt_data"](service.home, target, installer)))
    yield service, flow
    service.unload()


def launch(service, flow):
    proposal = flow.prepare(service, "owner", "realm")
    return flow.start(service, "owner", "realm", proposal["consent"], {"runtime_session_id": "runtime"})


def test_setup_native_actually_attempts_owned_start_and_rolls_back(setup_service, monkeypatch):
    service, flow = setup_service
    attempts = []
    original = Path.mkdir

    def unavailable_runtime(path, *args, **kwargs):
        if str(path).startswith("/run/user/") and path.name.startswith("hr-"):
            attempts.append(path)
            raise OSError("fixture private runtime unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", unavailable_runtime)
    result = wait_job(flow, service, launch(service, flow))
    assert attempts, "successful setup must attempt real Manager.start, not lazy /realm on"
    assert result["state"] == "failed"
    assert service.owners.mode("owner", "realm") == "ask"
    assert service.kind("owner") == "realm"
    assert not service.records("owner")
    assert not service._attachments


@pytest.mark.parametrize("boundary", ["off", "finalize", "unload", "other-process-off"])
def test_late_setup_cannot_reactivate_revoked_owner(setup_service, monkeypatch, boundary):
    service, flow = setup_service
    entered, release = threading.Event(), threading.Event()

    def paused_install_io(*args, **kwargs):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(load("setup_worker"), "run_child", paused_install_io)
    job = launch(service, flow)
    assert entered.wait(5)
    try:
        if boundary == "off":
            service.command("off", session_id="owner")
        elif boundary == "finalize":
            service.finalize(session_id="owner")
        elif boundary == "unload":
            service.unload()
        else:
            # A separate store/service proves revocation is not an in-memory flag.
            other = load("integration").RealmIntegration(service.home)
            other.command("off", session_id="owner")
        # Aliases deliberately survive every lifecycle boundary.
        assert service.owners.resolve(runtime_session_id="runtime") == "owner"
    finally:
        release.set()
    result = wait_job(flow, service, job)
    assert result["state"] == "failed", "durable aliases must not authorize late activation"
    assert service.owners.mode("owner", "realm") == ("host" if "off" in boundary else "ask")
    assert not service._attachments
    assert not service.records("owner")


def test_readonly_ownership_store_does_not_create_or_migrate(tmp_path):
    integration = load("integration")
    missing = tmp_path / "missing"
    store = integration.OwnershipStore(missing, readonly=True)
    with pytest.raises(sqlite3.OperationalError):
        store.resolve(session_id="owner")
    assert not missing.exists()
    writable = integration.OwnershipStore(tmp_path / "existing")
    writable.bind(session_id="owner")
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in writable.root.iterdir()}
    readonly = integration.OwnershipStore(tmp_path / "existing", readonly=True)
    assert readonly.resolve(session_id="owner") == "owner"
    assert before == {p: (p.read_bytes(), p.stat().st_mode) for p in writable.root.iterdir()}
    with pytest.raises(sqlite3.OperationalError):
        readonly.set_mode("owner", "host")


def test_activation_rejects_config_drift_before_any_start(setup_service, monkeypatch):
    service, flow = setup_service
    entered, release = threading.Event(), threading.Event()

    def paused_install_io(*args, **kwargs):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(load("setup_worker"), "run_child", paused_install_io)
    job = launch(service, flow)
    assert entered.wait(5)
    try:
        (service.home / "config.yaml").write_text("plugins:\n  realms:\n    size: 1280x720\n")
        # Keep the readiness receipt current: rejection must bind consent/config,
        # not accidentally succeed because describe spots a stale driver receipt.
        source = load("setup_plan").setup_module()
        target = load("config").driver_path(service.home)
        target.with_name(".realms-setup.json").write_text(json.dumps(source["_receipt_data"](service.home, target, load("install_driver"))))
    finally:
        release.set()
    assert wait_job(flow, service, job)["state"] == "failed"
    assert service.owners.mode("owner", "realm") == "ask"


def test_vm_manager_cache_drops_removed_custom_installer(tmp_path):
    service = load("integration").RealmIntegration(tmp_path / "profile")
    config_path = service.home / "config.yaml"
    config_path.write_text("plugins:\n  realms:\n    vm:\n      omarchy_vm_path: /fixture/custom-installer\n")
    cached = service.vm
    config_path.write_text("plugins:\n  realms: {}\n")
    refreshed = service.vm
    assert refreshed is not cached
    assert refreshed.config == load("config").Config.load(service.home)
    assert not refreshed.config.vm.omarchy_vm_path


def test_teardown_serializes_with_startup_across_services(setup_service, monkeypatch):
    service, _ = setup_service
    other = load("integration").RealmIntegration(service.home)
    generation = service.reserve_setup("owner")
    entered, release, off_entered, off_done = (threading.Event() for _ in range(4))
    errors = []
    original = Path.mkdir

    def paused_runtime_io(path, *args, **kwargs):
        if str(path).startswith("/run/user/") and path.name.startswith("hr-"):
            entered.set()
            assert release.wait(10)
            raise OSError("fixture startup failed after teardown was requested")
        return original(path, *args, **kwargs)

    def activate():
        try:
            service.activate_setup("owner", "realm", asdict(load("config").Config.load(service.home)),
                                   generation, {"runtime_session_id": "runtime"})
        except Exception as exc:
            errors.append(exc)

    def turn_off():
        off_entered.set()
        try:
            other.command("off", session_id="owner")
        except Exception as exc:
            errors.append(exc)
        finally:
            off_done.set()

    monkeypatch.setattr(Path, "mkdir", paused_runtime_io)
    startup = threading.Thread(target=activate)
    teardown = threading.Thread(target=turn_off)
    startup.start()
    try:
        assert entered.wait(5)
        # Startup must not hold a long SQLite writer transaction.
        other.bind(session_id="unrelated-owner")
        teardown.start()
        assert off_entered.wait(5)
        assert not off_done.wait(.1), "off must not race past an in-flight activation"
    finally:
        release.set()
        startup.join(5)
        if teardown.ident is not None:
            teardown.join(5)
        service.finish_setup("owner")
    assert not startup.is_alive() and not teardown.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], OSError)
    assert service.owners.mode("owner", "realm") == "host"
    assert service.owners.setup_generation("owner") != generation
    assert not service.records("owner")


def test_unload_releases_cached_service_without_revalidating_old_lease(tmp_path):
    integration = load("integration")
    service = integration.get_integration(tmp_path / "profile")
    service.bind(session_id="owner")
    lease = service.reserve_setup("owner")
    service.unload()
    replacement = integration.get_integration(service.home)
    assert replacement is not service
    assert replacement.owners.setup_generation("owner") != lease
    with pytest.raises(integration.OwnerError, match="unloaded"):
        service.reserve_setup("owner")
    replacement.unload()
