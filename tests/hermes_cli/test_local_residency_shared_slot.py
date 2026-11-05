"""Heavy residency is ONE slot, seen through two names, and llama's hooks start in shadow.

The two heavy engines on a machine are the llama.cpp router and the media sidecar. Admission can only arbitrate them against each other if
they ask for the same resource, and renaming a key would silently let both boot side by side for anyone holding a lease under the old name.
So the sharing is a canonical alias inside the store: an existing record is found under either name, and nothing has to migrate.

The llama side starts in shadow mode: it records the decision it WOULD have made and boots anyway. Honoring refusals is a later, separate
change, made once the records have been seen to match reality. A shadow hook that could block a boot would not be a shadow hook.
"""
from __future__ import annotations

import json
import threading
import time

import psutil
import pytest

from hermes_cli.local_runtime import admission as adm
from hermes_cli.local_runtime.admission import AdmissionStore, Grant, Refused

BOOT = "boot-A"
LEASE = 30.0
SHARED = adm.SHARED_RESIDENCY_RESOURCE
LEGACY_MEDIA = adm.LEGACY_MEDIA_RESIDENCY_RESOURCE


def _acquire(store, resource, *, controller="ctrl-1", pid=1000, ident="ident-1", engine="llama.cpp", now=100.0,
             boot=BOOT):
    return store.acquire(resource, owner_scope="scope-a", controller_id=controller, pid=pid,
                         process_start_identity=ident, engine=engine, model="TestModel-Q4",
                         boot_id=boot, now=now, lease_seconds=LEASE)


def _rows(store):
    conn = store._connect(create_parent=False, read_only=True)
    try:
        return conn.execute("SELECT resource, residency FROM admission_resources").fetchall()
    finally:
        conn.close()


# -- the shared slot ---------------------------------------------------------------

def test_media_and_llama_contend_on_one_slot_whichever_name_they_ask_with(tmp_path):
    store = AdmissionStore(tmp_path)
    assert isinstance(_acquire(store, SHARED, engine="llama.cpp"), Grant)
    refusal = _acquire(store, LEGACY_MEDIA, controller="ctrl-2", pid=2000, ident="ident-2", engine="unsloth-studio")
    assert isinstance(refusal, Refused) and refusal.reason == "busy", refusal


def test_the_slot_is_found_in_reverse_order_too(tmp_path):
    store = AdmissionStore(tmp_path)
    assert isinstance(_acquire(store, LEGACY_MEDIA, engine="unsloth-studio"), Grant)
    refusal = _acquire(store, SHARED, controller="ctrl-2", pid=2000, ident="ident-2")
    assert isinstance(refusal, Refused) and refusal.reason == "busy", refusal


def test_the_shared_record_lands_under_one_canonical_name(tmp_path):
    """One slot means one row: the second name must not create a second record beside the first."""
    store = AdmissionStore(tmp_path)
    assert isinstance(_acquire(store, SHARED), Grant)
    assert store.status(SHARED) is not None
    assert store.status(LEGACY_MEDIA) is not None
    assert len(_rows(store)) == 1, _rows(store)


def test_a_third_resource_is_still_its_own_slot(tmp_path):
    """Canonicalising two names must not collapse every resource into one."""
    store = AdmissionStore(tmp_path)
    assert isinstance(_acquire(store, SHARED), Grant)
    assert isinstance(_acquire(store, "llama:res-2", controller="ctrl-2", pid=2000, ident="ident-2"), Grant)


def test_media_asks_for_the_shared_slot_by_default():
    """The media controller is the other half of the contention; if it asks for a private key, the shared slot proves nothing."""
    from hermes_cli.local_runtime import media_runtime as mr
    assert mr.DEFAULT_RESOURCE == SHARED


# -- shadow mode -------------------------------------------------------------------

@pytest.fixture
def bootable(tmp_path, monkeypatch):
    """A managed boot that reaches a fake supervisor: staged model, an installed engine, no real process, no GPU probe."""
    from hermes_cli.local_runtime import bootstrap, supervisor as sup_mod
    from hermes_cli.local_runtime.binaries import Engine

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_SERVING_ENGINE", None)
    mdir = bootstrap.models_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "stub-Q4_K_M.gguf").touch()
    monkeypatch.setattr(bootstrap, "_generate_presets", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_presets_stale", lambda: False)
    monkeypatch.setattr(bootstrap, "_detect_gpu_vendor", lambda: None)
    monkeypatch.setattr(bootstrap, "_stop_at_exit", lambda: None)
    monkeypatch.setattr(bootstrap, "_start_idle_sweeper", lambda sup: None)
    monkeypatch.setattr("hermes_cli.local_runtime.binaries.installed_engine",
                        lambda backend: Engine("cpu", "b1", tmp_path / "llama-server"))

    spawns = []

    class _FakeSupervisor:
        def __init__(self, *a, **k):        # the real constructor takes binary, models dir, presets, port
            self.port, self.api_key, self.proc = 18471, "k", None

        @property
        def base_url(self):
            return f"http://127.0.0.1:{self.port}/v1"

        def start(self, timeout_s=120):
            spawns.append(1)
            time.sleep(0.4)                 # widen the window a racing caller boots into
            # publish what a real router publishes, so the second caller adopts instead of spawning
            _publish_live_state(sup_mod.state_path(), base_url=self.base_url, api_key=self.api_key)

        def stop(self):
            pass

    monkeypatch.setattr(sup_mod, "LlamaServerSupervisor", _FakeSupervisor)
    return {"spawns": spawns, "bootstrap": bootstrap}


def _publish_live_state(path, *, base_url, api_key):
    """A modern state record for the process hosting the stub, so a second boot can adopt it (a bare pid is not evidence of a live router)."""
    proc = psutil.Process()
    parent = proc.parent()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"base_url": base_url, "api_key": api_key, "pid": proc.pid,
                                "create_time": proc.create_time(), "executable": proc.exe(),
                                "owner_pid": parent.pid, "owner_create_time": parent.create_time()}), encoding="utf-8")


def _shadow_lines(bootstrap):
    path = bootstrap.runtimes_root() / "residency-shadow.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_shadow_records_the_would_be_decision_and_still_boots(tmp_path, bootable, monkeypatch):
    """The whole point of shadow mode: a slot someone else holds is recorded, and the router boots anyway."""
    b = bootable["bootstrap"]
    # The same OS boot: a record from a different boot is refused for its own reason (boot_mismatch), and this test is about the collision.
    # The same OS boot, and a store under this test's tmp_path: the real store is machine-scoped, so a test must not write into it. (A record
    # from a different boot is refused for its own reason, boot_mismatch; this test is about the collision.)
    store = AdmissionStore(tmp_path / "home")
    monkeypatch.setattr(b, "_shadow_boot_id", lambda: BOOT)
    monkeypatch.setattr(b, "_shadow_store", lambda: store)
    # now=None: a lease stamped at an injected past time is expired against the boot path's real clock, and would be refused as
    # stale_unverified rather than as the live collision this records.
    incumbent = _acquire(store, SHARED, controller="media-ctrl", pid=4242, ident="media-ident",
                         engine="unsloth-studio", boot=BOOT, now=None)
    assert isinstance(incumbent, Grant)

    sup = b.ensure_local_runtime({"local_runtime": {"enabled": True, "residency_mode": "shadow"}})

    assert sup is not None and bootable["spawns"] == [1], "shadow mode blocked or failed a boot"
    lines = _shadow_lines(b)
    assert len(lines) == 1, lines
    rec = lines[0]
    assert rec["decision"] == "would_refuse", rec
    assert rec["reason"] == "busy", rec
    assert rec["incumbent_controller"] == "media-ctrl", rec
    assert rec["incumbent_engine"] == "unsloth-studio", rec
    assert rec["action"] == "boot_anyway", rec
    assert rec["engine"] == "b1" and rec["backend"] == "cpu", rec   # the stub Engine is ("cpu", "b1", path): backend cpu, build tag b1


def test_shadow_records_would_grant_when_the_slot_is_free(bootable):
    sup = bootable["bootstrap"].ensure_local_runtime({"local_runtime": {"enabled": True, "residency_mode": "shadow"}})
    assert sup is not None and bootable["spawns"] == [1]
    lines = _shadow_lines(bootable["bootstrap"])
    assert len(lines) == 1 and lines[0]["decision"] == "would_grant", lines
    assert lines[0]["reason"] == "" and lines[0]["incumbent_controller"] == "", lines


def test_a_shadow_probe_leaves_no_grant_behind_for_the_next_holder(tmp_path, bootable, monkeypatch):
    """The probe asks a real question with a real acquire. If it kept the grant, the next process to ask would be refused by a ghost."""
    b = bootable["bootstrap"]
    store = AdmissionStore(tmp_path / "home")
    monkeypatch.setattr(b, "_shadow_store", lambda: store)
    assert b.ensure_local_runtime({"local_runtime": {"enabled": True, "residency_mode": "shadow"}}) is not None
    other = _acquire(store, SHARED, controller="a-different-process", pid=9999, ident="ident-9999", now=None)
    assert isinstance(other, Grant), other


def test_a_record_written_under_the_old_name_is_the_slot_the_canonical_name_reads(tmp_path):
    """The point of the alias is that an existing deployment's row is the one everybody contends on, so the row must live under one name."""
    store = AdmissionStore(tmp_path)
    assert isinstance(_acquire(store, LEGACY_MEDIA, engine="unsloth-studio"), Grant)
    assert [r[0] for r in _rows(store)] == [SHARED], _rows(store)
    assert store.status(LEGACY_MEDIA).controller_id == "ctrl-1"


@pytest.mark.parametrize("config", [
    {"local_runtime": {"enabled": True}},                                  # unset means off: nothing is read, written or logged
    {"local_runtime": {"enabled": True, "residency_mode": "off"}},         # explicitly off
])
def test_shadow_off_writes_nothing_and_changes_nothing(tmp_path, monkeypatch, bootable, config):
    """The default must be inert. Each config boots in a fresh HERMES_HOME, because a boot that adopted the previous one would prove nothing."""
    from hermes_cli.local_runtime import bootstrap, supervisor as sup_mod
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / f"home-{config['local_runtime'].get('residency_mode', 'unset')}"))
    (bootstrap.models_dir()).mkdir(parents=True, exist_ok=True)
    (bootstrap.models_dir() / "stub-Q4_K_M.gguf").touch()
    sup_mod.state_path().unlink(missing_ok=True)
    bootable["spawns"].clear()
    assert bootstrap.ensure_local_runtime(config) is not None and bootable["spawns"] == [1], config
    assert _shadow_lines(bootstrap) == [], config


def test_a_broken_shadow_hook_cannot_stop_a_boot(bootable, monkeypatch, caplog):
    """The shadow path touches a SQLite file on a boot path whose contract is 'never raises, never hangs'."""
    b = bootable["bootstrap"]
    monkeypatch.setattr(AdmissionStore, "acquire", lambda *a, **k: (_ for _ in ()).throw(OSError("disk on fire")))
    monkeypatch.setattr(b, "_shadow_store", lambda: AdmissionStore())
    sup = b.ensure_local_runtime({"local_runtime": {"enabled": True, "residency_mode": "shadow"}})
    assert sup is not None and bootable["spawns"] == [1]
    assert _shadow_lines(b) == [], "a failed shadow record must not be written as if it were a decision"
    assert "residency shadow" in caplog.text.lower(), caplog.text


def test_an_unimplemented_mode_is_refused_loudly_rather_than_silently_booting():
    """An unread mode would mean no residency arbitration at all while looking configured. 'enforce' lands with a later change."""
    from hermes_cli.local_runtime import bootstrap
    assert bootstrap.RESIDENCY_MODES == ("off", "shadow")
    with pytest.raises(ValueError, match="residency_mode"):
        bootstrap.residency_mode({"local_runtime": {"enabled": True, "residency_mode": "enforce"}})


def test_residency_mode_defaults_to_off_and_tolerates_case_and_space():
    from hermes_cli.local_runtime import bootstrap
    assert bootstrap.residency_mode({}) == "off"
    assert bootstrap.residency_mode({"local_runtime": {"residency_mode": "shadow"}}) == "shadow"
    assert bootstrap.residency_mode({"local_runtime": {"residency_mode": " SHADOW "}}) == "shadow"


def test_a_shadow_record_is_one_line_of_valid_json_with_no_secrets(bootable):
    """A human reads this file before any refusal is honored: one parseable line per boot, no keys, no absolute paths."""
    assert bootable["bootstrap"].ensure_local_runtime({"local_runtime": {"enabled": True, "residency_mode": "shadow"}}) is not None
    path = bootable["bootstrap"].runtimes_root() / "residency-shadow.jsonl"
    raw = path.read_text(encoding="utf-8")
    assert raw.count("\n") == 1
    assert set(json.loads(raw)) == {"at", "decision", "reason", "incumbent_controller", "incumbent_engine", "action",
                                    "engine", "backend", "model_count", "detail"}, sorted(json.loads(raw))
    assert str(bootable["bootstrap"].runtimes_root()) not in raw, raw


def test_two_racing_boots_write_one_shadow_line_each(bootable):
    """One line per boot attempt, including the loser that adopts: a human must be able to count boots in this file."""
    b = bootable["bootstrap"]
    results, barrier = [], threading.Barrier(2)

    def _boot():
        barrier.wait()
        results.append(b.ensure_local_runtime({"local_runtime": {"enabled": True, "residency_mode": "shadow"}}))

    threads = [threading.Thread(target=_boot) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    lines = _shadow_lines(b)
    assert len(lines) == 2, lines
    assert sorted(line["action"] for line in lines) == ["adopt", "boot_anyway"], lines
