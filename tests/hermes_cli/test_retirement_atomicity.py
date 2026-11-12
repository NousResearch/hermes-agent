"""Freeze admission before consulting ledgers, without holding a lock across their locks."""

from concurrent.futures import ThreadPoolExecutor
import threading


def test_prepare_freezes_admission_without_blocking_other_work_locks(monkeypatch):
    from hermes_cli.backend_retirement import RetirementFence
    from hermes_cli import web_server_idle_proof

    fence = RetirementFence()
    probing, release, attempted = (threading.Event() for _ in range(3))

    def probe():
        probing.set()
        assert release.wait(10)
        return {"idle": None}

    def admit():
        with fence.work() as admitted:
            attempted.set()
            return admitted

    monkeypatch.setattr(web_server_idle_proof, "idle_proof", probe)
    with ThreadPoolExecutor(max_workers=2) as pool:
        preparing = pool.submit(fence.prepare)
        try:
            assert probing.wait(10)
            arriving = pool.submit(admit)
            assert attempted.wait(3), "admission must not wait on a ledger lock held by its own caller"
            assert arriving.result(10) is False
        finally:
            release.set()
        assert preparing.result(10) == {"ok": False, "idle": None}
    with fence.work() as admitted:
        assert admitted  # unreadable/busy probe did not leave a fence behind


def test_probe_failure_refuses_the_permit_without_wedging_admission(monkeypatch):
    from hermes_cli.backend_retirement import RetirementFence
    from hermes_cli import web_server_idle_proof

    def broken_probe():
        raise OSError("ledger unavailable")

    fence = RetirementFence()
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", broken_probe)
    assert fence.prepare() == {"ok": False, "idle": None}
    with fence.work() as admitted:
        assert admitted
