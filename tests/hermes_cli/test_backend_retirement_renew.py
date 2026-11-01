"""A hand-off that is still preparing (a long download) keeps its admission permit by renewing it."""


def test_a_renewed_permit_outlives_30_seconds_and_an_expired_one_cannot_be_revived(monkeypatch):
    from hermes_cli import web_server_idle_proof
    from hermes_cli.backend_retirement import RetirementFence

    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda **_: {"idle": True})
    fence = RetirementFence()
    now = [100.0]
    monkeypatch.setattr(fence, "_now", lambda: now[0], raising=False)
    token = fence.prepare()["token"]

    for _ in range(6):  # two minutes of 20 s heartbeats
        now[0] += 20.0
        assert fence.renew(token) == {"ok": True}
        assert fence.acquire() is False  # new work still waits
    assert fence.renew("wrong") == {"ok": False}
    assert fence.cancel(token) == {"ok": True}
    assert fence.renew(token) == {"ok": False}

    lapsed = fence.prepare()["token"]
    now[0] += 30.0
    assert fence.renew(lapsed) == {"ok": False}  # work may have started since: re-prove instead
    assert fence.acquire() is True
    fence.release()
