"""A2A counters are per profile: one adapter module serves every profile in the gateway process."""


def _bare_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter
    return A2AAdapter(PlatformConfig(enabled=True))


class TestMetricsProfileScope:
    def test_counters_stay_in_the_owning_profile(self, tmp_path):
        """One protocol module serves every profile: an adapter counts into its own profile's
        metrics (captured at construction; its HTTP threads carry no scope) and ``metrics``
        reads the current profile's."""
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from plugins.platforms.a2a import protocol

        def in_home(name, fn):
            token = set_hermes_home_override(str(tmp_path / name))
            try:
                return fn()
            finally:
                reset_hermes_home_override(token)

        def counts():
            return protocol.metrics.inbound_total, protocol.metrics.outbound_total

        base_a, base_b = in_home("a", counts), in_home("b", counts)
        in_home("b", lambda: setattr(protocol.metrics, "outbound_total", protocol.metrics.outbound_total + 1))
        assert in_home("a", counts) == base_a
        assert in_home("b", counts) == (base_b[0], base_b[1] + 1)

        adapter_a = in_home("a", _bare_adapter)
        adapter_a._metrics.inbound_total += 1  # what A's request thread does, outside any scope
        assert in_home("a", counts) == (base_a[0] + 1, base_a[1])
        assert in_home("b", counts) == (base_b[0], base_b[1] + 1)

    def test_outcome_counters_from_an_unscoped_thread_land_in_the_owning_profile(self, tmp_path, monkeypatch):
        """_record_outcome runs on ThreadingHTTPServer threads (no profile scope): profile B's
        adapter must count completed/failed/latency/outbound into B, never the process home."""
        import threading

        from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
        from plugins.platforms.a2a import protocol

        token = set_hermes_home_override(str(tmp_path / "b"))
        try:
            adapter = _bare_adapter()
        finally:
            reset_hermes_home_override(token)
        monkeypatch.setattr(protocol, "persist_message", lambda *a, **k: None)
        monkeypatch.setattr(adapter, "_send_push_notification", lambda *a, **k: None)
        mine, default = protocol.metrics_for(hermes_home_key(tmp_path / "b")), protocol.metrics_for()
        assert adapter._metrics is mine and mine is not default

        def counts(m):
            return {n: getattr(m, n) for n in protocol.Metrics._COUNTERS} | {"latencies": len(m._latencies)}

        before_mine, before_default = counts(mine), counts(default)

        def http_thread():
            adapter._record_outcome("t-ok", "c1", "peer", protocol.STATE_COMPLETED, "ok", started=0.0)
            adapter._record_outcome("t-bad", "c1", "peer", protocol.STATE_FAILED, "no")

        worker = threading.Thread(target=http_thread)
        worker.start()
        worker.join()
        moved = {k: v - before_mine[k] for k, v in counts(mine).items() if v != before_mine[k]}
        assert moved == {"tasks_completed": 1, "tasks_failed": 1, "outbound_total": 1, "latencies": 1}
        assert counts(default) == before_default
