from benchmarks.interface import BenchmarkConfig, BenchmarkableStore
from benchmarks.runner import preflight_backend, register_backend


class PassingPreflightStore(BenchmarkableStore):
    def __init__(self, **kwargs):
        self.items = []

    def store(self, content, category="factual", scope="global", importance=0.5):
        self.items.append(content)

    def recall(self, query, top_k=10, scope=None):
        return self.items[:top_k]

    def simulate_time(self, days):
        pass

    def simulate_access(self, content_substring):
        pass

    def consolidate(self):
        pass

    def get_stats(self):
        return {"fact_count": len(self.items)}

    def reset(self):
        self.items.clear()


class FailingPreflightStore(PassingPreflightStore):
    def store(self, content, category="factual", scope="global", importance=0.5):
        raise RuntimeError("boom")


def test_preflight_backend_passes_for_store_recall_reset_cycle():
    register_backend("passing-preflight-test", PassingPreflightStore)
    config = BenchmarkConfig(backend_name="passing-preflight-test")

    result = preflight_backend(config)

    assert result["ok"] is True
    assert result["steps"]["reset_before"] == "ok"
    assert result["steps"]["store"] == "ok"
    assert result["steps"]["recall"] == "ok"
    assert result["steps"]["reset_after"] == "ok"


def test_preflight_backend_reports_failures_without_raising():
    register_backend("failing-preflight-test", FailingPreflightStore)
    config = BenchmarkConfig(backend_name="failing-preflight-test")

    result = preflight_backend(config)

    assert result["ok"] is False
    assert "RuntimeError" in result["error"]
