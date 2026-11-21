"""A failed SWR refresh must release its key so the next picker open retries.

Drives the real ``cached_fetch_api_models`` and the real ``_spawn_swr_refresh`` worker over a real
temporary JSON cache; only the thread start and the network result are replaced.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from hermes_cli import models


class _QueuedThread:
    """Thread double: ``start`` queues the target so the test runs it deterministically."""
    queue: list = []

    def __init__(self, target, **_kwargs):
        self._target = target

    def start(self):
        self.queue.append(self._target)


@pytest.mark.parametrize("failure", ["raises", "returns_none"])
def test_failed_refresh_releases_key_and_next_open_recovers(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "hermes").mkdir()
    queue: list = []
    monkeypatch.setattr(_QueuedThread, "queue", queue)
    monkeypatch.setattr(models, "threading", SimpleNamespace(Thread=_QueuedThread, Lock=models.threading.Lock))
    monkeypatch.setattr(models, "_swr_refresh_inflight", set())

    api_key, url = "synthetic-key", "http://synthetic.invalid/v1"
    fp = models._custom_endpoint_fingerprint(api_key, None, None)
    expired = time.time() - models._PROVIDER_MODELS_CACHE_TTL - 60
    target_key = f"custom:{url}#{fp}"
    rows = {
        target_key: {"fp": fp, "at": expired, "models": ["old-a"], "native_catalog": False},
        "custom:http://sibling.invalid/v1#abc": {"fp": "abc", "at": expired - 5, "models": ["sib-a"]},
    }
    cache_path = tmp_path / "hermes" / "provider_models_cache.json"
    cache_path.write_text(json.dumps(rows))
    before = cache_path.read_bytes()

    fetches: list = []
    results = iter([OSError("synthetic outage") if failure == "raises" else None, ["new-a", "new-b"]])

    def fetch():
        fetches.append(1)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    def picker():
        return models.cached_fetch_api_models(api_key, url, cache_only=True, fetch_models=fetch)

    assert picker() == ["old-a"]
    assert (len(queue), len(fetches)) == (1, 0)

    queue.pop(0)()  # worker fails
    assert len(fetches) == 1
    assert cache_path.read_bytes() == before

    assert picker() == ["old-a"]
    assert (len(queue), len(fetches)) == (1, 1)  # retry queued despite the earlier failure

    queue.pop(0)()  # worker succeeds
    assert len(fetches) == 2

    assert picker() == ["new-a", "new-b"]
    assert (len(queue), len(fetches)) == (0, 2)  # fresh row: no further fetch or refresh
    on_disk = json.loads(cache_path.read_text())
    assert on_disk[target_key]["models"] == ["new-a", "new-b"]
    assert on_disk["custom:http://sibling.invalid/v1#abc"] == rows["custom:http://sibling.invalid/v1#abc"]
