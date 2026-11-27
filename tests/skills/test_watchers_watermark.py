"""Tests for the watchers skill's shared watermark helper.

The bounded seen-ID set is what keeps a watcher from re-reporting items it has
already delivered. Once a feed saturates the cap, every run has to evict
something, and evicting the wrong entries is silent: no error, exit code 0,
just an old item showing up as new.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WATERMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/devops/watchers/scripts/_watermark.py"
)


def _seed_state(state_dir: Path, name: str, ids: list[str]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{name}.json").write_text(
        json.dumps({"seen_ids": ids, "first_run": False}), encoding="utf-8"
    )


@pytest.fixture
def watermark_module(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHER_STATE_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "watchers_watermark_test", WATERMARK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_truncation_evicts_oldest_first(watermark_module, tmp_path):
    _seed_state(tmp_path, "feed", [f"id-{i}" for i in range(10)])
    wm = watermark_module.Watermark.load("feed", max_seen=10)

    wm.filter_new([{"id": "id-10"}, {"id": "id-11"}], id_key="id")

    assert wm.seen == [f"id-{i}" for i in range(2, 12)]


def test_retained_ids_are_not_re_emitted(watermark_module, tmp_path):
    _seed_state(tmp_path, "feed", [f"id-{i}" for i in range(10)])
    wm = watermark_module.Watermark.load("feed", max_seen=10)
    wm.filter_new([{"id": "id-10"}, {"id": "id-11"}], id_key="id")

    # id-2 is the oldest survivor: seeing it again must stay quiet. Before the
    # fix it was evicted roughly at random, and a feed still carrying it would
    # get it delivered a second time.
    assert wm.filter_new([{"id": "id-2"}], id_key="id") == []


def test_duplicate_stored_ids_do_not_inflate_the_cap(watermark_module, tmp_path):
    _seed_state(tmp_path, "feed", ["a", "b", "b", "c"])
    wm = watermark_module.Watermark.load("feed", max_seen=4)

    wm.filter_new([{"id": "d"}], id_key="id")

    assert wm.seen == ["a", "b", "c", "d"]


# Run in a subprocess: PYTHONHASHSEED is read at interpreter start, so the
# randomization this guards against cannot be toggled from inside a test.
_DETERMINISM_SNIPPET = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("wm", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
wm = module.Watermark.load("feed", max_seen=10)
wm.filter_new([{"id": "id-10"}, {"id": "id-11"}], id_key="id")
print(",".join(wm.seen))
"""


def test_truncation_is_deterministic_across_hash_seeds(tmp_path):
    _seed_state(tmp_path, "feed", [f"id-{i}" for i in range(10)])

    results = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, WATCHER_STATE_DIR=str(tmp_path))
        proc = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SNIPPET, str(WATERMARK_PATH)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        results.add(proc.stdout.strip())

    assert len(results) == 1, f"eviction set depends on the hash seed: {results}"
