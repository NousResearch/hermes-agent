"""Regression: sample_and_compress must not materialize the full dataset pool."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path


def _load_sample_and_compress():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sample_and_compress.py"
    spec = importlib.util.spec_from_file_location("sample_and_compress", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ImmediatePool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, fn, iterable, chunksize=1):
        for item in iterable:
            yield fn(item)


def test_sample_from_datasets_does_not_sample_from_full_materialized_pool(monkeypatch):
    """#84703: reservoir-sample qualifying rows; do not random.sample(all_filtered)."""
    sac = _load_sample_and_compress()

    population = [{"conversations": [{"value": f"row-{i}"}]} for i in range(40)]

    def fake_load(_name):
        return list(population)

    def fake_iter(_name):
        yield from population

    monkeypatch.setattr(sac, "load_dataset_from_hf", fake_load)
    if hasattr(sac, "iter_dataset_entries"):
        monkeypatch.setattr(sac, "iter_dataset_entries", fake_iter)
    monkeypatch.setattr(sac, "_count_tokens_for_entry", lambda entry: (entry, 20_000))
    monkeypatch.setattr("multiprocessing.Pool", _ImmediatePool)

    sample_sizes: list[int] = []
    real_sample = sac.random.sample

    def spy_sample(population_arg, k):
        sample_sizes.append(len(population_arg))
        return real_sample(population_arg, k)

    monkeypatch.setattr(sac.random, "sample", spy_sample)

    sampled = sac.sample_from_datasets(
        ["fake/ds"],
        total_samples=5,
        min_tokens=1,
        num_proc=1,
        seed=0,
    )

    assert sample_sizes == [], (
        "sample_from_datasets materialized a full qualifying pool and called "
        f"random.sample on {sample_sizes}"
    )
    assert len(sampled) == 5


def test_iter_dataset_entries_labels_non_streaming_fallback(monkeypatch, capsys):
    """TypeError from streaming=True must warn and say Loaded, not Streamed."""
    import sys
    import types

    sac = _load_sample_and_compress()

    def fake_load_dataset(name, split="train", streaming=False):
        if streaming:
            raise TypeError("streaming is not supported")
        return [{"conversations": [{"value": "row"}]}]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    rows = list(sac.iter_dataset_entries("fake/ds"))
    captured = capsys.readouterr().out
    assert len(rows) == 1
    assert "does not support streaming" in captured
    assert "Loaded 1 entries" in captured
    assert "Streamed" not in captured


def test_merge_output_normalizes_pretty_json_to_jsonl(tmp_path):
    sac = _load_sample_and_compress()
    src_dir = tmp_path / "parts"
    src_dir.mkdir()
    (src_dir / "pretty.jsonl").write_text(
        '{ "id": 1 }\n',
        encoding="utf-8",
    )
    out = tmp_path / "merged.jsonl"

    sac.merge_output_to_single_jsonl(src_dir, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0] == '{"id": 1}'
    assert json.loads(lines[0]) == {"id": 1}


def test_merge_output_streams_jsonl(tmp_path):
    sac = _load_sample_and_compress()
    src_dir = tmp_path / "parts"
    src_dir.mkdir()
    (src_dir / "a.jsonl").write_text(
        json.dumps({"id": 1}) + "\n" + json.dumps({"id": 2}) + "\n",
        encoding="utf-8",
    )
    (src_dir / "b.jsonl").write_text(json.dumps({"id": 3}) + "\n", encoding="utf-8")
    out = tmp_path / "merged.jsonl"

    sac.merge_output_to_single_jsonl(src_dir, out)

    rows = [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [1, 2, 3]


def test_reservoir_add_caps_memory_at_k():
    sac = _load_sample_and_compress()
    rng = random.Random(0)
    reservoir: list[int] = []
    for seen, value in enumerate(range(200), start=1):
        sac.reservoir_add(reservoir, value, seen, 8, rng)
    assert len(reservoir) == 8
    assert set(reservoir) <= set(range(200))
