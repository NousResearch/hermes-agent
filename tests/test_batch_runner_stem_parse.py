"""Guard batch_file stem parsing used in run() join-output logging."""

from pathlib import Path


def _batch_num(stem: str) -> str:
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else "?"


def test_batch_num_happy_path():
    assert _batch_num(Path("batch_12.jsonl").stem) == "12"


def test_batch_num_missing_underscore():
    assert _batch_num(Path("batchonly.jsonl").stem) == "?"


def test_batch_num_extra_parts():
    assert _batch_num(Path("batch_3_extra.jsonl").stem) == "3"
