"""Protect archive accounting, not report wording or markup."""
from scripts.ci.cache_report import parse_cache_log


def test_cache_accounting() -> None:
    log = """##[group]Run actions/cache@fixture
##[endgroup]
Cache hit for: hit
Received 0 of 100 (0.0%), 0.0 MBs/sec
Received 100 of 100 (100.0%), 1.0 MBs/sec
Cache restored from key: hit
##[group]Run actions/cache@fixture
##[endgroup]
Cache not found for input keys: miss
Post job cleanup.
Sent 0 of 200 (0.0%), 0.0 MBs/sec
Sent 200 of 200 (100.0%), 1.0 MBs/sec
Cache saved with key: miss
##[group]Run actions/cache@fixture
##[endgroup]
Cache hit for: failed
Received 80 of 80 (100.0%), 1.0 MBs/sec
##[warning]Failed to restore: tar failed
Post job cleanup.
Sent 90 of 90 (100.0%), 1.0 MBs/sec
##[group]Run actions/cache@fixture
##[endgroup]
Cache hit for: unknown
Cache restored from key: unknown
"""
    entries = parse_cache_log(log)
    assert [(e["restore_status"], e["restored_bytes"], e["save_status"], e["saved_bytes"])
            for e in entries] == [
        ("exact", 100, "not-attempted", None),
        ("miss", None, "saved", 200),
        ("failed", None, "unknown", None),
        ("exact", None, "not-attempted", None),
    ]
