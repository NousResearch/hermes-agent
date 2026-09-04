"""Regression tests for the read_user_config_raw() parse memo.

The memo exists because the desktop sidebar fan-out (list_profiles over 50+
profiles) re-parsed the same config.yaml files dozens of times per refresh and
saturated both cores of a 2-vCore VPS (HostHighCPU). These tests pin the
semantics the memo must never break:

* a hit returns a FRESH dict — write-back mutation cannot corrupt the cache;
* an on-disk edit (or an atomic replace onto a new inode) reparses;
* unparseable YAML raises on EVERY call — errors are never cached;
* a missing file returns {} on every call without poisoning the memo.
"""

import pytest

from hermes_cli import config as config_mod


@pytest.fixture(autouse=True)
def _cold_memo():
    config_mod._USER_CONFIG_RAW_CACHE.clear()
    yield
    config_mod._USER_CONFIG_RAW_CACHE.clear()


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_hit_returns_deepcopy_so_mutation_cannot_leak(tmp_path):
    p = tmp_path / "config.yaml"
    _write(p, "display:\n  skin: warm\n")

    first = config_mod.read_user_config_raw(p)
    first["display"]["skin"] = "mutated"
    first["injected"] = True

    second = config_mod.read_user_config_raw(p)
    assert second["display"]["skin"] == "warm"
    assert "injected" not in second
    assert second is not first


def test_edit_invalidates_via_fingerprint(tmp_path):
    p = tmp_path / "config.yaml"
    _write(p, "model: aaaa\n")
    assert config_mod.read_user_config_raw(p)["model"] == "aaaa"

    # Different size AND mtime_ns — the fingerprint changes, memo misses.
    _write(p, "model: bbbbbbbbbb\n")
    assert config_mod.read_user_config_raw(p)["model"] == "bbbbbbbbbb"


def test_atomic_replace_onto_new_inode_reparses(tmp_path):
    p = tmp_path / "config.yaml"
    _write(p, "model: old\n")
    assert config_mod.read_user_config_raw(p)["model"] == "old"

    # save_config() writes a temp file and os.replace()s it — new inode, and
    # on coarse-mtime filesystems possibly even a matching mtime. (dev, ino)
    # in the fingerprint is what catches this.
    tmp = tmp_path / "config.yaml.tmp"
    _write(tmp, "model: new\n")
    tmp.replace(p)
    assert config_mod.read_user_config_raw(p)["model"] == "new"


def test_parse_errors_are_never_cached(tmp_path):
    p = tmp_path / "config.yaml"
    _write(p, "key: [unclosed\n")
    with pytest.raises(Exception):
        config_mod.read_user_config_raw(p)
    # Still raises on the second call — a cached error would mask recovery.
    with pytest.raises(Exception):
        config_mod.read_user_config_raw(p)

    # And once the file is fixed, the next call sees the fixed content.
    _write(p, "key: [ok]\n")
    assert config_mod.read_user_config_raw(p) == {"key": ["ok"]}


def test_missing_file_returns_empty_every_time(tmp_path):
    p = tmp_path / "config.yaml"
    assert config_mod.read_user_config_raw(p) == {}
    _write(p, "model: late\n")
    assert config_mod.read_user_config_raw(p)["model"] == "late"


def test_non_dict_root_normalizes_to_empty(tmp_path):
    p = tmp_path / "config.yaml"
    _write(p, "- just\n- a\n- list\n")
    assert config_mod.read_user_config_raw(p) == {}


def test_memo_reuses_parse_across_calls(tmp_path, monkeypatch):
    """The whole point: a second read of an unchanged file skips the parser."""
    p = tmp_path / "config.yaml"
    _write(p, "model: memoized\n")

    calls = []
    real_load = config_mod.fast_safe_load

    def counting_load(stream):
        calls.append(1)
        return real_load(stream)

    monkeypatch.setattr(config_mod, "fast_safe_load", counting_load)
    config_mod.read_user_config_raw(p)
    config_mod.read_user_config_raw(p)
    assert len(calls) == 1
