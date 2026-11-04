"""``read_yaml_layers`` parses an unchanged user config.yaml once, and every caller gets its own copy.

The gateway loader and the cron ticker read the same profile files on every tick. Before this seam
cached, each read re-parsed the file in pure Python (GATEWAY-LOOP-STALL: the cron-scheduler thread
held the GIL for that parse in the 07:00 burst). The cache is keyed by the backend's config signature,
so an edit reparses, including a same-size rewrite inside one timestamp tick. The managed overlay and
``${VAR}`` expansion run on every call, and no caller can reach the cached document.
"""

import threading
import time

import pytest

from gateway.config_loader import read_yaml_layers

BASE = "model:\n  default: alpha\n"
EXPECTED = {"model": {"default": "alpha"}}


class _FrozenTime:
    """``utils``' view of the ``time`` module with ``time_ns`` frozen at one tick."""

    def __init__(self, now):
        self.now = now

    def time_ns(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def coarse_clock(monkeypatch):
    """Every stat reports one frozen mtime/ctime tick, as a coarse filesystem clock does, so a
    same-size rewrite inside the test keeps the stat signature. The racy window is frozen too, so the
    content digest (when the key uses one) stays in force for the whole test."""
    import utils

    real_signature = utils.file_signature
    tick = time.time_ns()
    monkeypatch.setattr(
        utils, "file_signature", lambda st: (tick, *real_signature(st)[1:3], tick)
    )
    monkeypatch.setattr(utils, "time", _FrozenTime(tick))


@pytest.fixture
def parse_counter(monkeypatch):
    """Record the file of every parse through the shared safe reader (the leaf of each config read)."""
    import utils

    real = utils.fast_safe_load
    parsed = []

    def counting(stream):
        parsed.append(getattr(stream, "name", ""))
        return real(stream)

    monkeypatch.setattr(utils, "fast_safe_load", counting)
    return parsed


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    profile = tmp_path / "profile"
    profile.mkdir()
    return profile


def _config_parses(parsed):
    return [name for name in parsed if name.endswith("config.yaml")]


def test_three_reads_of_an_unchanged_config_parse_once(home, parse_counter):
    (home / "config.yaml").write_text(BASE, encoding="utf-8")
    for _ in range(3):
        assert read_yaml_layers(home) == EXPECTED
    assert len(_config_parses(parse_counter)) == 1


def test_an_edited_config_is_reparsed(home, parse_counter):
    path = home / "config.yaml"
    path.write_text(BASE, encoding="utf-8")
    assert read_yaml_layers(home) == EXPECTED
    path.write_text("model:\n  default: alpha-but-longer\n", encoding="utf-8")
    assert read_yaml_layers(home) == {"model": {"default": "alpha-but-longer"}}
    assert len(_config_parses(parse_counter)) == 2


def test_same_size_rewrite_in_one_timestamp_tick_is_reparsed(home, coarse_clock):
    path = home / "config.yaml"
    path.write_text("model:\n  default: aaaaa\n", encoding="utf-8")
    assert read_yaml_layers(home)["model"]["default"] == "aaaaa"
    # Same length, same frozen stat signature: only the content differs.
    path.write_text("model:\n  default: bbbbb\n", encoding="utf-8")
    assert read_yaml_layers(home)["model"]["default"] == "bbbbb"


def test_caller_mutation_does_not_reach_the_next_read(home):
    (home / "config.yaml").write_text(BASE, encoding="utf-8")
    first = read_yaml_layers(home)
    first["model"]["default"] = "mutated"
    first["injected"] = {"x": 1}
    second = read_yaml_layers(home)
    assert second == EXPECTED
    assert second is not first


def test_a_malformed_config_raises_on_every_read_and_is_never_cached(
    home, parse_counter
):
    from ruamel.yaml.error import YAMLError

    (home / "config.yaml").write_text("model: [unclosed\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(YAMLError):
            read_yaml_layers(home)
    assert len(_config_parses(parse_counter)) == 2


def test_an_absent_config_is_an_empty_layer(home):
    assert read_yaml_layers(home) == {}


def test_env_refs_expand_on_every_read_of_an_unchanged_file(
    home, monkeypatch, parse_counter
):
    (home / "config.yaml").write_text(
        "platforms:\n  hook:\n    token: ${HERMES_T32_TOKEN}\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_T32_TOKEN", "first")
    assert read_yaml_layers(home)["platforms"]["hook"]["token"] == "first"
    monkeypatch.setenv("HERMES_T32_TOKEN", "second")
    assert read_yaml_layers(home)["platforms"]["hook"]["token"] == "second"
    assert len(_config_parses(parse_counter)) == 1


def test_managed_overlay_applies_on_every_read(
    home, tmp_path, monkeypatch, parse_counter
):
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (home / "config.yaml").write_text(BASE, encoding="utf-8")
    (managed / "config.yaml").write_text(
        "model:\n  default: pinned\n", encoding="utf-8"
    )
    assert read_yaml_layers(home)["model"]["default"] == "pinned"
    (managed / "config.yaml").write_text(
        "model:\n  default: repinned-longer\n", encoding="utf-8"
    )
    assert read_yaml_layers(home)["model"]["default"] == "repinned-longer"
    assert len(_config_parses(parse_counter)) == 1


@pytest.mark.parametrize(
    "clock_offset_ns", [0, 10_000_000_000], ids=["racy-clock", "settled-clock"]
)
def test_eight_concurrent_readers_see_consistent_results_from_one_parse(
    home, parse_counter, monkeypatch, clock_offset_ns
):
    # The racy window closes 2 s after the write. A clock that crosses it mid-run changes the cache
    # key, so readers straddling the boundary re-parse (correct, but not one parse). Freeze the clock
    # so the window is constant for the whole run and the count below is exact.
    import utils

    monkeypatch.setattr(utils, "time", _FrozenTime(time.time_ns() + clock_offset_ns))
    (home / "config.yaml").write_text(BASE, encoding="utf-8")
    assert read_yaml_layers(home) == EXPECTED  # warm
    errors, matches = [], []
    barrier = threading.Barrier(8)

    def reader():
        try:
            barrier.wait()
            for _ in range(25):
                scratch = read_yaml_layers(home)
                scratch["model"]["default"] = (
                    "scratch"  # must not leak into any other reader
                )
                matches.append(read_yaml_layers(home) == EXPECTED)
        except Exception as exc:  # collected for the assertion below, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert len(matches) == 200 and all(matches)
    assert len(_config_parses(parse_counter)) == 1


def test_a_bom_prefixed_config_reads_as_the_utf8_sig_parse(home):
    from utils import fast_safe_load

    path = home / "config.yaml"
    path.write_text("\ufeff" + BASE, encoding="utf-8")
    with open(path, encoding="utf-8-sig") as f:
        expected = fast_safe_load(f)
    assert read_yaml_layers(home) == expected == EXPECTED


def test_each_profile_home_keeps_its_own_parse(tmp_path, monkeypatch, parse_counter):
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    profiles = {}
    for name, value in (("one", "alpha"), ("two", "bravo")):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            f"model:\n  default: {value}\n", encoding="utf-8"
        )
        profiles[home] = value
    for _ in range(2):
        for home, value in profiles.items():
            assert read_yaml_layers(home) == {"model": {"default": value}}
    assert len(_config_parses(parse_counter)) == 2


def test_a_caller_mutation_of_a_set_leaf_does_not_reach_the_cache(home):
    # _expand_env_vars rebuilds dicts and lists but returns other leaves by identity, so only the deep
    # copy in read_yaml_layers keeps a caller's change to a set (a YAML !!set) out of the cache.
    (home / "config.yaml").write_text(
        "tags: !!set\n  ? alpha\n  ? beta\n", encoding="utf-8"
    )
    first = read_yaml_layers(home)
    assert first["tags"] == {"alpha", "beta"}
    first["tags"].add("mutated")
    assert read_yaml_layers(home)["tags"] == {"alpha", "beta"}
