"""Contracts for Hermes' shared YAML reader and writer."""

import io
from concurrent.futures import ThreadPoolExecutor

import pytest

import hermes_yaml as yaml


def test_safe_load_accepts_existing_config_boolean_spellings():
    document = "flags: [on, off, yes, no, true, false]\nquoted: ['off', 'yes']\n"
    expected = {"flags": [True, False, True, False, True, False], "quoted": ["off", "yes"]}
    for stream in (document, document.encode(), io.StringIO(document), io.BytesIO(document.encode())):
        assert yaml.safe_load(stream) == expected
    assert yaml.safe_load("") is None


def test_safe_load_rejects_python_object_construction():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("!!python/object/apply:builtins.str ['must not construct']")


def test_safe_dump_preserves_data_and_readable_block_layout():
    data = {"z": [{"mode": "off", "choice": "y", "label": "こんにちは 🦀"}], "a": "yes"}
    text = yaml.safe_dump(data, sort_keys=False)
    assert yaml.safe_load(text) == data
    assert text.startswith("z:\n  - ")
    assert "こんにちは 🦀" in text
    stream = io.StringIO()
    assert yaml.safe_dump(data, stream, sort_keys=False) is None
    assert stream.getvalue() == text
    with pytest.raises(yaml.YAMLError):
        yaml.safe_dump({"object": object()})


def test_safe_dump_honors_the_options_used_by_callers():
    data = {"zebra": {"zed": 1, "alpha": 2}, "alpha": "hé"}
    for sort_keys in (False, True):
        loaded = yaml.safe_load(yaml.safe_dump(data, sort_keys=sort_keys))
        assert list(loaded) == (sorted(data) if sort_keys else list(data))
        assert list(loaded["zebra"]) == (sorted(data["zebra"]) if sort_keys else list(data["zebra"]))
    escaped = yaml.safe_dump(data, allow_unicode=False)
    assert "hé" not in escaped
    assert yaml.safe_load(escaped) == data
    flow = yaml.safe_dump(data, default_flow_style=True, width=100000)
    assert flow.startswith("{") and len(flow.splitlines()) == 1
    assert yaml.safe_load(flow) == data


def test_roundtrip_preserves_comments_quotes_and_scalar_types():
    editor = yaml.roundtrip_yaml()
    original = '# keep this\nname: "hello 🦀"  # note\nflag: off\n'
    data = editor.load(original)
    assert data["flag"] is False
    data["mode"] = "off"
    stream = io.StringIO()
    editor.dump(data, stream)
    text = stream.getvalue()
    assert text.startswith('# keep this\nname: "hello 🦀"  # note\n')
    assert yaml.safe_load(text) == {"name": "hello 🦀", "flag": False, "mode": "off"}


def test_native_yaml11_scalars_and_duplicate_key_policy():
    for load in (yaml.safe_load, yaml.roundtrip_yaml().load):
        assert load("[y, n, Y, N, 'y', 'n']") == [True, False, True, False, "y", "n"]
        with pytest.raises(yaml.YAMLError):
            load("model: first\nmodel: second\n")
    # Merge keys override defaults, not duplicates in the mapping itself.
    merged = yaml.safe_load("defaults: &defaults {enabled: true}\nlocal: {<<: *defaults, enabled: false}\n")
    assert merged["local"]["enabled"] is False


def test_parallel_calls_do_not_share_parser_or_emitter_state():
    def roundtrip(index):
        data = {"index": index, "words": ["yes", "no", "on", "off", "y", "n"]}
        text = yaml.safe_dump(data, sort_keys=bool(index % 2))
        assert yaml.safe_load(text) == data
        assert yaml.roundtrip_yaml().load(text) == data
        return data

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert [data["index"] for data in pool.map(roundtrip, range(32))] == list(range(32))
    # A failed parse/dump must not poison the next operation.
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("key: [unterminated")
    with pytest.raises(yaml.YAMLError):
        yaml.safe_dump(object())
    assert yaml.safe_load(yaml.safe_dump({"healthy": True})) == {"healthy": True}
