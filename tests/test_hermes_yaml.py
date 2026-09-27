"""Contracts for Hermes' shared YAML reader and writer."""

import io
from concurrent.futures import ThreadPoolExecutor

import pytest

import hermes_yaml as yaml
from utils import fast_safe_load


@pytest.mark.parametrize("load", [yaml.safe_load, fast_safe_load])
def test_safe_load_accepts_existing_config_boolean_spellings(load):
    document = "flags: [on, off, yes, no, true, false]\nquoted: ['off', 'yes']\n"
    expected = {"flags": [True, False, True, False, True, False], "quoted": ["off", "yes"]}
    for stream in (document, document.encode(), io.StringIO(document), io.BytesIO(document.encode())):
        assert load(stream) == expected
    assert load("") is None


@pytest.mark.parametrize("load", [yaml.safe_load, fast_safe_load])
def test_safe_load_rejects_python_object_construction(load):
    with pytest.raises(yaml.YAMLError):
        load("!!python/object/apply:builtins.str ['must not construct']")


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
    for load in (yaml.safe_load, fast_safe_load, yaml.roundtrip_yaml().load):
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


@pytest.mark.parametrize("load", [yaml.safe_load, fast_safe_load])
def test_safe_load_keeps_legacy_id_shaped_scalars_strings(load):
    # Session IDs PyYAML dumped unquoted (``20260820_093237_089e44``) load back as floats under
    # ruamel's YAML 1.1 resolver, and the first profile.yaml rewrite persists the float (#124901).
    document = "ui_meta:\n  hermes-bots:\n    chat: 20260820_093237_089e44\ntoken: 987e654321\nversion: 1e5\n"
    loaded = load(document)
    assert loaded["ui_meta"]["hermes-bots"]["chat"] == "20260820_093237_089e44"
    assert loaded["token"] == "987e654321"
    assert loaded["version"] == "1e5"


@pytest.mark.parametrize("load", [yaml.safe_load, fast_safe_load])
def test_safe_load_keeps_reading_dotted_floats(load):
    # Only the dot-less unsigned exponent form is treated as a legacy string; every float
    # representation with a dot in the mantissa (all Python ``repr`` shapes) stays a float.
    document = "a: 1.5e5\nb: 2.026082009323709e+60\nc: .5\nd: 5.\ne: 1.0e-05\nf: .inf\ng: 1:30.5\n"
    loaded = load(document)
    assert loaded == {
        "a": 150000.0, "b": 2.026082009323709e60, "c": 0.5, "d": 5.0,
        "e": 1e-05, "f": float("inf"), "g": 90.5,
    }


def test_safe_dump_writes_dotted_mantissa_floats_for_roundtrip():
    # The emitter used to write ``1e-05``; under the PyYAML-parity read rule that reloads as a
    # string, silently re-typing the value on every save. Floats with an exponent now carry a
    # dot in the mantissa, so ``safe_dump`` output reloads as the same float.
    text = yaml.safe_dump({"v": 1e-05, "w": 2.026082009323709e60})
    assert "1.0e-05" in text and "2.026082009323709e+60" in text
    assert yaml.safe_load(text) == {"v": 1e-05, "w": 2.026082009323709e60}


def test_roundtrip_preserves_a_legacy_id_scalar_and_quotes_it_on_change():
    editor = yaml.roundtrip_yaml()
    original = "# keep\nui_meta:\n  hermes-bots:\n    chat: 20260820_093237_089e44\n"
    data = editor.load(original)
    assert data["ui_meta"]["hermes-bots"]["chat"] == "20260820_093237_089e44"
    stream = io.StringIO()
    editor.dump(data, stream)
    # The re-quote is deliberate: a bare ID-shaped scalar is exactly the form a YAML 1.1 reader
    # re-types as a float, so once the editor owns the line it writes the safe spelling.
    assert stream.getvalue() == "# keep\nui_meta:\n  hermes-bots:\n    chat: '20260820_093237_089e44'\n"
    data["ui_meta"]["hermes-bots"]["chat"] = "20260821_101112_089e44"
    data["threshold"] = 1e-05
    stream = io.StringIO()
    editor.dump(data, stream)
    text = stream.getvalue()
    # A reassigned ID-shaped string is quoted (bare, a YAML 1.1 reader would re-type it as a
    # float); exponents keep a dot so the value survives its own next load.
    assert "chat: '20260821_101112_089e44'" in text
    assert "threshold: 1.0e-05" in text
    assert yaml.safe_load(text)["ui_meta"]["hermes-bots"]["chat"] == "20260821_101112_089e44"
