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


@pytest.mark.platforms("posix")
def test_safe_load_has_the_c_parser_where_its_wheel_ships():
    # Every YAML read (plugin manifests, config) goes through safe_load, and the pure parser
    # cost ~5 s of gateway startup on 1 vCPU. ruamel.yaml 0.18.16 only declares its C parser
    # below 3.14, so on 3.14 it exists only because pyproject.toml declares ruamel.yaml.clib.
    import platform
    import sys

    wheel_ships = sys.platform == "darwin" or (
        sys.platform == "linux" and platform.machine() in {"x86_64", "aarch64", "i686"})
    if platform.python_implementation() != "CPython" or not wheel_ships:
        pytest.skip("ruamel.yaml.clib ships no wheel for this interpreter or machine")
    import _ruamel_yaml  # the module safe_load's pure=False parser binds to


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


_NO_DOT_FLOAT_PROBE = """
import json

import hermes_yaml

print(json.dumps({
    "safe": hermes_yaml.safe_load("threshold: 5e-1\\ntarget_ratio: 2e-1\\n"),
    "roundtrip": hermes_yaml.roundtrip_yaml().load("threshold: 5e-1\\n"),
}))
"""


def test_bare_exponent_floats_are_not_linted_onto_stderr():
    # YAML 1.1's float type wants a dot in the mantissa, so ruamel warns about scalars like "5e-1"
    # even though it parses them correctly as 0.5 — ~28 lines of stderr per scalar, per load, in
    # every process that reads config. A config with eight of them put 32 warning blocks (~900
    # lines) around a 14-line `hermes status`, burying the output and breaking stderr parsers.
    #
    # A fresh interpreter is the only honest way to assert this: once a warning has been shown for
    # a call site, the same process will not show it again, so an in-process check cannot tell a
    # suppressed lint from an already-consumed one.
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(yaml.__file__).resolve().parent
    proc = subprocess.run([sys.executable, "-c", _NO_DOT_FLOAT_PROBE],
                          cwd=repo_root, capture_output=True, text=True, check=False)

    assert proc.returncode == 0, proc.stderr
    assert "MantissaNoDot" not in proc.stderr
    assert "Correct your float" not in proc.stderr
    # Silencing the lint must not change what the scalars parse to.
    loaded = json.loads(proc.stdout)
    assert loaded["safe"] == {"threshold": 0.5, "target_ratio": 0.2}
    assert loaded["roundtrip"] == {"threshold": 0.5}


def test_the_float_lint_filter_is_narrow():
    # The 1.1 float lint is answered for that one category only — a blanket ignore would also hide
    # the diagnostics that make a bad config visible in the first place.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("still visible", UserWarning)

    assert [entry.category for entry in caught] == [UserWarning]
