"""Invariants for utils.fast_safe_load.

fast_safe_load is a drop-in for yaml.safe_load that prefers the libyaml
CSafeLoader C extension for speed. These tests assert the behavior contract
(it parses identically to safe_load across input shapes), not a snapshot of
any particular document.
"""

import io

import yaml

from utils import fast_safe_load, _get_fast_yaml_loader


_DOCS = [
    "",  # empty document -> None
    "a: 1\nb: two\nc: 3.5\n",
    "list: [1, 2, 3]\nnested:\n  k: v\n  flag: true\n  empty: null\n",
    "name: skill-x\nmetadata:\n  hermes:\n    tags: [alpha, beta]\n    category: devops\n",
    "- one\n- two\n- three\n",  # top-level sequence
    "scalar string",  # bare scalar
]


def test_equivalent_to_safe_load_for_strings():
    for doc in _DOCS:
        assert fast_safe_load(doc) == yaml.safe_load(doc), repr(doc)


def test_equivalent_to_safe_load_for_file_objects():
    for doc in _DOCS:
        assert fast_safe_load(io.StringIO(doc)) == yaml.safe_load(io.StringIO(doc)), repr(doc)


def test_empty_document_returns_none():
    # Callers rely on ``fast_safe_load(...) or {}`` — empty must be falsy.
    assert fast_safe_load("") is None


def test_prefers_c_loader_when_available():
    loader = _get_fast_yaml_loader()
    # If libyaml is compiled in, we must be using the C loader; otherwise the
    # pure-Python SafeLoader is an acceptable fallback. Either way it must be a
    # safe loader (never the unsafe full Loader).
    c_loader = getattr(yaml, "CSafeLoader", None)
    if c_loader is not None:
        assert loader is c_loader
    else:
        assert loader is yaml.SafeLoader


def test_rejects_arbitrary_python_objects_like_safe_load():
    # Safe loaders must not construct arbitrary Python objects. This tag is
    # accepted by the unsafe Loader but rejected by Safe/CSafe loaders.
    dangerous = "!!python/object/apply:os.system ['echo pwned']\n"
    try:
        fast_safe_load(dangerous)
        raised = False
    except yaml.YAMLError:
        raised = True
    assert raised, "fast_safe_load must reject python/object tags like safe_load"
