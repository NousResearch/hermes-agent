"""The fast-load entry point follows the shared YAML policy."""

import io

import pytest

import hermes_yaml as yaml
from utils import fast_safe_load


_DOCS = [
    "",
    "a: 1\nb: two\nc: 3.5\n",
    "list: [1, 2, 3]\nnested:\n  k: v\n  flag: true\n  empty: null\n",
    "name: skill-x\nmetadata:\n  hermes:\n    tags: [alpha, beta]\n    category: devops\n",
    "- one\n- two\n- three\n",
    "scalar string",
    "flags: [on, off, yes, no, y, n]\n",
]


def test_equivalent_to_safe_load_for_strings():
    for doc in _DOCS:
        assert fast_safe_load(doc) == yaml.safe_load(doc), repr(doc)


def test_equivalent_to_safe_load_for_file_objects():
    for doc in _DOCS:
        assert fast_safe_load(io.StringIO(doc)) == yaml.safe_load(io.StringIO(doc)), repr(doc)


def test_empty_document_returns_none():
    assert fast_safe_load("") is None


def test_duplicate_keys_are_rejected_instead_of_silently_overwriting():
    with pytest.raises(yaml.YAMLError):
        fast_safe_load("model: first\nmodel: second\n")


def test_rejects_arbitrary_python_objects_like_safe_load():
    with pytest.raises(yaml.YAMLError):
        fast_safe_load("!!python/object/apply:builtins.str ['must not construct']\n")
