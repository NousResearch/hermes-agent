"""
Tests for skills/research/arxiv/scripts/search_arxiv.py's arXiv id parsing.

parse_arxiv_id() must split old-style ids (which may contain a literal 'v' in
the archive name, e.g. solv-int, adap-org, chao-dyn, patt-sol, comp-gas) on
the trailing version suffix only, not on the first 'v' anywhere in the string.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "research"
    / "arxiv"
    / "scripts"
    / "search_arxiv.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("search_arxiv", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_arxiv_id_modern_versioned():
    mod = load_module()
    assert mod.parse_arxiv_id("2402.03300v2") == ("2402.03300", "v2")


def test_parse_arxiv_id_modern_unversioned():
    mod = load_module()
    assert mod.parse_arxiv_id("2402.03300") == ("2402.03300", "")


def test_parse_arxiv_id_old_style_archive_name_contains_v():
    mod = load_module()
    # Regression: a naive split on the first 'v' truncates this to 'sol'.
    assert mod.parse_arxiv_id("solv-int/9701001v1") == ("solv-int/9701001", "v1")


def test_parse_arxiv_id_old_style_no_v_in_archive_name():
    mod = load_module()
    assert mod.parse_arxiv_id("hep-th/0601001v3") == ("hep-th/0601001", "v3")
