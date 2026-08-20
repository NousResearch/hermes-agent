"""Tests for the fingerprints module (Phase 3 D1).

Strict TDD: every function tested. Tests written first, then implementation.
"""
import json
import os
import sys
import tempfile
import textwrap

from pathlib import Path

# Add code-scan scripts to sys.path so `fingerprints` can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "code-scan"))

import pytest

FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "fingerprints"
)


# ── Import test ──────────────────────────────────────────────────────────

class TestModuleImport:
    """Verify the module can be imported and all public functions exist."""

    def test_imports(self):
        from fingerprints import (
            extract_fingerprint,
            _extract_functions,
            _extract_classes,
            _extract_content_hash,
            load_fingerprint_file,
            save_fingerprint_file,
            compare_fingerprints,
            build_fingerprint_map,
            get_fingerprint_path,
            _extract_imports_from_source,
        )


# ── _extract_content_hash ───────────────────────────────────────────────

class TestExtractContentHash:
    """Content hash computation."""

    def test_same_file_same_hash(self):
        from fingerprints import _extract_content_hash
        path = os.path.join(FIXTURES_DIR, "original", "main.py")
        h1 = _extract_content_hash(path)
        h2 = _extract_content_hash(path)
        assert h1 == h2

    def test_different_content_different_hash(self):
        from fingerprints import _extract_content_hash
        p1 = os.path.join(FIXTURES_DIR, "original", "main.py")
        p2 = os.path.join(FIXTURES_DIR, "cosmetic", "main.py")
        assert _extract_content_hash(p1) != _extract_content_hash(p2)

    def test_empty_file_valid_hash(self):
        from fingerprints import _extract_content_hash
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            path = f.name
        try:
            h = _extract_content_hash(path)
            assert h.startswith("sha256:")
            assert len(h) == len("sha256:") + 64
        finally:
            os.unlink(path)

    def test_unchanged_files_identical_hash(self):
        from fingerprints import _extract_content_hash
        p1 = os.path.join(FIXTURES_DIR, "original", "main.py")
        p2 = os.path.join(FIXTURES_DIR, "unchanged", "main.py")
        assert _extract_content_hash(p1) == _extract_content_hash(p2)


# ── _extract_functions ─────────────────────────────────────────────────

class TestExtractFunctions:
    """Function name extraction via regex."""

    def test_simple_functions(self):
        from fingerprints import _extract_functions
        source = textwrap.dedent("""\
            import os

            def main():
                pass

            def helper():
                pass
        """)
        result = _extract_functions(source)
        assert result == ["helper", "main"]

    def test_async_functions(self):
        from fingerprints import _extract_functions
        source = textwrap.dedent("""\
            async def fetch():
                pass

            def sync_func():
                pass
        """)
        result = _extract_functions(source)
        assert result == ["fetch", "sync_func"]

    def test_nested_functions_captured(self):
        from fingerprints import _extract_functions
        source = textwrap.dedent("""\
            def outer():
                def inner():
                    pass
                pass
        """)
        result = _extract_functions(source)
        assert result == ["inner", "outer"]

    def test_deduplicated_and_sorted(self):
        from fingerprints import _extract_functions
        source = textwrap.dedent("""\
            def zebra():
                pass
            def alpha():
                pass
            def alpha():
                pass
        """)
        result = _extract_functions(source)
        assert result == ["alpha", "zebra"]

    def test_no_functions(self):
        from fingerprints import _extract_functions
        source = "x = 42\ny = 'hello'\n"
        assert _extract_functions(source) == []


# ── _extract_classes ────────────────────────────────────────────────────

class TestExtractClasses:
    """Class name extraction via regex."""

    def test_simple_classes(self):
        from fingerprints import _extract_classes
        source = textwrap.dedent("""\
            class Foo:
                pass

            class Bar:
                pass
        """)
        result = _extract_classes(source)
        assert result == ["Bar", "Foo"]

    def test_nested_classes(self):
        from fingerprints import _extract_classes
        source = textwrap.dedent("""\
            class Outer:
                class Inner:
                    pass
        """)
        result = _extract_classes(source)
        assert result == ["Inner", "Outer"]

    def test_deduplicated_and_sorted(self):
        from fingerprints import _extract_classes
        source = textwrap.dedent("""\
            class Zebra:
                pass
            class Alpha:
                pass
            class Alpha:
                pass
        """)
        result = _extract_classes(source)
        assert result == ["Alpha", "Zebra"]

    def test_no_classes(self):
        from fingerprints import _extract_classes
        source = "def foo():\n    pass\n"
        assert _extract_classes(source) == []


# ── _extract_imports_from_source ────────────────────────────────────────

class TestExtractImportsFromSource:
    """Import extraction for enrichment."""

    def test_python_imports(self):
        from fingerprints import _extract_imports_from_source
        source = textwrap.dedent("""\
            import os
            import sys
            from pathlib import Path
            from collections import OrderedDict
        """)
        result = _extract_imports_from_source(source, "python")
        assert result == ["collections", "os", "pathlib", "sys"]

    def test_python_sorts_and_dedups(self):
        from fingerprints import _extract_imports_from_source
        source = textwrap.dedent("""\
            import sys
            import os
            from os import path
        """)
        result = _extract_imports_from_source(source, "python")
        assert result == ["os", "sys"]

    def test_javascript_imports(self):
        from fingerprints import _extract_imports_from_source
        source = textwrap.dedent("""\
            import { useState } from 'react';
            import express from 'express';
        """)
        result = _extract_imports_from_source(source, "javascript")
        assert result == ["express", "react"]

    def test_js_require_and_dynamic_imports(self):
        """Test require() and dynamic import() extraction for JS/TS."""
        from fingerprints import _extract_imports_from_source
        source = textwrap.dedent("""\
            const fs = require('fs');
            const path = require('path');
            async function load() {
                const mod = await import('./dynamic-module');
            }
            import { useState } from 'react';
        """)
        result = _extract_imports_from_source(source, "javascript")
        assert "fs" in result
        assert "path" in result
        assert "react" in result
        # Dynamic import with relative path – stripped of ./ prefix
        assert "dynamic-module" in result

    def test_ts_require_and_dynamic_imports(self):
        """Test require() and dynamic import() extraction for TypeScript."""
        from fingerprints import _extract_imports_from_source
        source = textwrap.dedent("""\
            const axios = require('axios');
            import('./lazy-component').then(m => m.default());
            import Vue from 'vue';
        """)
        result = _extract_imports_from_source(source, "typescript")
        assert "axios" in result
        assert "vue" in result
        assert "lazy-component" in result

    def test_unknown_language_empty_list(self):
        from fingerprints import _extract_imports_from_source
        source = "import foo"
        result = _extract_imports_from_source(source, "rust")
        assert result == []


# ── extract_fingerprint ─────────────────────────────────────────────────

class TestExtractFingerprint:
    """Full fingerprint extraction from a file."""

    def _get_fp(self, rel_path, imports=None):
        from fingerprints import extract_fingerprint
        full_path = os.path.join(FIXTURES_DIR, rel_path)
        content = open(full_path, "rb").read()
        line_count = content.count(b"\n") + (1 if not content.endswith(b"\n") and content else 0)
        return extract_fingerprint(
            file_path=full_path,
            scan_root=FIXTURES_DIR,
            line_count=line_count,
            size_bytes=len(content),
            imports=imports,
        )

    def test_all_required_keys_present(self):
        fp = self._get_fp("original/main.py")
        for key in ("content_hash", "line_count", "size_bytes", "functions", "classes", "imports"):
            assert key in fp, f"Missing key: {key}"

    def test_content_hash_format(self):
        fp = self._get_fp("original/main.py")
        assert fp["content_hash"].startswith("sha256:")
        assert len(fp["content_hash"]) == len("sha256:") + 64

    def test_line_count_and_size(self):
        fp = self._get_fp("original/main.py")
        assert fp["line_count"] > 0
        assert fp["size_bytes"] > 0
        assert isinstance(fp["line_count"], int)
        assert isinstance(fp["size_bytes"], int)

    def test_functions_sorted_deduped(self):
        fp = self._get_fp("original/main.py")
        assert fp["functions"] == sorted(fp["functions"])
        assert len(fp["functions"]) == len(set(fp["functions"]))

    def test_classes_sorted_deduped(self):
        fp = self._get_fp("original/main.py")
        assert fp["classes"] == sorted(fp["classes"])
        assert len(fp["classes"]) == len(set(fp["classes"]))

    def test_imports_sorted_deduped(self):
        fp = self._get_fp("original/main.py")
        assert fp["imports"] == sorted(fp["imports"])
        assert len(fp["imports"]) == len(set(fp["imports"]))

    def test_original_functions(self):
        fp = self._get_fp("original/main.py")
        assert "main" in fp["functions"]
        assert "helper_utility" in fp["functions"]

    def test_original_classes(self):
        fp = self._get_fp("original/main.py")
        assert "ConfigParser" in fp["classes"]

    def test_provided_imports_used(self):
        fp = self._get_fp("original/main.py", imports=["custom_import"])
        assert fp["imports"] == ["custom_import"]

    def test_structural_func_has_extra_function(self):
        fp = self._get_fp("structural_func/main.py")
        assert "new_function" in fp["functions"]

    def test_structural_class_has_extra_class(self):
        fp = self._get_fp("structural_class/main.py")
        assert "NewClass" in fp["classes"]

    def test_structural_import_has_extra_import(self):
        fp = self._get_fp("structural_import/main.py")
        assert "new_module" in fp["imports"]

    def test_mixed_composite(self):
        fp = self._get_fp("mixed/main.py")
        assert len(fp["functions"]) >= 4
        assert len(fp["classes"]) >= 2
        assert len(fp["imports"]) >= 2


# ── save_fingerprint_file ───────────────────────────────────────────────

class TestSaveFingerprintFile:
    """Fingerprint file persistence."""

    def _sample_files(self):
        return {
            "main.py": {
                "content_hash": "sha256:abc123",
                "line_count": 10,
                "size_bytes": 100,
                "functions": ["main"],
                "classes": [],
                "imports": ["os"],
            }
        }

    def test_creates_parent_dirs(self):
        from fingerprints import save_fingerprint_file
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path = os.path.join(tmpdir, "nested", "deep", "fingerprints.json")
            project_root = tmpdir
            result = save_fingerprint_file(fp_path, project_root, self._sample_files())
            assert os.path.isfile(result)

    def test_valid_json_schema(self):
        from fingerprints import save_fingerprint_file
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path = os.path.join(tmpdir, "fingerprints.json")
            project_root = os.path.abspath(tmpdir)
            save_fingerprint_file(fp_path, project_root, self._sample_files())
            with open(fp_path) as f:
                data = json.load(f)
            assert data["schema_version"] == "1.0.0"
            assert data["project_root"] == project_root
            assert "captured_at" in data
            assert "main.py" in data["files"]

    def test_captured_at_is_iso_8601(self):
        from fingerprints import save_fingerprint_file
        from datetime import datetime
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path = os.path.join(tmpdir, "fingerprints.json")
            save_fingerprint_file(fp_path, tmpdir, self._sample_files())
            with open(fp_path) as f:
                data = json.load(f)
            # Should parse as ISO 8601
            dt = datetime.fromisoformat(data["captured_at"].replace("Z", "+00:00"))
            assert dt is not None

    def test_returns_path(self):
        from fingerprints import save_fingerprint_file
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path = os.path.join(tmpdir, "fingerprints.json")
            result = save_fingerprint_file(fp_path, tmpdir, self._sample_files())
            assert result == fp_path

    def test_absolute_project_root(self):
        from fingerprints import save_fingerprint_file
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path = os.path.join(tmpdir, "fingerprints.json")
            save_fingerprint_file(fp_path, tmpdir, self._sample_files())
            with open(fp_path) as f:
                data = json.load(f)
            assert os.path.isabs(data["project_root"])


# ── load_fingerprint_file ───────────────────────────────────────────────

class TestLoadFingerprintFile:
    """Fingerprint file loading."""

    def test_valid_file_loads(self):
        from fingerprints import save_fingerprint_file, load_fingerprint_file
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path = os.path.join(tmpdir, "fingerprints.json")
            files = {"main.py": {"content_hash": "sha256:abc", "line_count": 10, "size_bytes": 100, "functions": [], "classes": [], "imports": []}}
            save_fingerprint_file(fp_path, tmpdir, files)
            data = load_fingerprint_file(fp_path)
            assert data is not None
            assert data["schema_version"] == "1.0.0"

    def test_missing_file_returns_none(self):
        from fingerprints import load_fingerprint_file
        result = load_fingerprint_file("/nonexistent/path/fingerprint.json")
        assert result is None

    def test_corrupt_json_returns_none(self):
        from fingerprints import load_fingerprint_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json!!!")
            path = f.name
        try:
            assert load_fingerprint_file(path) is None
        finally:
            os.unlink(path)

    def test_missing_schema_version_returns_none(self):
        from fingerprints import load_fingerprint_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"files": {}, "captured_at": "2026-01-01T00:00:00Z"}, f)
            path = f.name
        try:
            assert load_fingerprint_file(path) is None
        finally:
            os.unlink(path)

    def test_wrong_schema_version_returns_none(self):
        from fingerprints import load_fingerprint_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"schema_version": "2.0.0", "files": {}}, f)
            path = f.name
        try:
            assert load_fingerprint_file(path) is None
        finally:
            os.unlink(path)


# ── compare_fingerprints ────────────────────────────────────────────────

class TestCompareFingerprints:
    """Change classification."""

    def _make_fp(self, path="main.py", content_hash="sha256:abc",
                 functions=None, classes=None, imports=None):
        return {
            "path": path,
            "files": {
                path: {
                    "content_hash": content_hash,
                    "line_count": 10,
                    "size_bytes": 100,
                    "functions": functions or [],
                    "classes": classes or [],
                    "imports": imports or [],
                }
            }
        }

    def test_unchanged_same_hash(self):
        from fingerprints import compare_fingerprints
        old = self._make_fp(content_hash="sha256:same_hash")
        new = self._make_fp(content_hash="sha256:same_hash")
        result = compare_fingerprints(old, new)
        assert result["main.py"] == "UNCHANGED"

    def test_cosmetic_hash_differs_structures_same(self):
        from fingerprints import compare_fingerprints
        old = self._make_fp(
            content_hash="sha256:old_hash",
            functions=["main"],
            classes=["Config"],
            imports=["os"],
        )
        new = self._make_fp(
            content_hash="sha256:new_hash",
            functions=["main"],
            classes=["Config"],
            imports=["os"],
        )
        result = compare_fingerprints(old, new)
        assert result["main.py"] == "COSMETIC"

    def test_structural_function_added(self):
        from fingerprints import compare_fingerprints
        old = self._make_fp(
            content_hash="sha256:old",
            functions=["main"], classes=[], imports=[],
        )
        new = self._make_fp(
            content_hash="sha256:new",
            functions=["main", "new_func"], classes=[], imports=[],
        )
        result = compare_fingerprints(old, new)
        assert result["main.py"] == "STRUCTURAL"

    def test_structural_class_added(self):
        from fingerprints import compare_fingerprints
        old = self._make_fp(
            content_hash="sha256:old",
            functions=[], classes=["OldClass"], imports=[],
        )
        new = self._make_fp(
            content_hash="sha256:new",
            functions=[], classes=["NewClass"], imports=[],
        )
        result = compare_fingerprints(old, new)
        assert result["main.py"] == "STRUCTURAL"

    def test_structural_import_added(self):
        from fingerprints import compare_fingerprints
        old = self._make_fp(
            content_hash="sha256:old",
            functions=[], classes=[], imports=["os"],
        )
        new = self._make_fp(
            content_hash="sha256:new",
            functions=[], classes=[], imports=["os", "sys"],
        )
        result = compare_fingerprints(old, new)
        assert result["main.py"] == "STRUCTURAL"

    def test_new_file_is_structural(self):
        from fingerprints import compare_fingerprints
        old = {"files": {}}
        new = self._make_fp(path="new_file.py")
        result = compare_fingerprints(old, new)
        assert result["new_file.py"] == "STRUCTURAL"

    def test_deleted_file_is_structural(self):
        from fingerprints import compare_fingerprints
        old = self._make_fp()
        new = {"files": {}}
        result = compare_fingerprints(old, new)
        assert result["main.py"] == "STRUCTURAL"

    def test_mixed_file_scenario(self):
        from fingerprints import compare_fingerprints
        old = {
            "files": {
                "a.py": self._make_fp("a.py", functions=["foo"])["files"]["a.py"],
                "b.py": self._make_fp("b.py", functions=["bar"])["files"]["b.py"],
            }
        }
        new = {
            "files": {
                "a.py": self._make_fp("a.py", "sha256:different", functions=["foo", "baz"])["files"]["a.py"],
                "b.py": self._make_fp("b.py", functions=["bar"])["files"]["b.py"],
                "c.py": self._make_fp("c.py", functions=["qux"])["files"]["c.py"],
            }
        }
        result = compare_fingerprints(old, new)
        assert result["a.py"] == "STRUCTURAL"  # function added
        assert result["b.py"] == "UNCHANGED"   # same hash
        assert result["c.py"] == "STRUCTURAL"  # new file

    @classmethod
    def test_fixture_based_unchanged(cls):
        from fingerprints import extract_fingerprint, compare_fingerprints
        orig_path = os.path.join(FIXTURES_DIR, "original", "main.py")
        unch_path = os.path.join(FIXTURES_DIR, "unchanged", "main.py")
        for p in (orig_path, unch_path):
            content = open(p, "rb").read()
        fp1 = extract_fingerprint(orig_path, FIXTURES_DIR, line_count=1, size_bytes=1, imports=None)
        fp2 = extract_fingerprint(unch_path, FIXTURES_DIR, line_count=1, size_bytes=1, imports=None)
        result = compare_fingerprints(
            {"files": {"main.py": fp1}},
            {"files": {"main.py": fp2}},
        )
        assert result["main.py"] == "UNCHANGED"

    @classmethod
    def test_fixture_based_cosmetic(cls):
        from fingerprints import extract_fingerprint, compare_fingerprints
        orig_path = os.path.join(FIXTURES_DIR, "original", "main.py")
        cosm_path = os.path.join(FIXTURES_DIR, "cosmetic", "main.py")
        fp1 = extract_fingerprint(orig_path, FIXTURES_DIR, line_count=1, size_bytes=1, imports=None)
        fp2 = extract_fingerprint(cosm_path, FIXTURES_DIR, line_count=1, size_bytes=1, imports=None)
        result = compare_fingerprints(
            {"files": {"main.py": fp1}},
            {"files": {"main.py": fp2}},
        )
        assert result["main.py"] == "COSMETIC"

    @classmethod
    def test_fixture_based_structural_func(cls):
        from fingerprints import extract_fingerprint, compare_fingerprints
        orig_path = os.path.join(FIXTURES_DIR, "original", "main.py")
        sfunc_path = os.path.join(FIXTURES_DIR, "structural_func", "main.py")
        fp1 = extract_fingerprint(orig_path, FIXTURES_DIR, line_count=1, size_bytes=1, imports=None)
        fp2 = extract_fingerprint(sfunc_path, FIXTURES_DIR, line_count=1, size_bytes=1, imports=None)
        result = compare_fingerprints(
            {"files": {"main.py": fp1}},
            {"files": {"main.py": fp2}},
        )
        assert result["main.py"] == "STRUCTURAL"


# ── build_fingerprint_map ───────────────────────────────────────────────

class TestBuildFingerprintMap:
    """Build fingerprint map from scan data."""

    def _scan_data(self):
        return {
            "files": [
                {
                    "relative_path": "src/main.py",
                    "path": os.path.join(FIXTURES_DIR, "original", "main.py"),
                    "lines": 30,
                    "size_bytes": 608,
                }
            ]
        }

    def test_basic_build(self):
        from fingerprints import build_fingerprint_map
        result = build_fingerprint_map(self._scan_data(), FIXTURES_DIR, None)
        assert "src/main.py" in result
        entry = result["src/main.py"]
        assert "content_hash" in entry
        assert "functions" in entry
        assert "classes" in entry
        assert "imports" in entry

    def test_import_enrichment(self):
        from fingerprints import build_fingerprint_map
        scan_data = self._scan_data()
        import_data = {
            "src/main.py": ["custom_a", "custom_b"],
        }
        result = build_fingerprint_map(scan_data, FIXTURES_DIR, import_data)
        assert result["src/main.py"]["imports"] == ["custom_a", "custom_b"]

    def test_import_enrichment_real_phase2_schema(self):
        """Support Phase 2 extract_imports.py nested schema:
        {"schema_version": "1.0.0", "files": {"src/main.py": {"imports": [...]}}}"""
        from fingerprints import build_fingerprint_map
        scan_data = self._scan_data()
        import_data = {
            "schema_version": "1.0.0",
            "files": {
                "src/main.py": {"imports": ["nested_a", "nested_b"], "warnings": []},
            },
        }
        result = build_fingerprint_map(scan_data, FIXTURES_DIR, import_data)
        assert result["src/main.py"]["imports"] == ["nested_a", "nested_b"]

    def test_without_import_data_extracts_imports(self):
        from fingerprints import build_fingerprint_map
        result = build_fingerprint_map(self._scan_data(), FIXTURES_DIR, None)
        # Should auto-extract imports from the source
        assert isinstance(result["src/main.py"]["imports"], list)
        assert len(result["src/main.py"]["imports"]) > 0

    def test_all_fingerprint_keys_present(self):
        from fingerprints import build_fingerprint_map
        result = build_fingerprint_map(self._scan_data(), FIXTURES_DIR, None)
        fp = result["src/main.py"]
        for key in ("content_hash", "line_count", "size_bytes", "functions", "classes", "imports"):
            assert key in fp


# ── get_fingerprint_path ────────────────────────────────────────────────

class TestGetFingerprintPath:
    """Fingerprint path resolution."""

    def test_default_path(self):
        from fingerprints import get_fingerprint_path
        path = get_fingerprint_path("/tmp/testproj")
        assert path == os.path.join("/tmp/testproj", ".hermes", "code-state", "fingerprints.json")

    def test_absolute_project_root(self):
        from fingerprints import get_fingerprint_path
        path = get_fingerprint_path("/home/user/myrepo")
        assert path.startswith("/home/user/myrepo")
        assert path.endswith("fingerprints.json")
