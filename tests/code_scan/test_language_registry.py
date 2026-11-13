"""Tests for scripts/code-scan/language_registry.py."""
import os
import sys
import pytest
from pathlib import Path

# Ensure project root is on sys.path so we can import the script module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "code-scan"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from language_registry import (
    LANGUAGE_BY_EXT,
    CATEGORY_BY_EXT,
    INFRA_FILENAMES,
    FRAMEWORK_MANIFEST_PATTERNS,
    get_language,
    get_category,
    detect_frameworks,
)


class TestLanguageLookup:
    """Test get_language() for various extensions and filenames."""

    def test_python_extension(self):
        assert get_language("main.py") == "python"

    def test_typescript_extension(self):
        assert get_language("index.ts") == "typescript"

    def test_typescriptx_extension(self):
        assert get_language("component.tsx") == "typescript"

    def test_javascript_extension(self):
        assert get_language("app.js") == "javascript"

    def test_rust_extension(self):
        assert get_language("lib.rs") == "rust"

    def test_go_extension(self):
        assert get_language("main.go") == "go"

    def test_ruby_extension(self):
        assert get_language("setup.rb") == "ruby"

    def test_shell_extension(self):
        assert get_language("deploy.sh") == "shell"

    def test_yaml_extension(self):
        assert get_language("config.yaml") == "yaml"

    def test_json_extension(self):
        assert get_language("data.json") == "json"

    def test_markdown_extension(self):
        assert get_language("README.md") == "markdown"

    def test_unknown_extension(self):
        assert get_language("file.xyz") == "unknown"

    def test_no_extension(self):
        assert get_language("Makefile") == "unknown"

    def test_unknown_dockerfile(self):
        assert get_language("Dockerfile") == "dockerfile"

    def test_pyproject_toml(self):
        assert get_language("pyproject.toml") == "toml"

    def test_path_with_python_extension(self):
        assert get_language("src/main.py") == "python"


class TestCategoryLookup:
    """Test get_category() for various files and paths."""

    def test_python_code_category(self):
        assert get_category("main.py") == "code"

    def test_typescript_code_category(self):
        assert get_category("index.ts") == "code"

    def test_html_template_category(self):
        assert get_category("index.html") == "template"

    def test_yaml_config_category(self):
        assert get_category("config.yaml") == "config"

    def test_markdown_doc_category(self):
        assert get_category("README.md") == "doc"

    def test_dockerfile_infra_category(self):
        assert get_category("Dockerfile") == "infra"

    def test_tf_infra_category(self):
        assert get_category("main.tf") == "infra"

    def test_test_path_detection(self):
        """Files in test directories should be categorized as 'test'."""
        assert get_category("tests/test_main.py") == "test"
        assert get_category("test_utils.py") == "test"

    def test_pyproject_infra(self):
        """pyproject.toml is an infra filename."""
        assert get_category("pyproject.toml") == "infra"

    def test_package_json_infra(self):
        """package.json is an infra filename."""
        assert get_category("package.json") == "infra"

    def test_unknown_category(self):
        assert get_category("file.xyz") == "other"


class TestFrameworkDetection:
    """Test detect_frameworks() against fixture projects."""

    def test_detect_react_in_mixed_project(self):
        fixtures_dir = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"
        mixed_dir = fixtures_dir / "mixed_project"
        frameworks = detect_frameworks(str(mixed_dir))
        assert "react" in frameworks

    def test_detect_nextjs_in_mixed_project(self):
        fixtures_dir = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"
        mixed_dir = fixtures_dir / "mixed_project"
        frameworks = detect_frameworks(str(mixed_dir))
        assert "nextjs" in frameworks

    def test_no_frameworks_in_small_project(self):
        fixtures_dir = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"
        small_dir = fixtures_dir / "small_project"
        frameworks = detect_frameworks(str(small_dir))
        assert isinstance(frameworks, list)

    def test_missing_project_root(self):
        frameworks = detect_frameworks("/nonexistent/path/xyz")
        assert frameworks == []
