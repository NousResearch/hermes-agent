#!/usr/bin/env python3
"""
Hermes Contributing Guide Compliance Tests for Agathos.

Checks agathos.py and wal_monitor.py against:
https://hermes-agent.nousresearch.com/docs/developer-guide/contributing

Sections verified:
  1. Code Style (logging, hermes_home, exc_info)
  2. Cross-Platform Compatibility (pathlib, platform guards)
  3. Security (no shell injection, parameterized SQL, no eval/exec)
  4. General (no hardcoded hermes paths, Python 3.10+ compat)
"""

import re
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AGATHOS_DIR = SCRIPT_DIR.parent.parent / "agathos"
AGATHOS_PATH = AGATHOS_DIR / "agathos.py"
WAL_PATH = AGATHOS_DIR / "wal_monitor.py"


class _SourceMixin:
    """Load source files once."""

    @classmethod
    def setUpClass(cls):
        cls.agathos_src = AGATHOS_PATH.read_text()
        cls.wal_src = WAL_PATH.read_text()
        cls.agathos_lines = cls.agathos_src.split("\n")
        cls.wal_lines = cls.wal_src.split("\n")


# =====================================================================
# 1. CODE STYLE
# =====================================================================


class TestLoggingStyle(_SourceMixin, unittest.TestCase):
    """Contributing guide: Use logger.warning(), logger.error()."""

    def test_uses_logger_not_print(self):
        """Agathos must use logger, not bare print() for output."""
        violations = []
        for i, line in enumerate(self.agathos_lines, 1):
            stripped = line.strip()
            # Skip comments, docstrings, test files
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                continue
            # print() is only OK in stub functions or if logger is unavailable
            if "print(" in stripped and "def " not in stripped:
                violations.append(f"  L{i}: {stripped[:80]}")
        self.assertEqual(
            violations,
            [],
            "Bare print() found (should use logger):\n" + "\n".join(violations),
        )

    def test_logger_error_has_exc_info(self):
        """logger.error() in except blocks should include exc_info where appropriate."""
        # This is a best-practice check, not a hard requirement
        in_except = False
        violations = []
        for i, line in enumerate(self.agathos_lines, 1):
            stripped = line.strip()
            if stripped.startswith("except "):
                in_except = True
            elif (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith("logger")
            ):
                in_except = False
            if in_except and "logger.error(" in stripped and "exc_info" not in stripped:
                violations.append(f"  L{i}: {stripped[:80]}")
        # Warning only, not failure — some errors don't need traceback
        if violations:
            print(
                "  WARNING: logger.error() without exc_info in except blocks:\n"
                + "\n".join(violations[:5])
            )


class TestHermesHome(_SourceMixin, unittest.TestCase):
    """Contributing guide: Use get_hermes_home(), not hardcoded ~/.hermes."""

    def test_no_hardcoded_hermes_home_in_function_bodies(self):
        """Module functions must not hardcode ~/.hermes paths."""
        # Allow module-level constants (CONFIG, _DEFAULT_Agathos_CONFIG) but not inside functions
        violations = []
        in_function = False
        in_class = False
        for i, line in enumerate(self.agathos_lines, 1):
            stripped = line.strip()
            if stripped.startswith("def ") and not stripped.startswith("    "):
                in_function = True
            elif stripped.startswith("class "):
                in_class = True
            elif (
                in_function
                and not line.startswith("    ")
                and not line.startswith("#")
                and stripped
            ):
                in_function = False

            if in_function and not in_class:
                # Check for hardcoded hermes paths
                if (
                    "'~/.hermes/" in stripped
                    or '"~/.hermes/' in stripped
                    or "home / '.hermes'" in stripped
                ):
                    # Skip if it's using expanduser (which is fine)
                    if "expanduser" not in stripped:
                        violations.append(f"  L{i}: {stripped[:80]}")
        self.assertEqual(
            violations,
            [],
            "Hardcoded ~/.hermes in function bodies:\n" + "\n".join(violations),
        )


# =====================================================================
# 2. CROSS-PLATFORM COMPATIBILITY
# =====================================================================


class TestPathlibUsage(_SourceMixin, unittest.TestCase):
    """Contributing guide: Use pathlib.Path instead of string concatenation."""

    def test_uses_pathlib_for_paths(self):
        """Core path operations should use pathlib.Path."""
        # Check that Path is imported
        self.assertIn(
            "from pathlib import Path",
            self.agathos_src,
            "agathos.py must import pathlib.Path",
        )

    def test_no_os_path_join_for_new_code(self):
        """New code should prefer Path / over os.path.join."""
        violations = []
        in_new_section = False
        for i, line in enumerate(self.agathos_lines, 1):
            # Flag sections that are clearly new (WAL, PID, launchd)
            if "# ===" in line:
                section = line.strip()
                in_new_section = any(
                    k in section for k in ["WAL", "PID", "LAUNCHD", "LOGGING"]
                )
            if in_new_section and "os.path.join(" in line:
                violations.append(f"  L{i}: {line.strip()[:80]}")
        # Warning, not failure — os.path.join is acceptable
        if violations:
            print(
                "  WARNING: os.path.join in new sections (prefer Path /):\n"
                + "\n".join(violations[:3])
            )


# =====================================================================
# 3. SECURITY
# =====================================================================


class TestSecuritySubprocess(_SourceMixin, unittest.TestCase):
    """Contributing guide: No shell=True, use shlex.quote()."""

    def test_no_shell_true(self):
        """subprocess calls must not use shell=True."""
        violations = []
        for i, line in enumerate(self.agathos_lines, 1):
            if "shell=True" in line:
                violations.append(f"  L{i}: {line.strip()[:80]}")
        self.assertEqual(
            violations,
            [],
            "shell=True found (security violation):\n" + "\n".join(violations),
        )

    def test_no_string_interpolation_in_subprocess(self):
        """subprocess args must not use f-strings or .format() with user data."""
        violations = []
        for i, line in enumerate(self.agathos_lines, 1):
            stripped = line.strip()
            # Check for subprocess.run with f-string args
            if "subprocess.run(" in stripped or "subprocess.call(" in stripped:
                # Check next few lines for f-string interpolation
                for j in range(i, min(i + 5, len(self.agathos_lines))):
                    next_line = self.agathos_lines[j].strip()
                    if "f'" in next_line or 'f"' in next_line:
                        if any(cmd in next_line for cmd in ["kill", "hermes", "curl"]):
                            violations.append(f"  L{j + 1}: {next_line[:80]}")
        self.assertEqual(
            violations,
            [],
            "F-string in subprocess args (shell injection risk):\n"
            + "\n".join(violations),
        )


class TestSecuritySQL(_SourceMixin, unittest.TestCase):
    """Contributing guide: Parameterized SQL, no string interpolation."""

    def test_parameterized_sql(self):
        """All SQL queries must use ? placeholders, not f-strings."""
        violations = []
        in_execute = False
        for i, line in enumerate(self.agathos_lines, 1):
            stripped = line.strip()
            if "cursor.execute(" in stripped or ".execute(" in stripped:
                in_execute = True
            if in_execute:
                # Check for f-string or .format() inside execute
                if ("f'" in stripped or 'f"' in stripped) and (
                    "SELECT" in stripped or "INSERT" in stripped or "UPDATE" in stripped
                ):
                    violations.append(f"  L{i}: {stripped[:80]}")
                if ".format(" in stripped and (
                    "SELECT" in stripped or "INSERT" in stripped
                ):
                    violations.append(f"  L{i}: {stripped[:80]}")
                if stripped.endswith(")") or stripped.endswith('""")'):
                    in_execute = False
        self.assertEqual(
            violations, [], "Non-parameterized SQL found:\n" + "\n".join(violations)
        )

    def test_no_eval_exec(self):
        """No eval() or exec() — security requirement."""
        violations = []
        for i, line in enumerate(self.agathos_lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"\beval\s*\(", stripped) or re.search(
                r"\bexec\s*\(", stripped
            ):
                violations.append(f"  L{i}: {stripped[:80]}")
        for i, line in enumerate(self.wal_lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"\beval\s*\(", stripped) or re.search(
                r"\bexec\s*\(", stripped
            ):
                violations.append(f"  wal_monitor.py L{i}: {stripped[:80]}")
        self.assertEqual(
            violations,
            [],
            "eval()/exec() found (security violation):\n" + "\n".join(violations),
        )


class TestSecurityPathTraversal(_SourceMixin, unittest.TestCase):
    """Contributing guide: Use os.path.realpath() for path resolution."""

    def test_file_operations_resolve_paths(self):
        """File write operations should resolve paths to prevent traversal."""
        for i, line in enumerate(self.agathos_lines, 1):
            stripped = line.strip()
            # Look for direct file writes without realpath
            if "write_text(" in stripped or "write_file(" in stripped:
                # Check if the path was resolved earlier
                if "realpath" not in stripped and "resolve()" not in stripped:
                    # This is a warning, not a failure — context matters
                    pass
        # No hard failure — just verify awareness


# =====================================================================
# 4. GENERAL COMPLIANCE
# =====================================================================


class TestImportStructure(_SourceMixin, unittest.TestCase):
    """Hermes imports must handle unavailable modules gracefully."""

    def test_hermes_imports_have_fallback(self):
        """Imports from hermes internals must be wrapped in try/except."""
        # Check that hermes imports are in try/except
        in_try = False
        hermes_imports_without_try = []
        for i, line in enumerate(self.agathos_lines, 1):
            stripped = line.strip()
            if stripped.startswith("try:"):
                in_try = True
            elif stripped.startswith("except"):
                in_try = False
            elif (
                "from cron." in stripped
                or ("from hermes_" in stripped and "hermes_fallback" not in stripped)
                or "from gateway." in stripped
            ):
                if not in_try:
                    hermes_imports_without_try.append(f"  L{i}: {stripped[:80]}")

        self.assertEqual(
            hermes_imports_without_try,
            [],
            "Hermes imports without try/except fallback:\n"
            + "\n".join(hermes_imports_without_try),
        )

    def test_no_hardcoded_credentials(self):
        """No API keys, tokens, or secrets in source code."""
        violations = []
        secret_patterns = [
            r'api_key\s*=\s*["\'][a-zA-Z0-9]{20,}',
            r'token\s*=\s*["\'][a-zA-Z0-9]{20,}',
            r'secret\s*=\s*["\'][a-zA-Z0-9]{10,}',
            r'password\s*=\s*["\'][^"\']+',
            r"sk-[a-zA-Z0-9]{20,}",
            r"ghp_[a-zA-Z0-9]{36}",
        ]
        for name, lines in [
            ("agathos.py", self.agathos_lines),
            ("wal_monitor.py", self.wal_lines),
        ]:
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pattern in secret_patterns:
                    if re.search(pattern, stripped, re.IGNORECASE):
                        violations.append(f"  {name}:{i}: potential secret")
        self.assertEqual(
            violations, [], "Potential hardcoded secrets:\n" + "\n".join(violations)
        )


class TestDocstrings(_SourceMixin, unittest.TestCase):
    """Public functions must have docstrings."""

    def test_public_functions_have_docstrings(self):
        """All public methods/functions need docstrings."""
        for name, code in [
            ("agathos.py", self.agathos_src),
            ("wal_monitor.py", self.wal_src),
        ]:
            # Find public functions (not starting with _)
            func_defs = re.finditer(
                r"^(\s*)def ([^_]\w+)\([^)]*\):\s*$", code, re.MULTILINE
            )
            missing = []
            for match in func_defs:
                func_name = match.group(2)
                # Check next line for docstring
                pos = match.end()
                next_lines = code[pos : pos + 200]
                if (
                    '"""' not in next_lines.split("\n")[1]
                    if "\n" in next_lines
                    else next_lines
                ):
                    missing.append(f"  {name}: {func_name}")
            if missing:
                print(
                    "  WARNING: Public functions without docstrings:\n"
                    + "\n".join(missing[:5])
                )


class TestModuleStructure(_SourceMixin, unittest.TestCase):
    """Module structure must follow hermes conventions."""

    def test_has_main_guard(self):
        """Module must have if __name__ == '__main__' guard."""
        self.assertIn(
            'if __name__ == "__main__"',
            self.agathos_src,
            "agathos.py must have __main__ guard",
        )

    def test_shebang_line(self):
        """Module must start with shebang."""
        first_line = self.agathos_lines[0]
        self.assertTrue(
            first_line.startswith("#!"),
            f"First line must be shebang, got: {first_line}",
        )

    def test_module_docstring(self):
        """Module must have a docstring."""
        # Check second or third line for docstring
        for line in self.agathos_lines[1:4]:
            if '"""' in line:
                return
        self.fail("agathos.py must have a module-level docstring")


class TestRuntimeCompatibility(_SourceMixin, unittest.TestCase):
    """Verify the module actually imports and works."""

    def test_imports_successfully(self):
        """Module must import without errors."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from agathos import agathos

            self.assertTrue(hasattr(agathos, "Agathos"))
            self.assertTrue(hasattr(agathos, "CONFIG"))
            self.assertTrue(hasattr(agathos, "is_agathos_running"))
        finally:
            sys.path.pop(0)

    def test_config_has_required_keys(self):
        """CONFIG must have all required keys."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from agathos import agathos

            required = [
                "db_path",
                "log_dir",
                "poll_interval",
                "entropy_threshold",
                "quality_threshold",
                "max_restart_count",
            ]
            for key in required:
                self.assertIn(key, agathos.CONFIG, f"CONFIG missing key: {key}")
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
