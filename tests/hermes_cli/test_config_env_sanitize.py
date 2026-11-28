"""Sanitizer write-path regressions for #76571; no new reader contract."""
import os
from unittest.mock import patch

import pytest

from hermes_cli.config import _sanitize_env_lines, sanitize_env_file


class TestSanitizeEnvMultiline:
    def test_multiline_quoted_value_is_preserved_while_sibling_is_normalized(self):
        lines = [
            "  MULTI='opening  \n",
            "  continuation # value content  \n",
            "closing'  # comment\n",
            "  SINGLE=value  \n",
        ]

        assert _sanitize_env_lines(lines) == [
            "  MULTI='opening  \n",
            "  continuation # value content  \n",
            "closing'  # comment\n",
            "SINGLE=value\n",
        ]


    def test_double_quoted_multiline_value_with_escaped_quote_is_preserved(self):
        lines = [
            '  MULTI="opening  \n',
            '  escaped \\" quote # value content  \n',
            'closing"  # comment\n',
        ]

        assert _sanitize_env_lines(lines) == lines


    def test_unclosed_quote_leaves_entire_file_untouched(self):
        lines = [
            "  BROKEN='opening  \n",
            "  continuation # ambiguous  \n",
            "  SINGLE=value  \n",
        ]

        assert _sanitize_env_lines(lines) == lines


    def test_sanitize_env_file_preserves_multiline_dotenv_value(self, tmp_path):
        from dotenv import dotenv_values

        env_file = tmp_path / ".env"
        original = (
            "  MULTI='opening  \n"
            "  continuation # value content  \n"
            "closing'  # comment\n"
            "  SINGLE=value  \n"
        )
        env_file.write_text(original, encoding="utf-8")
        expected_value = dotenv_values(env_file, interpolate=False)["MULTI"]

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            fixes = sanitize_env_file()

        assert fixes == 1
        assert dotenv_values(env_file, interpolate=False)["MULTI"] == expected_value
        assert env_file.read_text(encoding="utf-8") == (
            "  MULTI='opening  \n"
            "  continuation # value content  \n"
            "closing'  # comment\n"
            "SINGLE=value\n"
        )



class TestSanitizeEnvLinesBoundaryAdditions:
    def test_unicode_line_separators_inside_a_value_are_not_physical_lines(self):
        # readlines() splits only on \n / \r / \r\n; str.splitlines() would also
        # split on U+2028, U+2029, U+0085 and \x0c and inject a real newline.
        for separator in ("\u2028", "\u2029", "\x85", "\x0c"):
            lines = [f"MULTI='left{separator}right'\n", "  SINGLE=value  \n"]

            assert _sanitize_env_lines(lines) == [
                f"MULTI='left{separator}right'\n",
                "SINGLE=value\n",
            ], repr(separator)

    def test_blank_lines_before_a_binding_do_not_make_it_multiline(self):
        lines = ["\n", "   \n", "  SINGLE=value  \n"]

        assert _sanitize_env_lines(lines) == ["\n", "   \n", "SINGLE=value\n"]

    def test_blank_lines_before_a_multiline_binding_keep_its_value_bytes(self):
        lines = ["\n", "  MULTI='x  \n", "  y'  \n", "  SINGLE=v \n"]

        assert _sanitize_env_lines(lines) == [
            "\n",
            "  MULTI='x  \n",
            "  y'  \n",
            "SINGLE=v\n",
        ]


@pytest.mark.parametrize("operation", ["sanitize", "save", "append", "remove"])
def test_sibling_write_preserves_existing_value_and_native_reader_view(tmp_path, monkeypatch, operation):
    from dotenv import dotenv_values
    from agent.secret_scope import _parse_env_text
    from hermes_cli.config import remove_env_value, save_env_value

    original = "\n  SANITIZER_MULTI='opening\u2028  \n  continuation # value  \nclosing'  \n  SIBLING=old  \n"
    path = tmp_path / ".env"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expected = dotenv_values(path, interpolate=False)["SANITIZER_MULTI"]
    native_before = _parse_env_text(original)["SANITIZER_MULTI"]

    if operation == "sanitize":
        assert sanitize_env_file() == 1
        assert sanitize_env_file() == 0
    elif operation == "save":
        save_env_value("SIBLING", "updated")
    elif operation == "append":
        save_env_value("ADDED", "new")
    else:
        assert remove_env_value("SIBLING")

    content = path.read_text(encoding="utf-8")
    assert content.startswith(original.rsplit("  SIBLING=", 1)[0])
    assert dotenv_values(path, interpolate=False)["SANITIZER_MULTI"] == expected
    assert _parse_env_text(content)["SANITIZER_MULTI"] == native_before
    assert sanitize_env_file() == 0


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_startup_load_preserves_multiline_value_on_disk_and_in_environment(tmp_path, monkeypatch, encoding):
    from hermes_cli.env_loader import load_hermes_dotenv

    original = "\r\n  SANITIZER_MULTI='opening\u2028  \r\n  continuation # value  \r\nclosing'  \r\n  SIBLING=value  \r\n"
    path = tmp_path / ".env"
    path.write_bytes(original.encode(encoding))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("SANITIZER_MULTI", raising=False)
    monkeypatch.delenv("SIBLING", raising=False)

    assert load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False) == [path]

    assert os.environ["SANITIZER_MULTI"] == "opening\u2028  \n  continuation # value  \nclosing"
    expected = original.replace("\r\n", "\n").replace("  SIBLING=value  \n", "SIBLING=value\n")
    assert path.read_bytes() == expected.encode("utf-8")
    assert sanitize_env_file() == 0


def test_line_alignment_idempotence_and_native_reader_contract():
    from io import StringIO
    from dotenv import dotenv_values
    from agent.secret_scope import _parse_env_text

    text = "\n\n  SINGLE=one  \n  MULTI='left\u2028  \n  # right'  \n# comment  \n"
    lines = StringIO(text).readlines()
    sanitized = _sanitize_env_lines(lines)
    result = "".join(sanitized)

    assert len(sanitized) == len(lines)
    assert _sanitize_env_lines(sanitized) == sanitized
    assert dotenv_values(stream=StringIO(result), interpolate=False) == dotenv_values(stream=StringIO(text), interpolate=False)
    # Deliberately preserve the native reader's existing partial view, not add multiline support.
    assert _parse_env_text(result) == _parse_env_text(text)
