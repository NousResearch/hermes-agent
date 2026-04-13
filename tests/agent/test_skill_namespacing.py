"""Unit tests for skill namespace inference and qualified name helpers."""

from agent.skill_utils import (
    build_qualified_name,
    is_valid_namespace,
    parse_qualified_name,
)


class TestParseQualifiedName:
    def test_with_namespace(self):
        assert parse_qualified_name("superpowers:writing-plans") == (
            "superpowers",
            "writing-plans",
        )

    def test_without_namespace_returns_none(self):
        assert parse_qualified_name("writing-plans") == (None, "writing-plans")

    def test_empty_bare_name(self):
        # "foo:" → ("foo", "") — validation happens elsewhere
        assert parse_qualified_name("foo:") == ("foo", "")

    def test_empty_namespace(self):
        # ":foo" → ("", "foo") — validation happens elsewhere
        assert parse_qualified_name(":foo") == ("", "foo")

    def test_multiple_colons_first_wins(self):
        # "a:b:c" splits on the first ':' only
        assert parse_qualified_name("a:b:c") == ("a", "b:c")


class TestBuildQualifiedName:
    def test_with_namespace(self):
        assert build_qualified_name("superpowers", "writing-plans") == (
            "superpowers:writing-plans"
        )

    def test_with_none_namespace(self):
        assert build_qualified_name(None, "writing-plans") == "writing-plans"

    def test_with_empty_namespace(self):
        assert build_qualified_name("", "writing-plans") == "writing-plans"


class TestIsValidNamespace:
    def test_simple_lowercase(self):
        assert is_valid_namespace("superpowers")

    def test_with_dash(self):
        assert is_valid_namespace("my-plugin")

    def test_with_underscore(self):
        assert is_valid_namespace("my_plugin")

    def test_mixed_case(self):
        assert is_valid_namespace("MyPlugin")

    def test_with_digits(self):
        assert is_valid_namespace("plugin123")

    def test_rejects_colon(self):
        assert not is_valid_namespace("a:b")

    def test_rejects_dot(self):
        assert not is_valid_namespace("a.b")

    def test_rejects_slash(self):
        assert not is_valid_namespace("a/b")

    def test_rejects_space(self):
        assert not is_valid_namespace("my plugin")

    def test_rejects_empty(self):
        assert not is_valid_namespace("")

    def test_rejects_none(self):
        assert not is_valid_namespace(None)
