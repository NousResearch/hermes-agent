"""Tests for ``_format_acp_import_error`` — the helper that turns
ImportError raised during ``hermes acp`` startup into user-facing messages.

Regression coverage for issue #24959: a fresh hermes install (with a
pre-existing ``marker-pdf`` in the same env that conflicts with the
project's openai pin) reported "ACP dependencies not installed" even
when the actual import failure was elsewhere, sending users in circles
reinstalling the ``[acp]`` extra.
"""

from hermes_cli.main import _format_acp_import_error


def test_missing_acp_module_reports_install_extra():
    exc = ImportError("No module named 'acp'", name="acp")
    lines = _format_acp_import_error(exc)
    assert lines[0] == "ACP dependencies not installed."
    assert any("pip install -e '.[acp]'" in line for line in lines)


def test_missing_acp_submodule_reports_install_extra():
    exc = ImportError("No module named 'acp.schema'", name="acp.schema")
    lines = _format_acp_import_error(exc)
    assert lines[0] == "ACP dependencies not installed."


def test_unrelated_missing_module_does_not_blame_acp_extra():
    exc = ImportError("No module named 'pydantic'", name="pydantic")
    lines = _format_acp_import_error(exc)
    joined = "\n".join(lines)
    assert "ACP dependencies not installed" not in joined
    assert "pydantic" in joined
    assert "not an ACP packaging issue" in joined


def test_missing_name_attribute_still_renders():
    exc = ImportError("cannot import name 'Foo'")
    lines = _format_acp_import_error(exc)
    joined = "\n".join(lines)
    assert "hermes acp failed to start" in joined
    assert "cannot import name 'Foo'" in joined
    assert "failing import could not be identified" in joined
    assert "unrelated module" not in joined
