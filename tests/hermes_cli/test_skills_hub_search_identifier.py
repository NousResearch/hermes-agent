"""Regression tests for #33674: skills search identifier truncation.

``hermes skills search`` rendered a Rich table where the Identifier column
was truncated (Rich default), cutting off the hash suffix like ``-1uezib``
from browse-sh skill slugs.  Users copied the truncated identifier and
``hermes skills install`` failed.

Two fixes were applied:
  1. ``no_wrap=True`` on the Identifier column so Rich never truncates it.
  2. ``--json`` flag that outputs full machine-readable JSON including
     full identifiers, so scripting workflows can reliably capture slugs.
"""

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from hermes_cli.skills_hub import do_search


# ── helpers ──────────────────────────────────────────────────────────────


def _make_result(identifier, name="weather-forecast"):
    """Build a fake search result object with a browse-sh-style identifier."""
    return SimpleNamespace(
        name=name,
        description="Get NOAA weather forecasts for a location",
        source="browse-sh",
        trust_level="community",
        identifier=identifier,
    )


def _run_do_search(identifiers, *, json_output=False, console_width=200):
    """Run do_search with mocked dependencies and return captured output."""
    results = [_make_result(ident) for ident in identifiers]
    buf = StringIO()
    console = Console(file=buf, width=console_width, highlight=False, no_color=True)

    with (
        patch("tools.skills_hub.unified_search", return_value=results),
        patch("tools.skills_hub.create_source_router", return_value={}),
        patch("tools.skills_hub.GitHubAuth"),
    ):
        do_search("weather", source="all", limit=10, console=console,
                  json_output=json_output)

    return buf.getvalue()


def _extract_json(output: str):
    """Extract the first JSON array from Rich console output."""
    start = output.find("[")
    if start == -1:
        raise ValueError(f"No JSON array found in output: {output!r}")
    return json.loads(output[start:].strip())


# ── tests: no_wrap on Identifier column ──────────────────────────────────


class TestIdentifierColumnNoWrap:
    """Identifier column must not be truncated in the table output."""

    def test_short_identifier_fully_shown(self):
        """Plain short identifiers are unaffected."""
        output = _run_do_search(["openai/weather/forecast"])
        assert "openai/weather/forecast" in output, (
            "Short identifier should appear verbatim in table output"
        )

    def test_long_browse_sh_identifier_not_truncated(self):
        """browse-sh slugs with hash suffix must never be truncated."""
        long_id = "browse-sh/weather.gov/get-forecast-1uezib"
        output = _run_do_search([long_id])

        assert long_id in output, (
            f"Full identifier {long_id!r} not found in table output.\n"
            "This means no_wrap=True is not set on the Identifier column. "
            "Users cannot install skills with truncated identifiers (#33674)."
        )
        assert "…" not in output, (
            "Identifier column is truncating with '…' — set no_wrap=True on the column"
        )

    def test_multiple_long_identifiers_all_shown(self):
        """All rows must show their full identifier."""
        identifiers = [
            "browse-sh/weather.gov/get-forecast-1uezib",
            "browse-sh/weather.gov/get-hourly-abc123",
            "browse-sh/weather.gov/get-alerts-xyz789",
        ]
        output = _run_do_search(identifiers)
        for ident in identifiers:
            assert ident in output, (
                f"Identifier {ident!r} was truncated in table output (#33674)"
            )


# ── tests: --json output ──────────────────────────────────────────────────


class TestJsonOutput:
    """--json flag must emit machine-readable JSON with full identifiers."""

    def test_json_output_is_valid_json(self):
        """JSON mode must produce parseable JSON."""
        output = _run_do_search(
            ["browse-sh/weather.gov/get-forecast-1uezib"],
            json_output=True,
        )
        parsed = _extract_json(output)
        assert isinstance(parsed, list), "JSON output must be a list"

    def test_json_output_contains_full_identifier(self):
        """JSON output must include the full, untruncated identifier field."""
        full_id = "browse-sh/weather.gov/get-forecast-1uezib"
        output = _run_do_search([full_id], json_output=True)
        parsed = _extract_json(output)
        assert len(parsed) == 1
        assert parsed[0]["identifier"] == full_id, (
            f"Expected full identifier {full_id!r} in JSON output, "
            f"got {parsed[0].get('identifier')!r} (#33674)"
        )

    def test_json_output_has_expected_fields(self):
        """JSON rows must include name, description, source, trust_level, identifier."""
        output = _run_do_search(
            ["browse-sh/test-skill/action-hash99"],
            json_output=True,
        )
        parsed = _extract_json(output)
        row = parsed[0]
        for field in ("name", "description", "source", "trust_level", "identifier"):
            assert field in row, f"JSON output missing field '{field}' (#33674)"

    def test_json_mode_omits_table_header(self):
        """When --json is active, no Rich table header text should appear."""
        output = _run_do_search(["openai/skills/pptx"], json_output=True)
        assert "Skills Hub" not in output, (
            "Table header 'Skills Hub' should not appear in JSON mode"
        )

    def test_json_multiple_results_all_present(self):
        """All results must appear in JSON output."""
        identifiers = [
            "browse-sh/weather.gov/get-forecast-1uezib",
            "browse-sh/weather.gov/get-hourly-abc123",
        ]
        output = _run_do_search(identifiers, json_output=True)
        parsed = _extract_json(output)
        assert len(parsed) == 2
        returned_ids = {r["identifier"] for r in parsed}
        assert returned_ids == set(identifiers), (
            f"Expected {identifiers!r} in JSON, got {returned_ids!r}"
        )
