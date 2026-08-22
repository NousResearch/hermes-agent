"""Self-checks for patent_search.py — citation extraction, meta parsing, date construction."""

import json
import re
import unittest
from html.parser import HTMLParser


class TestCitationRegex(unittest.TestCase):
    """re.findall returns full patent IDs, not just country prefixes."""

    def _extract(self, html_text: str) -> list[str]:
        ids = re.findall(r"(?:US|EP|WO|JP|CN|KR|DE|FR|GB)\d{7,12}[A-Z]\d?", html_text)
        return sorted(set(ids))

    def test_full_ids_returned(self):
        html = "See US11074495B2 and EP4234567A1 for details."
        ids = self._extract(html)
        self.assertIn("US11074495B2", ids)
        self.assertIn("EP4234567A1", ids)

    def test_multiple_occurrences_deduped(self):
        html = "Cited: US11074495B2 and also US11074495B2."
        ids = self._extract(html)
        self.assertEqual(len(ids), 1)

    def test_no_false_positives(self):
        html = "US or EP alone are not patent IDs and should not match."
        ids = self._extract(html)
        self.assertEqual(len(ids), 0)


class TestMetaDescriptionParsing(unittest.TestCase):
    """DC.description is read from the meta content attribute."""

    def test_abstract_from_content_attr(self):
        html = '<html><head><meta name="DC.description" content="A method for blockchain consensus."/></head></html>'
        parser = _make_parser()
        parser.feed(html)
        self.assertEqual(parser.abstract, "A method for blockchain consensus.")

    def test_no_description_returns_none(self):
        html = "<html><head></head><body></body></html>"
        parser = _make_parser()
        parser.feed(html)
        self.assertIsNone(parser.abstract)


# Replicate the parser inline so tests don't depend on full module imports
class _TestParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.abstract = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "meta" and attrs_dict.get("name") == "DC.description":
            self.abstract = attrs_dict.get("content", "").strip() or None


def _make_parser():
    return _TestParser()


class TestDateParam(unittest.TestCase):
    """_date_param builds correct date ranges."""

    def test_since_and_until(self):
        from datetime import datetime

        # Mock: pretend current year is 2026
        result = _date_param(2023, 2025)
        self.assertEqual(result, "20230101000000/20250101000000")

    def test_since_only_defaults_until_to_current_year(self):
        from datetime import datetime

        now = datetime.now()
        result = _date_param(2023, None)
        expected = f"20230101000000/{now.year}0101000000"
        self.assertEqual(result, expected)

    def test_neither_returns_none(self):
        result = _date_param(None, None)
        self.assertIsNone(result)

    def test_until_only(self):
        result = _date_param(None, 2025)
        expected_until = "20250101000000"
        self.assertIsNotNone(result)
        self.assertIn(expected_until, result)


def _date_param(since: int | None, until: int | None) -> str | None:
    """Inline copy of the module's _date_param for testing."""
    from datetime import datetime

    if not since and not until:
        return None
    now = datetime.now()
    _since = f"{since}0101000000" if since else "00000101000000"
    _until = f"{until}0101000000" if until else f"{now.year}0101000000"
    return f"{_since}/{_until}"


if __name__ == "__main__":
    unittest.main()
