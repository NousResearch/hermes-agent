"""Devin CLI model auto-discovery (invalid --model → Available: …)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from hermes_cli.models import (
    fetch_devin_cli_models,
    parse_devin_cli_available_models,
    provider_model_ids,
)


SAMPLE_ERR = (
    "Error: Unknown model: '__hermes_devin_probe__'\n"
    "Available: adaptive, claude-opus-4.8, swe-1.7, gpt-5.5\n"
)


class TestParseDevinAvailable(unittest.TestCase):
    def test_parses_available_line(self):
        models = parse_devin_cli_available_models(SAMPLE_ERR)
        self.assertEqual(models, ["adaptive", "claude-opus-4.8", "swe-1.7", "gpt-5.5"])

    def test_empty_when_missing(self):
        self.assertEqual(parse_devin_cli_available_models("no available here"), [])
        self.assertEqual(parse_devin_cli_available_models(""), [])

    def test_dedupes_preserving_order(self):
        text = "Available: a, b, a, c"
        self.assertEqual(parse_devin_cli_available_models(text), ["a", "b", "c"])


class TestFetchDevinCliModels(unittest.TestCase):
    def test_fetch_parses_stderr(self):
        mock_proc = MagicMock()
        mock_proc.stderr = SAMPLE_ERR
        mock_proc.stdout = ""
        with patch("subprocess.run", return_value=mock_proc) as run:
            with patch("shutil.which", return_value="/bin/devin"):
                models = fetch_devin_cli_models(command="devin")
        self.assertEqual(models, ["adaptive", "claude-opus-4.8", "swe-1.7", "gpt-5.5"])
        argv = run.call_args[0][0]
        self.assertEqual(argv[0], "/bin/devin")
        self.assertIn("--model", argv)
        self.assertIn("-p", argv)

    def test_fetch_returns_empty_on_timeout(self):
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="devin", timeout=1),
        ):
            with patch("shutil.which", return_value="/bin/devin"):
                self.assertEqual(fetch_devin_cli_models(command="devin"), [])

    def test_provider_model_ids_uses_live(self):
        with patch(
            "hermes_cli.models.fetch_devin_cli_models",
            return_value=["swe-1.7", "claude-opus-4.8"],
        ):
            ids = provider_model_ids("devin-acp")
        self.assertEqual(ids, ["swe-1.7", "claude-opus-4.8"])

    def test_provider_model_ids_falls_back_to_curated(self):
        with patch("hermes_cli.models.fetch_devin_cli_models", return_value=[]):
            ids = provider_model_ids("devin-acp")
        self.assertIn("swe-1.7", ids)
        self.assertIn("claude-opus-4.8", ids)
        self.assertIn("devin-acp", ids)


if __name__ == "__main__":
    unittest.main()
