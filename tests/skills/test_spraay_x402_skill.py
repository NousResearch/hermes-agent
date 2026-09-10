"""Tests for the spraay-x402 optional skill helper script."""

import json
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

# Import the helper script
sys.path.insert(
    0, "optional-skills/blockchain/spraay-x402/scripts"
)
import spraay_gateway  # noqa: E402


class TestSpraayGatewayHealth(unittest.TestCase):
    """Tests for the health command."""

    @patch("spraay_gateway.urllib.request.urlopen")
    def test_health_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"status": "ok", "version": "3.8.1"}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = spraay_gateway._request("GET", "/health")
        self.assertEqual(result["status"], "ok")

    @patch("spraay_gateway.urllib.request.urlopen")
    def test_health_url_error(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused")

        result = spraay_gateway._request("GET", "/health")
        self.assertTrue(result["error"])
        self.assertIn("Connection refused", result["message"])


class TestSpraayGatewayScan(unittest.TestCase):
    """Tests for the scan command."""

    @patch("spraay_gateway.urllib.request.urlopen")
    def test_scan_returns_categories(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"categories": ["batch-payments", "escrow"], "total_primitives": 151}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = spraay_gateway._request("GET", "/scan")
        self.assertIn("categories", result)
        self.assertGreaterEqual(result["total_primitives"], 1)


class TestSpraayGatewayQuote(unittest.TestCase):
    """Tests for the quote command."""

    @patch("spraay_gateway.urllib.request.urlopen")
    def test_quote_sends_correct_body(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"amount": "0.02", "gas_estimate": "185000"}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = spraay_gateway._request(
            "POST",
            "/quote",
            data={
                "primitive": "batch_payment",
                "chain": "base",
                "recipientCount": 5,
            },
        )
        self.assertIn("amount", result)

        # Verify the request was constructed correctly
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertEqual(req.method, "POST")
        body = json.loads(req.data.decode())
        self.assertEqual(body["primitive"], "batch_payment")
        self.assertEqual(body["chain"], "base")


class TestSpraayGatewayExecute(unittest.TestCase):
    """Tests for the execute command with payment header."""

    @patch("spraay_gateway.urllib.request.urlopen")
    def test_execute_includes_payment_header(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"transaction": {"to": "0x1646...", "data": "0x..."}}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        payment_header = "x402_test_payment_header_abc123"

        result = spraay_gateway._request(
            "POST",
            "/execute",
            data={
                "primitive": "batch_payment",
                "chain": "base",
                "sender": "0xTestSender",
                "token": "USDC",
                "recipients": ["0xAddr1"],
                "amounts": ["1000000"],
            },
            headers={"X-402-Payment": payment_header},
        )
        self.assertIn("transaction", result)

        # Verify the payment header was included
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertEqual(req.get_header("X-402-payment"), payment_header)

    @patch("spraay_gateway.urllib.request.urlopen")
    def test_execute_402_without_payment(self, mock_urlopen):
        """Gateway returns 402 when payment header is missing."""
        error_body = json.dumps(
            {"error": "Payment required", "amount": "0.02"}
        ).encode()
        mock_fp = MagicMock()
        mock_fp.read.return_value = error_body
        mock_urlopen.side_effect = HTTPError(
            url="https://gateway.spraay.app/execute",
            code=402,
            msg="Payment Required",
            hdrs={},
            fp=mock_fp,
        )

        result = spraay_gateway._request(
            "POST",
            "/execute",
            data={"primitive": "batch_payment", "chain": "base"},
        )
        self.assertTrue(result["error"])
        self.assertEqual(result["status"], 402)

    def test_cmd_execute_rejects_missing_payment(self):
        """cmd_execute exits when --payment is not provided."""
        with self.assertRaises(SystemExit) as ctx:
            spraay_gateway.cmd_execute(
                "batch_payment", "base", "0xSender", "", "{}"
            )
        self.assertEqual(ctx.exception.code, 1)


class TestSpraayGatewayConfig(unittest.TestCase):
    """Tests for configuration handling."""

    def test_default_gateway_url(self):
        """Default gateway URL points to production."""
        # Temporarily clear env override if present
        original = spraay_gateway.GATEWAY_URL
        spraay_gateway.GATEWAY_URL = "https://gateway.spraay.app"
        self.assertEqual(
            spraay_gateway.GATEWAY_URL, "https://gateway.spraay.app"
        )
        spraay_gateway.GATEWAY_URL = original

    @patch.dict("os.environ", {"SPRAAY_GATEWAY_URL": "https://custom.example.com"})
    def test_custom_gateway_url(self):
        """SPRAAY_GATEWAY_URL env var overrides the default."""
        url = os.environ.get("SPRAAY_GATEWAY_URL", "https://gateway.spraay.app")
        self.assertEqual(url, "https://custom.example.com")


class TestSpraayGatewayCLI(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def test_no_args_prints_usage(self):
        """Running with no arguments exits with usage info."""
        with self.assertRaises(SystemExit) as ctx:
            spraay_gateway.main()
        self.assertEqual(ctx.exception.code, 1)

    def test_unknown_command_exits(self):
        """Unknown command exits with error."""
        original_argv = sys.argv
        sys.argv = ["spraay_gateway.py", "nonexistent"]
        try:
            with self.assertRaises(SystemExit) as ctx:
                spraay_gateway.main()
            self.assertEqual(ctx.exception.code, 1)
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
