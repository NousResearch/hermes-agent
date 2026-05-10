"""Regression tests for #22489 — Gemini doctor probe uses x-goog-api-key."""

import os
import re
import unittest


class TestGeminiDoctorProbeHeaders(unittest.TestCase):
    """Verify _probe_apikey_provider sends x-goog-api-key for Gemini endpoints."""

    def _get_doctor_source(self):
        with open("/tmp/hermes-agent-fork/hermes_cli/doctor.py") as f:
            return f.read()

    def test_gemini_header_branch_exists(self):
        """doctor.py must contain a Gemini-specific header branch."""
        src = self._get_doctor_source()
        # Must check for generativelanguage.googleapis.com host
        self.assertIn(
            'generativelanguage.googleapis.com',
            src,
            "Missing Gemini host detection in doctor.py",
        )
        # Must use x-goog-api-key somewhere
        self.assertIn(
            'x-goog-api-key',
            src,
            "Missing x-goog-api-key header in doctor.py",
        )

    def test_gemini_branch_is_conditional(self):
        """The x-goog-api-key header must be conditional, not unconditional."""
        src = self._get_doctor_source()
        # Find the block around the Gemini check
        gemini_check = 'generativelanguage.googleapis.com'
        idx = src.find(gemini_check)
        self.assertGreater(idx, 0, "Gemini host check not found")
        # Look at surrounding 400 chars
        block = src[idx - 100:idx + 300]
        # Must be inside an if/else — should have both x-goog-api-key and Bearer
        self.assertIn('x-goog-api-key', block)
        self.assertIn('Authorization', block)
        self.assertIn('Bearer', block)
        # Must NOT unconditionally overwrite all headers with x-goog-api-key
        # (i.e. there should be an else branch)
        self.assertIn('else:', block.lower())

    def test_no_bearer_sent_to_gemini(self):
        """The Gemini branch must NOT include Authorization: Bearer."""
        src = self._get_doctor_source()
        # Find the if block for Gemini
        lines = src.splitlines()
        gemini_line = next(i for i, l in enumerate(lines) if 'generativelanguage.googleapis.com' in l)
        # Collect lines until the else
        branch_lines = []
        for i in range(gemini_line, min(gemini_line + 20, len(lines))):
            if lines[i].strip().startswith('else:') or lines[i].strip().startswith('elif '):
                break
            branch_lines.append(lines[i])
        branch = "\n".join(branch_lines)
        # Gemini branch should set x-goog-api-key
        self.assertIn('x-goog-api-key', branch)
        # Gemini branch should NOT set Authorization/Bearer
        self.assertNotIn(
            'Authorization',
            branch,
            "Gemini branch incorrectly sends Authorization header",
        )


if __name__ == "__main__":
    unittest.main()
