import unittest

from scripts.audit_local_stack_readiness import parse_model_block, sanitize_text


class AuditLocalStackReadinessTests(unittest.TestCase):
    def test_parse_model_block_reads_top_level_fields_only(self) -> None:
        config = """
model:
  provider: ollama
  default: hermes-local-tools:latest
  base_url: http://127.0.0.1:11434/v1
  context_length: 65536
toolsets:
  - hermes-cli
"""
        parsed = parse_model_block(config)
        self.assertEqual(
            parsed,
            {
                "provider": "ollama",
                "default": "hermes-local-tools:latest",
                "base_url": "http://127.0.0.1:11434/v1",
                "context_length": "65536",
            },
        )

    def test_sanitize_text_redacts_bearer_and_query_tokens(self) -> None:
        raw = "Bearer abc.def token=secret-value keep=this"
        sanitized = sanitize_text(raw)
        self.assertNotIn("abc.def", sanitized)
        self.assertNotIn("secret-value", sanitized)
        self.assertIn("Bearer <redacted>", sanitized)
        self.assertIn("token=<redacted>", sanitized)


if __name__ == "__main__":
    unittest.main()
