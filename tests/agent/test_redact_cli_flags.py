"""CLI flag regressions; runnable with stdlib unittest as well as pytest."""
import json
import unittest

from agent.redact import redact_sensitive_text, redact_terminal_output


class TestCLIFlagRegressions(unittest.TestCase):
    @staticmethod
    def redact(text, surface):
        if surface == "ps":
            return redact_terminal_output(text, command="ps auxww", force=True)
        return redact_sensitive_text(text, force=True, **{
            "ordinary": {}, "code": {"code_file": True}, "file": {"file_read": True},
        }[surface])

    def test_json_closing_escape_survives_without_leaking_backslash_values(self):
        for depth in (1, 2, 3):
            for secret in ("SYNTHETICopaque", r"SYNTHETIC\opaque", r"SYNTHETIC\\opaque"):
                for suffix in ("", " --port 80"):
                    original = {"cmd": f"server --token {secret}{suffix}"}
                    text = original
                    for _ in range(depth):
                        text = json.dumps(text)
                    for surface in ("ordinary", "code", "file", "ps"):
                        with self.subTest(depth=depth, secret=secret, suffix=suffix, surface=surface):
                            result = self.redact(text, surface)
                            decoded = result
                            for _ in range(depth):
                                decoded = json.loads(decoded)
                            mask = "«redacted-secret»" if surface == "file" else "***"
                            self.assertEqual(decoded, {"cmd": f"server --token {mask}{suffix}"})
                            self.assertEqual(self.redact(result, surface), result)

    def test_complete_references_preserved_on_output_and_file_surfaces(self):
        for reference in ("$TOKEN", "${TOKEN}", "${{ secrets.GH_TOKEN }}",
                          "$(op read vault/item/password)", "$1", "$env:API_KEY"):
            for quote in ("", '"', "'", '\\"'):
                for surface in ("ordinary", "code", "file", "ps"):
                    with self.subTest(reference=reference, quote=quote, surface=surface):
                        text = f"server --password {quote}{reference}{quote} --port 80"
                        self.assertEqual(self.redact(text, surface), text)

    def test_dollar_prefix_and_embedded_backslashes_do_not_exempt_literals(self):
        for secret in ("$2b$12$SYNTHETICdigest0123456789", "$argon2id$v=19$SYNTHETICdigest",
                       "$NotAReference!", r"SYNTHETIC\opaque", r"SYNTHETIC\\opaque",
                       "$(op read vault/item/password)SYNTHETICliteralSuffix"):
            for surface in ("ordinary", "code", "file", "ps"):
                with self.subTest(secret=secret, surface=surface):
                    text = f"server --password '{secret}' --port 80"
                    mask = "«redacted-secret»" if surface == "file" else "***"
                    self.assertEqual(self.redact(text, surface), f"server --password '{mask}' --port 80")
                    # Also check unquoted backslashes and expression-plus-literal concatenation.
                    if not secret.startswith("$argon2id"):
                        text = f"server --password {secret} --port 80"
                        self.assertEqual(self.redact(text, surface), f"server --password {mask} --port 80")

    def test_control_split_prefix_is_masked_before_cli_consumes_its_head(self):
        for separator in ("\n", "\r", "\t", "\x1b", "\u200b"):
            for surface in ("ordinary", "code", "file", "ps"):
                with self.subTest(separator=separator, surface=surface):
                    text = f"server --api-key ghp_abcdef{separator}1234567890ABCDEFGHIJKLMNOPQRSTUV"
                    mask = "«redacted:ghp_…»" if surface == "file" else "***"
                    result = self.redact(text, surface)
                    self.assertEqual(result, f"server --api-key {mask}")
                    self.assertEqual(self.redact(result, surface), result)

    def test_prefix_pass_does_not_leave_a_quoted_credential_suffix(self):
        text = 'server --password "ghp_abcdef0123456789 SYNTHETICliteralSuffix" --port 80'
        for surface in ("ordinary", "code", "file", "ps"):
            with self.subTest(surface=surface):
                mask = "«redacted:ghp_…»" if surface == "file" else "***"
                result = self.redact(text, surface)
                self.assertEqual(result, f'server --password "{mask}" --port 80')
                self.assertEqual(self.redact(result, surface), result)


if __name__ == "__main__":
    unittest.main()
