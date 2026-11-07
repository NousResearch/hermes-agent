"""test suite for the aws-ses skill scripts.

no real aws calls — boto3 clients are stubbed with botocore.stub.stubber,
which validates request parameters against the real sesv2 / cloudwatch
service models. the smtp password derivation is tested against the
aws-documented example key with independently-measured output.
"""

import io
import json
import os
import stat
import sys
import tempfile
import unittest
from datetime import datetime, timezone, UTC
from unittest import mock

import boto3
from botocore.exceptions import ClientError
from botocore.stub import Stubber

# make scripts importable regardless of cwd
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "optional-skills", "email", "aws-ses", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import bounce_stats
import check_ses_status
import gen_smtp_password
import send_test_email
import setup_domain
import verify_dns


# aws public example key (from the sigv4 documentation) — not a real secret.
EXAMPLE_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# computed from the documented algorithm by a second, independent
# implementation of the sigv4 chain — not a vector aws publishes directly.
SMTP_VECTOR_US_EAST_1 = "BLBM/9hSUELfq8Gw+rU1YcBjkOxGbhT2XG763xVLGWL9"
SMTP_VECTOR_EU_NORTH_1 = "BPfy85A/4ChxYc0s4mIG5bpIlvRl5AP1uupODe8KaHpf"


def sesv2_stub():
    client = boto3.client(
        "sesv2",
        region_name="eu-north-1",
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="secret",
    )
    return client, Stubber(client)


def cw_stub():
    client = boto3.client(
        "cloudwatch",
        region_name="eu-north-1",
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="secret",
    )
    return client, Stubber(client)


def sample_account(**overrides):
    acct = {
        "ProductionAccessEnabled": True,
        "SendingEnabled": True,
        "EnforcementStatus": "HEALTHY",
        "SendQuota": {
            "Max24HourSend": 50000.0,
            "MaxSendRate": 14.0,
            "SentLast24Hours": 1234.0,
        },
        "Details": {
            "MailType": "MARKETING",
            "WebsiteURL": "https://example.com",
            "ReviewDetails": {"Status": "GRANTED", "CaseId": "000000000000000"},
        },
    }
    acct.update(overrides)
    return acct


def sample_identity(**overrides):
    ident = {
        "VerificationStatus": "SUCCESS",
        "VerifiedForSendingStatus": True,
        "DkimAttributes": {
            "SigningEnabled": True,
            "Status": "SUCCESS",
            "Tokens": ["tok1", "tok2", "tok3"],
            "SigningAttributesOrigin": "AWS_SES",
            "CurrentSigningKeyLength": "RSA_2048_BIT",
        },
        "MailFromAttributes": {
            "MailFromDomain": "bounce.example.com",
            "MailFromDomainStatus": "SUCCESS",
            "BehaviorOnMxFailure": "USE_DEFAULT_VALUE",
        },
    }
    ident.update(overrides)
    return ident


# ─ gen_smtp_password ──────────────────────────────────────────────

class TestGenSmtpPassword(unittest.TestCase):
    def test_known_vector_us_east_1(self):
        # aws-documented example key -> independently measured output.
        self.assertEqual(
            gen_smtp_password.derive_smtp_password(EXAMPLE_SECRET, "us-east-1"),
            SMTP_VECTOR_US_EAST_1,
        )

    def test_known_vector_eu_north_1(self):
        self.assertEqual(
            gen_smtp_password.derive_smtp_password(EXAMPLE_SECRET, "eu-north-1"),
            SMTP_VECTOR_EU_NORTH_1,
        )

    def test_region_binding_changes_password(self):
        a = gen_smtp_password.derive_smtp_password(EXAMPLE_SECRET, "us-east-1")
        b = gen_smtp_password.derive_smtp_password(EXAMPLE_SECRET, "eu-north-1")
        self.assertNotEqual(a, b)

    def test_output_is_base64_no_ascii_prefix(self):
        pw = gen_smtp_password.derive_smtp_password(EXAMPLE_SECRET, "eu-north-1")
        # must NOT start with the literal ascii "04" (the old bug)
        self.assertFalse(pw.startswith("04"))
        self.assertRegex(pw, r"^[A-Za-z0-9+/]+=*$")

    def test_empty_secret_raises(self):
        with self.assertRaises(ValueError):
            gen_smtp_password.derive_smtp_password("", "eu-north-1")

    def test_empty_region_raises(self):
        with self.assertRaises(ValueError):
            gen_smtp_password.derive_smtp_password(EXAMPLE_SECRET, "")

    def test_deterministic(self):
        a = gen_smtp_password.derive_smtp_password(EXAMPLE_SECRET, "eu-north-1")
        b = gen_smtp_password.derive_smtp_password(EXAMPLE_SECRET, "eu-north-1")
        self.assertEqual(a, b)

    def test_cli_missing_region_exits_1(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("sys.argv", ["gen_smtp_password.py"]), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            with self.assertRaises(SystemExit) as ctx:
                gen_smtp_password.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("region", err.getvalue().lower())

    def test_cli_has_no_secret_argv_flag(self):
        with mock.patch("sys.argv", ["gen_smtp_password.py", "--secret", "x"]), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as ctx:
                gen_smtp_password.main()
            self.assertEqual(ctx.exception.code, 2)

    def _run_cli(self, argv, stdin=""):
        with mock.patch.dict(os.environ, {"AWS_DEFAULT_REGION": "eu-north-1"}, clear=True), \
                mock.patch("sys.argv", ["gen_smtp_password.py", *argv]), \
                mock.patch("sys.stdin", io.StringIO(stdin)), \
                mock.patch.object(gen_smtp_password.boto3, "Session") as session, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            session.return_value.get_credentials.return_value = mock.Mock(
                access_key="AKIAEXAMPLE", secret_key=EXAMPLE_SECRET,
            )
            gen_smtp_password.main()
            return out.getvalue()

    def test_cli_hides_password_by_default(self):
        out = self._run_cli([])
        self.assertIn("SMTP_USERNAME: AKIAEXAMPLE", out)
        self.assertNotIn(SMTP_VECTOR_EU_NORTH_1, out)

    def test_cli_show_password_prints_it(self):
        out = self._run_cli(["--show-password"])
        self.assertIn(SMTP_VECTOR_EU_NORTH_1, out)

    def test_cli_secret_stdin(self):
        out = self._run_cli(["--secret-stdin", "--show-password"], stdin=EXAMPLE_SECRET + "\n")
        self.assertIn(SMTP_VECTOR_EU_NORTH_1, out)

    def test_cli_output_file_is_0600_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "smtp.env")
            out = self._run_cli(["--output", path])
            self.assertNotIn(SMTP_VECTOR_EU_NORTH_1, out)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            with open(path) as fh:
                self.assertIn(f"SMTP_PASSWORD={SMTP_VECTOR_EU_NORTH_1}", fh.read())
            with self.assertRaises(SystemExit) as ctx:
                self._run_cli(["--output", path])
            self.assertEqual(ctx.exception.code, 1)


# ─ check_ses_status ───────────────────────────────────────────────

class TestCheckSesStatus(unittest.TestCase):
    def test_collect_status_reads_nested_details(self):
        client, stub = sesv2_stub()
        stub.add_response("get_account", sample_account(), {})
        stub.add_response("get_email_identity", sample_identity(), {"EmailIdentity": "example.com"})
        stub.activate()
        result = check_ses_status.collect_status(client, domain="example.com")

        self.assertTrue(result["production_access"])
        self.assertEqual(result["review"]["Status"], "GRANTED")
        self.assertEqual(result["review"]["CaseId"], "000000000000000")
        self.assertEqual(result["mail_type"], "MARKETING")
        self.assertEqual(result["website"], "https://example.com")
        self.assertEqual(result["dkim"]["Status"], "SUCCESS")
        self.assertEqual(len(result["dkim"]["Tokens"]), 3)

    def test_account_error_short_circuits(self):
        client, stub = sesv2_stub()
        stub.add_client_error("get_account", "AccessDeniedException", "no perms", 403)
        stub.activate()
        result = check_ses_status.collect_status(client, domain=None)
        self.assertIn("account_error", result)

    def test_unknown_domain_sets_domain_error(self):
        client, stub = sesv2_stub()
        stub.add_response("get_account", sample_account(), {})
        stub.add_client_error(
            "get_email_identity", "NotFoundException",
            "not found", 404, {"EmailIdentity": "nope.example.com"},
        )
        stub.activate()
        result = check_ses_status.collect_status(client, domain="nope.example.com")
        self.assertIn("domain_error", result)
        self.assertIn("not a verified identity", result["domain_error"])


# ─ setup_domain ───────────────────────────────────────────────────

class TestSetupDomain(unittest.TestCase):
    def _created(self):
        return {
            "IdentityType": "DOMAIN",
            "VerifiedForSendingStatus": False,
            "DkimAttributes": {
                "Status": "PENDING",
                "Tokens": ["tok1abcd", "tok2efgh", "tok3ijkl"],
            },
        }

    def test_create_new_identity_omits_signing_attributes(self):
        # stubber validates params against the real model: passing
        # SigningAttributes would raise ParamValidationError, so this test
        # proves the script uses the valid, aws-managed (easy dkim) path.
        client, stub = sesv2_stub()
        stub.add_response("create_email_identity", self._created(), {"EmailIdentity": "example.com"})
        stub.activate()
        result = setup_domain.create_identity(client, "example.com")
        self.assertEqual(result["DkimAttributes"]["Tokens"], ["tok1abcd", "tok2efgh", "tok3ijkl"])

    def test_already_exists_falls_back_to_get(self):
        client, stub = sesv2_stub()
        stub.add_client_error(
            "create_email_identity", "AlreadyExistsException",
            "already exists", 400, {"EmailIdentity": "example.com"},
        )
        stub.add_response(
            "get_email_identity", self._created(), {"EmailIdentity": "example.com"}
        )
        stub.activate()
        result = setup_domain.create_identity(client, "example.com")
        self.assertEqual(result["DkimAttributes"]["Tokens"], ["tok1abcd", "tok2efgh", "tok3ijkl"])

    def test_print_dns_records_output(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            setup_domain.print_dns_records("example.com", ["abc123def", "ghi456jkl", "mno789pqr"])
            text = out.getvalue()
            self.assertIn("abc123def._domainkey.example.com", text)
            self.assertIn("abc123def.dkim.amazonses.com", text)
            self.assertIn("include:amazonses.com", text)


# ─ verify_dns ─────────────────────────────────────────────────────

class TestVerifyDns(unittest.TestCase):
    def test_get_dkim_tokens_from_api(self):
        client, stub = sesv2_stub()
        stub.add_response("get_email_identity", sample_identity(), {"EmailIdentity": "example.com"})
        stub.activate()
        tokens = verify_dns.get_dkim_tokens(client, "example.com")
        self.assertEqual(tokens, ["tok1", "tok2", "tok3"])

    def test_dkim_pass_when_cname_targets_amazonses(self):
        with mock.patch.object(verify_dns, "dig_short", return_value=[
            "abc123.dkim.amazonses.com"
        ]) as dig:
            results = verify_dns.check_dkim("example.com", ["sel1"])
            self.assertTrue(results["sel1"]["passed"])
            dig.assert_called_once_with("sel1._domainkey.example.com", "CNAME")

    def test_dkim_fail_when_no_cname(self):
        with mock.patch.object(verify_dns, "dig_short", return_value=[]):
            results = verify_dns.check_dkim("example.com", ["sel1"])
            self.assertFalse(results["sel1"]["passed"])

    def test_spf_pass_with_amazonses_include(self):
        with mock.patch.object(verify_dns, "dig_short", return_value=[
            '"v=spf1 +a +mx include:amazonses.com ~all"'
        ]):
            spf = verify_dns.check_spf("example.com")
            self.assertTrue(spf["includes_amazonses"])

    def test_dig_short_strips_trailing_dot(self):
        with mock.patch(
            "subprocess.run",
            return_value=mock.Mock(stdout="abc.dkim.amazonses.com.\n", returncode=0),
        ):
            answers = verify_dns.dig_short("sel1._domainkey.example.com")
            self.assertEqual(answers, ["abc.dkim.amazonses.com"])

    def test_dig_short_pins_name_and_type(self):
        with mock.patch(
            "subprocess.run",
            return_value=mock.Mock(stdout="", returncode=0),
        ) as run:
            verify_dns.dig_short("example.com", "TXT")
            self.assertEqual(run.call_args.args[0],
                             ["dig", "+short", "-q", "example.com", "-t", "TXT"])

    def test_dig_short_rejects_invalid_names(self):
        with mock.patch("subprocess.run") as run:
            for bad in ["-f/etc/passwd", "a b.com", "x.com;rm", "@8.8.8.8", ""]:
                with self.assertRaises(ValueError):
                    verify_dns.dig_short(bad)
            run.assert_not_called()

    def test_valid_dns_names(self):
        for good in ["example.com", "abc123._domainkey.example.com", "mail.example.co.uk."]:
            self.assertTrue(verify_dns.is_valid_dns_name(good), good)


# ─ send_test_email ────────────────────────────────────────────────

class TestSendTestEmail(unittest.TestCase):
    def test_build_request_text_only(self):
        kwargs = send_test_email.build_email_request(
            from_address="a@x.com",
            to_addresses=["b@x.com"],
            subject="Test",
            body_text="Hello.",
        )
        self.assertEqual(kwargs["FromEmailAddress"], "a@x.com")
        self.assertEqual(kwargs["Content"]["Simple"]["Body"]["Text"]["Data"], "Hello.")
        self.assertNotIn("Html", kwargs["Content"]["Simple"]["Body"])

    def test_build_request_multipart(self):
        kwargs = send_test_email.build_email_request(
            from_address="a@x.com", to_addresses=["b@x.com"], subject="S",
            body_text="plain", body_html="<p>html</p>",
        )
        self.assertEqual(kwargs["Content"]["Simple"]["Body"]["Html"]["Data"], "<p>html</p>")

    def test_successful_send_prints_message_id(self):
        client, stub = sesv2_stub()
        stub.add_response(
            "get_account",
            {"ProductionAccessEnabled": True, "SendingEnabled": True},
            {},
        )
        stub.add_response("send_email", {"MessageId": "msg-456"})
        stub.activate()
        with mock.patch.object(send_test_email, "get_client", return_value=client), \
                mock.patch("sys.argv", [
                    "send_test_email.py", "--from", "a@x.com", "--to", "b@x.com",
                    "--subject", "Hi", "--body", "Hello.",
                ]), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            send_test_email.main()
            self.assertIn("MessageId: msg-456", out.getvalue())

    def test_client_error_not_verified_exits_1(self):
        client, stub = sesv2_stub()
        stub.add_response(
            "get_account",
            {"ProductionAccessEnabled": False, "SendingEnabled": True},
            {},
        )
        stub.add_client_error(
            "send_email", "MessageRejected",
            "Email address is not verified.", 400,
        )
        stub.activate()
        with mock.patch.object(send_test_email, "get_client", return_value=client), \
                mock.patch("sys.argv", [
                    "send_test_email.py", "--from", "a@x.com", "--to", "u@x.com",
                    "--subject", "Hi", "--body", "Hello.",
                ]), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err, \
                self.assertRaises(SystemExit) as ctx:
            send_test_email.main()
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("not verified", err.getvalue().lower())


# ─ bounce_stats ───────────────────────────────────────────────────

class TestBounceStats(unittest.TestCase):
    def _entry(self, email, reason, updated="2025-01-15T10:00:00Z"):
        return {"EmailAddress": email, "Reason": reason, "LastUpdateTime": updated}

    def test_list_suppressed_uses_reasons_list(self):
        # stubber validates the param name: SuppressionListReason would raise.
        client, stub = sesv2_stub()
        stub.add_response(
            "list_suppressed_destinations",
            {"SuppressedDestinationSummaries": [
                self._entry("b1@x.com", "BOUNCE"),
            ]},
        )
        stub.activate()
        items = bounce_stats.list_suppressed(client, reason_filter="BOUNCE")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["reason"], "BOUNCE")

    def test_collect_stats_success(self):
        client, stub = sesv2_stub()
        stub.add_response("get_account", sample_account(), {})
        stub.add_response(
            "list_suppressed_destinations",
            {"SuppressedDestinationSummaries": [
                self._entry("b1@x.com", "BOUNCE"),
                self._entry("c1@x.com", "COMPLAINT"),
            ]},
        )
        stub.activate()
        stats = bounce_stats.collect_stats(client)
        self.assertEqual(stats["suppression"]["bounces"], 1)
        self.assertEqual(stats["suppression"]["complaints"], 1)
        emails = {s["email"] for s in stats["suppression"]["samples"]}
        self.assertEqual(emails, {"b***@x.com", "c***@x.com"})

    def test_collect_stats_reveal_addresses(self):
        client, stub = sesv2_stub()
        stub.add_response("get_account", sample_account(), {})
        stub.add_response(
            "list_suppressed_destinations",
            {"SuppressedDestinationSummaries": [self._entry("b1@x.com", "BOUNCE")]},
        )
        stub.activate()
        stats = bounce_stats.collect_stats(client, reveal_addresses=True)
        self.assertEqual(stats["suppression"]["samples"][0]["email"], "b1@x.com")

    def test_mask_email(self):
        self.assertEqual(bounce_stats.mask_email("alice@example.com"), "a***@example.com")
        self.assertEqual(bounce_stats.mask_email("not-an-address"), "***")

    def test_complaint_rate_is_ratio(self):
        # Reputation.ComplaintRate is a ratio: 0.2 == 20%, not 0.2%.
        client, stub = cw_stub()
        stub.add_response(
            "get_metric_statistics",
            {"Datapoints": [
                {"Timestamp": datetime.now(UTC), "Average": 0.2},
            ]},
        )
        stub.activate()
        with mock.patch.object(bounce_stats, "get_cloudwatch_client", return_value=client):
            rate = bounce_stats.get_complaint_rate("eu-north-1")
            self.assertEqual(rate, 0.2)  # raw ratio, unmultiplied


if __name__ == "__main__":
    unittest.main(verbosity=2)
