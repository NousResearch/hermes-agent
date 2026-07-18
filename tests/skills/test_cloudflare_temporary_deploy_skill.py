"""Tests for optional-skills/web-development/cloudflare-temporary-deploy/scripts/parse_deploy_output.py"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "web-development"
    / "cloudflare-temporary-deploy"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import parse_deploy_output as pdo


CREATED = """\
Continuing means you accept Cloudflare's Terms of Service and Privacy Policy.

Temporary account ready:
     Account:        swift-otter (created)
     Claim within:   60 minutes
     Claim URL:      https://dash.cloudflare.com/claim-preview?claimToken=TOKEN_AAA

Uploaded my-worker
Deployed my-worker triggers
     https://my-worker.swift-otter.workers.dev
"""

REUSED = """\
Temporary account ready:
     Account:        swift-otter (reused)
     Claim within:   17 minutes
     Claim URL:      https://dash.cloudflare.com/claim-preview?claimToken=TOKEN_BBB
Deployed my-worker triggers
     https://my-worker.swift-otter.workers.dev
"""

NOT_LOGGED_IN = """\
✘ [ERROR] You are not logged in.

To continue without logging in, rerun this command with `--temporary`.
"""

AUTH_PRESENT_ERROR = """\
✘ [ERROR] The --temporary flag cannot be used while Wrangler is authenticated.
Run `wrangler logout` first, or remove CLOUDFLARE_API_TOKEN.
"""


class TestParseCreated:
    def test_live_url(self):
        assert pdo.parse(CREATED)["live_url"] == "https://my-worker.swift-otter.workers.dev"

    def test_claim_url(self):
        assert (
            pdo.parse(CREATED)["claim_url"]
            == "https://dash.cloudflare.com/claim-preview?claimToken=TOKEN_AAA"
        )

    def test_account_and_state(self):
        r = pdo.parse(CREATED)
        assert r["account"] == "swift-otter"
        assert r["account_state"] == "created"

    def test_expiry_and_deployed(self):
        r = pdo.parse(CREATED)
        assert r["expires_minutes"] == 60
        assert r["deployed"] is True


class TestParseReused:
    def test_state_is_reused(self):
        assert pdo.parse(REUSED)["account_state"] == "reused"

    def test_expiry_window_can_shrink(self):
        assert pdo.parse(REUSED)["expires_minutes"] == 17

    def test_live_url_stable(self):
        assert pdo.parse(REUSED)["live_url"] == "https://my-worker.swift-otter.workers.dev"


class TestNoDeploy:
    def test_not_logged_in_has_no_urls(self):
        r = pdo.parse(NOT_LOGGED_IN)
        assert r["live_url"] is None
        assert r["claim_url"] is None
        assert r["account"] is None
        assert r["deployed"] is False

    def test_auth_present_error_has_no_urls(self):
        r = pdo.parse(AUTH_PRESENT_ERROR)
        assert r["live_url"] is None
        assert r["claim_url"] is None
        assert r["deployed"] is False


class TestRealWorldOutput:
    """Regression: real wrangler output uses tab-indent + multi-word account names."""

    REAL = (
        "⛅️ wrangler 4.103.0\n"
        "Continuing means you accept Cloudflare's Terms of Service and Privacy Policy.\n"
        "Solving proof-of-work challenge…\n"
        "Temporary account ready:\n"
        "\tAccount: Serene Temple (created)\n"
        "\tClaim within: 60 minutes\n"
        "\tClaim URL: https://dash.cloudflare.com/claim-preview?claimToken=fxLzyAD-vlTzMQmClpg\n"
        "Total Upload: 0.19 KiB / gzip: 0.16 KiB\n"
        "Uploaded hermes-temp-hello (0.74 sec)\n"
        "Deployed hermes-temp-hello triggers (0.42 sec)\n"
        "  https://hermes-temp-hello.serene-temple.workers.dev\n"
    )

    def test_multiword_account_name(self):
        r = pdo.parse(self.REAL)
        assert r["account"] == "Serene Temple"
        assert r["account_state"] == "created"

    def test_all_fields_from_real_output(self):
        r = pdo.parse(self.REAL)
        assert r["live_url"] == "https://hermes-temp-hello.serene-temple.workers.dev"
        assert r["claim_url"].endswith("claimToken=fxLzyAD-vlTzMQmClpg")
        assert r["expires_minutes"] == 60
        assert r["deployed"] is True


class TestUrlHygiene:
    def test_trailing_punctuation_stripped(self):
        text = "Deployed\n  see https://w.acct.workers.dev. for details"
        assert pdo.parse(text)["live_url"] == "https://w.acct.workers.dev"

    def test_does_not_match_plain_cloudflare_com(self):
        # A generic cloudflare.com link without a claimToken must not be taken as the claim URL.
        text = "Privacy Policy: https://www.cloudflare.com/privacypolicy/\nDeployed x"
        assert pdo.parse(text)["claim_url"] is None


class TestCli:
    def test_selftest_exits_zero(self):
        assert pdo.main(["--selftest"]) == 0

    def test_main_prints_json_and_exit_zero_on_live(self, capsys):
        with mock.patch.object(sys.stdin, "read", return_value=CREATED):
            rc = pdo.main([])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["live_url"] == "https://my-worker.swift-otter.workers.dev"

    def test_main_exit_one_when_no_live_url(self, capsys):
        with mock.patch.object(sys.stdin, "read", return_value=NOT_LOGGED_IN):
            rc = pdo.main([])
        out = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert out["live_url"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
