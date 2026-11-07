"""Real botocore chains: a served profile's SigV4 identity never falls through to the launch identity."""
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import httpx

from agent import bedrock_adapter as b
from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_context, reset_multiplex_context
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


@contextmanager
def scope(home, values):
    h = set_hermes_home_override(home)
    s = set_secret_scope(values, profile_home=home)
    m = set_multiplex_context(True)
    try:
        yield
    finally:
        reset_multiplex_context(m)
        reset_secret_scope(s)
        reset_hermes_home_override(h)


class AmbientCredentialTests(unittest.TestCase):
    def setUp(self):
        b.reset_client_cache()
        self.addCleanup(b.reset_client_cache)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        config = os.path.join(tmp.name, "config")
        creds = os.path.join(tmp.name, "credentials")
        with open(config, "w") as fh:
            fh.write("[profile regiononly]\nregion = us-east-1\n")
        with open(creds, "w") as fh:
            fh.write("[withkeys]\naws_access_key_id = PROFILEKEY\naws_secret_access_key = profile-secret\n")
        # The launch process: its own keys, a container role and a bearer/base URL for the SDK.
        launch = {
            "AWS_CONFIG_FILE": config, "AWS_SHARED_CREDENTIALS_FILE": creds,
            "AWS_ACCESS_KEY_ID": "LAUNCHKEY", "AWS_SECRET_ACCESS_KEY": "launch-secret",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://127.0.0.1:9/launch-role",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_BEARER_TOKEN_BEDROCK": "launch-bearer",
            "ANTHROPIC_BEDROCK_BASE_URL": "https://launch-gateway.example.com",
        }
        patcher = patch.dict(os.environ, launch)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_named_profile_without_credentials_does_not_use_launch_container_role(self):
        with scope("/tmp/hermes-profile-a", {"AWS_PROFILE": "regiononly"}):
            self.assertIsNone(b._signing_session().get_credentials())

    def test_named_profile_keeps_its_own_shared_credentials(self):
        with scope("/tmp/hermes-profile-a", {"AWS_PROFILE": "withkeys"}):
            self.assertEqual(b._signing_session().get_credentials().access_key, "PROFILEKEY")

    def test_scoped_anthropic_bedrock_ignores_launch_bearer_and_base_url(self):
        from agent.anthropic_adapter import build_anthropic_bedrock_client
        with scope("/tmp/hermes-profile-a", {"AWS_ACCESS_KEY_ID": "PROFILEKEY",
                                             "AWS_SECRET_ACCESS_KEY": "profile-secret"}):
            client = build_anthropic_bedrock_client("us-east-1")
            self.assertEqual(client.base_url.host, "bedrock-runtime.us-east-1.amazonaws.com")
            request = httpx.Request("POST", str(client.base_url) + "/model/m/invoke", content=b"{}")
            client._prepare_request(request)
        self.assertTrue(request.headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=PROFILEKEY/"))


if __name__ == "__main__":
    unittest.main()
