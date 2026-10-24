"""Regression tests for secret values riding on command-line flag forms (#143982).

Two shapes slipped past ``redact_for_egress`` / the approval-card scrub while the
equals form (``--password=…``) was already covered by the ENV-assignment passes:
space-separated flags (``--password <value>``, ``docker login -p <value>``) and
Azure connection-string keys (``AccountKey=``/``SharedAccessKey=``) whose
CamelCase names fall between the all-caps and the underscore-boundary ENV rules.
The no-overreach negative cases (technical flags, English words, path values)
pin the same gate the #77484 lessons fixed for the assignment passes.
"""

from agent.redact import redact_for_egress

SECRET = "hunter2secret"


class TestFlagSpaceSeparatedForm:
    def test_long_flag_space_form_password_masked(self):
        out = redact_for_egress(f"mysql -h db --user admin --password {SECRET} app")
        assert SECRET not in out
        assert "--password" in out
        assert "mysql -h db" in out  # command structure stays judgeable

    def test_long_flag_space_form_token_masked(self):
        out = redact_for_egress(f"vault login --token {SECRET}")
        assert SECRET not in out

    def test_long_flag_space_form_api_key_masked(self):
        out = redact_for_egress(f"tool --api-key {SECRET} run")
        assert SECRET not in out

    def test_long_flag_equals_form_still_masked(self):
        out = redact_for_egress(f"mysql --password={SECRET} app")
        assert SECRET not in out

    def test_quoted_space_form_value_masked(self):
        out = redact_for_egress(f'serve --password "{SECRET}"')
        assert SECRET not in out


class TestDockerLoginShortFlag:
    def test_docker_login_p_space_form_masked(self):
        out = redact_for_egress("docker login -u bob -p mypass123 registry.example.com")
        assert "mypass123" not in out
        assert "registry.example.com" in out

    def test_docker_login_port_map_behind_separator_untouched(self):
        raw = "docker login -u bob -p mypass123; docker run -p 8080:80 nginx"
        out = redact_for_egress(raw)
        assert "mypass123" not in out
        assert "docker run -p 8080:80 nginx" in out  # second command's -p is a port map


class TestAzureConnectionStringKeys:
    def test_storage_account_key_masked(self):
        conn = (
            "DefaultEndpointsProtocol=https;AccountName=st;AccountKey=abc123def456==;"
            "EndpointSuffix=core.windows.net"
        )
        out = redact_for_egress(conn)
        assert "abc123def456==" not in out

    def test_shared_access_key_masked(self):
        out = redact_for_egress(
            "Endpoint=sb://ns.servicebus.windows.net/;SharedAccessKey=zxy987sig=="
        )
        assert "zxy987sig==" not in out

    def test_camel_case_key_alone_masked(self):
        out = redact_for_egress("AccountKey=abc123def456==;Endpoint=portal.check")
        assert "abc123def456==" not in out


class TestFlagPassNoOverreach:
    def test_technical_flag_values_untouched(self):
        raw = "build --max-tokens 100 --token-count 5 --config-file app.conf"
        assert redact_for_egress(raw) == raw

    def test_flag_value_referencing_a_path_untouched(self):
        raw = "serve --auth-token ~/.config/hermes/token --config /etc/app.yaml"
        assert redact_for_egress(raw) == raw

    def test_english_word_after_flag_masked_conservatively(self):
        # ``--password please`` reads as a placeholder for the real value: the equals
        # form (``--password=please``) is already masked by the same keyword-gated
        # assignment rule, so the flag pass keeps that stance (宁漏勿误) instead of
        # trying to guess which plain words are not credentials.
        out = redact_for_egress("docs example: --password please fill in your own")
        assert "--password" in out
        assert "please" not in out
        assert "fill in your own" in out

    def test_prose_around_a_real_credential_flag(self):
        out = redact_for_egress(f"run the migration with --db-password {SECRET} then verify")
        assert SECRET not in out
        assert "then verify" in out


class TestApprovalSurface:
    def test_approval_command_masks_flag_space_form(self):
        from gateway.run import _redact_approval_command

        out = _redact_approval_command(f"deploy --api-token {SECRET} prod")
        assert SECRET not in out
        assert "deploy" in out and "prod" in out
