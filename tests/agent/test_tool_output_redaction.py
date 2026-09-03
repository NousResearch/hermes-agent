"""Final-boundary tool-output redaction contracts.

Every credential-shaped fixture in this file is synthesized locally. None is a
live or previously issued secret.
"""

import base64
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.redact import (
    ToolOutputRedactionPolicy,
    normalize_tool_output,
    normalize_tool_output_content,
    resolve_tool_output_redaction_policy,
)
from agent.tool_dispatch_helpers import make_tool_result_message
from tools.tool_result_storage import (
    _last_retention_sweep,
    maybe_persist_tool_result,
    sweep_expired_results,
)


def _synthetic_jwt() -> str:
    def _part(value):
        encoded = base64.urlsafe_b64encode(json.dumps(value).encode("utf-8"))
        return encoded.decode("ascii").rstrip("=")

    return f"{_part({'alg': 'HS256', 'typ': 'JWT'})}.{_part({'sub': 'synthetic-user'})}.{'S1gN4tUr3' * 4}"


def _synthetic_high_entropy() -> str:
    # Deterministic bytes encoded locally; this is not credential material.
    return base64.urlsafe_b64encode(bytes(range(1, 49))).decode("ascii")


@pytest.mark.parametrize("variant", ["a", "b", "y"])
def test_bcrypt_variants_are_class_labelled(variant):
    literal = f"$2{variant}$12$" + "A" * 53
    result = normalize_tool_output(literal)
    assert result == f"[REDACTED:BCRYPT_2{variant.upper()}]"


def test_bcrypt_grep_regression_redacts_all_matched_lines():
    literals = [
        "$2a$04$" + "A" * 53,
        "$2b$10$" + "B" * 53,
        "$2y$12$" + "C" * 53,
    ]
    grep_output = "\n".join(
        f"src/synthetic_users.py:{line}:{literal}"
        for line, literal in enumerate(literals, start=17)
    )
    result = normalize_tool_output(grep_output)
    assert all(literal not in result for literal in literals)
    assert result.count("[REDACTED:BCRYPT_") == 3
    assert "src/synthetic_users.py:17:" in result


@pytest.mark.parametrize("label", ["PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY", "OPENSSH PRIVATE KEY"])
def test_pem_private_key_blocks_are_redacted(label):
    block = (
        f"-----BEGIN {label}-----\n"
        "U1lOVEhFVElDLU5PVC1LRVktBVEVSSUFM\n"
        f"-----END {label}-----"
    )
    assert normalize_tool_output(block) == "[REDACTED:PEM_PRIVATE_KEY]"


@pytest.mark.parametrize(
    "prefix",
    ["ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"],
)
def test_github_token_prefixes_are_redacted(prefix):
    literal = prefix + ("Synthetic123" * 4)
    result = normalize_tool_output(literal)
    assert literal not in result
    assert result == "[REDACTED:GITHUB_TOKEN]"


def test_jwt_is_redacted():
    literal = _synthetic_jwt()
    assert normalize_tool_output(literal) == "[REDACTED:JWT]"


@pytest.mark.parametrize("prefix", ["AKIA", "ASIA", "AIDA", "AROA"])
def test_aws_access_key_ids_are_redacted(prefix):
    literal = prefix + "SYNTHETIC1234567"
    assert len(literal) == 20
    assert normalize_tool_output(literal) == "[REDACTED:AWS_ACCESS_KEY_ID]"


def test_configured_secret_name_is_resolved_at_runtime_without_a_value_lookup():
    raw_config = {
        "security": {
            "tool_output_redaction": {
                "secret_names": ["deployment_credential"],
            }
        }
    }
    with patch("hermes_cli.config.read_raw_config", return_value=raw_config):
        policy = resolve_tool_output_redaction_policy()
        result = normalize_tool_output(
            'deployment_credential = "s3cr3t-literal-42"',
            policy=policy,
        )
    assert result == 'deployment_credential = "[REDACTED:NAME:DEPLOYMENT_CREDENTIAL]"'


def test_truncated_escaped_json_does_not_backtrack_pathologically():
    escaped_markdown = (
        '---\\nname: synthetic-skill\\ndescription: \\"quoted words\\"\\n'
        * 500
    )
    truncated_tool_result = json.dumps({"content": escaped_markdown})[:-2]

    started = time.monotonic()
    result = normalize_tool_output(truncated_tool_result)
    elapsed = time.monotonic() - started

    assert result == truncated_tool_result
    assert elapsed < 1.0


def test_literal_incident_systemd_environment_string_is_redacted():
    incident = "Environment=DB_PASSWORD=s3cr3tValue123"
    result = normalize_tool_output(incident)
    assert result == "Environment=DB_PASSWORD=[REDACTED:NAME:DB_PASSWORD]"


@pytest.mark.parametrize(
    ("literal", "label"),
    [
        ("DATABASE_PASSWORD=hunter2corrhorse", "DATABASE_PASSWORD"),
        (
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI7MDENGbPxRfiCYEXAMPLEKEY",
            "AWS_SECRET_ACCESS_KEY",
        ),
        ("export API_TOKEN=abc123randomstring456xyz", "API_TOKEN"),
        ('{"database_password": "hunter2corrhorse"}', "DATABASE_PASSWORD"),
    ],
)
def test_compound_secret_names_are_redacted(literal, label):
    result = normalize_tool_output(literal)
    assert literal not in result
    assert f"[REDACTED:NAME:{label}]" in result


@pytest.mark.parametrize(
    ("literal", "label"),
    [
        ("dbPassword=s3cr3tValue123", "DB_PASSWORD"),
        ('{"clientSecret":"s3cr3tValue123"}', "CLIENT_SECRET"),
        ('{"accessToken":"s3cr3tValue123"}', "ACCESS_TOKEN"),
    ],
)
def test_camel_case_secret_names_are_redacted(literal, label):
    result = normalize_tool_output(literal)
    assert literal not in result
    assert f"[REDACTED:NAME:{label}]" in result


@pytest.mark.parametrize(
    ("literal", "label"),
    [
        ("API_KEY_V2=8fJ2kD9sLmQ4xZ7v", "API_KEY_V2"),
        ("MY_API_KEY_PROD=aB3dE5fG7hJ9kL1m", "MY_API_KEY_PROD"),
        ("DATABASE_PASSWORD_STAGING=Str0ngPassw0rd", "DATABASE_PASSWORD_STAGING"),
        ("GITHUB_TOKEN_FOR_CI=aB3dE5fG7hJ9kL1m", "GITHUB_TOKEN_FOR_CI"),
        ("SECRET_VALUE=8fJ2kD9sLmQ4xZ7v", "SECRET_VALUE"),
        ("API_KEY=8fJ2kD9sLmQ4xZ7v", "API_KEY"),
        ("DB_PASSWORD=Str0ngPassw0rd", "DB_PASSWORD"),
        (
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI7MDENGbPxRfiCYEXAMPLEKEY",
            "AWS_SECRET_ACCESS_KEY",
        ),
    ],
)
def test_printenv_secret_names_are_redacted(literal, label):
    result = normalize_tool_output(literal)
    assert result == f"{literal.partition('=')[0]}=[REDACTED:NAME:{label}]"


def test_printenv_known_shape_is_redacted_independently_of_name_matching():
    literal = "STRIPE_KEY_LIVE=sk_live_1234567890"
    result = normalize_tool_output(literal)
    assert literal not in result
    assert "STRIPE_KEY_LIVE=" in result


def test_generic_high_entropy_uses_configurable_thresholds():
    literal = _synthetic_high_entropy()
    assert normalize_tool_output(literal) == "[REDACTED:HIGH_ENTROPY]"

    policy = ToolOutputRedactionPolicy(
        secret_names=(),
        entropy_min_length=40,
        entropy_floor=7.0,
    )
    assert normalize_tool_output(literal, policy=policy) == literal


@pytest.mark.parametrize(
    "literal",
    [
        (
            "integrity: sha512-"
            "3q7Q/5V1wS9nE0f4J6mY2xC8pK1uR9tL5bN7dH0sA4gF6jQ8vW2zX9cM"
            "1kP3rT7uY5iO0lK6nB4eS8dG2fA=="
        ),
        (
            '"integrity": "sha512-'
            "3q7Q/5V1wS9nE0f4J6mY2xC8pK1uR9tL5bN7dH0sA4gF6jQ8vW2zX9cM"
            '1kP3rT7uY5iO0lK6nB4eS8dG2fA=="'
        ),
        (
            "resolution: {integrity: sha512-"
            "3q7Q/5V1wS9nE0f4J6mY2xC8pK1uR9tL5bN7dH0sA4gF6jQ8vW2zX9cM"
            "1kP3rT7uY5iO0lK6nB4eS8dG2fA==}"
        ),
        (
            '"integrity": "sha512-'
            "V7Qr52IhZmdKPVr+Vtw8o+WLsQJYCTd8loIfpDaMRWGUZfBOYEJeyJIkq"
            'DgqkQzHElZDM7A5Y5DqTqvS7A=="'
        ),
        '"registry-auth-token": "^5.0.1",',
    ],
)
def test_package_manager_integrity_digests_are_not_redacted(literal):
    assert normalize_tool_output(literal) == literal


def test_digest_shield_marker_forgery_cannot_alias_multiple_digests():
    first = "sha512-" + _synthetic_high_entropy()
    second = "sha256-" + _synthetic_high_entropy()[::-1]
    text = f"\x00HERMES_TOOL_DIGEST_0\x00 {first} {second}"
    assert normalize_tool_output(text) == text


def test_high_entropy_after_whitespace_only_prefix_does_not_crash():
    literal = _synthetic_high_entropy()
    result = normalize_tool_output(f"{' ' * 128}{literal}")
    assert result.endswith("[REDACTED:HIGH_ENTROPY]")


def test_redaction_is_idempotent():
    text = (
        "password=synthetic-password\n"
        + "$2b$12$"
        + "D" * 53
        + "\n"
        + _synthetic_jwt()
    )
    once = normalize_tool_output(text)
    assert normalize_tool_output(once) == once


def test_serializer_normalizes_before_returning_to_model():
    literal = "$2b$12$" + "E" * 53
    message = make_tool_result_message(
        "terminal",
        f"src/synthetic_users.py:31:{literal}",
        "call-synthetic",
    )
    assert literal not in message["content"]
    assert "[REDACTED:BCRYPT_2B]" in message["content"]


def test_serializer_normalizes_multimodal_text_and_preserves_image_part():
    bcrypt = "$2b$12$" + "M" * 53
    incident = "Environment=DB_PASSWORD=s3cr3tValue123"
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,c3ludGhldGlj"},
    }
    content = [
        {"type": "text", "text": f"{bcrypt}\n{incident}"},
        image_part,
    ]

    message = make_tool_result_message("computer_use", content, "call-multimodal")

    assert bcrypt not in message["content"][0]["text"]
    assert incident not in message["content"][0]["text"]
    assert "[REDACTED:BCRYPT_2B]" in message["content"][0]["text"]
    assert "[REDACTED:NAME:DB_PASSWORD]" in message["content"][0]["text"]
    assert message["content"][1] is image_part
    assert content[0]["text"] == f"{bcrypt}\n{incident}"


@pytest.mark.parametrize("part_type", ["text", "input_text", "output_text"])
def test_multimodal_text_bearing_part_types_are_normalized(part_type):
    incident = "Environment=DB_PASSWORD=s3cr3tValue123"
    part = {"type": part_type, "text": incident, "annotations": []}

    result = normalize_tool_output_content([part])

    assert result[0]["text"] == "Environment=DB_PASSWORD=[REDACTED:NAME:DB_PASSWORD]"
    assert result[0]["annotations"] == []
    assert part["text"] == incident


@pytest.mark.parametrize("resource_at_top_level", [False, True])
def test_multimodal_resource_text_is_normalized(resource_at_top_level):
    incident = "Environment=DB_PASSWORD=s3cr3tValue123"
    if resource_at_top_level:
        part = {"type": "resource", "text": incident, "uri": "memory://synthetic"}
    else:
        part = {
            "type": "resource",
            "resource": {
                "uri": "memory://synthetic",
                "mimeType": "text/plain",
                "text": incident,
            },
        }

    result = normalize_tool_output_content([part])
    result_text = result[0]["text"] if resource_at_top_level else result[0]["resource"]["text"]

    assert result_text == "Environment=DB_PASSWORD=[REDACTED:NAME:DB_PASSWORD]"
    original_text = part["text"] if resource_at_top_level else part["resource"]["text"]
    assert original_text == incident


def test_multimodal_binary_payloads_with_secret_shapes_are_byte_identical():
    raw_image = b"synthetic image bytes AKIAIOSFODNN7EXAMPLE ghp_" + b"A" * 40
    encoded = base64.b64encode(raw_image).decode("ascii")
    image_url = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }
    anthropic_image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
    }

    result = normalize_tool_output_content([image_url, anthropic_image])

    assert result[0] is image_url
    assert result[1] is anthropic_image
    assert base64.b64decode(result[0]["image_url"]["url"].partition(",")[2]) == raw_image
    assert base64.b64decode(result[1]["source"]["data"]) == raw_image


def test_verbose_tool_result_log_text_is_normalized():
    from agent.tool_executor import _normalized_tool_result_log_text

    incident = "Environment=DB_PASSWORD=s3cr3tValue123"
    bcrypt = "$2b$12$" + "L" * 53
    result = _normalized_tool_result_log_text(f"{incident}\n{bcrypt}")
    assert incident not in result
    assert bcrypt not in result
    assert "[REDACTED:NAME:DB_PASSWORD]" in result
    assert "[REDACTED:BCRYPT_2B]" in result


def test_spill_path_writes_only_normalized_content(tmp_path, monkeypatch):
    literal = "$2b$12$" + "F" * 53
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    env = MagicMock()
    env.get_temp_dir.return_value = "/tmp"
    env.execute.return_value = {"output": "", "returncode": 0}
    policy = ToolOutputRedactionPolicy(secret_names=())

    result = maybe_persist_tool_result(
        content=f"src/synthetic_users.py:44:{literal}\n" + "ordinary output\n" * 20,
        tool_name="terminal",
        tool_use_id="call-spill-redaction",
        env=env,
        threshold=0,
        redaction_policy=policy,
    )

    spill_path = tmp_path / "cache" / "spillover" / "call-spill-redaction.txt"
    persisted = spill_path.read_text(encoding="utf-8")
    assert literal not in persisted
    assert "[REDACTED:BCRYPT_2B]" in persisted
    assert literal not in result


def test_retention_sweep_uses_configured_max_age_and_regular_files_only():
    env = MagicMock()
    env.execute.return_value = {"output": "", "returncode": 0}
    _last_retention_sweep.clear()
    assert sweep_expired_results(
        env,
        storage_dir="/tmp/hermes-results",
        max_age_seconds=24 * 60 * 60,
        force=True,
        now=100.0,
    )
    command = env.execute.call_args.args[0]
    assert "find /tmp/hermes-results -maxdepth 1 -type f -mmin +1440 -delete" in command


def test_ordinary_source_code_is_not_mangled():
    source = """\
def calculate_token_count(document):
    api_key_name = "OPENAI_API_KEY"
    max_tokens = 4096
    digest = "sha512-AbCdEf0123456789AbCdEf0123456789AbCdEf0123456789AbCdEf0123456789"
    return document, api_key_name, max_tokens, digest
"""
    assert normalize_tool_output(source) == source


@pytest.mark.parametrize(
    "source",
    [
        "api_key: Optional[str] = None",
        "def connect(self, token: str, ...)",
        "Client(api_key=api_key, ...)",
        '{"access_token": access_token, ...}',
        "token = argv[i]",
        "const API_KEY_OPTIONS: ApiKeyOption[]",
        'AutoTokenizer(sep_token="[SEP]", ...)',
    ],
)
def test_named_value_matcher_preserves_source_expressions(source):
    assert normalize_tool_output(source) == source


def test_legacy_pass_preserves_bare_shell_variable_references():
    text = (
        'curl -H "Authorization: token $GITHUB_TOKEN" https://example.invalid\n'
        "printf '%s' $API_KEY"
    )

    result = normalize_tool_output(text)

    assert "$GITHUB_TOKEN" in result
    assert "$API_KEY" in result
    literal_header = "Authorization: token opaque-credential-1234567890"
    assert literal_header not in normalize_tool_output(literal_header)


@pytest.mark.parametrize(
    "placeholder",
    [
        'api_key="your-api-key"',
        'token="synthetic-token"',
        'api_key="copilot-acp"',
        "https://example.invalid/image.png?token=3077792b-90ff-459d-aa52-57abcf219adf",
        '_OAUTH_TOKEN_USER_AGENT = "axios/1.7.9"',
    ],
)
def test_named_value_matcher_preserves_placeholders_and_urls(placeholder):
    assert normalize_tool_output(placeholder) == placeholder


def test_ordinary_prose_is_not_mangled():
    prose = (
        "The deployment guide explains how credentials are provisioned, but it "
        "does not print them. The token count and password policy are ordinary "
        "documentation terms, not assignments or credential literals."
    )
    assert normalize_tool_output(prose) == prose


def test_inline_source_map_is_not_mangled():
    source_map = (
        "//# sourceMappingURL=data:application/json;charset=utf-8;base64,"
        + _synthetic_high_entropy()
    )
    assert normalize_tool_output(source_map) == source_map


def test_high_entropy_url_path_is_not_mangled():
    path_segment = _synthetic_high_entropy().rstrip("=")
    url = f"https://github.com/synthetic-org/{path_segment}/module.py#L847"
    assert normalize_tool_output(url) == url


def test_jwt_like_version_string_is_not_mangled():
    text = "artifact eyJnot-json.payload-segment.signature-segment-123 release"
    assert normalize_tool_output(text) == text
