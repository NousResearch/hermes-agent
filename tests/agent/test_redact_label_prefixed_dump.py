"""Structured field dumps must mask values rather than type annotations (#157615)."""

import time

import pytest

from agent.redact import redact_sensitive_text


@pytest.mark.parametrize("secret_file", [False, True])
@pytest.mark.parametrize("prefix,value", [
    ('field password: CONCEALED = ', '"synthpass.d\'X9"'),
    ('field token: STRING = ', "'synthtok123'"),
    ('field token: STRING = ', 'synthtok123'),
    ('field api_key: STRING = ', '"synthkey789"'),
    ('db field token: STRING = ', '"synthtok123"'),
    ('5|      field password: CONCEALED = ', '"synthpass.d\'X9"'),
    ('field token: Optional[str] = ', '"synthtok123"'),
    ('field token: STRING = ', '<Zq8x>'),
    ('field password: ', '"quoted-direct"'),
    ('  field api_key:\tSTRING\t=\t', '"synthkey789"'),
    ('field password: CONCEALED = ', '"escaped\\\"secret"'),
    ('field password: CONCEALED = ', "'doubled''secret'"),
])
def test_mask_preserves_dump_format(prefix, value, secret_file):
    quote = value[0] if value[0] in "\"'" else ""
    marker = "«redacted-secret»" if secret_file else "***"
    assert redact_sensitive_text(prefix + value, force=True, file_read=secret_file,
                                 secret_file=secret_file) == prefix + quote + marker + quote


@pytest.mark.parametrize("secret_file", [False, True])
@pytest.mark.parametrize("text", [
    "Secretary: J.Smith", "tokenizer: cl100k_base", "token: discussão", "author=Smith",
    "The Secretary: J.Smith signed the token: discussão", "debug: password is required",
    'Discuss field password: "not a structured dump"',
    'field tokenizer: STRING = "cl100k_base"',
    "field password:", "field token: STRING", "",
    'field password: CONCEALED = ""',
    'field password: CONCEALED = "${API_KEY}"',
    'field token: STRING = "${OPENAI_API_KEY}"',
    'field password: CONCEALED = "{{ .Values.dbPassword }}"',
    'field token: STRING = "true"',
    'field token: STRING = "<your token>"',
    'field token: STRING = <your token>',
    'field password: CONCEALED = ${API_KEY}',
    'field password: CONCEALED = "$DB_PASSWORD"',
    'field password: "***"', 'field password: "${PASSWORD}"',
    'field password: CONCEALED = "os.getenv(\'PASSWORD\')"',
    'field token: "CPU"', 'field token: STRING = "/home/user/token"',
])
def test_placeholders_and_prose_unchanged(text, secret_file):
    assert redact_sensitive_text(text, force=True, file_read=secret_file,
                                 secret_file=secret_file) == text


@pytest.mark.parametrize("text", [
    "field " + "a" * 100_000 + ': STRING = "synthetic"',
    "field password: " + "[" * 100_000,
    'field password: CONCEALED = "' + "\\" * 100_000,
])
def test_adversarial_dump_is_linear(text):
    started = time.perf_counter()
    redact_sensitive_text(text, force=True)
    assert time.perf_counter() - started < 2.0
