"""Sequential display redaction follows current static policy without rescanning tokens."""
import pytest

from agent import redact
from agent import streaming_redact as streaming


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    redact._reset_plugin_redaction_patterns()
    yield
    redact._reset_plugin_redaction_patterns()


@pytest.mark.parametrize("payload", [
    "token: CPU\n", '"token": "local"', "password: hunter2\n",
    "spring.datasource.password=hunter2\n", "openai_key=opaque_value_123456789\n",
    "client_secret=hunter2\n", "I have password=foo\n",
    'OPENAI_API_KEY=os.getenv("SECRET")\n',
    '{"password": "hunter2"}',
    "data=x&client_secret=opaque123456789",
    "https://example.com/file?token=opaqueSignedValue123",
    "postgresql://operator:hunter2@example.com/db ",
    "sk-abcdef12345678901234567890 ",
    "x-api-key: opaque123456789\n",
    "Authorization: Bearer opaque123456789\n",
])
@pytest.mark.parametrize("fragment_size", [1, 7])
def test_fragmented_text_matches_current_policy(payload, fragment_size):
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(payload[i:i + fragment_size]) for i in range(0, len(payload), fragment_size))
    visible += sanitizer.flush()
    assert visible == redact.redact_sensitive_text(payload, force=True)
    assert sanitizer.pending_length == 0


@pytest.mark.parametrize("pattern,token", [
    (r"acme_[A-Za-z0-9]{20,}", "acme_abcdefghijklmnopqrstuvwxyz"),
    (r"zeta_[A-Z]+:[0-9]+", "zeta_ABCDEFGHIJK:123456789"),
    (r"acme(?:prod|test):[A-Z]{20,}", "acmeprod:ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    (r"zeta_[A-Z]+:[0-9]{20,}", "zeta_ABC:1234567890123456789012345"),
])
def test_pattern_registered_after_sanitizer_construction(pattern, token):
    sanitizer = streaming.StreamingSecretSanitizer()
    redact.register_redaction_patterns([pattern], source="dynamic test")
    visible = "".join(sanitizer.feed(char) for char in token + " ") + sanitizer.flush()
    assert token not in visible
    assert visible == redact.redact_sensitive_text(token + " ", force=True)


@pytest.mark.parametrize("prefix,suffix", [
    ('"password": "', '"'), ("OPENAI_API_KEY=", "\n"),
    ("sk-", " "), ("postgres://", ":password@example.com "),
    ("spring.datasource.password=", "\n"),
])
def test_validated_one_byte_continuations_have_linear_scan_cost(prefix, suffix, monkeypatch):
    original = streaming._partial_context_opener_start
    scanned = []
    def counted(text, **kwargs):
        scanned.append(len(text))
        return original(text, **kwargs)
    monkeypatch.setattr(streaming, "_partial_context_opener_start", counted)
    sanitizer = streaming.StreamingSecretSanitizer()
    sanitizer.feed(prefix)
    for _ in range(10000):
        sanitizer.feed("x")
    sanitizer.feed(suffix)
    sanitizer.flush()
    assert sum(scanned) < 50000
    assert len(scanned) < 30


@pytest.mark.parametrize("control", ["\x1b", "\u200b", "\n"])
def test_control_split_vendor_tokens_match_current_policy(control):
    token = "ghp_abcdef1234567890ABCDEF1234567890abcdef"
    payload = token[:10] + control + token[10:] + " "
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(char) for char in payload) + sanitizer.flush()
    assert token[10:] not in visible
    assert visible == redact.redact_sensitive_text(payload, force=True)


def test_escaped_json_value_remains_one_secret():
    payload = '{"password": "head\\"opaqueEscapedTail123"}'
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(char) for char in payload) + sanitizer.flush()
    assert "opaqueEscapedTail123" not in visible
    import json
    assert json.loads(visible) == {"password": '***'}


def test_private_key_body_advances_incrementally(monkeypatch):
    original = streaming._partial_context_opener_start
    lengths = []
    def counted(text, **kwargs):
        lengths.append(len(text))
        return original(text, **kwargs)
    monkeypatch.setattr(streaming, "_partial_context_opener_start", counted)
    sanitizer = streaming.StreamingSecretSanitizer()
    sanitizer.feed("-----BEGIN PRIVATE KEY-----\n")
    for _ in range(3000):
        sanitizer.feed("x")
    visible = sanitizer.feed("\n-----END PRIVATE KEY-----\n") + sanitizer.flush()
    assert "x" * 30 not in visible
    assert sum(lengths) < 15000


@pytest.mark.parametrize("control", ["\x1b", "\u200b", "\n"])
@pytest.mark.parametrize("position", [1, 3, 40])
def test_controls_inside_prefix_and_after_complete_token(control, position):
    token = "ghp_abcdef1234567890ABCDEF1234567890abcdef"
    payload = token[:position] + control + token[position:] + " prose "
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(char) for char in payload) + sanitizer.flush()
    assert visible == redact.redact_sensitive_text(payload, force=True)


def test_pattern_reregistered_while_candidate_is_pending():
    sanitizer = streaming.StreamingSecretSanitizer()
    assert sanitizer.feed("acme_") == ""
    redact.register_redaction_patterns([r"acme_[A-Za-z0-9]{20,}"], source="late")
    token = "acme_abcdefghijklmnopqrstuvwxyz"
    visible = sanitizer.feed(token[5:] + " ") + sanitizer.flush()
    assert visible == redact.redact_sensitive_text(token + " ", force=True)


@pytest.mark.parametrize("prefix,continuation,suffix", [
    ("Authorization:", " ", "Bearer opaqueCredential123\n"),
    ("Authorization: Bearer ", "x", "\n"),
    ("OPENAI_API_KEY", " ", "=opaqueCredential123\n"),
    ("OPENAI_API_KEY=", " ", "opaqueCredential123\n"),
])
def test_unbounded_openers_advance_incrementally(prefix, continuation, suffix, monkeypatch):
    original = streaming._partial_context_opener_start
    scanned = []
    def counted(text, **kwargs):
        scanned.append(len(text))
        return original(text, **kwargs)
    monkeypatch.setattr(streaming, "_partial_context_opener_start", counted)
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = sanitizer.feed(prefix)
    for _ in range(1000):
        visible += sanitizer.feed(continuation)
    visible += sanitizer.feed(suffix) + sanitizer.flush()
    assert visible == redact.redact_sensitive_text(prefix + continuation * 1000 + suffix, force=True)
    assert sum(scanned) < 15000


@pytest.mark.parametrize("progress_only", [False, True])
@pytest.mark.parametrize("payload", [
    '"token"' + " " * 2000 + ': "opaqueChunkedJsonCredential123"',
    '"token"' + " " * 2000 + "is ordinary prose",
    '"token"' + " " * 2000 + ": " + "not-a-string",
    'OPENAI_API_KEY=""opaqueConcatenatedCredential123 tail',
    'OPENAI_API_KEY=opaqueChunkedCredential123\n',
    "OPENAI_API_KEY='opaqueChunkedCredential123' tail",
    'OPENAI_API_KEY="opaqueChunkedCredential123" tail',
])
def test_source_whitespace_and_quoted_grammar_survives_one_byte_feeds(progress_only, payload):
    sanitizer = streaming.StreamingSecretSanitizer(token_candidates_only=progress_only)
    visible = "".join(sanitizer.feed(char) for char in payload) + sanitizer.flush()
    assert visible == redact.redact_sensitive_text(payload, force=True)
    assert sanitizer.pending_length == 0


@pytest.mark.parametrize("payload", [
    '{"token": "unfinishedOpaqueCredential',
    "postgresql://operator:unfinishedOpaqueCredential",
    "-----BEGIN PRIVATE KEY-----\nunfinishedOpaqueCredential",
])
def test_terminal_incomplete_structured_values_are_opaque(payload):
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(char) for char in payload) + sanitizer.flush()
    assert "unfinishedOpaqueCredential" not in visible
    assert visible == streaming.sanitize_terminal_secret_text(payload)


@pytest.mark.parametrize("tail,accepted", [(" PRIVATE K", True), (" PRIVATE KEY-----", True), ("PRIVATE KEY------", False), ("a", False)])
def test_unbounded_pem_type_label_uses_linear_character_checks(tail, accepted, monkeypatch):
    checks = 0
    def counted(items):
        nonlocal checks
        for item in items:
            checks += 1
            if not item:
                return False
        return True
    monkeypatch.setattr(streaming, "all", counted, raising=False)
    payload = "-----BEGIN" + "A" * 3000 + tail
    assert streaming._could_be_private_key_opener(payload) is accepted
    assert checks <= 2 * len(payload)


@pytest.mark.parametrize("payload", [
    "  password: hunter2\n", "\tpassword: hunter2\n",
    "hello\n  password: hunter2\n", "\n password: hunter2\n",
    "  token: opaqueCredential123456\n", "Prose password: hunter2\n",
    "  export password=hunter2\n", "hello\n\tEXPORT password=hunter2\n",
])
@pytest.mark.parametrize("fragment_size", [1, 2, 7, 64])
def test_line_indentation_preserves_canonical_assignment_context(payload, fragment_size):
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(payload[i:i + fragment_size]) for i in range(0, len(payload), fragment_size))
    visible += sanitizer.flush()
    assert visible == redact.redact_sensitive_text(payload, force=True)


@pytest.mark.parametrize("key", sorted(redact._SENSITIVE_QUERY_PARAMS))
@pytest.mark.parametrize("initial", ["abc", "nonSensitive123"])
@pytest.mark.parametrize("fragment_size", [1, 7])
def test_form_keeps_initial_nonsensitive_pair_context(key, initial, fragment_size):
    payload = f"data={initial}&{key}=opaqueCredential123456"
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(payload[i:i + fragment_size]) for i in range(0, len(payload), fragment_size)) + sanitizer.flush()
    assert visible == redact.redact_sensitive_text(payload, force=True)


@pytest.mark.parametrize("escape", [r"\q", r"\x20", r"\uZZZZ"])
def test_invalid_json_escape_cannot_release_sensitive_value_tail(escape):
    payload = '{"password":"head' + escape + r'\"opaqueSecretTail123456"}'
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(char) for char in payload) + sanitizer.flush()
    assert "opaqueSecretTail123456" not in visible
    assert "opaqueSecretTail123456" not in streaming.sanitize_terminal_secret_text(payload)


@pytest.mark.parametrize("options", [{"token_candidates_only": True}, {"embedded_prefixes": False}])
@pytest.mark.parametrize("event", ["echo one", "https://example.org/tools/tool-progress", "npm install package@latest"])
def test_event_mode_releases_complete_ordinary_commands_and_urls(options, event):
    sanitizer = streaming.StreamingSecretSanitizer(**options)
    assert sanitizer.feed(event) == event
    assert sanitizer.pending_length == 0


def test_event_mode_benign_prefix_divergence_stays_separate():
    sanitizer = streaming.StreamingSecretSanitizer(embedded_prefixes=False)
    assert sanitizer.feed("g") == ""
    assert sanitizer.feed("ood") == "good"
    assert sanitizer.pending_length == 0


def test_progress_database_event_does_not_retain_an_unkeyed_password():
    sanitizer = streaming.StreamingSecretSanitizer(token_candidates_only=True)
    event = "postgresql://user:opaqueProgressDbCredentialValue123"
    visible = sanitizer.feed(event)
    assert "opaqueProgressDbCredentialValue123" not in visible
    assert sanitizer.pending_length == 0
    assert sanitizer.feed("safe-second-event") == "safe-second-event"
    assert sanitizer.flush() == ""


@pytest.mark.parametrize("indent", ["  ", "\t"])
@pytest.mark.parametrize("key", ["token", "client_secret", "password"])
@pytest.mark.parametrize("fragment_size", [1, 3, 7, 64])
def test_indented_whole_form_uses_opaque_query_masks(indent, key, fragment_size):
    payload = f"{indent}data=abc&{key}=opaqueCredential123456"
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(payload[i:i + fragment_size]) for i in range(0, len(payload), fragment_size)) + sanitizer.flush()
    # Harmless indentation may already have streamed before the form is known.
    assert visible.strip() == redact.redact_sensitive_text(payload, force=True)


@pytest.mark.parametrize("prefix", ["\n ", "Intro\n\t"])
def test_form_after_newline_keeps_static_multiline_policy(prefix):
    payload = prefix + "data=abc&client_secret=opaqueCredential123456"
    sanitizer = streaming.StreamingSecretSanitizer()
    visible = "".join(sanitizer.feed(char) for char in payload) + sanitizer.flush()
    assert visible == redact.redact_sensitive_text(payload, force=True)
