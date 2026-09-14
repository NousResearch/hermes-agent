"""Full-progress redaction contracts at the real callback/enqueue boundary."""

from copy import deepcopy
import json
from queue import Queue
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


def _enqueue_args(args):
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="redaction-test"),
        _run_still_current=lambda: True,
        progress_mode="full",
        tool_progress_enabled=True,
        progress_queue=Queue(),
    )
    turn = TurnRunner(SimpleNamespace(_adapter_for_source=lambda _source: None), ctx)
    turn.progress_callback("tool.started", "credential_test_tool", "short", args)
    marker, content = ctx.progress_queue.get_nowait()
    assert marker == "__full__"
    assert ctx.progress_queue.empty()
    return json.loads(content.split("\n", 1)[1])


@pytest.mark.parametrize(
    "credential, masked",
    [
        *[
            pair
            for name in ("passphrase", "passwd", "credential", "credentials",
                         "authentication", "signature")
            for pair in (
                (f"--{name} opaque_cli_916", f"--{name} ***"),
                (f"--{name}=opaque_cli_916", f"--{name}=***"),
                (f"--{name}='opaque cli 916'", f"--{name}='***'"),
                (f'"--{name}=opaque cli 916"', f'"--{name}=***"'),
                (f'"--{name}" "opaque cli 916"', f'"--{name}" "***"'),
            )
        ],
        ("--oauth2-bearer opaque_cli_916", "--oauth2-bearer ***"),
        ("--oauth2-bearer='opaque cli 916'", "--oauth2-bearer='***'"),
        ('"--oauth2-bearer=opaque cli 916"', '"--oauth2-bearer=***"'),
        ('"--oauth2-bearer" "opaque cli 916"', '"--oauth2-bearer" "***"'),
        ("--proxy-user alice:opaque_cli_916", "--proxy-user ***"),
        ('--proxy-user="alice:opaque cli 916"', '--proxy-user="***"'),
        ("'--proxy-user=alice:opaque cli 916'", "'--proxy-user=***'"),
        ("-U 'alice:opaque cli 916'", "-U '***'"),
        ("-Ualice:opaque_cli_916", "-U***"),
        ('-U"alice:opaque cli 916"', '-U"***"'),
        ("'-Ualice:opaque cli 916'", "'-U***'"),
        ("-ualice:opaque_cli_916", "-u***"),
        ("-u'alice:opaque cli 916'", "-u'***'"),
        ("--cookie opaque_cli_916", "--cookie ***"),
        ("--cookie '--token'", "--cookie '***'"),
        ("--oauth2-bearer '--password'", "--oauth2-bearer '***'"),
        ('--cookie "sid=opaque cli 916; theme=dark"', '--cookie "***"'),
        ("--cookie='sid=opaque cli 916; theme=dark'", "--cookie='***'"),
        ('"--cookie=sid=opaque cli 916"', '"--cookie=***"'),
        ("-b 'sid=opaque cli 916; theme=dark'", "-b '***'"),
        ("-bsid=opaque_cli_916", "-b***"),
        ('-b"sid=opaque cli 916"', '-b"***"'),
        ("'-bsid=opaque cli 916'", "'-b***'"),
    ],
)
def test_full_cli_credentials_preserve_shell_structure(monkeypatch, credential, masked):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    # Mix new options with the existing shell-mask provenance and URL scanners.
    prefix = "__HERMES_FULL_MASKED_WORD_0__ export API_TOKEN='opaque phrase'; curl "
    suffix = (
        " --proxy-user alice -U alice -Ualice --user alice -u alice -ualice"
        " --cookie-jar ./public.jar --oauth2-bearer-format text"
        " --proxy-username alice --token-count 4 --password-file ./public.txt"
        " --passphrase-file ./public.txt --credentials-file ./public.json"
        " --authentication-method plain --signature-algorithm sha256"
        " 'https://example.test/health?visible=hidden#fragment'; echo 'keep  spacing'"
    )
    args = {"command": prefix + credential + suffix, "workdir": "/tmp/visible"}
    original = deepcopy(args)

    safe = _enqueue_args(args)

    assert safe == {
        "command": prefix.replace("'opaque phrase'", "'***'") + masked
        + suffix.replace("?visible=hidden#fragment", "?visible=***#***"),
        "workdir": original["workdir"],
    }
    assert args == original


def test_full_masks_cli_values_exposed_by_native_assignment_scanning(monkeypatch):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    command = 'app.password="x --passphrase opaque-cascade-271"'
    args = {"command": command}
    assert _enqueue_args(args) == {"command": "app.password=*** --passphrase ***"}
    assert args == {"command": command}


@pytest.mark.parametrize("key", [
    "ssh_key", "sshKey", "ssh-key", "recovery_key", "recoveryKey", "recovery-key",
    "sshkey", "recoverykey", "SSH_KEY", "RECOVERY_KEY",
])
def test_full_structured_credential_keys_preserve_benign_metadata(monkeypatch, key):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    secret = "opaque recovery phrase 314"
    metadata = {
        "sort_key": "created_at", "primary_key": "id", "foreign_key": "owner_id",
        "monkey": "capuchin", "keyboard_layout": "de", "key_count": 3,
        "key_format": "pem", "ssh_key_path": "./id_ed25519",
        "recovery_key_configured": True, "ssh_keyboard_layout": "de",
        "available_recovery_key_count": 2,
    }
    for prefix in ("", "matrix_", "service-", "tenant.config."):
        credential_key = prefix + key
        prefixed_metadata = {prefix + name: value for name, value in metadata.items()}
        args = {"records": [{credential_key: secret}, [credential_key, secret],
                            (credential_key, secret)], **prefixed_metadata}
        original = deepcopy(args)
        assert _enqueue_args(args) == {
            "records": [{credential_key: "***"}, [credential_key, "***"],
                        [credential_key, "***"]], **prefixed_metadata,
        }
        assert args == original

    env_name = "MATRIX" + "".join(char for char in key if char.isalnum()).upper()
    flag_name = env_name.lower()
    assert _enqueue_args({"command": f"{env_name}='{secret}' tool --{flag_name} public"}) == {
        "command": f"{env_name}='***' tool --{flag_name} public",
    }
    monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})
    redact.register_vault_redaction_value(secret)
    for metadata_key in ("ssh_key_path", "matrix_recovery_key_configured"):
        assert _enqueue_args({metadata_key: secret}) == {
            metadata_key: "«redacted-vault-secret»",
        }


@pytest.mark.parametrize("header", ["Cookie", "Set-Cookie", "Authorization", "X-Api-Key"])
@pytest.mark.parametrize("quote", ["'", '"'])
def test_full_curl_headers_follow_shell_word_boundaries(monkeypatch, header, quote):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    prefix = "__HERMES_FULL_MASKED_WORD_0__ curl  "
    suffix = " --header-file public.txt -H 'Accept: application/json' --output public.txt https://example.test/health; echo 'keep  spacing'"
    other_quote = '"' if quote == "'" else "'"
    # The same header reaches curl through separate, equals and attached words.
    # Keep original non-secret syntax, including quotes around option names.
    for option in ("--header=", "-H"):
        for value in ("opaque header 941", "opaque" + other_quote + "header"):
            argument = quote + option + header + ": " + value + quote
            masked = quote + option + header + ": ***" + quote
            args = {"command": prefix + argument + suffix}
            original = deepcopy(args)
            assert _enqueue_args(args) == {"command": prefix + masked + suffix}
            assert args == original

    for option in (quote + "--header" + quote + " ", quote + "-H" + quote + " ",
                   "--hea" + quote + "der" + quote + "="):
        argument = option + quote + header + ": opaque header 941" + quote
        masked = option + quote + header + ": ***" + quote
        assert _enqueue_args({"command": prefix + argument + suffix}) == {
            "command": prefix + masked + suffix,
        }

    for argument, masked in (
        ("--header=" + header + r":opaque\ header", "--header=" + header + ":***"),
        ("--header=" + header + r"\:opaque", "--header=" + header + r"\:***"),
        ("--header=" + header + ":" + quote + "opaque header" + quote,
         "--header=" + header + ":" + quote + "***" + quote),
        ("--header=" + quote + header + ": opaque" + quote + other_quote + "tail" + other_quote,
         "--header=" + quote + header + ": ***" + quote),
    ):
        assert _enqueue_args({"command": prefix + argument + suffix}) == {
            "command": prefix + masked + suffix,
        }

    # A value may close and reopen quotes; its trailing pieces remain secret.
    argument = "--header=" + quote + header + ": opaque" + quote + "tail"
    masked = "--header=" + quote + header + ": ***" + quote
    assert _enqueue_args({"command": prefix + argument + suffix}) == {
        "command": prefix + masked + suffix,
    }
    # A header option at a command boundary must not consume the next command.
    assert _enqueue_args({"command": "curl " + quote + "--header" + quote + "; echo public"}) == {
        "command": "curl " + quote + "--header" + quote + "; echo public",
    }


@pytest.mark.parametrize("header", ["Cookie", "Set-Cookie", "Authorization", "X-Api-Key"])
def test_full_curl_header_equals_preserves_shell_structure(monkeypatch, header):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    suffix = " --output public.txt https://example.test/health; echo 'keep  spacing'"
    for option in ("--header=", "--header ", "-H "):
        for quote, spacing, value in (("'", " ", "opaque header 927"),
                                      ('"', " ", "opaque header 927"),
                                      ("", "", "opaque_header_927")):
            prefix = "curl  " + option + quote + header + ":" + spacing
            args = {"command": prefix + value + quote + suffix,
                    "note": "--header-file ./public.txt"}
            original = deepcopy(args)

            assert _enqueue_args(args) == {
                **original, "command": prefix + "***" + quote + suffix,
            }
            assert args == original

    if header == "Authorization":
        monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})
        redact.register_vault_redaction_value(header)
        safe = _enqueue_args({"command": f"curl --header='{header}: opaque' --output public.txt"})
        assert header not in safe["command"]
        assert "opaque" not in safe["command"]
        assert "--output public.txt" in safe["command"]


@pytest.mark.parametrize(
    "url, masked",
    [
        ("custom+db://alice:cedar'maple@example.test/path", "custom+db://***@example.test/path"),
        ("//alice:cedar'maple@example.test/path", "//alice:***@example.test/path"),
        ("https://alice:cedar;maple@example.test/path", "https://***@example.test/path"),
        ("https://example.test/owner(path)?field=cedar;maple", "https://example.test/owner(path)?field=***"),
        ("//alice:cedar;maple@example.test/path", "//alice:***@example.test/path"),
        ("/owner(path)?field=cedar;maple", "/owner(path)?field=***"),
        *[
            (prefix + "?fi'eld=cedar'maple&next=birch'willow#elm'ash",
             prefix + "?fi'eld=***&next=***#***")
            for prefix in (
                "https://example.test/owner's/path", "//example.test/owner's/path",
                "/owner's/path", "./owner's/path", "../owner's/path", "owner's/path", "",
            )
        ],
        *[
            (prefix + "#cedar'maple", prefix + "#***")
            for prefix in ("https://example.test/path", "//example.test/path",
                           "/path", "./path", "../path", "path")
        ],
    ],
)
def test_full_url_apostrophes_remain_secret_before_enqueue(monkeypatch, url, masked):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    other = url.replace("cedar", "spruce").replace("maple", "beech")
    args = {
        "url": url,
        "records": [{"label": url}, [url], (url,)],
        "mapping": {url: "first", other: "second"},
    }
    original = deepcopy(args)

    safe = _enqueue_args(args)

    for half in ("cedar", "maple", "birch", "willow", "elm", "ash", "spruce", "beech"):
        assert half not in json.dumps(safe)
    assert safe == {
        "url": masked,
        "records": [{"label": masked}, [masked], [masked]],
        "mapping": {masked: "first", masked + " «redacted-key-collision-2»": "second"},
    }
    assert args == original


@pytest.mark.parametrize(
    "url, masked",
    [
        ('"https://alice:cedar\'maple@example.test/owner\'s/path?fi\'eld=birch\'willow#elm\'ash"',
         '"https://***@example.test/owner\'s/path?fi\'eld=***#***"'),
        *[
            ('"' + prefix + "?fi'eld=cedar'maple#elm'ash\"",
             '"' + prefix + "?fi'eld=***#***\"")
            for prefix in ("//example.test/owner's/path", "/owner's/path", "./owner's/path",
                           "../owner's/path", "owner's/path", "")
        ],
        ("'https://example.test/path?field=cedar#elm'", "'https://example.test/path?field=***#***'"),
        ("'/path?field=cedar#elm'", "'/path?field=***#***'"),
        ("'path?field=cedar#elm'", "'path?field=***#***'"),
        ("'?field=cedar#elm'", "'?field=***#***'"),
        (r"https://example.test/path?field=cedar\'maple", "https://example.test/path?field=***"),
        (r"'https://example.test/path?field=cedar'\''maple'", "'https://example.test/path?field=***'"),
        ("'https://example.test/path?field=cedar'\"'\"'maple'", "'https://example.test/path?field=***'"),
        ("'//alice:cedar'\"'\"'maple@example.test/path'", "'//alice:***@example.test/path'"),
        ("'path?field=cedar'\"'\"'maple'", "'path?field=***'"),
        ("'?field=cedar'\"'\"'maple'", "'?field=***'"),
        ('"https://example.test/owner\'s/path"', '"https://example.test/owner\'s/path"'),
        ('"/tmp/owner\'s/path"', '"/tmp/owner\'s/path"'),
        ('"owner\'s/path"', '"owner\'s/path"'),
    ],
)
def test_full_url_apostrophes_preserve_surrounding_bytes(monkeypatch, url, masked):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    suffix = "; echo 'keep  spacing' --format json"
    args = {"command": "curl  " + url + suffix,
            "prose": "It's the owner's guide. Ready? code = public.",
            "path": "/tmp/owner's/path"}
    original = deepcopy(args)

    safe = _enqueue_args(args)

    assert safe == {**original, "command": "curl  " + masked + suffix}
    assert args == original


def test_full_url_failed_lookahead_completes_before_enqueue():
    # Run the real callback in a killable child: a regex regression must not
    # hang this test process or rely on a platform-specific signal handler.
    program = """
import runpy
import sys
from unittest.mock import patch

enqueue = runpy.run_path(sys.argv[1])["_enqueue_args"]
spelling = "'" + chr(92) + "''"
with patch("agent.redact._redact_enabled", return_value=False):
    for prefix in ("?", "/path?", "path?"):
        value = prefix + spelling * 40
        assert enqueue({"note": value}) == {"note": value}
    value = "https://example.test/path?field=" + spelling * 40
    assert enqueue({"url": value}) == {
        "url": "https://example.test/path?field=***"
    }
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, __file__],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("secret_kind", ["vendor", "vault"])
def test_full_metadata_stays_visible_without_exempting_values(monkeypatch, secret_kind):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})
    secret = "ghp_" + "a" * 36 if secret_kind == "vendor" else "isolated-vault-value-619"
    if secret_kind == "vault":
        redact.register_vault_redaction_value(secret)
    marked = "***" + secret
    scanned = redact.redact_sensitive_text(marked, force=True)
    assert secret not in scanned
    visible = {"author": "Alice", "token_count": 4, "signature_algorithm": "sha256"}
    # Exact metadata exceptions must not exempt similarly named credentials.
    credentials = {
        "author_token": "opaque-author-value",
        "token_count_secret": "opaque-count-value",
        "signature_algorithm_key": "opaque-signing-value",
    }
    args = {
        **visible,
        **credentials,
        "records": [
            {key: marked for key in visible},
            {"author": {"password": "opaque-nested-value", "label": marked}},
            ["token_count", 4],
            ("signature_algorithm", "sha256"),
            {"Author": "Alice", "tokenCount": 4, "signatureAlgorithm": "sha256"},
        ],
    }
    original = deepcopy(args)

    safe = _enqueue_args(args)

    assert secret not in json.dumps(safe)
    assert safe["records"][0] == {key: scanned for key in visible}
    assert safe["records"][1] == {"author": {"password": "***", "label": scanned}}
    assert safe == {
        **visible,
        **dict.fromkeys(credentials, "***"),
        "records": [
            {key: scanned for key in visible},
            {"author": {"password": "***", "label": scanned}},
            ["token_count", 4],
            ["signature_algorithm", "sha256"],
            {"Author": "Alice", "tokenCount": 4, "signatureAlgorithm": "sha256"},
        ],
    }
    assert args == original
