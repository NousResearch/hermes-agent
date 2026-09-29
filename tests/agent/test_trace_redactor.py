"""Seeded-secret tests for agent/trace_redactor.py.

Acceptance per task t_c0942a1a:
- seeded secrets in nested JSON args, string results, and message content
- assert zero seeded secrets survive in output
- JSON structure is preserved (same shape post-redaction)
- always-safe fields (tool name, exit_code, duration_ms, join keys) untouched
- marker shape conforms to specs/trace-v1.schema.json $defs/redactionMarker
- fail-closed on detector exceptions
- detectors driven by config, not hardcoded
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest
import yaml

from agent import trace_redactor as tr


# ----------------------------------------------------------------------
# Fixtures / seeds
# ----------------------------------------------------------------------
REPO_ROOT = tr.Path(__file__).resolve().parents[2] if hasattr(tr, "Path") else None

SECRETS = {
    "anthropic": "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx",
    "openai_style": "sk-AbCdEfGhIjKlMnOpQrSt",  # 16 chars after sk-
    "aws": "AKIAIOSFODNN7EXAMPLE",
    "ghp": "ghp_" + "a1B2c3D4" * 4 + "wxyz",  # 36 chars after ghp_
    "gh_pat": "github_pat_11ABCDEFG0" + "x" * 40,
    "slack": "xoxb-123456789012-abcdefghijkl",
    "bearer": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "basic_b64": "dXNlcjpwYXNzd29yZA==",
    "password_val": "hunter2-not-a-real-password",
    "email": "jane.doe+ci@example-corp.com",
    "phone": "+1 (415) 555-0132",
    "card": "4242 4242 4242 4242",  # Luhn-valid test card
}

PRIVATE_KEY = """-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABFwAAAAdzc2gtcn
NhAAAAAwEAAQAAAQEA1234567890abcdefghijklmno
-----END OPENSSH PRIVATE KEY-----"""


@pytest.fixture(scope="module")
def redactor():
    """Redactor built from the packaged default config."""
    cfg_path = (
        tr.Path(__file__).resolve().parents[2]
        / "specs"
        / "trace_redaction_default.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    return tr.TraceRedactor(cfg, config_source=str(cfg_path))


def ctx(path: str = "", always_safe: bool = False) -> tr.RedactionContext:
    return tr.RedactionContext(path=path, always_safe=always_safe)


def assert_no_secrets(payload, seeds=SECRETS, extra=()):
    blob = json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload
    for name, s in list(seeds.items()) + list(extra):
        fragments = {s}
        if name == "bearer":
            fragments.add(s.split(None, 1)[1])  # token without scheme
        if name == "card":
            fragments.add(s.replace(" ", ""))
        for frag in fragments:
            assert frag not in blob, "seeded secret %r (%s) survived in %r" % (
                frag,
                name,
                blob[:500],
            )
    assert PRIVATE_KEY.splitlines()[1] not in blob, "private key body survived"


def is_marker(v):
    return (
        isinstance(v, dict)
        and v.get("redacted") is True
        and v.get("reason") in ("credential", "pii", "redactor_error")
    )


# ----------------------------------------------------------------------
# Always-safe and scalar pass-through (schema S4.1, S3 table)
# ----------------------------------------------------------------------
class TestPassThrough:
    def test_always_safe_context_skips_detectors(self, redactor):
        v = redactor.redact(SECRETS["anthropic"], context=ctx("tool_calls[0].name", True))
        assert v == SECRETS["anthropic"]

    def test_always_safe_keys_untouched_inside_arguments(self, redactor):
        args = {
            "name": "sk-ant-api03-notarealkeybutmatches",  # always-safe structural key
            "exit_code": 0,
            "duration_ms": 1234,
            "is_terminal": False,
            "id": "call_" + SECRETS["ghp"],  # join key — never redacted
        }
        out = redactor.redact(args, context=ctx("tool_calls[0].arguments"))
        assert out == args

    def test_scalars_pass_through(self, redactor):
        assert redactor.redact(42, context=ctx()) == 42
        assert redactor.redact(4.25, context=ctx()) == 4.25
        assert redactor.redact(True, context=ctx()) is True
        assert redactor.redact(None, context=ctx()) is None
        assert redactor.redact("", context=ctx()) == ""

    def test_benign_text_untouched(self, redactor):
        s = "Ran pytest, 47 passed. The monkey-patch works."
        assert redactor.redact(s, context=ctx("tool_calls[0].result")) == s

    def test_marker_passthrough(self, redactor):
        marker = {"redacted": True, "reason": "credential", "sha256": "a" * 64}
        assert redactor.redact(marker, context=ctx()) == marker


# ----------------------------------------------------------------------
# Seeded secrets in nested JSON arguments
# ----------------------------------------------------------------------
class TestNestedArguments:
    def test_all_seeds_redacted_from_nested_args(self, redactor):
        args = {
            "command": "curl -H 'Authorization: Bearer %s' https://api.example.com"
            % SECRETS["bearer"].split(None, 1)[1],
            "env": {
                "ANTHROPIC_API_KEY": SECRETS["anthropic"],
                "AWS_CREDS": SECRETS["aws"],
                "GH": SECRETS["ghp"],
                "FINE_GRAINED": SECRETS["gh_pat"],
                "SLACK_BOT": SECRETS["slack"],
                "OPENAI": SECRETS["openai_style"],
            },
            "headers": ["Authorization: Basic " + SECRETS["basic_b64"]],
            "config": {
                "database": {"password": SECRETS["password_val"], "host": "db.internal"},
                "deploy_key": PRIVATE_KEY,
                "session_jwt": SECRETS["jwt"],
            },
            "contacts": {
                "owner_email": SECRETS["email"],
                "phone": SECRETS["phone"],
                "billing_card": SECRETS["card"],
            },
            "notes": "plain text, no secrets here",
            "count": 3,
        }
        out = redactor.redact(args, context=ctx("tool_calls[3].arguments"))
        assert_no_secrets(out)

        # Structure preserved
        assert out["count"] == 3
        assert out["notes"] == args["notes"]
        assert out["config"]["database"]["host"] == "db.internal"
        assert out["env"]["OPENAI"] is None or is_marker(out["env"]["OPENAI"])
        # password_field: the whole value becomes a marker object
        pw = out["config"]["database"]["password"]
        assert is_marker(pw) and pw["reason"] == "credential"
        assert re.fullmatch(r"[0-9a-f]{64}", pw["sha256"])
        # bearer prefix survives inline; token replaced with span placeholder
        assert "Bearer «REDACTED:credential:sha256:" in out["command"]
        # basic_auth keeps the scheme prefix, redacts only the blob
        hdr = out["headers"][0]
        assert hdr.startswith("Authorization: Basic «REDACTED:credential:sha256:")
        assert is_marker(out["env"]["ANTHROPIC_API_KEY"])
        assert is_marker(out["config"]["session_jwt"])
        assert out["config"]["session_jwt"]["reason"] == "credential"
        # whole-string private key -> marker object
        assert is_marker(out["config"]["deploy_key"])
        # PII reasons
        assert is_marker(out["contacts"]["owner_email"])
        assert out["contacts"]["owner_email"]["reason"] == "pii"
        assert is_marker(out["contacts"]["phone"])
        assert out["contacts"]["phone"]["reason"] == "pii"
        card = out["contacts"]["billing_card"]
        assert is_marker(card) and card["reason"] == "pii"

    def test_output_remains_json_serializable_and_valid(self, redactor):
        args = {"nested": [{"secret": SECRETS["anthropic"], "ok": "fine"}]}
        out = redactor.redact(args, context=ctx("tool_calls[0].arguments"))
        json.dumps(out)  # must not raise
        assert out["nested"][0]["ok"] == "fine"

    def test_json_string_result_redacted_as_structure(self, redactor):
        result = json.dumps({"user": {"token": SECRETS["ghp"], "id": 7}})
        out = redactor.redact(result, context=ctx("tool_calls[1].result"))
        assert isinstance(out, str)  # stays a string
        reparsed = json.loads(out)  # valid JSON
        assert reparsed["user"]["id"] == 7
        assert is_marker(reparsed["user"]["token"])
        assert_no_secrets(out)

    def test_deeply_nested_list_of_dicts(self, redactor):
        args = {"matrix": [[{"apikey": SECRETS["openai_style"]}], [["plain"]]]}
        out = redactor.redact(args, context=ctx("tool_calls[2].arguments"))
        assert is_marker(out["matrix"][0][0]["apikey"])
        assert out["matrix"][1][0][0] == "plain"
        assert_no_secrets(out)


# ----------------------------------------------------------------------
# Message content redaction
# ----------------------------------------------------------------------
class TestMessageContent:
    def test_assistant_content_with_embedded_secret(self, redactor):
        content = "I found key %s in the config file." % SECRETS["anthropic"]
        out = redactor.redact(content, context=ctx("messages[5].content"))
        assert isinstance(out, str)
        assert "«REDACTED:credential:sha256:" in out
        assert_no_secrets(out)

    def test_tool_message_result_with_email_and_card(self, redactor):
        content = "Contact %s or charge %s for the order." % (
            SECRETS["email"],
            SECRETS["card"],
        )
        out = redactor.redact(content, context=ctx("messages[6].content"))
        assert "«REDACTED:pii:sha256:" in out
        assert_no_secrets(out)

    def test_whole_content_becomes_marker(self, redactor):
        out = redactor.redact(SECRETS["jwt"], context=ctx("messages[7].content"))
        assert is_marker(out)
        assert out["reason"] == "credential"

    def test_null_assistant_content_untouched(self, redactor):
        assert redactor.redact(None, context=ctx("messages[8].content")) is None


# ----------------------------------------------------------------------
# Detector specificity: no false positives on benign lookalikes
# ----------------------------------------------------------------------
class TestNoFalsePositives:
    def test_non_luhn_number_survives(self, redactor):
        # Off-by-one check digit fails Luhn -> the card detector must not fire
        # (regression guard for "reject non-Luhn digit runs" in S4.2).
        assert redactor._luhn_matches("order id 4242424242424241 placed") == []

    def test_short_digit_runs_survive(self, redactor):
        # 7 digits after strip: below the schema's >=8 phone threshold.
        assert redactor.redact("call 415 5550", context=ctx()) == "call 415 5550"
        assert redactor.redact("exit code 4242", context=ctx()) == "exit code 4242"

    def test_monkey_and_name_keys_untouched(self, redactor):
        args = {"monkey": "banana", "name": "sk-ant-api03-XXXXXXXXXXXXXXXXXXXX",
                "rename_plan": SECRETS["aws"]}
        out = redactor.redact(args, context=ctx())
        assert out["monkey"] == "banana"
        assert out["name"] == args["name"]  # always-safe key wins over content
        assert is_marker(out["rename_plan"])  # AKIA caught as embedded span? whole value

    def test_public_key_block_survives(self, redactor):
        pub = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBg\n-----END PUBLIC KEY-----"
        assert redactor.redact(pub, context=ctx()) == pub

    def test_nonmatching_basic_auth_blob_survives(self, redactor):
        s = "Authorization: Basic dGV"  # < 8 chars
        assert redactor.redact(s, context=ctx()) == s


# ----------------------------------------------------------------------
# Marker semantics: dedup via stable sha256
# ----------------------------------------------------------------------
class TestMarkerSemantics:
    def test_same_secret_same_hash(self, redactor):
        a = redactor.redact({"password": SECRETS["password_val"]}, context=ctx("a"))
        b = redactor.redact({"password": SECRETS["password_val"]}, context=ctx("b"))
        assert a["password"]["sha256"] == b["password"]["sha256"] == hashlib.sha256(
            SECRETS["password_val"].encode()
        ).hexdigest()

    def test_distinct_secrets_distinct_hashes(self, redactor):
        c1 = ctx()
        out = redactor.redact(
            {"a": SECRETS["anthropic"], "b": SECRETS["ghp"]}, context=c1
        )
        assert out["a"]["sha256"] != out["b"]["sha256"]

    def test_redaction_log_records_events(self, redactor):
        c = ctx("tool_calls[0].arguments")
        redactor.redact({"password": SECRETS["password_val"]}, context=c)
        assert len(c.log) == 1
        entry = c.log[0]
        assert set(entry) == {"path", "reason", "sha256"}
        assert entry["path"].endswith("password")
        assert entry["reason"] == "credential"


# ----------------------------------------------------------------------
# Fail-closed behavior (schema S4.1)
# ----------------------------------------------------------------------
class TestFailClosed:
    def test_detector_exception_yields_redactor_error_marker(self):
        cfg = {
            "detector_reasons": {"boom": "credential"},
            "detectors": {"boom": {"patterns": ["x(?"],}},  # invalid regex
        }
        with pytest.raises(tr.RedactionError):
            tr.TraceRedactor(cfg)

    def test_runtime_exception_fail_closed(self, redactor, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("detector exploded")

        monkeypatch.setattr(tr.TraceRedactor, "_redact_string", boom)
        c = ctx("messages[0].content")
        out = redactor.redact("contains sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv", context=c)
        assert is_marker(out)
        assert out["reason"] == "redactor_error"
        assert out["sha256"] is None
        assert c.log and c.log[-1]["reason"] == "redactor_error"

    def test_missing_config_raises_redaction_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_TRACE_REDACTION_CONFIG", str(tmp_path / "nope.yaml"))
        monkeypatch.setattr(tr, "_DEFAULT_CONFIG_PATH", str(tmp_path / "also-nope.yaml"))
        monkeypatch.setattr(tr, "_PACKAGED_DEFAULT_CONFIG", tmp_path / "nope3.yaml")
        with pytest.raises(tr.RedactionError):
            tr.TraceRedactor.from_default_config()


# ----------------------------------------------------------------------
# Config-driven detectors
# ----------------------------------------------------------------------
class TestConfigDriven:
    def test_custom_detector_from_config(self):
        cfg = {
            "detector_reasons": {"internal_key": "credential"},
            "detectors": {
                "internal_key": {"patterns": [r"ACME-[A-Z0-9]{12}"]},
            },
        }
        r = tr.TraceRedactor(cfg)
        out = r.redact("code ACME-0123456789AB leaked", context=ctx("m.content"))
        assert "«REDACTED:credential:sha256:" in out
        assert "ACME-0123456789AB" not in out

    def test_disabled_detector_not_applied(self):
        cfg = {"detector_reasons": {"email": "pii"}, "detectors": {
            "email": {"patterns": [r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"]},
        }}
        r = tr.TraceRedactor(cfg)  # no password_field or api_key detectors configured
        out = r.redact({"password": "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv"}, context=ctx())
        # password key not configured -> no redaction; proves nothing is hardcoded
        assert out["password"] == "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv"

    def test_packaged_default_loads(self, redactor):
        assert redactor.config_source.endswith("trace_redaction_default.yaml")
        assert len(redactor._regex_detectors) >= 8

    def test_default_config_load_path(self, monkeypatch):
        monkeypatch.delenv("HERMES_TRACE_REDACTION_CONFIG", raising=False)
        monkeypatch.setattr(tr, "_DEFAULT_CONFIG_PATH", "/nonexistent/x.yaml")
        r = tr.TraceRedactor.from_default_config()
        assert r.config_source.endswith("trace_redaction_default.yaml")


# ----------------------------------------------------------------------
# t_d55b6db3: password-literal string-leaf + ghp_ variable-length
# ----------------------------------------------------------------------
class TestPasswordLiteralStringLeaf:
    """password_field must also fire inside string leaves (code, shell,
    connection strings), not only when the secret sits under a JSON key in
    the argument structure (QA t_0b40eccc blocker)."""

    def test_code_dict_literal(self, redactor):
        code = "pw = {'password': 'hunter2-QA-fake-password'}"
        out = redactor.redact(code, context=ctx("tool_calls[0].arguments.code"))
        assert isinstance(out, str)
        assert "hunter2-QA-fake-password" not in out
        assert "«REDACTED:credential:sha256:" in out
        # The `key =` prefix survives so the trace keeps shape.
        assert "'password':" in out

    def test_unquoted_yaml_style(self, redactor):
        s = "db:\n  password=hunter2\n  host: x"
        out = redactor.redact(s, context=ctx("tool_calls[0].arguments.command"))
        assert "hunter2" not in out
        assert "password=" in out

    def test_json_embedded_in_string(self, redactor):
        s = 'export CONF=\'{"password": "hunter2secret"}\''
        out = redactor.redact(s, context=ctx("tool_calls[0].arguments.command"))
        assert "hunter2secret" not in out

    def test_prose_no_separator_untouched(self, redactor):
        s = "reset your password here; the password is unknown"
        assert redactor.redact(s, context=ctx()) == s

    def test_short_value_untouched(self, redactor):
        # 3-char value below the 4-char floor: too weak to be a live cred,
        # too noisy to eat in prose.
        s = "try password=foo first"
        assert redactor.redact(s, context=ctx()) == s

    def test_credential_reason_in_log(self, redactor):
        c = ctx("tool_calls[0].arguments.code")
        redactor.redact("pw = {'password': 'hunter2-QA-fake'}", context=c)
        assert any(e["reason"] == "credential" for e in c.log)

    def test_value_key_shape_still_works(self, redactor):
        # The pre-existing whole-value path under a JSON key is unaffected.
        out = redactor.redact({"password": "hunter2-QA-fake"}, context=ctx())
        assert is_marker(out["password"])
        assert out["password"]["reason"] == "credential"


class TestGhpVariableLength:
    """ghp_ must be eaten whole at 36+ chars; fixed-{36} leaked a live
    suffix on disk (QA t_0b40eccc minor: trailing 'd6' survived)."""

    def test_canonical_36(self, redactor):
        tok = "ghp_" + "a1B2c3D4" * 4 + "wxyz"  # 36 chars after ghp_
        out = redactor.redact("token %s end" % tok, context=ctx())
        assert tok not in out

    def test_longer_than_36_eaten_whole(self, redactor):
        tail = "d6"
        tok = "ghp_" + "a1B2c3D4" * 4 + "wxyz" + tail  # 38 after ghp_
        out = redactor.redact("token %s end" % tok, context=ctx())
        assert tok not in out
        # The strict tail check: whole trailing sequence must be gone, not
        # just replaced-prefix + surviving tail.
        assert "wxyzd6" not in out

    def test_40_char_shape(self, redactor):
        tok = "ghp_" + "Z9y8X7w6V5" * 4  # 40 after ghp_
        out = redactor.redact("x %s y" % tok, context=ctx())
        assert tok not in out

    def test_shorter_than_36_not_a_ghp(self, redactor):
        # 35 chars: too short to be a ghp_ classic token. Boundary-less
        # fallback also requires >= 36; this string must survive untouched.
        s = "ghp_" + "a" * 35 + " end"
        assert redactor.redact(s, context=ctx()) == s

    def test_two_tokens_one_string(self, redactor):
        a = "ghp_" + "a1B2c3D4" * 4 + "wxyz"
        b = "ghp_" + "d4C3b2A1" * 4 + "pqrs" + "zz"
        out = redactor.redact("a %s b %s c" % (a, b), context=ctx())
        assert a not in out and b not in out
        # Both markers land.
        assert out.count("«REDACTED:credential:sha256:") == 2


# ----------------------------------------------------------------------
# Integration: full record shape against normative JSON Schema
# ----------------------------------------------------------------------
class TestSchemaConformance:
    def test_redacted_record_validates_against_schema(self, redactor):
        jsonschema = pytest.importorskip("jsonschema")
        schema_path = (
            tr.Path(__file__).resolve().parents[2] / "specs" / "trace-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text())

        record = {
            "schema_version": "1.0.0",
            "run_id": 839,
            "session_id": "20260929_090555_b960e5",
            "profile_slug": "viiy-privacy-engineer",
            "board_slug": "specialized",
            "task_id": "t_c0942a1a",
            "trace_id": "5f0d1a9c-3b2e-4f8a-9c1d-2e7f6a5b4c3d",
            "model": "kimi-k3",
            "provider": "ollama-cloud",
            "harness_version": {
                "prompt_builder_sha": "89cba9d8459191276f195ba73e183673d7709f58",
                "config_sha256": "a" * 64,
            },
            "messages": [
                {"role": "system", "content": "You are...", "tool_call_id": None,
                 "name": None, "index": 0},
                {"role": "assistant", "content": "Using token %s" % SECRETS["ghp"],
                 "tool_call_id": None, "name": None, "index": 1},
                {"role": "tool", "content": "user %s card %s" % (
                    SECRETS["email"], SECRETS["card"]),
                 "tool_call_id": "call_1", "name": "terminal", "index": 2},
            ],
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "terminal",
                    "arguments": {
                        "command": "curl -H 'Authorization: Basic %s' https://x"
                        % SECRETS["basic_b64"],
                        "env": {"api_key": SECRETS["aws"]},
                    },
                    "result": "ok: %s" % SECRETS["anthropic"],
                    "exit_code": 0,
                    "duration_ms": 812,
                    "is_terminal": False,
                }
            ],
            "outcome": "completed",
            "terminal_call": "kanban_complete",
            "exit_code": 0,
            "error_class": None,
            "killswitch": None,
            "safety_tier_hit": None,
            "token_counts": {"input": 100, "output": 50, "cache_read": 0,
                             "cache_write": 0, "total": 150},
            "cost_usd": 0.001,
            "token_counts_complete": True,
            "started_at": 1790687465,
            "ended_at": 1790689001,
            "wall_clock_seconds": 1536,
        }

        # Run the hook over the redactable fields exactly as the writer would.
        root_ctx = tr.RedactionContext()
        for msg in record["messages"]:
            msg["content"] = redactor.redact(
                msg["content"],
                context=tr.RedactionContext(
                    path="messages[%d].content" % msg["index"], log=root_ctx.log
                ),
            )
        for tc in record["tool_calls"]:
            tc["arguments"] = redactor.redact(
                tc["arguments"],
                context=tr.RedactionContext(
                    path="tool_calls[%s].arguments" % tc["id"], log=root_ctx.log
                ),
            )
            tc["result"] = redactor.redact(
                tc["result"],
                context=tr.RedactionContext(
                    path="tool_calls[%s].result" % tc["id"], log=root_ctx.log
                ),
            )
        if root_ctx.log:
            record["redaction_log"] = root_ctx.log

        assert_no_secrets(record)
        v = jsonschema.Draft202012Validator(schema)
        errors = list(v.iter_errors(record))
        assert not errors, [e.message for e in errors]

        # Credential-bearing marker present in tool_call arguments/result.
        assert any(
            e["reason"] == "credential" for e in record["redaction_log"]
        )
