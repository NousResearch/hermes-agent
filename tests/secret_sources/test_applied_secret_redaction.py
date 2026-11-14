"""Secret material stays redacted after line-oriented output splits it from its markers.

Inspired by Devin's Oct 7 2026 release note ("Private Key Strings Are Redacted More
Aggressively": each line of a saved private key is redacted in command output, so the
key stays hidden even when only part of it is printed).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from contextlib import contextmanager

import pytest

from agent import redact_secret_sources
from agent.redact import redact_sensitive_text, redact_terminal_output, register_vault_redaction_value
from agent.secret_sources import registry as reg
from agent.secret_sources.base import FetchResult, SecretSource
from hermes_cli import env_loader
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _key(label: str, kind: str, lines: int) -> tuple[str, list[str]]:
    """A PEM-shaped key of ``lines`` 64-char base64 body lines (deterministic, not a real key)."""
    blob = b"".join(hashlib.sha512(f"{label}{i}".encode()).digest() for i in range(lines))
    b64 = base64.b64encode(blob).decode()
    body = [b64[i:i + 64] for i in range(0, 64 * lines, 64)]
    dashes = "-" * 5
    return "\n".join([f"{dashes}BEGIN {kind}{dashes}", *body, f"{dashes}END {kind}{dashes}"]), body


SSH_KEY, SSH_BODY = _key("ssh", "OPENSSH PRIVATE KEY", 6)
RSA_KEY, RSA_BODY = _key("rsa", "RSA PRIVATE KEY", 25)
BODY = SSH_BODY + RSA_BODY


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reg._reset_registry_for_tests()
    monkeypatch.setattr(reg, "_ensure_builtin_sources", lambda: None)
    monkeypatch.setattr(redact_secret_sources, "_VALUES_BY_HOME", {})
    monkeypatch.setattr("agent.redact._VAULT_REDACTION_VALUES", {})
    env_loader.reset_secret_source_cache()
    environ_before = dict(os.environ)
    yield
    for name in set(os.environ) - set(environ_before):
        del os.environ[name]
    os.environ.update(environ_before)
    env_loader.reset_secret_source_cache()
    reg._reset_registry_for_tests()


@contextmanager
def _profile(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _home(tmp_path, name):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text("secrets:\n  bulk:\n    enabled: true\n", encoding="utf-8")
    return home


def _bulk_source(pulls):
    """Register a bulk source whose Nth fetch returns ``pulls[N]`` (the last one repeats)."""
    pulls = list(pulls)

    class _Bulk(SecretSource):
        name, label, shape, scheme = "bulk", "Bulk", "bulk", "bulk://x"
        api_version = reg.SECRET_SOURCE_API_VERSION

        def fetch(self, cfg, home_path):
            return FetchResult(secrets=dict(pulls.pop(0) if len(pulls) > 1 else pulls[0]))

    assert reg.register_source(_Bulk())


def _apply(home):
    env_loader.reset_secret_source_cache(home)
    env_loader._apply_external_secret_sources(home)


def _leaked(home, secrets: dict[str, str]) -> dict[str, list[str]]:
    """Body lines each shell pipeline over the env dump still shows after redaction."""
    env_lines = [ln for k, v in secrets.items() for ln in f"{k}={v}".splitlines()] + ["PATH=/usr/bin"]
    pipelines = {
        "env": env_lines,
        "env | sort": sorted(env_lines),
        "env | sort | grep -v -i key": [ln for ln in sorted(env_lines) if "key" not in ln.lower()],
        "env | cut -d= -f1": [ln.split("=", 1)[0] for ln in env_lines],
        "env | grep -v -e BEGIN -e END": [ln for ln in env_lines if "BEGIN" not in ln and "END" not in ln],
        "printenv SSH_KEY | tr -d '\\n'": [SSH_KEY.replace("\n", "")],
    }
    with _profile(home):
        out = {cmd: redact_terminal_output("\n".join(lines), command=cmd, force=True) for cmd, lines in pipelines.items()}
    return {cmd: leaked for cmd, text in out.items() if (leaked := [b for b in BODY if b in text])}


def test_applied_key_lines_stay_masked_past_any_number_of_values_and_vault_fills(tmp_path):
    # Bulk blob of 82 vars (2 keys + 80 tokens), then 64 unrelated browser-vault/OTP fills in the
    # same profile: nothing may evict the still-applied keys' body lines.
    secrets = {"SSH_KEY": SSH_KEY, "RSA_KEY": RSA_KEY,
               **{f"SVC{i}_API_TOKEN": hashlib.sha256(str(i).encode()).hexdigest()[:32] for i in range(80)}}
    _bulk_source([secrets])
    home_a, home_b = _home(tmp_path, "a"), _home(tmp_path, "b")
    _apply(home_a)
    with _profile(home_a):
        for i in range(64):
            register_vault_redaction_value(f"{i:06d}")

    assert _leaked(home_a, secrets) == {}
    with _profile(home_b):  # another profile's output is never scrubbed against A's values
        assert SSH_BODY[0] in redact_sensitive_text(SSH_BODY[0])


def test_ordinary_config_and_non_key_multiline_lines_stay_unmasked(tmp_path):
    sa_key, _ = _key("sa", "PRIVATE KEY", 4)
    service_account = json.dumps({"type": "service_account", "project_id": "myapp-prod",
                                  "private_key": sa_key + "\n",
                                  "token_uri": "https://oauth2.googleapis.com/token"}, indent=2)
    ordinary = {"AWS_REGION": "eu-central-1", "PROJECT_DIR": "/home/user/projects/myapp",
                "API_BASE": "https://api.example.com/v1", "APP_VERSION": "v2.14.0-beta",
                "APP_NAME": "Payments Service 2"}
    _bulk_source([{**ordinary, "GOOGLE_CREDENTIALS": service_account, "SSH_KEY": SSH_KEY}])
    home = _home(tmp_path, "a")
    _apply(home)

    text = "\n".join([
        "region eu-central-1 ok", "cd /home/user/projects/myapp", "curl https://api.example.com/v1/users",
        "release v2.14.0-beta", "Payments Service 2", '  "type": "service_account",',
        '  "token_uri": "https://oauth2.googleapis.com/token"', '  "project_id": "myapp-prod",',
    ])
    with _profile(home):
        assert redact_sensitive_text(text, force=True) == text
        assert SSH_BODY[0] not in redact_sensitive_text(f"x {SSH_BODY[0]}")  # the key itself still registers


def test_a_revoked_key_is_unmasked_and_its_replacement_masked_on_reload(tmp_path):
    other, other_body = _key("rotated", "OPENSSH PRIVATE KEY", 6)
    _bulk_source([{"SSH_KEY": SSH_KEY}, {"DEBUG_MODE": "verbose"}, {"SSH_KEY": other}])
    home = _home(tmp_path, "a")
    _apply(home)
    with _profile(home):
        assert SSH_BODY[1] not in redact_sensitive_text(SSH_BODY[1])
    _apply(home)  # revoked: the source stopped supplying SSH_KEY
    with _profile(home):
        assert redact_sensitive_text(SSH_BODY[1]) == SSH_BODY[1]
    _apply(home)  # reload with a new key: only the live one is masked
    with _profile(home):
        assert redact_sensitive_text(SSH_BODY[1]) == SSH_BODY[1]
        assert other_body[1] not in redact_sensitive_text(other_body[1])


def test_partial_private_key_print_masks_the_body_next_to_a_lone_armor_line():
    lines = SSH_KEY.splitlines()
    head = "\n".join(lines[:4]) + "\n"    # head -n 4 id_ed25519
    tail = "\n".join(lines[-4:]) + "\n"   # tail -n 4 id_ed25519
    for out in (head, tail):
        redacted = redact_sensitive_text(out, force=True)
        assert not [b for b in SSH_BODY if b in redacted]
        assert "[REDACTED PRIVATE KEY]" in redacted
    # A base64 run with no private-key armor next to it (a public key, a data blob) is kept, and so
    # is a hex digest listed after a lone BEGIN line in docs.
    prose = lines[0].replace("OPENSSH PRIVATE", "PUBLIC") + "\n" + SSH_BODY[0] + "\n"
    assert redact_sensitive_text(prose, force=True) == prose
    digest = hashlib.sha256(b"doc").hexdigest()
    assert digest in redact_sensitive_text(f"{lines[0]}\n{digest}\n", force=True)
