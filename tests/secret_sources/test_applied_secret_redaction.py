"""Secret material stays redacted after line-oriented output splits it from its markers.

Inspired by Devin's Oct 7 2026 release note ("Private Key Strings Are Redacted More
Aggressively": each line of a saved private key is redacted in command output, so the
key stays hidden even when only part of it is printed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import redact
from agent.redact import redact_sensitive_text, redact_terminal_output
from agent.secret_sources import registry as reg
from agent.secret_sources.base import FetchResult, SecretSource
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

BEGIN, END = "-----BEGIN OPENSSH PRIVATE KEY-----", "-----END OPENSSH PRIVATE KEY-----"
BODY = [f"b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAt{i:x}" for i in range(6)]
KEY = "\n".join([BEGIN, *BODY, END])


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reg._reset_registry_for_tests()
    monkeypatch.setattr(reg, "_ensure_builtin_sources", lambda: None)
    monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})
    yield
    reg._reset_registry_for_tests()


def _under(home: Path, text: str, command: str) -> str:
    token = set_hermes_home_override(str(home))
    try:
        return redact_terminal_output(text, command=command, force=True)
    finally:
        reset_hermes_home_override(token)


def test_applied_multiline_secret_is_scrubbed_per_line_in_its_own_profile_only(tmp_path):
    class _Bulk(SecretSource):
        name, label, shape, scheme = "bulk", "Bulk", "bulk", "bulk://x"
        api_version = reg.SECRET_SOURCE_API_VERSION

        def fetch(self, cfg, home_path):
            return FetchResult(secrets={"SSH_KEY": KEY, "DEBUG": "true", "LOG_LEVEL": "info"})

    assert reg.register_source(_Bulk())
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir(), home_b.mkdir()
    reg.apply_all({"bulk": {"enabled": True}}, home_a, environ={})

    env_lines = ["DEBUG=true", *f"SSH_KEY={KEY}".splitlines(), "PATH=/usr/bin"]
    pipelines = {  # what each shell pipeline leaves of the env dump
        "env | sort": sorted(env_lines),
        "env | sort | grep -v -i key": [ln for ln in sorted(env_lines) if "key" not in ln.lower()],
        "env | cut -d= -f1": [ln.split("=", 1)[0] for ln in env_lines],
    }
    for command, lines in pipelines.items():
        assert not [b for b in BODY if b in _under(home_a, "\n".join(lines), command)], command
    # Isolation: profile B's output is never scrubbed against profile A's values.
    assert BODY[0] in _under(home_b, BODY[0], "cat notes.txt")
    # A bulk blob's mundane values are not turned into global substring scrubs.
    assert _under(home_a, "level info, debug true", "echo") == "level info, debug true"


def test_partial_private_key_print_masks_the_body_next_to_a_lone_armor_line():
    head = "\n".join([BEGIN, *BODY[:3]]) + "\n"   # head -n 4 id_ed25519
    tail = "\n".join([*BODY[3:], END]) + "\n"     # tail -n 4 id_ed25519
    for out in (head, tail):
        redacted = redact_sensitive_text(out, force=True)
        assert not [b for b in BODY if b in redacted]
        assert "[REDACTED PRIVATE KEY]" in redacted
    # A base64 run with no private-key armor next to it (a public key, a data blob) is kept.
    prose = "-----BEGIN PUBLIC KEY-----\n" + BODY[0] + "\n"
    assert redact_sensitive_text(prose, force=True) == prose
