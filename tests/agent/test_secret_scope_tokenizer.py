"""The profile secret scope and ``os.environ`` read one ``.env`` with one tokenizer.

``load_env_file`` is the tokenizer every reader is meant to share, "so no two boundaries disagree
on which keys/values a file defines". It parsed line by line while ``load_hermes_dotenv`` parses
with python-dotenv, so a quoted multi-line value (a PEM key) reached ``os.environ`` whole and the
scope truncated to its first line, with the quote still on it.
"""

from __future__ import annotations

import os

import agent.secret_scope as ss
from hermes_cli.env_loader import load_hermes_dotenv

PEM = "-----BEGIN PRIVATE KEY-----\nline one\nline two\n-----END PRIVATE KEY-----"


def test_the_scope_and_the_environment_read_one_file_alike(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    (home / ".env").write_bytes(
        f'OPENAI_API_KEY=sk-test-123\nCERT_PEM="{PEM}"\nESCAPED="line1\\nline2"\nAFTER=still-parsed\n'.encode()
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("CERT_PEM", "ESCAPED", "AFTER"):
        monkeypatch.delenv(name, raising=False)
    ss.invalidate_env_file_cache()

    scope = ss.build_profile_secret_scope(home)
    load_hermes_dotenv()

    assert scope["CERT_PEM"] == PEM
    assert scope["ESCAPED"] == "line1\nline2"
    for name in ("CERT_PEM", "ESCAPED", "AFTER"):
        assert scope[name] == os.environ[name], name
