"""Credential-store reads (#124999): ``~/.aws/credentials``, ``.netrc``, ``.pgpass``, ``.npmrc``,
``.pypirc`` and ``.git-credentials`` are secret-bearing sources whatever their line format."""

import subprocess
import sys
from pathlib import Path

import pytest

from agent.redact import _is_secret_file_arg, redact_sensitive_text, redact_terminal_output


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


class TestCredentialStoreReads:
    """Well-known credential stores (``~/.aws/credentials``, ``.netrc``, ``.pgpass``, ``.npmrc``,
    ``.pypirc``, ``.git-credentials``) are secret-bearing sources: a read_file-style read and a
    terminal ``cat`` of one must not hand the stored secret to the model, whatever its format."""

    WEAK = "hunter2weak"  # a human password: no vendor prefix, not opaque-looking

    @pytest.mark.parametrize("path, body", [
        ("~/.aws/credentials", "[default]\naws_access_key_id = AKIAIOSFODNN7EXAMPLE\naws_secret_access_key = {pw}\n"),
        ("C:\\Users\\bob\\.aws\\credentials", "[work]\naws_secret_access_key={pw}\n"),
        ("~/.netrc", "machine api.example.com\n  login bob\n  password {pw}\n"),
        ("/home/bob/.netrc", "machine api.example.com login bob password {pw}\n"),
        ("~/.pgpass", "db.example.com:5432:app:bob:{pw}\n"),
        ("~/.npmrc", "//registry.npmjs.org/:_authToken={pw}\n"),
        ("~/.pypirc", "[pypi]\nusername = __token__\npassword = {pw}\n"),
        ("~/.git-credentials", "https://bob:{pw}@github.com\n"),
    ])
    def test_read_file_and_terminal_cat_mask_the_stored_secret(self, path, body):
        text = body.format(pw=self.WEAK)
        rendered = "\n".join(f"{i}|{line}" for i, line in enumerate(text.splitlines(), 1))  # read_file gutter
        read = redact_sensitive_text(rendered, force=True, file_read=True, secret_file=_is_secret_file_arg(path))
        assert self.WEAK not in read
        assert self.WEAK not in redact_terminal_output(text, f"cat {path}", force=True)

    @pytest.mark.parametrize("path, body", [
        ("~/.netrc", 'machine h login bob password "a b {pw}"\n'),
        ("~/_netrc", "machine h login bob password {pw}\n"),
        ("~/.pypirc", "[pypi]\npassword = correct horse {pw}\n"),
        ("~/.pypirc", "[pypi]\npassword=correct horse {pw}\n"),
        ("~/.aws/config", "[profile x]\naws_session_token = {pw}\n"),
    ])
    def test_quoted_and_spaced_values_are_masked_whole(self, path, body):
        text = body.format(pw=self.WEAK)
        assert _is_secret_file_arg(path)
        assert self.WEAK not in redact_sensitive_text(text, force=True, file_read=True, secret_file=True)
        assert self.WEAK not in redact_terminal_output(text, f"cat {path}", force=True)

    @pytest.mark.parametrize("value", ["${NPM_TOKEN}", '"${NPM_TOKEN}"', "${NPM_TOKEN?}", "${NPM_TOKEN}\r"])
    def test_npmrc_env_reference_stays_readable(self, value):
        # npm expands ``${NAME}`` in .npmrc: the reference names a variable, not a stored token.
        text = f"//registry.npmjs.org/:_authToken={value}\n"
        assert value in redact_sensitive_text(text, force=True, file_read=True, secret_file=True)
        assert value in redact_terminal_output(text, "cat .npmrc", force=True)

    @pytest.mark.parametrize("value", ["${NPM_TOKEN} {pw}", "${NPM_TOKEN}\t{pw}", "{pw} tail"])
    def test_npmrc_literal_after_a_reference_is_masked(self, value):
        # Only a value that is nothing but the reference is exempt; the rest of the line is the secret.
        text = "//registry.npmjs.org/:_authToken=" + value.replace("{pw}", self.WEAK) + "\n"
        assert self.WEAK not in redact_sensitive_text(text, force=True, file_read=True, secret_file=True)
        assert self.WEAK not in redact_terminal_output(text, "cat .npmrc", force=True)

    def test_every_netrc_password_on_one_line_is_masked(self):
        # netrc is whitespace-separated tokens: several ``machine`` entries may share one line.
        text = f"machine a login u password {self.WEAK}1 machine b login v password {self.WEAK}2\n"
        for out in (redact_sensitive_text(text, force=True, file_read=True, secret_file=True),
                    redact_terminal_output(text, "cat ~/.netrc", force=True)):
            assert self.WEAK not in out
            assert "machine b login v password" in out

    @pytest.mark.parametrize("sep", [" = ", "="])
    def test_aws_credential_process_command_stays_readable(self, sep):
        # ``credential_process`` names the helper that fetches the credential; the secret beside it is masked.
        text = f"[profile p]\ncredential_process{sep}aws-vault export prod --format=json\naws_session_token{sep}{self.WEAK}\n"
        for out in (redact_sensitive_text(text, force=True, file_read=True, secret_file=True),
                    redact_terminal_output(text, "cat ~/.aws/config", force=True)):
            assert f"credential_process{sep}aws-vault export prod --format=json" in out
            assert self.WEAK not in out

    @pytest.mark.parametrize("path, line", [
        (".env", "credential_process={pw}"),
        ("~/.zshrc", "credential_process={pw}"),
        ("~/.hermes/config.yaml", "credential_process: {pw}"),
        ("~/.aws/config", "credential_process = {pw}"),  # no ``[section]`` above it
    ])
    def test_credential_process_exemption_is_scoped_to_aws_ini(self, path, line):
        # The exemption belongs to the AWS INI store: a ``.env`` / rc / config.yaml key of that name, or one
        # with no ``[section]`` above it, keeps the assignment gate.
        opaque = "aB3dE5fG7hJ9" + "kL1mN3pQ5rS7tU9"
        text = line.format(pw=opaque) + "\n"
        assert opaque not in redact_sensitive_text(text, force=True, file_read=True, secret_file=True)
        assert opaque not in redact_terminal_output(text, f"cat {path}", force=True)

    def test_lookalikes_and_shell_rc_lines_are_left_alone(self):
        # ``credentials`` alone is too generic: only the ``.aws`` parent makes it a store.
        for path in ("docs/credentials", "~/credentials", "~/config", "notes/netrc.md"):
            assert not _is_secret_file_arg(path)
            assert redact_terminal_output(f"password = {self.WEAK}\n", f"cat {path}") == f"password = {self.WEAK}\n"
        # The store formats never fire on ordinary secret-file lines.
        rc = "# password manager setup\nexport PATH=/a:/b:/c:/d:/e\nexport EDITOR=vim\n"
        assert redact_terminal_output(rc, "cat ~/.bashrc", force=True) == rc


def test_ini_pass_is_linear_on_a_long_keyword_line():
    # A secret word repeated with no ``=``: the INI key run backtracked O(n^2) (12s at 80k chars).
    # A subprocess with a hard timeout, because an in-process regex cannot be interrupted.
    code = "import sys; from agent.redact import redact_terminal_output as r; r(sys.stdin.read(), 'cat ~/.pypirc', force=True)"
    subprocess.run([sys.executable, "-c", code], input="token" * 16000, text=True, check=True, timeout=5,
                   cwd=Path(__file__).resolve().parents[2])
