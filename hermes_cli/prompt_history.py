"""Credential-safe on-disk prompt history for the interactive CLI.

The prompt_toolkit input area persists every submitted prompt to ``$HERMES_HOME/.hermes_history``
so Up/Down and auto-suggest can recall it later. That file outlives the session, is world-readable
by default, and is read by nothing that redacts — so a prompt like ``set OPENAI key sk-…`` or
``curl https://user:pass@host`` would sit in cleartext on disk until the file was trimmed by hand.

Two rules close that, inspired by Muse Code 1.4.0 (prompts carrying a recognizable credential are
neither saved to prompt history nor offered by recall; the history file is created owner-only):

* a prompt that Hermes's own secret redaction would mask (vendor-prefixed tokens, bearer headers,
  private keys, credentials in a URL, …) is kept out of both the on-disk file AND in-process recall —
  the user still gets their answer, they just cannot arrow-key back to the secret;
* the history file is created ``0600`` and an existing group/world-readable one is tightened the
  first time this process writes to it (POSIX only; Windows ACLs already scope ``%USERPROFILE%``).

The classifier is a round trip through :func:`agent.redact.redact_sensitive_text` with
``force=True`` (a ``security.redact_secrets: false`` preference must not reopen the on-disk leak)
and ``redact_url_credentials=True`` (tool flows keep OAuth/pre-signed URLs intact, but a history
file has no workflow to break, so the strict URL pass is the right one here).
"""

from __future__ import annotations

import os
import stat

from prompt_toolkit.history import FileHistory


def carries_credential(text: str) -> bool:
    """True when the redaction layer would mask anything in ``text``."""
    if not text:
        return False
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(text, force=True, redact_url_credentials=True) != text


def _ensure_owner_only(path: str) -> None:
    # O_CREAT with 0o600 covers the first write; the chmod covers a file an older Hermes created
    # with the umask default. Best-effort: a read-only or exotic filesystem must not lose history.
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
        if os.name == "posix" and stat.S_IMODE(os.stat(path).st_mode) & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        pass


class CredentialSafeFileHistory(FileHistory):
    """``FileHistory`` that refuses credential-bearing prompts and keeps its file owner-only."""

    def append_string(self, string: str) -> None:
        if carries_credential(string):
            return
        super().append_string(string)

    def store_string(self, string: str) -> None:
        _ensure_owner_only(str(self.filename))
        super().store_string(string)
