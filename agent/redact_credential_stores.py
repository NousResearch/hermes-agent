"""Credential-store line masking for agent.redact (#124999).

``.netrc``, ``.pgpass``, ``.npmrc`` and INI stores (``.pypirc``, ``~/.aws/credentials``) have no
``KEY=value`` shape the assignment passes understand. These passes run only for a source already
classified as secret-bearing (``_is_secret_file_arg``). Also home to the rendered-line grammar
(``_SECRET_CFG_NAMES``, ``_LINE_NUMBER_GUTTER``) that agent/redact.py's anchored passes share.
"""

import re

_SECRET_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential|auth)"
# Rendered line-number prefix: ``5|line`` (read_file), ``6:line`` (grep -n), ``7-line`` (grep -A/-B/-C
# context lines) and ``     8\tline`` (cat -n / nl: right-aligned number + TAB). Callers put the ONLY
# leading ``[ \t]*`` in front of it — stacking a second whitespace run around an optional gutter made
# the anchored passes quadratic on long indented lines (2s per 5k spaces).
_LINE_NUMBER_GUTTER = r"(?:[0-9]+(?:[|:\-]|\t)[ \t]*)?"

# The key group of the INI pass carries the gutter and the ``=`` separator.
_AWS_CREDENTIAL_PROCESS_KEY_RE = re.compile(r"(?:^|[^\w.\-])credential_process[ \t]*=?[ \t]*$")


# Credential-store lines that have no ``KEY=value`` shape, so the assignment passes miss them:
# ``.netrc`` (``machine h login u password X``), ``.pgpass`` (``host:port:db:user:X``)
# and ``.npmrc`` (``//registry/:_authToken=X``); a netrc password may be a quoted token with
# spaces. ``.git-credentials`` URL userinfo is masked by _redact_userinfo_passwords on every read.
# They run only for a source already classified as secret-bearing. Each is shaped so ``.env``,
# shell rc and config.yaml lines do not match: ``PATH=/a:/b:/c:/d:/e`` carries ``=`` in its first
# field and a ``# password ...`` comment does not start with a netrc keyword.
# A netrc line is ``keyword value`` pairs, several ``machine`` entries may share one line.
_NETRC_LINE_RE = re.compile(
    rf"^([ \t]*{_LINE_NUMBER_GUTTER})((?:machine|default|login|account|port|password)\b[^\r\n]*)", re.MULTILINE,
)
_NETRC_TOKEN_RE = re.compile(r'"[^"\r\n]*"?|\S+')
_CREDENTIAL_STORE_RES = (
    re.compile(rf"(^[ \t]*{_LINE_NUMBER_GUTTER}(?:[^:\s=#]*:){{4}})(\S[^\r\n]*)", re.MULTILINE),
    # npm expands ``${NAME}`` / ``${NAME?}`` here: that reference names a variable, it is not the secret.
    re.compile(r'(:_(?:authToken|auth|password)[ \t]*=[ \t]*)(?!"?\$\{\w+\??\}"?(?:\s|$))(\S+)'),
)
# INI ``password = X`` (``.pypirc``, ``~/.aws/credentials``): the anchored config pass only accepts
# ``key=value`` with no spaces around ``=``, and stops ``password=a b`` at the first space. The value
# runs to end of line: configparser keeps internal spaces.
# NOTE(perf): the key run is possessive and the secret word is checked in the callback
# (_CFG_SECRET_WORD_RE): ``run{NAMES}run`` backtracked O(n^2) on a long keyword line with no ``=``.
# ``(?<=api) (?=key)`` keeps the one key spelling with a space, ``api key = X``, in the key run.
_INI_SPACED_ASSIGN_RE = re.compile(
    rf"(^[ \t]*{_LINE_NUMBER_GUTTER}([A-Za-z0-9_.\-]*+(?:(?<=api) (?=key)[A-Za-z0-9_.\-]*+)?+)"
    r"(?:[ \t]++=[ \t]*+|=[ \t]++|=(?=\S+[ \t]+[^\s#])))(\S(?:[^\r\n]*\S)?)",
    re.IGNORECASE | re.MULTILINE,
)
_CFG_SECRET_WORD_RE = re.compile(_SECRET_CFG_NAMES, re.IGNORECASE)


def redact_credential_store_lines(text: str, mask, should_redact) -> str:
    """Mask the value of every credential-store line: in a credential store the value is a secret
    whatever it looks like, so a weak human password is masked too. ``should_redact(key, value)``
    is agent.redact's assignment gate, applied to the INI pass."""
    def _sub(m):
        value = m.group(2)
        if value == "***" or value.startswith("«redacted"):
            return m.group(0)
        return m.group(1) + mask(value)

    def _netrc_sub(m):
        line, out, pos, after_password = m.group(2), [], 0, False
        for tok in _NETRC_TOKEN_RE.finditer(line):
            value = tok.group(0)
            if after_password and value[0] not in "=:" and value != "***" and not value.startswith("«redacted"):
                out.append(line[pos:tok.start()] + mask(value))
                pos = tok.end()
            after_password = not after_password and value == "password"
        return m.group(1) + "".join(out) + line[pos:]

    def _ini_sub(m):
        key, value = m.group(1), m.group(3)
        if not _CFG_SECRET_WORD_RE.search(m.group(2)) or not should_redact(key, value):
            return m.group(0)
        return f"{key}{mask(value)}"

    text = _NETRC_LINE_RE.sub(_netrc_sub, text)
    for pattern in _CREDENTIAL_STORE_RES:
        text = pattern.sub(_sub, text)
    return _INI_SPACED_ASSIGN_RE.sub(_ini_sub, text)
