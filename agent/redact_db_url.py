"""DB URL query-string password masking and the userinfo placeholder test, for agent.redact.

``postgresql://db/app?user=app&password=…``: the userinfo form (``scheme://user:pass@host``) and the
rest of URL credential masking stay in agent/redact.py.
"""

import re


# A docs placeholder is a lower-case or ALL-CAPS name in angle brackets (``<password>``) or an
# upper-case env var reference (``${REDIS_PASSWORD}``); ``$ecretPassw0rd`` or ``<Zq8x…>`` is
# a real password that happens to start with the same character.
_DSN_PW_PLACEHOLDER_RE = re.compile(r"<(?:[a-z][a-z_ -]*|[A-Z][A-Z_ -]*)>|\$\{?[A-Z_][A-Z0-9_]*\}?|%[A-Z_][A-Z0-9_]*%")


# ``https://user:password@host`` in docs names the field instead of filling it in.
_USERINFO_PLACEHOLDER_WORDS = frozenset({"password", "passwd", "pass", "pw", "pwd", "secret", "token"})


def _is_userinfo_placeholder(password: str) -> bool:
    """A template (``{pw}``, ``{{ pw }}``), docs placeholder or field name, never a literal secret."""
    return ((password.startswith("{") and password.endswith("}"))
            or _DSN_PW_PLACEHOLDER_RE.fullmatch(password) is not None
            or password.casefold() in _USERINFO_PLACEHOLDER_WORDS)


# The same DB URLs (and their ``jdbc:`` forms) also take the password as a query parameter:
# ``postgresql://db/app?user=app&password=…``. Unlike web-URL queries these are never round-trip tokens.
# ``+driver`` covers SQLAlchemy URLs (``postgresql+psycopg2://``), which pass query params to the driver.
_DB_URL_QUERY_RE = re.compile(
    r"(?:postgres(?:ql)?|mysql|mariadb|mongodb|rediss?|amqps?)(?:\+[a-z0-9]++)?://(?:[^\s?#:]|:(?!//))*+\?[^\s#\"<>]+",
    re.IGNORECASE)  # the authority/path run stops at the next ``://``: linear on repeated schemes
_QUERY_PW_PARAM_RE = re.compile(r"([?&;](?:(?:ssl)?password|passwd|pwd)=)([^&;#\s\"'<>]+)", re.IGNORECASE)


def redact_db_url_query_passwords(text: str) -> str:
    """Mask ``password=``/``pwd=``/``passwd=``/``sslpassword=`` values in DB URL query strings."""
    # A raw ``{`` is not legal in a URL query, so a value opening one is a template cut at its first
    # space or quote (``{{ .Values.pw }}``, ``${{ secrets.PG }}``, ``{cfg['pw']}``), not a password.
    return _DB_URL_QUERY_RE.sub(lambda u: _QUERY_PW_PARAM_RE.sub(
        lambda m: m.group(0) if m.group(2).startswith(("{", "${")) or _is_userinfo_placeholder(m.group(2))
        else f"{m.group(1)}***", u.group(0)), text)
