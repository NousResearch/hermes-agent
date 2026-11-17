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
# The value runs to the next ``&``, ``#`` or whitespace: these URL schemes separate query params with
# ``&``, so a ``;`` is password data even before ``k=v`` (``password=Zq7x;a=b``). Quotes and ``<>`` still
# end it (JSON / autolink context). Fail closed: a ``;sslmode=…`` after the password is masked with it.
_QUERY_PW_KEY_RE = re.compile(r"[?&;](?:(?:ssl)?password|passwd|pwd)=", re.IGNORECASE)
_QUERY_PW_HEAD_RE = re.compile(r"[^;&#\s\"'<>]*+")  # the value up to its first ``;``
_QUERY_PW_REST_RE = re.compile(r"[^&#\s\"'<>]*+")
# The one exception: a template value followed by a ``;key=`` sibling (``password=${PG};password=…``)
# ends at that ``;``, so the sibling is judged on its own instead of riding the template exemption.
_SIBLING_KEY_RE = re.compile(r";[A-Za-z_][A-Za-z0-9_.-]*=")


def _is_template_value(value: str) -> bool:
    # A raw ``{`` is not legal in a URL query, so a value opening one is a template cut at its first
    # space or quote (``{{ .Values.pw }}``, ``${{ secrets.PG }}``, ``{cfg['pw']}``), not a password.
    return value.startswith(("{", "${")) or _is_userinfo_placeholder(value)


def _mask_query_params(query: str) -> str:
    out, pos = [], 0
    while (key := _QUERY_PW_KEY_RE.search(query, pos)) is not None:
        head_end = _QUERY_PW_HEAD_RE.match(query, key.end()).end()
        head = query[key.end():head_end]
        out.append(query[pos:key.end()])
        if _is_template_value(head) and _SIBLING_KEY_RE.match(query, head_end):
            out.append(head)  # rescan from the sibling's ``;``; nothing past it was read: linear
            pos = head_end
            continue
        pos = _QUERY_PW_REST_RE.match(query, head_end).end()
        # A template is kept only as the whole value: ``PASSWORD;Zq7x`` or ``{pw};Zq7x`` is data.
        out.append(head if pos == head_end and _is_template_value(head) else "***" if pos > key.end() else "")
    out.append(query[pos:])
    return "".join(out)


def redact_db_url_query_passwords(text: str) -> str:
    """Mask ``password=``/``pwd=``/``passwd=``/``sslpassword=`` values in DB URL query strings."""
    return _DB_URL_QUERY_RE.sub(lambda u: _mask_query_params(u.group(0)), text)
