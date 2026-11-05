"""Keyword/value connection-string password masking for agent.redact.

ADO.NET / ODBC / JDBC ``Host=db;Password=…`` and libpq ``host=db password=…`` conninfo; the
URL form (``scheme://user:pass@host``) stays in agent/redact.py.
"""

import re


# Keyword/value connection strings (ADO.NET, Npgsql, ODBC, JDBC SQL Server):
# ``Host=db;Username=app;Password=s3cret;Database=x``, and libpq's space-delimited conninfo
# (``psql "host=db user=app password=s3cret dbname=x"``). _DB_CONNSTR_RE only covers the
# ``scheme://user:pass@`` URL form, so a .NET connection string printed by ``kubectl get
# configmap``, ``cat appsettings.json`` or ``env`` leaked its password verbatim.
#
# Every key pattern starts AT the key name behind a fixed-width lookbehind. A leading
# ``(?:^|[;\s])\s*`` rescans runs of whitespace from every start position: quadratic, and
# a 16K-space line took ~30s. Value extents are found by _kv_value_end, not by regex, so
# each grammar (ODBC braces, ADO.NET doubled quotes, libpq backslash escapes) is exact.
# ``=`` is a companion-key boundary because an assigned connection string
# (``Environment=CONN=Server=db;…``, ``printenv`` output, a ``docker run -e CONN=…`` argv dump)
# puts the first key right after it. It is deliberately NOT added to _KV_KEY_BOUNDARY: a
# ``password=``/``pwd=`` key behind ``=`` is ``note=password=…`` prose far more often than a
# field. For assigned forms, only a password key immediately after ``;`` is eligible,
# so ``env X=host=db PWD=/home/app/project`` stays verbatim.
_KV_KEY_BOUNDARY = r"(?<![^\s;\"'])"
_KV_ASSIGNED_KEY_BOUNDARY = r"(?<![^\s;\"'=])"
_KV_CONNSTR_PW_RE = re.compile(
    _KV_KEY_BOUNDARY + r"(?:password|pwd)[ \t]*=[ \t]*", re.IGNORECASE)
_KV_CONNSTR_GATE_RE = re.compile(r"(?:password|pwd)\s*=", re.IGNORECASE)
# A ``password=``/``pwd=`` key is only treated as a connection-string field when its line
# also carries a companion connection key (``host=``, ``user=``, ``dbname=``...). Keeps prose
# (``the password= field``), form bodies (handled by _redact_form_body) and lone CLI flags out.
_KV_CONNSTR_KEYS = (
    r"host|server|data\s+source|address|addr|network\s+address|user\s*id|uid|"
    r"user(?:name)?|database|dbname|hostaddr|initial\s+catalog|port")
_KV_CONNSTR_COMPANION_RE = re.compile(
    _KV_KEY_BOUNDARY + r"(?:" + _KV_CONNSTR_KEYS + r")[ \t]*=", re.IGNORECASE)
_KV_CONNSTR_ASSIGNED_COMPANION_RE = re.compile(
    _KV_ASSIGNED_KEY_BOUNDARY + r"(?:" + _KV_CONNSTR_KEYS + r")[ \t]*=", re.IGNORECASE)
# Do not parse preceding fields: arbitrary options and quoted/braced values are legal.
_KV_ASSIGNED_PW_RE = re.compile(
    r"(?<=;)[ \t]*(?:password|pwd)[ \t]*=[ \t]*", re.IGNORECASE)
# ``;``-delimited grammar (ADO.NET/ODBC/JDBC) when a ``;`` is followed by a known field.
# libpq values may contain a bare ``;`` (``password=a;b dbname=x``), so a ``;`` alone
# does not decide the grammar.
_KV_SEMICOLON_FIELD_RE = re.compile(
    r";[ \t]*(?:" + _KV_CONNSTR_KEYS + r"|password|pwd|driver|encrypt|pooling|"
    r"integrated\s+security|trusted_connection|trustservercertificate|sslmode|ssl\s+mode|"
    r"timeout|connect(?:ion)?\s+timeout|application\s+name|search\s+path)[ \t]*=",
    re.IGNORECASE)


# ``{0}`` / ``{pw}`` / ``{cfg['pw']}`` / ``{pw!r}`` / ``{ get_pw() }``: a str.format, f-string or C#
# interpolation slot. An ODBC-braced literal is the rest: ``;`` inside, a ``}}`` escape, or content
# that does not open like an expression (``{Zq7x;LmP2}``).
# ``@ $ & #`` and a ``!`` that is not a ``!r``/``!s``/``!a`` conversion never appear in such a slot but
# do in passwords (``{P@ssw0rd!}``, ``{Tr0ub4dor&3}``).
_FORMAT_SLOT_RE = re.compile(r"\{\s*+(?:[A-Za-z0-9_](?:[^;{}@$&#!]|![rsa](?=\s*+[:}]))*+)?\}")


def _kv_value_end(line: str, i: int, semicolon: bool) -> int:
    """End index of the value starting at ``line[i]``; the scan never backtracks."""
    n = len(line)
    if i >= n:
        return i
    c = line[i]
    if semicolon:
        if c == "{":  # ODBC braced value, ``}}`` escapes a brace
            j = i + 1
            while j < n:
                if line[j] == "}":
                    if j + 1 < n and line[j + 1] == "}":
                        j += 2
                        continue
                    return j + 1
                j += 1
            return n
        if c in "\"'":  # ADO.NET quoting, doubled quote escapes
            j = i + 1
            while j < n:
                if line[j] == c:
                    if j + 1 < n and line[j + 1] == c:
                        j += 2
                        continue
                    return j + 1
                j += 1
            return n
        j = i
        while j < n and line[j] not in ";\"'\r\n":
            j += 1
        while j > i and line[j - 1] in " \t":  # trailing padding stays visible
            j -= 1
        return j
    # libpq conninfo: quoted or unquoted, backslash escapes any character.
    if c in "\"'":
        j = i + 1
        while j < n:
            if line[j] == "\\":
                j += 2
                continue
            if line[j] == c:
                return j + 1
            j += 1
        return n
    j = i
    while j < n:
        if line[j] == "\\":
            j += 2
            continue
        if line[j] in " \t\r\n\"'":  # quotes: the shell quoting that wraps the conninfo
            break
        j += 1
    return min(j, n)


def _quote_states(line: str, indexes: list[int]) -> list[tuple[bool, bool, bool]]:
    """Quoting context at each ascending ``index``: (in ``"..."``, in ``'...'``, in a ``{...}``
    ODBC value). One left-to-right walk over the whole list, so many candidates stay linear.

    The quote kind that is not active is literal inside the active one, and ODBC braces a literal
    value, so ``host="don't"`` and ``host={don't}`` do not open a string.
    """
    states, pos = [], 0
    in_dq = in_sq = in_brace = False
    for index in indexes:
        while pos < index:
            ch = line[pos]
            if in_dq:
                if ch == '"':
                    in_dq = False
            elif in_sq:
                if ch == "'":
                    in_sq = False
            elif in_brace:
                if ch == "}":
                    in_brace = False
            elif ch == '"':
                in_dq = True
            elif ch == "'":
                in_sq = True
            elif ch == "{" and pos and line[pos - 1] == "=":
                in_brace = True
            pos += 1
        states.append((in_dq, in_sq, in_brace))
    return states


def _redact_kv_connection_line(line: str, code_file: bool) -> str:
    if not _KV_CONNSTR_GATE_RE.search(line):
        return line
    semicolon = _KV_SEMICOLON_FIELD_RE.search(line) is not None
    companion = _KV_CONNSTR_COMPANION_RE.search(line)
    assigned = companion is None
    if assigned:
        companion = _KV_CONNSTR_ASSIGNED_COMPANION_RE.search(line)
        if companion is None:
            return line
    # Assigned companions only license semicolon-prefixed passwords, not a neighbouring PWD
    # environment variable.
    pw_re = _KV_ASSIGNED_PW_RE if assigned else _KV_CONNSTR_PW_RE
    matches = list(pw_re.finditer(line))
    if not matches:
        return line
    # An assigned companion and a ``;``-prefixed password can still sit in different shell
    # assignments (``env X=host=db NOTE=";password=off"``), so the password must share the
    # companion's quoting context.
    if assigned:
        companion_state = _quote_states(line, [companion.start()])[0]
        states = _quote_states(line, [m.start() for m in matches])
    else:
        companion_state, states = None, []
    out, pos = [], 0
    for i, m in enumerate(matches):
        if m.start() < pos:  # inside a value already masked
            continue
        if assigned and states[i] != companion_state:
            continue
        # ``the pwd= flag``: space after ``=`` but not before it is prose, not an empty value
        # followed by the next word. Padding on both sides (``password = x``) is still a conninfo
        # field. The guard is skipped only where a ``;`` already establishes the grammar on a
        # plain line; ``X=host=db; password= field`` is prose, not an assigned connection string.
        # It runs before the value scan so a rejected candidate cannot rescan the rest of the line.
        eq = line.rindex("=", m.start(), m.end())
        if (assigned or not semicolon) and line[m.end() - 1] in " \t" and line[eq - 1] not in " \t":
            continue
        # A semicolon-prefixed assigned field uses the ADO.NET/ODBC grammar, not libpq.
        field_semicolon = semicolon or assigned
        vstart = m.end()
        vend = _kv_value_end(line, vstart, field_semicolon)
        value = line[vstart:vend]
        if not value:
            continue
        bare = value.strip("\"'")
        # Templates, not credentials: ``Password={0}``, ``Password=${DB_PASS}``, ``{pw}``. ODBC braces a
        # literal value (``PWD={Zq7x;LmP2}``), so on a ``;`` line only a format slot or an identifier
        # counts as a template; ``cat appsettings.json`` output is read as code_file.
        if code_file and bare.startswith(("{", "${")) and bare.endswith("}") and (
                not field_semicolon or bare.startswith("${") or _FORMAT_SLOT_RE.fullmatch(bare)):
            continue
        if value[0] == "{" and field_semicolon:
            masked = "{***}"
        elif value[0] in "\"'" and len(value) > 1 and value[-1] == value[0]:
            masked = f"{value[0]}***{value[0]}"
        else:
            masked = "***"
        out.append(line[pos:vstart])
        out.append(masked)
        pos = vend
    if not out:
        return line
    out.append(line[pos:])
    return "".join(out)


def _redact_kv_connection_strings(text: str, code_file: bool) -> str:
    """Mask ``Password=``/``Pwd=`` values inside keyword/value connection strings, one
    companion-key scan per line."""
    return "".join(_redact_kv_connection_line(line, code_file)
                   for line in text.splitlines(keepends=True))
