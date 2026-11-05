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
# field. The assigned form also has to prove it is one run — a ``;`` between the companion and
# the password key — so ``env X=host=db PWD=/home/app/project`` stays verbatim.
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
# One ``;``-delimited field of an assigned connection string: ``Server=db;``, ``User Id=app;``.
# The value may not contain ``=``: in the ``;`` grammar an unescaped ``=`` inside a value is what
# a neighbouring assignment looks like (``host=db OPTIONS=';sslmode=require'``), and treating it
# as part of the run is how the ``PWD=`` next door got masked.
_KV_RUN_FIELD_RE = re.compile(
    r"[ \t]*(?:" + _KV_CONNSTR_KEYS + r"|password|pwd)[ \t]*=[^;=\r\n]*;", re.IGNORECASE)
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


def _assigned_run_end(line: str, start: int) -> int:
    """End of the ``;``-delimited run that starts at ``start``; ``start`` when it is not one."""
    pos = start
    while True:
        m = _KV_RUN_FIELD_RE.match(line, pos)
        if m is None or m.end() == pos:
            return pos
        pos = m.end()


def _redact_kv_connection_line(line: str, code_file: bool) -> str:
    if not _KV_CONNSTR_GATE_RE.search(line):
        return line
    semicolon = _KV_SEMICOLON_FIELD_RE.search(line) is not None
    plain_companion = _KV_CONNSTR_COMPANION_RE.search(line) is not None
    # A ``=``-prefixed companion key is evidence only for the run it starts: the password key has
    # to open a field of that same ``;``-delimited run. Another variable's ``OPTIONS=';sslmode=…'``
    # or a bare ``env X=host=db PWD=…`` therefore never activates the path.
    assigned_runs = ([(m.start(), _assigned_run_end(line, m.start()))
                      for m in _KV_CONNSTR_ASSIGNED_COMPANION_RE.finditer(line)]
                     if not plain_companion else [])
    if not plain_companion and not assigned_runs:
        return line
    out, pos = [], 0
    for m in _KV_CONNSTR_PW_RE.finditer(line):
        if m.start() < pos:  # inside a value already masked
            continue
        in_run = any(start <= m.start() <= end for start, end in assigned_runs)
        if not plain_companion and not in_run:
            continue
        # A field of an assigned ``;`` run is read with the ``;`` grammar, not the libpq one.
        field_semicolon = semicolon or in_run
        vstart = m.end()
        vend = _kv_value_end(line, vstart, field_semicolon)
        value = line[vstart:vend]
        # ``the pwd= flag``: space after ``=`` but not before it is prose, not an empty value followed
        # by the next word. Padding on both sides (``password = x``) is still a conninfo field.
        eq = line.rindex("=", m.start(), m.end())
        if not value or (not field_semicolon and line[m.end() - 1] in " \t"
                         and line[eq - 1] not in " \t"):
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
