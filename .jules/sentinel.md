## 2024-05-18 - Prevent Command Injection with Templated Strings in Subprocess
**Vulnerability:** Command Injection via Subprocess Template Strings (`shell=True`).
**Learning:** `tools/transcription_tools.py` executed local stt commands using `.format()` on user-controlled inputs with `shell=True`. Using `shlex.quote` in `shell=True` isn't foolproof enough when strings contain spaces.
**Prevention:** When mitigating command injection in subprocess calls using templated strings, tokenize the template with `shlex.split()` first, then substitute variables into the resulting list items. This ensures variables containing spaces (like file paths) are not improperly split and avoids argument injection. Finally pass the resulting list into `subprocess.run(args, check=True)` without `shell=True`.

## 2024-05-24 - Security Enhancement: YAML Parsing & Subprocess Execution
**Vulnerability:** Use of `yaml.load` and `subprocess.run(shell=True)`.
**Learning:** `yaml.load(value, Loader=yaml.CSafeLoader)` is structurally equivalent to `yaml.safe_load(value)` but has significant performance advantages, so it is safe to keep it this way. `subprocess.run(shell=True)` can introduce shell injection vulnerabilities and should be replaced with a list of arguments and `shell=False`. When updating `subprocess.run` to not use `shell=True`, be careful to update the test mocks to correctly handle `shlex.split()` list format.
**Prevention:** Avoid `shell=True` in `subprocess` unless absolutely necessary, and prefer passing commands as argument lists. Update test mocks accordingly.

## 2025-05-17 - Security Enhancement: SQL Injection Mitigation via fetchmany
**Vulnerability:** String-interpolating user-provided SQL to append `LIMIT`.
**Learning:** Wrapping a user-provided SQL query in a larger query using string interpolation (e.g., `f"SELECT * FROM ({sql}) LIMIT N"`) introduces an SQL injection vector. Attackers can provide queries like `SELECT 1) AS t; DROP TABLE users; --` to break out of the subquery context.
**Prevention:** Avoid string-interpolating dynamic SQL queries. Since SQLite does not support parameterizing subqueries in the `FROM` clause natively, execute the raw user query directly and enforce output limits in Python using `cursor.fetchmany(limit)`.
