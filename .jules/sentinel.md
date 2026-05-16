## 2024-05-24 - Security Enhancement: YAML Parsing & Subprocess Execution
**Vulnerability:** Use of `yaml.load` and `subprocess.run(shell=True)`.
**Learning:** `yaml.load(value, Loader=yaml.CSafeLoader)` is structurally equivalent to `yaml.safe_load(value)` but has significant performance advantages, so it is safe to keep it this way. `subprocess.run(shell=True)` can introduce shell injection vulnerabilities and should be replaced with a list of arguments and `shell=False`. When updating `subprocess.run` to not use `shell=True`, be careful to update the test mocks to correctly handle `shlex.split()` list format.
**Prevention:** Avoid `shell=True` in `subprocess` unless absolutely necessary, and prefer passing commands as argument lists. Update test mocks accordingly.
