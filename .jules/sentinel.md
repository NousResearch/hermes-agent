## 2024-05-24 - [Sanitize Subprocess Environments for `quick_commands` and `shell.exec`]
**Vulnerability:** The CLI and TUI Gateway executed user-defined `quick_commands` and arbitrary shell commands (`shell.exec`) using `subprocess.run(..., shell=True)` without sanitizing the environment variables passed to the child process.
**Learning:** This exposed sensitive API keys and credentials contained in the main Hermes process environment to these child processes, allowing for easy credential exfiltration by a malicious config or user interaction.
**Prevention:** Always use `tools.environments.local._sanitize_subprocess_env` to filter the environment before passing it to `subprocess` execution mechanisms when executing untrusted or user-supplied shell commands.

## 2025-02-26 - [Replace xml.etree with defusedxml to prevent XXE]
**Vulnerability:** The codebase was using `xml.etree.ElementTree.fromstring` and `xml.etree.ElementTree.parse` to parse untrusted XML data. This module is natively vulnerable to XML External Entity (XXE) injections, allowing malicious XML documents to access local files, conduct Server-Side Request Forgery (SSRF), or cause Denial of Service (Billion Laughs attack).
**Learning:** Python's built-in `xml.etree` is unsafe for parsing untrusted or external XML payloads.
**Prevention:** Always use `defusedxml.ElementTree` or `defusedxml.minidom` for parsing XML that could contain untrusted data.
