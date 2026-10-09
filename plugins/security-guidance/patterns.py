"""Regex-based security pattern definitions for the security-guidance plugin.

Pure data. No env-var reads, no I/O — importable in isolation.

Forked from Anthropic's claude-plugins-official repository
(plugins/security-guidance/hooks/patterns.py) under the Apache License 2.0:

    https://github.com/anthropics/claude-plugins-official

  Copyright (c) Anthropic, PBC. and the security-guidance contributors
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

NousResearch modifications: pattern data unchanged from upstream; the upstream RuleId
telemetry table (Claude Code PostToolUse metrics) is dropped — Hermes has no consumer.
Hermes-side wiring lives in __init__.py.
"""
_JS_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".vue", ".svelte")
_PY_EXTS = (".py", ".pyi", ".ipynb")
_DOC_EXTS = (".md", ".mdx", ".txt", ".rst", ".json", ".yaml", ".yml")

# Shared path_filter predicates. JS-only gating keeps bare `exec(` off Python's exec()
# and prose; Python-only gating keeps pickle/os.system rules off other languages;
# the eval rule skips doc/prose files entirely.
_JS_ONLY = lambda p: p.endswith(_JS_EXTS)  # noqa: E731
_PY_ONLY = lambda p: p.endswith(_PY_EXTS)  # noqa: E731
_NOT_DOCS = lambda p: not p.endswith(_DOC_EXTS)  # noqa: E731

_UNSAFE_DESERIALIZATION_REMINDER = """⚠️ Security Warning: Loading pickle data (or equivalents: cPickle, cloudpickle, dill, marshal, shelve, joblib, pandas.read_pickle, numpy with allow_pickle=True) from untrusted sources allows arbitrary code execution.

For simple data, prefer JSON or msgspec. For typed objects, prefer a schema-validated deserializer (msgspec.Struct, pydantic, marshmallow) that constructs only declared types.

If this is safe or is explicitly needed, briefly document that in a comment before continuing."""

_UNSAFE_YAML_LOAD_REMINDER = """⚠️ Security Warning: yaml.load() / yaml.unsafe_load() execute arbitrary Python via !!python/object tags.

Use yaml.safe_load() if the file only contains simple data structures (dicts, lists, strings, numbers). If you need typed objects, parse with safe_load and validate the result against a schema (pydantic, msgspec, marshmallow) — never use a custom Loader that constructs arbitrary types."""

_UNSAFE_TORCH_LOAD_REMINDER = """⚠️ Security Warning: torch.load() defaults to weights_only=False, which unpickles arbitrary Python objects and allows arbitrary code execution.

If the file only contains tensors and simple data structures, pass weights_only=True (or set TORCH_FORCE_WEIGHTS_ONLY_LOAD=1)."""

_GITHUB_ACTIONS_REMINDER = """⚠️ Security Warning: You are editing a GitHub Actions workflow file. Be aware of these security risks:

1. **Command Injection**: Never use untrusted input (like issue titles, PR descriptions, commit messages) directly in run: commands without proper escaping
2. **Use environment variables**: Instead of ${{ github.event.issue.title }}, use env: with proper quoting
3. **Review the guide**: https://github.blog/security/vulnerability-research/how-to-catch-github-actions-workflow-injections-before-attackers-do/

Example of UNSAFE pattern to avoid:
run: echo "${{ github.event.issue.title }}"

Example of SAFE pattern:
env:
  TITLE: ${{ github.event.issue.title }}
run: echo "$TITLE"

Other risky inputs to be careful with:
- github.event.issue.body
- github.event.pull_request.title
- github.event.pull_request.body
- github.event.comment.body
- github.event.review.body
- github.event.review_comment.body
- github.event.pages.*.page_name
- github.event.commits.*.message
- github.event.head_commit.message
- github.event.head_commit.author.email
- github.event.head_commit.author.name
- github.event.commits.*.author.email
- github.event.commits.*.author.name
- github.event.pull_request.head.ref
- github.event.pull_request.head.label
- github.event.pull_request.head.repo.default_branch
- github.event.client_payload.* (repository_dispatch events — attacker can set any field)

4. **Ref injection**: Never use untrusted input in `ref:` parameters of `actions/checkout`. For `client_payload.pr_number`, validate it matches `^[0-9]+$` before using in `ref: refs/pull/${{ ... }}/head`
- github.head_ref"""

_CHILD_PROCESS_EXEC_REMINDER = """⚠️ Security Warning: Using child_process.exec() can lead to command injection vulnerabilities.

exec() runs the command string through a shell, so any user input interpolated into it can inject arbitrary commands. Prefer child_process.execFile() (or spawn()) with an argument array instead of building a shell string.

Instead of:
  exec(`command ${userInput}`)

Use:
  import { execFile } from 'node:child_process'
  execFile('command', [userInput], callback)

Why execFile/spawn with an argument array is safer:
- No shell is involved, so shell metacharacters in arguments are not interpreted
- Arguments are passed directly to the program rather than interpolated into a command string

Only use exec() if you absolutely need shell features and the input is guaranteed to be safe."""

_SUBPROCESS_SHELL_REMINDER = """⚠️ Security Warning: Using subprocess with shell=True enables command injection.

UNSAFE:
  subprocess.run(f"ls {user_input}", shell=True)
  subprocess.call("grep " + pattern, shell=True)

SAFE - pass arguments as a list without shell:
  subprocess.run(["ls", user_input])
  subprocess.call(["grep", pattern])

When arguments are passed as a list without shell=True, special characters cannot be interpreted as shell metacharacters."""

_GO_EXEC_SHELL_REMINDER = """⚠️ Security Warning: Using exec.Command with a shell interpreter (sh/bash) enables command injection.

UNSAFE:
  exec.Command("sh", "-c", "ping -c 1 " + host)
  exec.Command("bash", "-c", fmt.Sprintf("df -h %s", path))

SAFE - pass arguments directly without a shell:
  exec.Command("ping", "-c", "1", host)
  exec.Command("df", "-h", path)

When arguments are passed directly (not through a shell), special characters in user input cannot be interpreted as shell metacharacters. This prevents command injection entirely.

Additionally, validate user inputs:
- For hostnames/IPs: use net.ParseIP() or a hostname regex
- For file paths: use filepath.Clean() and verify the result is within an allowed directory
- For numeric values: parse to int/float first"""


def _rule(name, reminder, **triggers):
    """Build one rule dict; only the trigger keys given (regex / substrings /
    path_filter / path_check) are present, so consumers can test with ``in``."""
    return {"ruleName": name, "reminder": reminder, **triggers}


# Regex notes: eval/exec lookbehinds exclude `.` so method calls (model.eval()) don't match; pickle
# matches deserialization only (load/loads/Unpickler) and `pkl_load` needs a word boundary;
# script_src_without_sri's negative lookahead after `<script` checks for integrity= anywhere in the
# tag; torch_unsafe_load is suppressed by weights_only=True on the same line (200 chars) — multi-line
# calls false-positive, same known limitation as unsafe_yaml_load; yaml_unsafe_load_variants covers
# yaml.unsafe_load plus wrapper names seen in the wild (bare yaml.load() is unsafe_yaml_load's job);
# pickle_wrapper_load: APIs that unpickle without saying "pickle" — numpy.load only with an explicit
# allow_pickle=True (default False since numpy 1.16.3).
SECURITY_PATTERNS = [
    _rule("github_actions_workflow", _GITHUB_ACTIONS_REMINDER,
          path_check=lambda path: ".github/workflows/" in path and (path.endswith(".yml") or path.endswith(".yaml"))),
    _rule("child_process_exec", _CHILD_PROCESS_EXEC_REMINDER, path_filter=_JS_ONLY, substrings=["child_process.exec", "execSync("], regex=r"(?<![a-zA-Z0-9_\.])exec\("),
    _rule("new_function_injection",
          "\u26a0\ufe0f Security Warning: Using new Function() with string interpolation is a CODE INJECTION vulnerability. If any variable is concatenated or interpolated into the function body string, an attacker controlling that variable can execute arbitrary code. Use safe alternatives: for property access use obj[key] or array.reduce((o, k) => o[k], root); for computation use a safe expression parser. NEVER interpolate untrusted strings into new Function() bodies.",
          substrings=["new Function"]),
    _rule("eval_injection",
          "⚠️ Security Warning: eval() executes arbitrary code and is a major security risk. Use JSON.parse() for data, ast.literal_eval() for Python literals, or a safe expression parser. If this is safe or is explicitly needed, briefly document that in a comment before continuing.",
          path_filter=_NOT_DOCS, regex=r"(?<![a-zA-Z0-9_\.])eval\("),
    _rule("react_dangerously_set_html",
          "⚠️ Security Warning: dangerouslySetInnerHTML can lead to XSS vulnerabilities if used with untrusted content. Ensure all content is properly sanitized using an HTML sanitizer library like DOMPurify, or use safe alternatives.",
          substrings=["dangerouslySetInnerHTML"]),
    _rule("document_write_xss",
          "⚠️ Security Warning: document.write() can be exploited for XSS attacks and has performance issues. Use DOM manipulation methods like createElement() and appendChild() instead.",
          substrings=["document.write"]),
    _rule("innerHTML_xss",
          "⚠️ Security Warning: Setting innerHTML with untrusted content can lead to XSS vulnerabilities. Use textContent for plain text or safe DOM methods for HTML content. If you need HTML support, consider using an HTML sanitizer library such as DOMPurify.",
          substrings=[".innerHTML =", ".innerHTML="]),
    _rule("pickle_deserialization", _UNSAFE_DESERIALIZATION_REMINDER, path_filter=_PY_ONLY, regex=r"(?<![a-zA-Z0-9_])pickle\.(loads?|Unpickler)\b|(?<![a-zA-Z0-9_])pkl_load\("),
    _rule("os_system_injection",
          "⚠️ Security Warning: os.system() runs a shell and is a command-injection sink. Use subprocess.run([...]) with a list of arguments instead. If this is safe or is explicitly needed, briefly document that in a comment before continuing.",
          path_filter=_PY_ONLY, regex=r"\bos\.system\s*\(", substrings=["from os import system"]),
    _rule("python_subprocess_shell", _SUBPROCESS_SHELL_REMINDER, regex=r"subprocess\.(?:run|call|Popen|check_output|check_call)\(.*shell\s*=\s*True"),
    # Go: exec.Command with a shell invocation (sh, bash, /bin/sh, /bin/bash)
    _rule("go_exec_shell_injection", _GO_EXEC_SHELL_REMINDER, regex=r'exec\.Command\(\s*"(?:sh|bash|/bin/sh|/bin/bash)"'),
    _rule("unsafe_yaml_load", _UNSAFE_YAML_LOAD_REMINDER, regex=r"\byaml\.load\s*\((?![^)\n]{0,80}\bSafe)"),
    _rule("node_createcipher_no_iv",
          "⚠️ Security Warning: Use crypto.createCipheriv() / createDecipheriv(). createCipher was removed in Node 22 and derives the key insecurely (no IV, MD5-based KDF).",
          regex=r"\bcrypto\.(createCipher|createDecipher)\b"),
    _rule("aes_ecb_mode",
          "⚠️ Security Warning: Use AES-GCM or AES-CBC with HMAC. ECB mode leaks plaintext structure (identical blocks encrypt to identical ciphertext).",
          regex=r"\bAES\.MODE_ECB\b|\bmodes\.ECB\s*\(|[\x22\x27]aes-\d+-ecb[\x22\x27]"),
    _rule("tls_verification_disabled",
          "⚠️ Security Warning: Don't disable TLS verification. This allows MITM attacks. For self-signed dev certs, add the CA to your trust store or use a properly-issued cert.",
          regex=r"\bverify\s*=\s*False\b|rejectUnauthorized\s*:\s*false|InsecureSkipVerify\s*:\s*true|NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*[\x22\x27]?0|ssl\._create_unverified_context|check_hostname\s*=\s*False"),
    _rule("marshal_loads", _UNSAFE_DESERIALIZATION_REMINDER, regex=r"\bmarshal\.loads?\s*\("),
    _rule("shelve_open", _UNSAFE_DESERIALIZATION_REMINDER, regex=r"\bshelve\.open\s*\("),
    _rule("xml_unsafe_parse",
          "⚠️ Security Warning: Use defusedxml.ElementTree. Python's stdlib XML parsers are vulnerable to XXE (external entity) and billion-laughs attacks by default.",
          regex=r"\b(xml\.etree\.ElementTree|ElementTree|ET)\.(parse|fromstring|XML)\s*\(|\bminidom\.(parse|parseString)\s*\(|\bxml\.sax\.(parse|make_parser)\b"),
    _rule("pickle_variants_load", _UNSAFE_DESERIALIZATION_REMINDER, regex=r"\b(cPickle|cloudpickle|dill)\.(load|loads)\s*\("),
    _rule("outerHTML_xss",
          "⚠️ Security Warning: Use textContent or sanitize with DOMPurify. outerHTML assignment is an XSS sink equivalent to innerHTML.",
          substrings=[".outerHTML =", ".outerHTML="]),
    _rule("insertAdjacentHTML_xss",
          "⚠️ Security Warning: Use insertAdjacentText() or sanitize with DOMPurify. insertAdjacentHTML is an XSS sink.",
          substrings=[".insertAdjacentHTML("]),
    _rule("script_src_without_sri",
          '⚠️ Security Warning: Add integrity="sha384-..." crossorigin="anonymous" to external script tags. Loading scripts without Subresource Integrity exposes you to CDN compromise.',
          regex=r"<script\s+(?![^>]{0,400}integrity\s*=)[^>]{0,200}src\s*=\s*[\x22\x27](?:https?:)?//[^\x22\x27]{1,300}[\x22\x27][^>]{0,100}>"),
    _rule("torch_unsafe_load", _UNSAFE_TORCH_LOAD_REMINDER, regex=r"(?:\btorch\.load|\.torch_load)\s*\((?![^)\n]{0,200}weights_only\s*=\s*True)"),
    _rule("yaml_unsafe_load_variants", _UNSAFE_YAML_LOAD_REMINDER, regex=r"(?:\byaml\.unsafe_load|\.yaml_unsafe_load)\s*\("),
    _rule("pickle_wrapper_load", _UNSAFE_DESERIALIZATION_REMINDER,
          regex=r"\bjoblib\.load\s*\(|\b(?:pd|pandas)\.read_pickle\s*\(|\.cloudpickle_load\s*\(|\b(?:np|numpy)\.load\s*\([^)\n]{0,200}allow_pickle\s*=\s*True"),
]
