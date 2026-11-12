"""Skill-only credential rules of the skills scanner (``hermes_agent.tools.skills_guard``).

A skill's scripts run in the execute_code or terminal child, which never receives a Hermes model-provider key
(the child-env scrub; GHSA-rhgp-j443-p4rf keeps a skill from declaring one back). Loading the Hermes .env defeats
that and hands the script every key of the profile, and reading a provider key means the skill depends on getting
it some other way. Both are high on a skill. Plugins run in-process and read their own keys, so ``plugin_guard``
scans without these rules. Leaf module: nothing here imports ``hermes_agent.tools.skills_guard``."""

import io
import re
import tokenize

# The Hermes name must sit inside the load_dotenv / dotenv_values call's own arguments, not in a trailing
# comment or a later statement. A regex cannot balance parens, so the span is spelled out for the call plus 3
# nested levels (pitfall 12 is ``load_dotenv(join(environ.get("HERMES_HOME", expanduser("~/.hermes")), ...))``).
def _closed(depth: int) -> str:  # a balanced (...) group at most *depth* levels deep
    return r'\([^()\n]*\)' if depth == 1 else rf'\((?:[^()\n]|{_closed(depth - 1)})*\)'


def _open_prefix(depth: int) -> str:  # text inside an open group, possibly entering one more nested group
    return r'[^()\n]*' if depth == 1 else rf'(?:[^()\n]|{_closed(depth - 1)})*(?:\({_open_prefix(depth - 1)})?'


# Case-sensitive, and ``.hermes`` must be a whole path segment, so ``.hermes-project.env``, ``myapp.hermes.env``
# or a ``hermes_home_env_path`` variable do not match. Prose that quotes the call ("Never call
# ``load_dotenv(HERMES_HOME/.env)``") matches too: that is a caution the user confirms, not a hard block.
SKILL_ONLY_PATTERNS = [(re.compile(
    rf'\b(?:load_dotenv|dotenv_values)\s*\({_open_prefix(4)}'
    r'(?:HERMES_HOME|get_hermes_home|[~/\\"\']\.hermes(?:[/\\"\']|$))'),
    "hermes_env_dotenv", "high", "exfiltration", "loads the Hermes .env (every key of the profile) into the script")]
_ENV_READ_IDS = frozenset({"python_getenv_secret", "python_environ_get_secret"})
# The name a getenv / environ.get / environ[...] call reads: its first argument only, so a provider name in a
# default argument or another literal on the line does not raise the read of the skill's own key.
_ENV_READ_NAME_RE = re.compile(r'(?:getenv|environ\s*\.\s*get|environ\s*\[)\s*\(?\s*["\']([A-Za-z_]\w*)["\']')
# Token types whose text is not code (see code_views). TSTRING_MIDDLE is the literal part of a 3.14 t-string.
_TEXT_TOKENS = {tokenize.COMMENT, tokenize.STRING} | {
    getattr(tokenize, name) for name in ("FSTRING_MIDDLE", "TSTRING_MIDDLE") if hasattr(tokenize, name)}
_FSTRING_PREFIX = re.compile(r"(?i)[rbu]*f")


def code_views(lines: list) -> tuple | None:
    """One tokenize pass over a .py file's *lines*: ``(view, in_text)`` per line. In *view* comments are
    blanked and parens inside string literals are spaces, so a call's argument span ends at its real closing
    paren; ``in_text[i][col]`` is set on string and comment columns. An f-string's replacement fields stay
    code: from 3.12 its literal parts are FSTRING_MIDDLE tokens, before that the whole f-string is one STRING
    token and is left as code. None when the file does not tokenize; callers then use the whole line."""
    view, in_text = [list(ln) for ln in lines], [bytearray(len(ln)) for ln in lines]
    try:
        for tok in tokenize.generate_tokens(io.StringIO("\n".join(lines)).readline):
            if tok.type not in _TEXT_TOKENS or tok.type == tokenize.STRING and _FSTRING_PREFIX.match(tok.string):
                continue
            (r0, c0), (r1, c1) = tok.start, tok.end
            for r in range(r0 - 1, min(r1, len(lines))):
                row = view[r]
                b = min(c1 if r == r1 - 1 else len(row), len(row))
                a = min(c0 if r == r0 - 1 else 0, b)
                in_text[r][a:b] = b"\x01" * (b - a)
                row[a:b] = [" " if tok.type == tokenize.COMMENT or ch in "()" else ch for ch in row[a:b]]
    except (tokenize.TokenError, SyntaxError):
        return None
    return ["".join(v) for v in view], in_text


def model_provider_key_names() -> frozenset:
    """Env names that hold a Hermes model-provider key: every provider's ``api_key_env_vars`` plus the
    secret provider entries of ``OPTIONAL_ENV_VARS`` (``OPENROUTER_API_KEY`` is declared only there), kept
    only where the child-env scrub removes them. A name the scrub passes on purpose
    (``CLAUDE_CODE_OAUTH_TOKEN``, #55878) does reach the skill, so reading it stays medium."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_agent.tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST
    names = {var.upper() for pconfig in PROVIDER_REGISTRY.values() for var in pconfig.api_key_env_vars}
    names.update(name.upper() for name, meta in OPTIONAL_ENV_VARS.items()
                 if meta.get("category") == "provider" and meta.get("password"))
    return frozenset(names & {n.upper() for n in _HERMES_PROVIDER_ENV_BLOCKLIST})


def raise_provider_key_reads(findings: list, lines: list, in_text: list | None = None) -> None:
    """A secret read that names a model-provider key becomes ``provider_key_env_read`` (high); a read of
    any other secret-shaped name stays medium (reading your own API key is normal, #60709). *in_text* is
    ``code_views``' mask for a tokenized .py file: there only a read that sits in code counts, not one in a
    comment or string text. Other files keep the whole line (a shell ``python3 -c "...getenv(...)..."``
    counts)."""
    reads = [f for f in findings if f.pattern_id in _ENV_READ_IDS]
    if not reads:
        return
    provider_keys = model_provider_key_names()
    for f in reads:
        line = lines[f.line - 1]
        named = [m[1] for m in _ENV_READ_NAME_RE.finditer(line) if m[1].upper() in provider_keys
                 and (in_text is None or not in_text[f.line - 1][m.start()])]
        if named:
            f.pattern_id, f.severity = "provider_key_env_read", "high"
            f.description = (f"reads Hermes model-provider key {named[0]}; skill scripts never receive it, "
                             "and the skill must not fetch it another way")
