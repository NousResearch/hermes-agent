"""Non-interactive HTTPS credentials for Hermes's internal git clones (private plugin/MCP/profile repos).

:func:`noninteractive_git_env` deliberately disables credential helpers, askpass and global git
config so a hostile repo cannot make our plumbing prompt or hang. The cost is that a *private*
repo the user can already clone from their shell fails inside ``hermes plugins install`` with
"could not read Username" (or hangs on a GUI askpass until the timeout). This module resolves a
credential up front, from sources the user already owns, and passes it to git as a one-shot
``http.<origin>/.extraheader`` in the environment — never in the URL and never in ``.git/config``,
so nothing is persisted into the installed checkout.

Resolution order for an ``https://`` URL:

1. ``GITHUB_TOKEN`` / ``GH_TOKEN`` (profile-scoped, GitHub hosts only).
2. ``gh auth token`` (GitHub hosts only; the gh CLI's own login).
3. ``git credential fill`` against the user's configured credential helpers (any host: GitLab,
   Bitbucket, self-hosted) with prompting disabled, so a stored credential is returned and a
   missing one fails in ~100 ms instead of asking.

The credential is attached only after an anonymous attempt is refused
(:func:`run_git_with_credential_fallback`), and a refused credential yields to the next one in
the list rather than ending the run — a stale ``GITHUB_TOKEN`` in ``.env`` otherwise shadows a
``gh auth login`` that still works. Most catalog repos are public, and a stale or
revoked stored token sent pre-emptively turns a clone that works anonymously into a 401 that git
can only answer with the prompt this module disables — "could not read Username for
'https://github.com': terminal prompts disabled".
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import subprocess
import urllib.parse
from typing import Iterator, Mapping, Optional

from hermes_cli._subprocess_compat import noninteractive_git_env, windows_hide_flags

logger = logging.getLogger(__name__)

_GITHUB_HOSTS = {"github.com", "gist.github.com"}

_HTTP_URL_RE = re.compile(r"https?://[^\s'\"<>|]+")


def without_credentials(text: str) -> str:
    """*text* with credentials stripped from every HTTP(S) URL it contains.

    Pass a bare URL and the URL comes back with its ``user:password@`` userinfo
    (and its query/fragment, which can carry a ``?token=``) removed. Pass a git
    error line and each embedded URL is scrubbed in place, so a remote URL git
    echoed into stderr cannot smuggle a PAT into a toast, a log, or a pasted bug
    report. One implementation for both directions — there is deliberately no
    second scrubber for a call site to pick instead.

    Non-HTTP transports (``git@host:owner/repo``) and already-clean URLs are
    returned unchanged; URLs with no hostname are left alone rather than
    mangled.
    """
    def _scrub(match: "re.Match[str]") -> str:
        parsed = urllib.parse.urlsplit(match.group(0))
        if not parsed.hostname:
            return match.group(0)
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))

    return _HTTP_URL_RE.sub(_scrub, text or "")


def _https_origin(url: str) -> Optional[str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return f"https://{host}"


def _env_github_token() -> Optional[str]:
    from agent.secret_scope import get_secret

    return get_secret("GITHUB_TOKEN") or get_secret("GH_TOKEN") or None


def _gh_cli_token() -> Optional[str]:
    gh = shutil.which("gh")
    if not gh:
        return None
    try:
        env = noninteractive_git_env()
        env["GH_PROMPT_DISABLED"] = "1"
        # gh echoes an exported GH_TOKEN/GITHUB_TOKEN back instead of its keyring login; that
        # token is already candidate 1, and a dead one would hide the login that still works.
        for var in ("GH_TOKEN", "GITHUB_TOKEN"):
            env.pop(var, None)
        result = subprocess.run(
            [gh, "auth", "token"], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, stdin=subprocess.DEVNULL, env=env, creationflags=windows_hide_flags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("gh auth token lookup failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _github_token() -> Optional[str]:
    return _env_github_token() or _gh_cli_token()


def _credential_fill(origin: str) -> Optional[tuple[str, str]]:
    """``(username, password)`` from the user's own git credential helpers, never prompting."""
    git = shutil.which("git")
    if not git:
        return None
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    # A GUI askpass (VS Code, ssh-askpass) would block on a dialog nobody sees.
    env.pop("GIT_ASKPASS", None)
    env.pop("SSH_ASKPASS", None)
    parsed = urllib.parse.urlsplit(origin)
    request = f"protocol=https\nhost={parsed.netloc}\n\n"
    try:
        result = subprocess.run(
            [git, "-c", "core.askPass=", "credential", "fill"], input=request, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=15, env=env,
            creationflags=windows_hide_flags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git credential fill failed for %s: %s", origin, exc)
        return None
    if result.returncode != 0:
        return None
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if fields.get("password"):
        return fields.get("username", ""), fields["password"]
    return None


def iter_git_basic_auth(url: str) -> Iterator[tuple[str, tuple[str, str]]]:
    """``(source, (username, password))`` for every credential the user owns for *url*'s host,
    in precedence order and without duplicates. The remote decides which one works: an expired
    ``GITHUB_TOKEN`` left in ``.env`` must not shadow a live ``gh auth login`` (#115257)."""
    origin = _https_origin(url)
    if origin is None:
        return
    seen: set[tuple[str, str]] = set()
    sources = []
    if urllib.parse.urlsplit(origin).hostname in _GITHUB_HOSTS:
        sources += [("GITHUB_TOKEN/GH_TOKEN", lambda: _token_pair(_env_github_token())),
                    ("gh auth token", lambda: _token_pair(_gh_cli_token()))]
    sources.append(("git credential helper", lambda: _credential_fill(origin)))
    for source, lookup in sources:
        auth = lookup()
        if auth is not None and auth not in seen:
            seen.add(auth)
            yield source, auth


def _token_pair(token: Optional[str]) -> Optional[tuple[str, str]]:
    return ("x-access-token", token) if token else None


def resolve_git_basic_auth(url: str) -> Optional[tuple[str, str]]:
    """``(username, password)`` for *url*, or None for non-HTTPS URLs / no stored credential."""
    return next((auth for _source, auth in iter_git_basic_auth(url)), None)


def with_git_auth(env: Mapping[str, str], url: str,
                  auth: Optional[tuple[str, str]] = None) -> dict[str, str]:
    """Copy of *env* (a :func:`noninteractive_git_env` result) that authenticates HTTPS requests to
    *url*'s origin via a ``GIT_CONFIG_*`` ``http.<origin>/.extraheader`` entry when a credential is
    available (*auth*, else the first stored one); unchanged otherwise. The header lives only in
    this process environment."""
    env = dict(env)
    origin = _https_origin(url)
    if origin is None:
        return env
    if auth is None:
        auth = resolve_git_basic_auth(url)
    if auth is None:
        return env
    encoded = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    idx = int(env.get("GIT_CONFIG_COUNT", "0") or 0)
    env[f"GIT_CONFIG_KEY_{idx}"] = f"http.{origin}/.extraheader"
    env[f"GIT_CONFIG_VALUE_{idx}"] = f"Authorization: basic {encoded}"
    env["GIT_CONFIG_COUNT"] = str(idx + 1)
    return env


def _auth_headers(env: Mapping[str, str]) -> set[str]:
    return {v for k, v in env.items() if k.startswith("GIT_CONFIG_VALUE_") and v.startswith("Authorization:")}


# What git prints when the remote wants a credential it could not obtain: the prompt that
# GIT_TERMINAL_PROMPT=0 refused, a rejected credential, or a bare 401/403 from the server.
_CREDENTIAL_REQUIRED_RE = re.compile(
    r"could not read (?:username|password)|terminal prompts disabled|authentication failed"
    r"|(?:error|status|http)[: ]+40[13]\b",
    re.IGNORECASE,
)


def is_credential_required_error(result: subprocess.CompletedProcess) -> bool:
    """True when a failed git run says the remote demands a credential (vs. a typo'd URL,
    a missing commit, a network error, ...)."""
    parts = []
    for stream in (result.stderr, result.stdout):
        if isinstance(stream, bytes):
            stream = stream.decode("utf-8", errors="replace")
        parts.append(stream or "")
    return _CREDENTIAL_REQUIRED_RE.search("\n".join(parts)) is not None


def run_git_with_credential_fallback(
    argv: list[str], url: str, *, env: Mapping[str, str], **run_kwargs,
) -> subprocess.CompletedProcess:
    """Run the git network verb *argv* against *url* anonymously; when the remote refuses with
    the credential-required class, rerun with each credential the user owns for that host, in
    precedence order, until one is accepted or the failure stops being about credentials. *env*
    is a :func:`noninteractive_git_env`; *run_kwargs* must capture output so the refusal can be
    classified. Empty *url* means no fallback (local verbs)."""
    run_kwargs.setdefault("stdin", subprocess.DEVNULL)
    env = dict(env)
    # An inherited askpass (VS Code terminal, ksshaskpass) would swallow the remote's 401 into a
    # dialog nobody answers: the run hits its timeout instead of failing with "could not read
    # Username", and the refusal below is never classified. Same drop _credential_fill does.
    env.pop("GIT_ASKPASS", None)
    env.pop("SSH_ASKPASS", None)
    result = subprocess.run(argv, env=env, **run_kwargs)
    if result.returncode == 0 or not url or not is_credential_required_error(result):
        return result
    auth_env = with_git_auth(env, url)
    if auth_env == env:
        return result
    result = subprocess.run(argv, env=auth_env, **run_kwargs)
    sent = _auth_headers(auth_env)
    rejected: list[str] = []
    # The first credential (typically GITHUB_TOKEN from .env) was refused too: it is stale or
    # revoked, not missing. Try the remaining ones the user owns instead of failing on it.
    for source, auth in iter_git_basic_auth(url):
        if result.returncode == 0 or not is_credential_required_error(result):
            break
        candidate_env = with_git_auth(env, url, auth)
        headers = _auth_headers(candidate_env)
        if headers <= sent:
            rejected.append(source)
            continue
        sent |= headers
        logger.warning("%s rejected the credential from %s; retrying with %s",
                       _https_origin(url), ", ".join(rejected) or "the stored credential", source)
        result = subprocess.run(argv, env=candidate_env, **run_kwargs)
        if result.returncode != 0 and is_credential_required_error(result):
            rejected.append(source)
    if result.returncode != 0 and "GITHUB_TOKEN/GH_TOKEN" in rejected and isinstance(result.stderr, str):
        result.stderr += ("\nhint: the GITHUB_TOKEN/GH_TOKEN in your .env was rejected by GitHub;"
                          " replace it or remove it (gh auth login is used when it is absent).")
    return result
