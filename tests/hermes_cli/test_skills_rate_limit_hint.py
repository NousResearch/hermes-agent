"""A quota-exhausted hub install says so, and when the quota returns (#140556).

Batch installs against the anonymous GitHub API (60 req/hr) exhaust the quota
mid-run; the remaining short-name installs then resolve to nothing and printed
"No skill named ... found in any source." — the refusal read as missing skills.
The terminal 403/429 carries ``X-RateLimit-Reset``; the adapter now keeps it so
every failure path can say "The limit resets around HH:MM." instead.
"""

import io
import time
from contextlib import nullcontext

import httpx
import pytest
from rich.console import Console

import tools.skills_hub as hub_mod
import tools.skills_hub_github as gh_mod

_RESET_EPOCH = 1893456000.0  # 2030-01-01T00:00:00Z; the local HH:MM is derived, not assumed


class _RateLimitedSource:
    """A source whose GitHub quota is exhausted, with a known reset time."""
    source_id = "github"
    is_rate_limited = True
    rate_limit_reset_at = _RESET_EPOCH
    auth = None


def _limit_response(status: int = 403, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, request=httpx.Request("GET", "https://api.github.com/test"))


def _string_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, no_color=True, width=200), buf


def test_terminal_rate_limit_records_reset_epoch(monkeypatch):
    """The terminal quota 403 flags the adapter AND keeps X-RateLimit-Reset as an epoch."""
    monkeypatch.setattr(gh_mod.GitHubAuth, "_resolve_token", lambda self: None)
    monkeypatch.setattr(hub_mod, "_skills_hub_http_get", lambda url, **kw: _limit_response(
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(_RESET_EPOCH))}))
    src = gh_mod.GitHubSource(gh_mod.GitHubAuth())
    resp = src._github_get("https://api.github.com/test", max_retries=1)
    assert resp is not None and resp.status_code == 403
    assert src.is_rate_limited
    assert src.rate_limit_reset_at == _RESET_EPOCH


def test_retry_after_header_sets_relative_reset(monkeypatch):
    """Retry-After (no X-RateLimit-Reset) also yields a reset epoch: now + the header."""
    monkeypatch.setattr(gh_mod.GitHubAuth, "_resolve_token", lambda self: None)
    monkeypatch.setattr(hub_mod, "_skills_hub_http_get", lambda url, **kw: _limit_response(
        status=429, headers={"Retry-After": "120"}))
    src = gh_mod.GitHubSource(gh_mod.GitHubAuth())
    src._github_get("https://api.github.com/test", max_retries=1)
    assert src.is_rate_limited
    assert src.rate_limit_reset_at == pytest.approx(time.time() + 120, abs=5)


def test_unresolvable_name_under_rate_limit_names_the_refusal(monkeypatch):
    """A short-name install that resolved to nothing while rate limited prints the
    rate-limit hint (with the reset time), not just 'No skill named ... found'."""
    import hermes_cli.skills_hub as cli_hub

    monkeypatch.setattr(cli_hub, "_sources", lambda: [_RateLimitedSource()])
    monkeypatch.setattr(cli_hub, "_full_identifier", lambda identifier, sources, c, attempt=None: "")
    monkeypatch.setattr(hub_mod, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub_mod, "skills_hub_http_session", nullcontext)
    c, buf = _string_console()
    attempt: dict = {}
    installed = cli_hub._install_skill("rate-limit-probe-skill", "", False, c, True, True, "", None, attempt=attempt)
    assert installed == (None, "failed", False)
    assert attempt["failure_class"] == "rate_limited"
    out = buf.getvalue()
    assert "rate limit exhausted" in out
    assert f"The limit resets around {time.strftime('%H:%M', time.localtime(_RESET_EPOCH))}" in out
    assert "GITHUB_TOKEN" in out


def test_fetch_failure_hint_carries_reset_time():
    """_print_fetch_failure's rate-limit branch includes 'The limit resets around HH:MM.'."""
    from hermes_cli.skills_hub import _print_fetch_failure

    c, buf = _string_console()
    outcome = _print_fetch_failure(c, [_RateLimitedSource()], "org/repo/some-skill")
    assert outcome == "rate_limited"
    out = buf.getvalue()
    assert "rate limit exhausted" in out
    assert f"The limit resets around {time.strftime('%H:%M', time.localtime(_RESET_EPOCH))}" in out
    # A stale-index misread must not fire while the quota is the real cause.
    assert "no longer exist upstream" not in out
