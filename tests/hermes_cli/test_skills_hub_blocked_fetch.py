"""Regression coverage for URL-safety-blocked Skills Hub downloads (#149201)."""

import socket
from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace

from rich.console import Console


def _console():
    sink = StringIO()
    return sink, Console(file=sink, force_terminal=False, color_system=None, width=200)


def _meta(source, identifier):
    from tools.skills_hub_models import SkillMeta
    return SkillMeta("example", "", source, identifier, "community")


def _clear_proxies(monkeypatch):
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_blocked_bundle_is_not_reported_as_stale_and_reason_does_not_leak(monkeypatch):
    import hermes_cli.skills_hub as cli_hub
    import tools.skills_hub as hub
    import tools.url_safety as url_safety

    class Source:
        blocked = True

        def source_id(self):
            return "clawhub"

        def inspect(self, _identifier):
            return _meta("clawhub", "clawhub/example")

        def fetch(self, _identifier):
            hub._guarded_http_get("https://clawhub.ai/api/v1/download")
            return None

    source = Source()
    not_found = SimpleNamespace(status_code=404, headers={})
    client = SimpleNamespace(get=lambda *_args, **_kwargs: not_found)

    def resolve(_host, port=None):
        address = "198.18.0.23" if source.blocked else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port or 443))]

    _clear_proxies(monkeypatch)
    monkeypatch.setattr(url_safety, "_getaddrinfo", resolve)
    monkeypatch.setattr(url_safety, "_global_allow_private_urls", lambda: False)
    monkeypatch.setattr(url_safety, "_is_declared_fake_ip", lambda _ip: False)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [source])
    monkeypatch.setattr(hub, "create_ssrf_safe_client", lambda **_kwargs: nullcontext(client))
    monkeypatch.setattr(hub, "check_website_access", lambda _url: None)

    sink, console = _console()
    assert cli_hub.do_install("clawhub/example", console=console, skip_confirm=True) is False
    blocked_output = sink.getvalue()
    assert "blocked by the URL-safety guard" in blocked_output
    assert "security.fake_ip_ranges" in blocked_output
    assert "Stale index entry" not in blocked_output

    sink.seek(0)
    sink.truncate()
    source.blocked = False

    assert cli_hub.do_install("clawhub/example", console=console, skip_confirm=True) is False
    stale_output = sink.getvalue()
    assert "Stale index entry" in stale_output
    assert "URL-safety guard" not in stale_output


def test_pooled_client_block_is_not_reported_as_stale(monkeypatch):
    import hermes_cli.skills_hub as cli_hub
    import tools.skills_hub as hub
    from tools.url_safety import SSRFConnectionBlocked

    class Source:
        def source_id(self):
            return "github"

        def inspect(self, _identifier):
            return _meta("github", "owner/repo/skills/example")

        def fetch(self, _identifier):
            try:
                hub._skills_hub_http_get("https://api.github.com/repos/owner/repo")
            except SSRFConnectionBlocked:
                pass
            return None

    def blocked(*_args, **_kwargs):
        raise SSRFConnectionBlocked(
            "Blocked request to private/internal address during connect: "
            "api.github.com -> 198.18.0.23"
        )

    monkeypatch.setattr(cli_hub, "_sources", lambda: [Source()])
    monkeypatch.setattr(
        hub, "create_ssrf_safe_client",
        lambda **_kwargs: nullcontext(SimpleNamespace(get=blocked)),
    )

    sink, console = _console()
    assert cli_hub.do_install(
        "owner/repo/skills/example", console=console, skip_confirm=True
    ) is False
    output = sink.getvalue()
    assert "blocked by the URL-safety guard" in output
    assert "security.fake_ip_ranges" in output
    assert "Stale index entry" not in output


def test_dns_failure_does_not_suggest_fake_ip_ranges(monkeypatch):
    import hermes_cli.skills_hub as cli_hub
    import tools.skills_hub as hub
    import tools.url_safety as url_safety

    class Source:
        def source_id(self):
            return "clawhub"

        def inspect(self, _identifier):
            return _meta("clawhub", "clawhub/example")

        def fetch(self, _identifier):
            hub._guarded_http_get("https://clawhub.ai/api/v1/download")
            return None

    def offline(*_args, **_kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    _clear_proxies(monkeypatch)
    monkeypatch.setattr(url_safety, "_getaddrinfo", offline)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [Source()])
    monkeypatch.setattr(hub, "check_website_access", lambda _url: None)

    sink, console = _console()
    assert cli_hub.do_install("clawhub/example", console=console, skip_confirm=True) is False
    output = sink.getvalue()
    assert "DNS resolution failed" in output
    assert "security.fake_ip_ranges" not in output
    assert "Stale index entry" not in output
