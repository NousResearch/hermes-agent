"""Regression coverage for URL-safety-blocked Skills Hub downloads (#149201)."""

from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace

from rich.console import Console


def test_blocked_bundle_is_not_reported_as_stale_and_reason_does_not_leak(monkeypatch):
    import hermes_cli.skills_hub as cli_hub
    import tools.skills_hub as hub
    from tools.skills_hub_models import SkillMeta

    class Source:
        blocked = True

        def source_id(self):
            return "clawhub"

        def inspect(self, _identifier):
            return SkillMeta("example", "", "clawhub", "clawhub/example", "community")

        def fetch(self, _identifier):
            hub._guarded_http_get("https://clawhub.ai/api/v1/download")
            return None

    source = Source()
    not_found = SimpleNamespace(status_code=404, headers={})
    client = SimpleNamespace(get=lambda *_args, **_kwargs: not_found)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [source])
    monkeypatch.setattr(hub, "create_ssrf_safe_client", lambda **_kwargs: nullcontext(client))
    monkeypatch.setattr(hub, "is_safe_url", lambda _url: not source.blocked)
    monkeypatch.setattr(hub, "check_website_access", lambda _url: None)

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None, width=200)

    assert cli_hub.do_install(
        "clawhub/example", console=console, skip_confirm=True
    ) is False
    blocked_output = sink.getvalue()
    assert "blocked by the URL-safety guard" in blocked_output
    assert "security.fake_ip_ranges" in blocked_output
    assert "Stale index entry" not in blocked_output

    sink.seek(0)
    sink.truncate()
    source.blocked = False

    assert cli_hub.do_install(
        "clawhub/example", console=console, skip_confirm=True
    ) is False
    stale_output = sink.getvalue()
    assert "Stale index entry" in stale_output
    assert "URL-safety guard" not in stale_output
