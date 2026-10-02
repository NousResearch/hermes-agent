"""Application inputs remain scoped and lazy at the Gateway execution boundary."""
from pathlib import Path

from gateway.slash_commands import _execute


def test_version_uses_current_application_label(monkeypatch):
    import hermes_cli.banner as banner
    labels = iter(["main label", "canary label"])
    monkeypatch.setattr(banner, "format_banner_version_label", lambda: next(labels))
    assert _execute("version").text == "main label"
    assert _execute("version").text == "canary label"


def test_profile_details_follow_each_source_without_mutating_options(monkeypatch):
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: Path(name))
    monkeypatch.setattr(profiles, "read_profile_meta", lambda path: {"display_name": str(path).upper()})
    for name in ("alpha", "beta", "alpha"):
        options = {"profile_name": name, "home_display": f"/profiles/{name}"}
        reply = _execute("profile", options=options)
        assert reply.data == {"profile": name, "home": f"/profiles/{name}"}
        assert name.upper() in reply.text
        assert options == {"profile_name": name, "home_display": f"/profiles/{name}"}


def test_profile_metadata_failure_keeps_identity(monkeypatch):
    import hermes_cli.profiles as profiles
    def unavailable(_):
        raise OSError("metadata unavailable")
    monkeypatch.setattr(profiles, "read_profile_meta", unavailable)
    reply = _execute("profile", options={"profile_name": "alpha", "home_display": "/profiles/alpha"})
    assert reply.text == "Profile: alpha\nHome: /profiles/alpha"


def test_catalog_filter_is_supplied_unchanged(monkeypatch):
    import gateway.command_presentation as presentation
    import gateway.slash_commands as slash
    received = []
    monkeypatch.setattr(presentation, "gateway_help_lines",
                        lambda allowed: received.append(allowed) or ["allowed help"])
    monkeypatch.setattr(slash, "t", lambda key, **kw: key)
    allowed = {"help", "whoami"}
    reply = _execute("help", options={"allowed_commands": allowed})
    assert reply.text == "gateway.help.header\nallowed help"
    assert received == [allowed]
    assert allowed == {"help", "whoami"}
