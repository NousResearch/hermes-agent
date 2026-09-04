"""Rollback-only profile deletion RPC contract."""

from tui_gateway import server


def test_profiles_delete_requires_explicit_confirmation(monkeypatch):
    import hermes_cli.profiles as profiles

    called = []
    monkeypatch.setattr(
        profiles,
        "delete_profile",
        lambda name, yes=False: called.append((name, yes)),
    )
    response = server.handle_request(
        {
            "id": "delete-no-confirm",
            "method": "profiles.delete",
            "params": {"name": "new-bot"},
        }
    )
    assert isinstance(response, dict)
    assert "error" in response
    assert not called


def test_profiles_delete_calls_canonical_delete_with_yes(monkeypatch):
    import hermes_cli.profiles as profiles

    called = []
    monkeypatch.setattr(
        profiles,
        "delete_profile",
        lambda name, yes=False: called.append((name, yes)) or "/tmp/removed",
    )
    response = server.handle_request(
        {
            "id": "delete-confirmed",
            "method": "profiles.delete",
            "params": {"name": "new-bot", "confirm": True},
        }
    )
    assert isinstance(response, dict)
    assert response.get("result", {}).get("ok") is True
    assert called == [("new-bot", True)]


def test_profiles_delete_sanitizes_rollback_errors(monkeypatch):
    import hermes_cli.profiles as profiles

    def fail(name, yes=False):
        raise OSError("cannot remove /home/private/profile-dir")

    monkeypatch.setattr(profiles, "delete_profile", fail)
    response = server.handle_request(
        {
            "id": "delete-error",
            "method": "profiles.delete",
            "params": {"name": "new-bot", "confirm": True},
        }
    )
    assert isinstance(response, dict)
    assert "/home/private/profile-dir" not in str(response)
    assert response["error"]["message"] == "profile deletion failed"
