"""The root ``default_model`` form must produce restorable local sessions.

A Desktop create following its profile sends no model of its own. The gateway
therefore has to resolve the profile's default before committing the receipt.
"""

import pytest

from tests.gateway.test_local_create_env_refs import _owner


@pytest.mark.asyncio
async def test_desktop_follow_profile_resolves_root_default_model(tmp_path, monkeypatch):
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "default_model: new-default\n"
        "platform_toolsets: {gui: []}\n"
        "providers:\n"
        "  named:\n"
        "    base_url: http://127.0.0.1:9/v1\n"
        "    models: {new-default: {}}\n"
    )
    owner, connection = await _owner(home, monkeypatch)
    try:
        ref = create_local_session(owner, connection.actor, {
            "request_id": "desktop-follow",
            "source": "gui",
            "cwd": str(home),
            "follow_profile_config": True,
            "toolsets": [],
        })
        policy = restore_policy(local_receipt(owner.db, ref.session_id)["policy"])
        assert policy.model == "new-default"
    finally:
        await connection.close()
        owner.db.close()
