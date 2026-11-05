"""``display.submit_key`` reaches native clients unmasked (#42648).

Ink reads the composer submit key from ``config.get full``. The projection masks every ``*_key``
leaf as a credential, which would hand the client ``"***"`` and silently pin the default.
"""
import asyncio
from types import SimpleNamespace

from gateway.session_config import _client_config, config_get
from hermes_constants import get_hermes_home


def _sessionless_reader(home):
    authority = SimpleNamespace(profile_id=str(home))
    actor = SimpleNamespace(profile_id=str(home), capabilities={'session:read'})
    return SimpleNamespace(authority=authority, actor=actor), SimpleNamespace(session_id=None)


def test_config_get_full_serves_the_configured_submit_key():
    home = get_hermes_home()
    (home / 'config.yaml').write_text('display:\n  submit_key: cmd_enter\n')
    connection, ref = _sessionless_reader(home)

    full = asyncio.run(config_get(connection, ref, {'key': 'full'}))

    assert full['config']['display']['submit_key'] == 'cmd_enter'


def test_credential_shaped_display_keys_stay_masked():
    projected = _client_config({'display': {'submit_key': 'enter', 'some_api_key': 'PRIVATE_DISPLAY'}})

    assert projected['display']['submit_key'] == 'enter'
    assert projected['display']['some_api_key'] != 'PRIVATE_DISPLAY'
