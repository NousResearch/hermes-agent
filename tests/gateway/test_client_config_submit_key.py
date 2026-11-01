"""``display.submit_key`` reaches native clients unmasked (#42648).

Ink reads the composer submit key from ``config.get full``. The projection masks every ``*_key``
leaf as a credential, which would hand the client ``"***"`` and silently pin the default.
"""
from gateway.session_config import _client_config


def test_submit_key_survives_the_client_projection():
    projected = _client_config({'display': {'submit_key': 'cmd_enter'}})

    assert projected['display']['submit_key'] == 'cmd_enter'


def test_credential_shaped_display_keys_stay_masked():
    projected = _client_config({'display': {'submit_key': 'enter', 'some_api_key': 'PRIVATE_DISPLAY'}})

    assert projected['display']['submit_key'] == 'enter'
    assert projected['display']['some_api_key'] != 'PRIVATE_DISPLAY'
