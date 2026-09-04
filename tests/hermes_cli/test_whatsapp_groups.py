from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_group_display_name_strips_terminal_controls():
    from hermes_cli.main import _sanitize_group_display_name

    assert _sanitize_group_display_name("Borderô\x1b[31m\nUBBO\x1b[0m") == "BorderôUBBO"


def test_group_display_name_strips_unicode_terminal_controls():
    from hermes_cli.main import _sanitize_group_display_name

    assert _sanitize_group_display_name("UBBO\u202e\u2066\u0085\u2028nome") == "UBBOnome"


def test_list_groups_rejects_invalid_port():
    from hermes_cli.main import _list_whatsapp_groups

    with pytest.raises(SystemExit) as exc_info:
        _list_whatsapp_groups(SimpleNamespace(bridge_port=0))
    assert exc_info.value.code == 1


def test_list_groups_rejects_noncanonical_jid(monkeypatch):
    from hermes_cli.main import _list_whatsapp_groups

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"groups":[{"group_jid":"legacy-123@g.us","name":"Group"}]}'

    with patch("urllib.request.urlopen", return_value=Response()):
        with pytest.raises(SystemExit) as exc_info:
            _list_whatsapp_groups(SimpleNamespace(bridge_port=3000))
    assert exc_info.value.code == 1
