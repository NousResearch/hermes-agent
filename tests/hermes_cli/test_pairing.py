from argparse import Namespace
from unittest.mock import patch

from gateway.pairing import PairingStore
from hermes_cli.pairing import pairing_command


def test_cli_listed_request_id_and_bot_code_can_be_approved(tmp_path, capsys):
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        store = PairingStore()
        store.generate_code("telegram", "listed-user", "Listed User")

        with patch("gateway.pairing.PairingStore", return_value=store):
            pairing_command(Namespace(pairing_action="list"))
            list_output = capsys.readouterr().out
            request_id = store.list_pending("telegram")[0]["request_id"]

            assert request_id in list_output

            pairing_command(
                Namespace(
                    pairing_action="approve",
                    platform="telegram",
                    code=request_id,
                )
            )
            request_approval_output = capsys.readouterr().out

            bot_code = store.generate_code("telegram", "code-user", "Code User")
            pairing_command(
                Namespace(
                    pairing_action="approve",
                    platform="telegram",
                    code=bot_code,
                )
            )
            code_approval_output = capsys.readouterr().out

        approved_ids = {entry["user_id"] for entry in store.list_approved("telegram")}

    assert "listed-user" in request_approval_output
    assert "code-user" in code_approval_output
    assert approved_ids == {"listed-user", "code-user"}
