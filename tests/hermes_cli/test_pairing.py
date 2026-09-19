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


def test_cli_approve_notify_delivers_receipt_to_paired_user(tmp_path, capsys):
    """A successful approval can notify the requester through ``hermes send``."""
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        store = PairingStore()
        code = store.generate_code("telegram", "requester-42", "Requester")

        with (
            patch("gateway.pairing.PairingStore", return_value=store),
            patch("hermes_cli.send_cmd.cmd_send", side_effect=SystemExit(0)) as send,
        ):
            status = pairing_command(
                Namespace(
                    pairing_action="approve",
                    platform="telegram",
                    code=code,
                    notify=True,
                    admin=False,
                )
            )

    assert status == 0
    assert send.call_args.args[0].to == "telegram:requester-42"
    assert "approved" in send.call_args.args[0].message.lower()
    assert "Receipt delivered" in capsys.readouterr().out


def test_cli_approve_notify_reports_delivery_failure_without_reversing_grant(tmp_path, capsys):
    """A failed receipt is visible and non-destructive after the pairing grant succeeds."""
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        store = PairingStore()
        code = store.generate_code("telegram", "requester-42", "Requester")

        with (
            patch("gateway.pairing.PairingStore", return_value=store),
            patch("hermes_cli.send_cmd.cmd_send", side_effect=SystemExit(1)),
        ):
            status = pairing_command(
                Namespace(
                    pairing_action="approve",
                    platform="telegram",
                    code=code,
                    notify=True,
                    admin=False,
                )
            )

    assert status == 1
    assert store.is_approved("telegram", "requester-42") is True
    assert "Receipt delivery failed" in capsys.readouterr().out


def test_cli_approve_admin_explicitly_sets_dm_admin_only(tmp_path, monkeypatch):
    """Only ``--admin`` promotes the approved user into the DM slash-admin scope."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        store = PairingStore()
        code = store.generate_code("telegram", "operator-7", "Operator")

        with patch("gateway.pairing.PairingStore", return_value=store):
            pairing_command(
                Namespace(
                    pairing_action="approve",
                    platform="telegram",
                    code=code,
                    notify=False,
                    admin=True,
                )
            )

    from hermes_cli.config import read_raw_config

    config = read_raw_config()
    assert config["platforms"]["telegram"]["allow_admin_from"] == ["operator-7"]
    assert "group_allow_admin_from" not in config["platforms"]["telegram"]


def test_cli_approve_without_admin_does_not_enable_slash_gating(tmp_path, monkeypatch):
    """A normal approval remains a pairing grant, not an implicit command-admin grant."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        store = PairingStore()
        code = store.generate_code("telegram", "paired-user", "Paired User")
        with patch("gateway.pairing.PairingStore", return_value=store):
            pairing_command(
                Namespace(
                    pairing_action="approve",
                    platform="telegram",
                    code=code,
                    notify=False,
                    admin=False,
                )
            )

    from hermes_cli.config import read_raw_config

    assert "allow_admin_from" not in read_raw_config().get("platforms", {}).get("telegram", {})
