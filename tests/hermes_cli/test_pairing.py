from argparse import Namespace
from unittest.mock import patch

from gateway.pairing import RATE_LIMIT_SECONDS, PairingStore
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


def test_cleanup_expired_prunes_stale_rate_limit_keys(tmp_path):
    now = 10_000.0
    with patch("gateway.pairing.PAIRING_DIR", tmp_path), patch(
        "gateway.pairing.time.time", return_value=now
    ):
        store = PairingStore()
        rate_path = store._rate_limit_path()
        store._save_json(
            rate_path,
            {
                "telegram:stale-user": now - RATE_LIMIT_SECONDS - 1,
                "telegram:fresh-user": now - RATE_LIMIT_SECONDS + 1,
                "telegram:malformed": "not-a-timestamp",
                "discord:stale-user": now - RATE_LIMIT_SECONDS - 1,
                "_lockout:telegram": now - 1,
                "_failures:telegram": 2,
            },
        )

        store._cleanup_expired("telegram")
        remaining = store._load_json(rate_path)

    assert "telegram:stale-user" not in remaining
    assert "telegram:malformed" not in remaining
    assert "_lockout:telegram" not in remaining
    assert remaining["telegram:fresh-user"] == now - RATE_LIMIT_SECONDS + 1
    assert "discord:stale-user" in remaining
    assert remaining["_failures:telegram"] == 2
