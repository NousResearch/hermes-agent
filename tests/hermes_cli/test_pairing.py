import argparse
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from gateway.pairing import PairingStore
from hermes_cli.subcommands.pairing import build_pairing_parser
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


def test_clear_pending_cli_clears_only_selected_home_and_keeps_approvals(
    tmp_path, monkeypatch, capsys
):
    user_home = tmp_path / "home"
    default_home = user_home / ".hermes"
    work_home = default_home / "profiles" / "work"
    for home in (default_home, work_home):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    monkeypatch.setattr(Path, "home", lambda: user_home)
    for env_var in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS"):
        monkeypatch.delenv(env_var, raising=False)

    def store_in(home):
        monkeypatch.setenv("HERMES_HOME", str(home))
        return PairingStore()

    def snapshot(root, only=None):
        return {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and (only is None or only(path))
        }

    default_store = store_in(default_home)
    assert default_store.generate_code("telegram", "default-sentinel")

    work_store = store_in(work_home)
    for platform, user_id in (("telegram", "approved-tg"), ("discord", "approved-dc")):
        assert work_store.approve_code(platform, work_store.generate_code(platform, user_id))
    assert work_store.generate_code("telegram", "pending-tg")
    assert work_store.generate_code("discord", "pending-dc")
    assert {r["platform"] for r in work_store.list_pending()} == {"telegram", "discord"}

    def is_preserved(path):
        return path.name.endswith("-approved.json") or path.name == "_rate_limits.json"

    preserved = snapshot(work_home, is_preserved)
    assert any(p.name == "_rate_limits.json" for p in preserved)
    assert len([p for p in preserved if p.name.endswith("-approved.json")]) == 2
    other_home = snapshot(default_home, lambda p: work_home not in p.parents)

    parser = argparse.ArgumentParser()
    build_pairing_parser(parser.add_subparsers(dest="command"), cmd_pairing=pairing_command)

    def run_clear_pending():
        monkeypatch.setenv("HERMES_HOME", str(work_home))
        args = parser.parse_args(["pairing", "clear-pending"])
        args.func(args)
        out = capsys.readouterr().out
        reopened = store_in(work_home)
        assert reopened.list_pending() == []
        assert reopened.is_approved("telegram", "approved-tg")
        assert reopened.is_approved("discord", "approved-dc")
        assert snapshot(work_home, is_preserved) == preserved
        assert snapshot(default_home, lambda p: work_home not in p.parents) == other_home
        sentinel = store_in(default_home).list_pending()
        assert [r["user_id"] for r in sentinel] == ["default-sentinel"]
        return out

    assert "Cleared 2 pending" in run_clear_pending()
    assert "No pending requests" in run_clear_pending()
