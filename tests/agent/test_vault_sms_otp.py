"""vault_sms_otp: allowlisted SMS codes from a Messages-shaped sqlite store."""

import json
import sqlite3
import time

import pytest

from agent import vault_sms_otp as sms

_APPLE_EPOCH = 978307200


def _messages_db(tmp_path, rows):
    db = tmp_path / "chat.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
    con.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, date INTEGER, handle_id INTEGER, "
                "is_from_me INTEGER, text TEXT, attributedBody BLOB)")
    for i, (age_s, sender, text, from_me) in enumerate(rows, start=1):
        con.execute("INSERT INTO handle VALUES (?, ?)", (i, sender))
        con.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?, NULL)",
                    (i, int((time.time() - age_s - _APPLE_EPOCH) * 1e9), i, int(from_me), text))
    con.commit()
    con.close()
    return db


CFG = {"enabled": True, "lookback_seconds": 120, "sites": {"amazon.com": ["amazon"]}}


pytestmark = pytest.mark.platforms("macos")  # the Messages store only exists on macOS


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))


def test_fills_recent_allowlisted_code_and_audits_without_the_code(tmp_path):
    db = _messages_db(tmp_path, [(30, "98626", "Amazon: Your code is 482913. Don't share it.", False)])
    code = sms.wait_for_code("https://www.amazon.com", cfg=CFG, db=db, now=time.time())
    assert code == "482913"
    log = (tmp_path / "home" / "logs" / "vault_sms_otp.log").read_text()
    assert json.loads(log.splitlines()[-1])["outcome"] == "filled"
    assert "482913" not in log


@pytest.mark.parametrize("rows, origin, cfg", [
    ([(600, "98626", "Amazon: Your code is 482913.", False)], "https://www.amazon.com", CFG),       # too old
    ([(30, "22000", "Chase: Your code is 482913.", False)], "https://www.amazon.com", CFG),         # wrong site text
    ([(30, "98626", "Amazon: Your code is 482913.", False)], "https://www.chase.com", CFG),         # site not allowlisted
    ([(30, "98626", "Amazon: Your code is 482913.", True)], "https://www.amazon.com", CFG),         # my own message
    ([(30, "98626", "Amazon: Your code is 482913.", False)], "https://www.amazon.com",
     {**CFG, "enabled": False}),                                                                   # feature off
])
def test_refuses_outside_the_guardrails(tmp_path, rows, origin, cfg):
    db = _messages_db(tmp_path, rows)
    assert sms.wait_for_code(origin, cfg=cfg, db=db, now=time.time()) is None


def test_ambiguous_text_yields_no_code():  # pure parsing: host-independent
    assert sms.extract_code("Amazon order 1234 ships; code 5678 or 9012") is None
    assert sms.extract_code("123456 is your Amazon verification code") == "123456"


def test_registrable_domain_matches_subdomains():
    assert sms.registrable_domain("https://www.amazon.com") == "amazon.com"
    assert sms.registrable_domain("https://signin.aws.amazon.co.uk") == "amazon.co.uk"


def test_config_yaml_gates_the_feature_through_the_real_loader(tmp_path):
    """vault.sms_otp in config.yaml is what enables a site; the shipped default leaves it off."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("vault:\n  sms_otp:\n    enabled: true\n    sites:\n      amazon.com: [amazon]\n")
    assert sms.site_keywords("https://www.amazon.com") == ["amazon"]
    assert sms.site_keywords("https://www.chase.com") is None
    (home / "config.yaml").write_text("model:\n  default: x\n")
    assert sms.site_keywords("https://www.amazon.com") is None
