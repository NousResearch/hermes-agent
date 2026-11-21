"""vault_sms_otp: allowlisted SMS codes from a Messages-shaped sqlite store."""

import json
import sqlite3
import time

import pytest

from agent import vault_sms_otp as sms

_APPLE_EPOCH = 978307200
AMAZON = "72656"


def _messages_db(tmp_path, rows, name="chat.db"):
    db = tmp_path / name
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


CFG = {"enabled": True, "lookback_seconds": 120,
       "sites": {"amazon.com": {"senders": [AMAZON], "keywords": ["amazon"]}}}


pytestmark = pytest.mark.platforms("macos")  # the Messages store only exists on macOS


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))


def _code(db, origin="https://www.amazon.com", cfg=CFG):
    return sms.wait_for_code(origin, cfg=cfg, db=db, now=time.time())


def test_fills_recent_allowlisted_code_and_audits_without_the_code(tmp_path):
    db = _messages_db(tmp_path, [(30, AMAZON, "Amazon: Your code is 482913. Don't share it.", False)])
    assert _code(db) == "482913"
    log = (tmp_path / "home" / "logs" / "vault_sms_otp.log").read_text()
    assert json.loads(log.splitlines()[-1])["outcome"] == "filled"
    assert "482913" not in log
    ledger = (tmp_path / "home" / "state" / "vault_sms_otp_used.json").read_text()
    assert "482913" not in ledger


@pytest.mark.parametrize("rows, origin, cfg", [
    ([(600, AMAZON, "Amazon: Your code is 482913.", False)], "https://www.amazon.com", CFG),        # too old
    ([(30, AMAZON, "Chase: Your code is 482913.", False)], "https://www.amazon.com", CFG),          # keyword missing
    ([(30, AMAZON, "Amazon: Your code is 482913.", False)], "https://www.chase.com", CFG),          # site not allowlisted
    ([(30, AMAZON, "Amazon: Your code is 482913.", True)], "https://www.amazon.com", CFG),          # my own message
    ([(30, AMAZON, "Amazon: Your code is 482913.", False)], "https://www.amazon.com",
     {**CFG, "enabled": False}),                                                                    # feature off
    ([(30, AMAZON, "Amazon: Your code is 482913.", False)], "https://www.amazon.com",
     {**CFG, "sites": {"amazon.com": ["amazon"]}}),                                                 # old keyword-only shape
    ([(30, AMAZON, "Amazon: Your code is 482913.", False)], "https://www.amazon.com",
     {**CFG, "sites": {"amazon.com": {"keywords": ["amazon"]}}}),                                   # no senders
])
def test_refuses_outside_the_guardrails(tmp_path, rows, origin, cfg):
    db = _messages_db(tmp_path, rows)
    assert _code(db, origin, cfg) is None


@pytest.mark.parametrize("rows, expected", [
    # A fresher text from an unlisted number must not outrank the real sender's code.
    ([(90, AMAZON, "Amazon: Your code is 111111.", False),
      (20, "+15550100100", "Amazon: your code is 999999", False)], "111111"),
    # An unlisted sender quoting the site name in the body is never eligible.
    ([(15, "+15550100100", "AMAZON verification code 424242", False)], None),
    # A sender ID spelled like the brand but not configured is not the configured short code.
    ([(15, "Amazon-SMS", "Amazon: Your code is 777777.", False)], None),
])
def test_only_configured_senders_supply_codes(tmp_path, rows, expected):
    db = _messages_db(tmp_path, rows)
    assert _code(db) == expected


def test_sender_matching_ignores_phone_formatting_but_stays_exact(tmp_path):
    cfg = {**CFG, "sites": {"amazon.com": {"senders": ["+1 (555) 010-0100"]}}}
    db = _messages_db(tmp_path, [(20, "+15550100100", "Amazon code 246810", False)])
    assert _code(db, cfg=cfg) == "246810"
    other = _messages_db(tmp_path, [(20, "+445550100100", "Amazon code 135790", False)], name="other.db")
    assert _code(other, cfg=cfg) is None


def test_keywords_match_whole_words_only(tmp_path):
    cfg = {**CFG, "sites": {"amazon.com": {"senders": [AMAZON], "keywords": ["am"]}}}
    db = _messages_db(tmp_path, [(20, AMAZON, "An amazing deal! Your code is 314159.", False)])
    assert _code(db, cfg=cfg) is None
    assert sms._has_word("your am code is 1234", "am")


def test_each_text_is_used_at_most_once(tmp_path):
    db = _messages_db(tmp_path, [(20, AMAZON, "Amazon: Your code is 111111.", False)])
    assert _code(db) == "111111"
    assert _code(db) is None
    assert _code(db, origin="https://signin.aws.amazon.com/") is None


def test_newest_unused_text_wins_and_older_ones_stay_available(tmp_path):
    db = _messages_db(tmp_path, [(60, AMAZON, "Amazon: Your code is 111111.", False),
                                 (10, AMAZON, "Amazon: Your code is 222222.", False)])
    assert _code(db) == "222222"
    assert _code(db) == "111111"
    assert _code(db) is None


def test_unwritable_ledger_fails_closed(tmp_path, monkeypatch):
    db = _messages_db(tmp_path, [(20, AMAZON, "Amazon: Your code is 111111.", False)])

    def boom(_message_id):
        raise OSError("read-only")

    monkeypatch.setattr(sms, "_mark_used", boom)
    assert _code(db) is None
    log = (tmp_path / "home" / "logs" / "vault_sms_otp.log").read_text()
    assert json.loads(log.splitlines()[-1])["outcome"].startswith("ledger_unwritable")


def test_ambiguous_text_yields_no_code():  # pure parsing: host-independent
    assert sms.extract_code("Amazon order 1234 ships; code 5678 or 9012") is None
    assert sms.extract_code("123456 is your Amazon verification code") == "123456"


def test_registrable_domain_matches_subdomains():
    assert sms.registrable_domain("https://www.amazon.com") == "amazon.com"
    assert sms.registrable_domain("https://signin.aws.amazon.co.uk") == "amazon.co.uk"


def test_normalize_sender():
    assert sms.normalize_sender("+1 (555) 010-0100") == "15550100100"
    assert sms.normalize_sender("72656") == "72656"
    assert sms.normalize_sender("Amazon-SMS") == "amazon-sms"


def test_config_yaml_gates_the_feature_through_the_real_loader(tmp_path):
    """vault.sms_otp in config.yaml is what enables a site; the shipped default leaves it off."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "vault:\n  sms_otp:\n    enabled: true\n    sites:\n      amazon.com:\n"
        "        senders: ['72656']\n        keywords: [amazon]\n")
    assert sms.site_rules("https://www.amazon.com") == {"senders": ["72656"], "keywords": ["amazon"]}
    assert sms.site_rules("https://www.chase.com") is None
    (home / "config.yaml").write_text("vault:\n  sms_otp:\n    enabled: true\n    sites:\n      amazon.com: [amazon]\n")
    assert sms.site_rules("https://www.amazon.com") is None  # keywords alone are not an identity
    (home / "config.yaml").write_text("model:\n  default: x\n")
    assert sms.site_rules("https://www.amazon.com") is None
