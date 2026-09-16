"""Hermetic tests for the unbroker skill.

Stdlib + pytest only; NO live network, NO browser, NO email. Each test runs against
an isolated temp PDD_DATA_DIR. Runnable with pytest or directly:

    python3 -m pytest tests/test_unbroker_skill.py -q
    python3 tests/test_unbroker_skill.py        # portable fallback runner
"""
from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Resolve the skill's scripts dir across layouts: standalone dev repo (tests/) and hermes-agent
# (tests/skills/ -> optional-skills/security/unbroker/scripts).
_HERE = Path(__file__).resolve()
_REL = ("optional-skills", "security", "unbroker", "scripts")
_CANDIDATES = [
    _HERE.parent.parent / "skill" / "scripts",           # standalone dev repo
    _HERE.parent.parent.joinpath(*_REL),                 # standalone layout
    _HERE.parent.parent.parent.joinpath(*_REL),          # hermes-agent (tests/skills/)
]
SCRIPTS = next((c for c in _CANDIDATES if (c / "pdd.py").exists()), _CANDIDATES[0])
sys.path.insert(0, str(SCRIPTS))

import autopilot        # noqa: E402
import contextlib as _ctx  # noqa: E402
import io as _io          # noqa: E402
import json as _json      # noqa: E402
import smtplib as _smtplib  # noqa: E402
import time as _time      # noqa: E402

import badbool          # noqa: E402
import brokers          # noqa: E402
import cdp              # noqa: E402
import config           # noqa: E402
import crypto           # noqa: E402
import dossier          # noqa: E402
import email_modes      # noqa: E402
import emailer          # noqa: E402
import pdd              # noqa: E402
import legal            # noqa: E402
import ledger           # noqa: E402
import paths            # noqa: E402
import registry         # noqa: E402
import report          # noqa: E402
import storage          # noqa: E402
import tiers            # noqa: E402
import vectors          # noqa: E402

_AGE = bool(shutil.which("age") and shutil.which("age-keygen"))


@contextlib.contextmanager
def temp_env():
    """Isolate every test in a fresh PDD_DATA_DIR."""
    prev = os.environ.get("PDD_DATA_DIR")
    with tempfile.TemporaryDirectory() as d:
        os.environ["PDD_DATA_DIR"] = str(Path(d) / "pdd")
        try:
            yield Path(os.environ["PDD_DATA_DIR"])
        finally:
            if prev is None:
                os.environ.pop("PDD_DATA_DIR", None)
            else:
                os.environ["PDD_DATA_DIR"] = prev


def _consenting(full_name="Jane Q. Public"):
    return {
        "subject_id": "sub_test01",
        "consent": {"authorized": True, "method": "self"},
        "identity": {
            "full_name": full_name,
            "emails": ["jane@example.com"],
            "phones": ["+1-415-555-0137"],
            "date_of_birth": "1987-04-12",
            "current_address": {"city": "Oakland", "state": "CA", "postal": "94601"},
        },
        "preferences": {"email_mode": "draft_only"},
    }


# --- config -------------------------------------------------------------------





def test_browser_clears_captcha_logic():
    assert config.browser_clears_captcha({"browser_backend": "browserbase"}) is True
    assert config.browser_clears_captcha({"browser_backend": "agent-browser"}) is False
    assert config.browser_clears_captcha({"browser_backend": "auto"}, env={}) is False
    assert config.browser_clears_captcha({"browser_backend": "auto"}, env={"BROWSERBASE_API_KEY": "x"}) is True


# --- storage ------------------------------------------------------------------

def test_storage_json_and_jsonl_roundtrip():
    with temp_env() as data:
        p = data / "x.json"
        storage.write_json(p, {"a": 1})
        assert storage.read_json(p) == {"a": 1}
        assert storage.read_json(data / "missing.json", []) == []
        log = data / "audit.jsonl"
        storage.append_jsonl(log, {"e": 1})
        storage.append_jsonl(log, {"e": 2})
        assert [r["e"] for r in storage.read_jsonl(log)] == [1, 2]


# --- at-rest encryption -------------------------------------------------------







# --- broker DB ----------------------------------------------------------------

def test_seed_broker_db_loads_and_is_well_formed():
    everyone = brokers.load_all()
    assert len(everyone) >= 10
    ids = {b["id"] for b in everyone}
    assert {"spokeo", "whitepages", "mylife"} <= ids
    for b in everyone:
        assert b.get("id") and b.get("name") and b.get("priority") in {"crucial", "high", "standard", "long_tail"}
        assert (b.get("optout") or {}).get("method")




def test_blocked_pass_records_and_cluster_coverage():
    # Records added from the blocked-tail pass load, resolve, and dedupe correctly.
    ids = {b["id"] for b in brokers.load_all()}
    assert {"addresses", "socialcatfish"} <= ids
    # addresses.com is a PeopleConnect/Intelius front-end -> covered by the intelius cluster (deduped).
    assert "addresses" in brokers.clusters().get("intelius", [])
    for bid in ("addresses", "socialcatfish"):
        b = brokers.get(bid)
        assert tiers.select_tier(b) in {"T0", "T1", "T2", "T3"}
        assert b["optout"]["method"]


# --- tier selection -----------------------------------------------------------

def test_every_broker_resolves_to_valid_tier():
    for b in brokers.load_all():
        assert tiers.select_tier(b) in {"T0", "T1", "T2", "T3"}




def test_captcha_tier_shifts_with_browser():
    tps = brokers.get("truepeoplesearch")
    assert tiers.select_tier(tps, "programmatic", browser_clears_captcha=False) == "T2"
    assert tiers.select_tier(tps, "programmatic", browser_clears_captcha=True) == "T1"




def test_plan_excludes_disallowed_fields():
    d = _consenting()
    actions = tiers.plan(d, brokers.load_all(), config.DEFAULT_CONFIG)
    for a in actions:
        assert "ssn" not in a["disclosure_fields"]
        assert "profile_url" not in a["disclosure_fields"]




def _mini_broker(bid, owns=None, requires=None, notes="", quirks=None):
    return {"id": bid, "name": bid.title(), "priority": "high",
            "search": {"by": ["name"]},
            "optout": {"method": "web_form", "url": f"https://{bid}.example/optout",
                       "requires": requires or {}, "inputs": ["full_name"], "owns": owns or [],
                       "notes": notes, "quirks": quirks or []},
            "owns": owns or []}


def test_batch_plan_groups_by_ledger_state():
    d = _consenting()
    bl = [_mini_broker("aaa"), _mini_broker("bbb"), _mini_broker("ccc"), _mini_broker("ddd")]
    ledger = {
        "aaa": {"state": "found"},
        "bbb": {"state": "not_found"},
        "ccc": {"state": "blocked"},
        # ddd absent -> unscanned/new
    }
    bp = tiers.batch_plan(d, bl, config.DEFAULT_CONFIG, ledger)
    assert bp["phase"] == "discover"                      # ddd is unscanned
    assert bp["counts"]["found"] == 1
    assert bp["counts"]["not_found"] == 1
    assert bp["counts"]["blocked"] == 1
    assert bp["counts"]["unscanned"] == 1
    assert any("PHASE 1" in t for t in bp["next_actions"])










# --- ledger / state machine ---------------------------------------------------

def test_ledger_valid_transition_and_audit():
    with temp_env():
        sid = "sub_test01"
        ledger.transition(sid, "spokeo", "searching")
        case = ledger.transition(sid, "spokeo", "found", found=True)
        assert case["state"] == "found" and case["found"] is True
        # found -> submitted must be allowed directly (action_selected is optional)
        case = ledger.transition(sid, "spokeo", "submitted")
        assert case["state"] == "submitted"
        audit = storage.read_jsonl(__import__("paths").audit_path(sid))
        assert any(e["to"] == "found" for e in audit)




def test_indirect_exposure_state_and_transitions():
    with temp_env():
        sid = "sub_test01"
        # a scan can land directly on indirect_exposure (PII on a relative's record)
        case = ledger.transition(sid, "thatsthem", "indirect_exposure",
                                  evidence={"summary": "email on relative record"})
        assert case["state"] == "indirect_exposure"
        # the lever from there is a targeted delete-my-PII request (-> submitted)
        assert ledger.transition(sid, "thatsthem", "submitted")["state"] == "submitted"
        # and a separate broker: not_found -> indirect_exposure is allowed (found on re-read)
        ledger.transition(sid, "radaris", "not_found")
        assert ledger.transition(sid, "radaris", "indirect_exposure")["state"] == "indirect_exposure"
        # re-scan can clear it
        assert ledger.transition(sid, "radaris", "not_found")["state"] == "not_found"






# --- dossier / consent / least-disclosure ------------------------------------



def test_least_disclosure_selection():
    d = _consenting()
    got = dossier.select_disclosure(d, ["full_name", "contact_email", "profile_url", "ssn", "date_of_birth"])
    assert set(got) == {"full_name", "contact_email", "date_of_birth"}
    assert "ssn" not in got and "profile_url" not in got




# --- alternates / search vectors ---------------------------------------------

def test_all_names_and_locations_dedupe():
    d = _consenting()
    d["identity"]["also_known_as"] = ["Jane Public", "Jane Q. Public"]   # 2nd dups primary
    d["identity"]["prior_addresses"] = [{"city": "Berkeley", "state": "CA"}, {"city": "Oakland", "state": "CA"}]
    assert dossier.all_names(d) == ["Jane Q. Public", "Jane Public"]
    assert [loc["city"] for loc in dossier.all_locations(d)] == ["Oakland", "Berkeley"]  # current first, deduped








# --- opaque ids / fan-out / antibot ------------------------------------------

def test_subject_id_is_opaque_no_name_leak():
    sid = dossier.new_subject_id("Maiden Married Person")
    assert sid.startswith("sub_")
    assert "maiden" not in sid.lower() and "person" not in sid.lower()
    assert dossier.new_subject_id("Maiden Married Person") != sid  # not derived from the name


def test_fanout_batches_large_runs():
    g = tiers.fanout([{"id": f"b{i}"} for i in range(20)], batch_size=8)
    assert g["broker_count"] == 20 and g["should_fanout"] is True
    assert len(g["batches"]) == 3 and g["batches"][0] == [f"b{i}" for i in range(8)]
    small = tiers.fanout([{"id": "x"}, {"id": "y"}], batch_size=8)
    assert small["should_fanout"] is False and small["batches"] == [["x", "y"]]




# --- cdp (operator browser over the DevTools protocol) --------------------------------------















# --- legal / templates --------------------------------------------------------



def test_render_optout_email_includes_listing_and_name():
    b = brokers.get("spokeo")
    out = legal.render_optout_email(b, {"full_name": "Jane Q. Public",
                                        "contact_email": "jane@example.com",
                                        "listing_urls": ["https://www.spokeo.com/jane"]})
    assert "Jane Q. Public" in out and "https://www.spokeo.com/jane" in out




# --- email verification-link extraction --------------------------------------





# --- BADBOOL live-pull parser -------------------------------------------------

BADBOOL_FIXTURE = """
## Search Engines
### Google
This is not a broker; ignore it.

## People Search Sites

### \U0001F490 BeenVerified
Find your information and opt out of [people search](https://www.beenverified.com/app/optout/search).

### \U0001F490 \U0001F4DE MyLife
[Find your information](https://www.mylife.com), and then [opt out](https://www.mylife.com/privacyrequest).

### \U0001F3AB PimEyes
To opt out, [upload an ID](https://pimeyes.com/en/opt-out-request-form).

## Special Circumstances
### Not A Broker
Ignore this section entirely.
"""


def test_badbool_parses_people_search_section_only():
    recs = badbool.parse(BADBOOL_FIXTURE)
    ids = {r["id"] for r in recs}
    assert ids == {"beenverified", "mylife", "pimeyes"}  # google + notabroker excluded
    bv = next(r for r in recs if r["id"] == "beenverified")
    assert bv["priority"] == "crucial"
    assert "beenverified.com/app/optout" in (bv["optout"]["url"] or "")
    assert bv["source"] == "BADBOOL-auto" and bv["confidence"] == "auto"




def test_badbool_merge_keeps_curated_and_adds_new():
    with temp_env():
        badbool.refresh(__import__("paths").brokers_cache_path(), markdown=BADBOOL_FIXTURE)
        merged = {b["id"]: b for b in brokers.load_all()}
        # curated record wins over the live one
        assert merged["beenverified"]["source"] == "BADBOOL"
        # a non-curated live record is added with auto confidence
        assert "pimeyes" in merged and merged["pimeyes"]["confidence"] == "auto"


# --- report -------------------------------------------------------------------



# --- autonomy: auto-configure ---------------------------------------------------------------



def test_auto_configure_picks_most_autonomous():
    with temp_env():
        # bare env -> draft_only floor, auto browser (still fully hands-off policy-wise)
        cfg = config.auto_configure(env={})
        assert cfg["autonomy"] == "full"
        assert cfg["email_mode"] == "draft_only"
        assert cfg["browser_backend"] == "auto"
        # SMTP creds -> programmatic email; Browserbase key -> cloud browser
        cfg = config.auto_configure(env={"EMAIL_ADDRESS": "agent@gmail.com",
                                         "EMAIL_PASSWORD": "app-pass",
                                         "BROWSERBASE_API_KEY": "bb"})
        assert cfg["email_mode"] == "programmatic"
        assert cfg["browser_backend"] == "browserbase"
        # AgentMail only -> alias mode
        assert config.auto_configure(env={"AGENTMAIL_API_KEY": "am"})["email_mode"] == "alias"
        # encryption auto-on exactly when age is installed (free privacy, zero human cost)
        assert config.auto_configure(env={})["encryption"] == ("age" if _AGE else "none")


# --- emailer: programmatic send + verification polling --------------------------------------



class _FakeSMTP:
    sent: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def ehlo(self):
        pass

    def starttls(self):
        pass

    def login(self, user, password):
        self.user = user

    def send_message(self, msg):
        _FakeSMTP.sent.append(msg)


def test_emailer_send_locks_recipient_to_broker():
    env = {"EMAIL_ADDRESS": "agent@gmail.com", "EMAIL_PASSWORD": "p"}
    broker = {"id": "radaris", "optout": {"email": "privacy@radaris.example"}}
    _FakeSMTP.sent = []
    out = emailer.send(broker, "Subject: Remove my listing\n\nBody here", env=env,
                       _smtp_factory=_FakeSMTP)
    assert out["to"] == "privacy@radaris.example"
    assert _FakeSMTP.sent[0]["Subject"] == "Remove my listing"
    assert "Body here" in _FakeSMTP.sent[0].get_content()
    # arbitrary recipients are refused -- this tool cannot be repurposed to email people
    try:
        emailer.send(broker, "Subject: x\n\nb", to="victim@example.com", env=env,
                     _smtp_factory=_FakeSMTP)
    except PermissionError:
        pass
    else:
        raise AssertionError("non-broker recipient must be refused")




def test_browser_send_payload_is_recipient_locked():
    broker = {"id": "radaris", "optout": {"email": "privacy@radaris.example"}}
    p = emailer.browser_send_payload(broker, "Subject: Remove my listing\n\nBody here")
    assert p["to"] == "privacy@radaris.example"
    assert p["subject"] == "Remove my listing" and "Body here" in p["body"]
    # the browser lane refuses arbitrary recipients too (same guard as SMTP send)
    try:
        emailer.browser_send_payload(broker, "Subject: x\n\nb", to="victim@example.com")
    except PermissionError:
        pass
    else:
        raise AssertionError("browser lane must refuse a non-broker recipient")




def test_verification_link_from_messages_is_domain_scoped():
    broker = {"id": "spokeo", "name": "Spokeo",
              "search": {"url": "https://www.spokeo.com/"},
              "optout": {"url": "https://www.spokeo.com/optout"}}
    phish = {"from": "phisher@evil.example", "subject": "verify now",
             "text": "click https://evil.example/optout/verify?x=1"}
    real = {"from": "no-reply@spokeo.com", "subject": "Confirm your opt out",
            "text": "Confirm here: https://www.spokeo.com/optout/verify/abc123"}
    hit = emailer.link_from_messages([phish, real], broker)
    assert hit["link"] == "https://www.spokeo.com/optout/verify/abc123"
    # a phishing-only inbox yields nothing (domain scoping + link scoring)
    assert emailer.link_from_messages([phish], broker) is None


# --- ledger: follow-up scheduling + due queue ------------------------------------------------







# --- autopilot: the autonomous action queue --------------------------------------------------

def _auto_cfg(**over):
    cfg = dict(config.DEFAULT_CONFIG)
    cfg.update(over)
    return cfg


def test_next_actions_scan_first_then_optouts_parents_first():
    with temp_env():
        d = _consenting()
        bl = [_mini_broker("parent", owns=["kid"]), _mini_broker("kid"), _mini_broker("solo")]
        q = autopilot.next_actions(d, bl, _auto_cfg(), {}, env={})
        types = [a["type"] for a in q["actions"]]
        assert "scan_inline" in types
        assert not any(t.startswith("optout") for t in types)   # never act before the crawl
        assert q["phase"] == "discover"
        led = {"parent": {"state": "found"}, "kid": {"state": "found"}, "solo": {"state": "found"}}
        q2 = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
        opt = [a for a in q2["actions"] if a["type"] == "optout_web_form"]
        assert [a["broker_id"] for a in opt] == ["parent", "solo"]  # kid covered by parent
        assert q2["phase"] == "delete"










def test_next_actions_blocked_stealth_or_operator_browser():
    with temp_env():
        d = _consenting()
        b = _mini_broker("gated")
        led = {"gated": {"state": "blocked"}}
        q = autopilot.next_actions(d, [b], _auto_cfg(), led, env={"BROWSERBASE_API_KEY": "bb"})
        assert any(a["type"] == "stealth_rescan" for a in q["actions"])
        q2 = autopilot.next_actions(d, [b], _auto_cfg(), led, env={})
        assert any("anti-bot" in t["reason"] for t in q2["human_digest"])






def test_parked_and_reappeared_states_group_correctly():
    # Regression: human_task_queued / action_selected / reappeared used to fall into "unscanned",
    # so the autonomous loop would try to re-scan parked or already-actioned cases forever.
    with temp_env():
        d = _consenting()
        bl = [_mini_broker("parked"), _mini_broker("chosen"), _mini_broker("back")]
        led = {"parked": {"state": "human_task_queued"},
               "chosen": {"state": "action_selected"},
               "back": {"state": "reappeared"}}
        bp = tiers.batch_plan(d, bl, config.DEFAULT_CONFIG, led)
        assert bp["counts"]["unscanned"] == 0
        assert bp["phase"] == "delete"
        assert [r["broker_id"] for r in bp["groups"]["human"]] == ["parked"]
        assert {r["broker_id"] for r in bp["groups"]["found"]} == {"chosen", "back"}
        q = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
        assert not any(a["type"] in ("scan_inline", "fanout_scan") for a in q["actions"])
        assert {a["broker_id"] for a in q["actions"] if a["type"] == "optout_web_form"} == {"chosen", "back"}


# --- cluster parents: verified deletion lanes + data-driven playbooks ------------------------



def test_curated_intelius_suppress_first_not_delete():
    # PeopleConnect is the EXCEPTION to deletion-beats-suppression: deleting user data wipes
    # your suppressions and does not stop public-records re-listing, so suppress-and-maintain.
    b = brokers.get("intelius")
    d = b["optout"]["deletion"]
    assert d["prefer"] is False and d["via"] == "in_flow"
    assert d["email"] == "privacy@peopleconnect.us"     # rights-request address for the data-purge path
    steps = " ".join(b["optout"]["playbook"]).upper()
    assert "SUPPRESS" in steps                          # the recommended action
    assert "DELETE MY USER DATA" in steps               # names the trap to avoid






def test_request_kind_is_residency_honest():
    ca = {"residency_jurisdiction": "US-CA"}
    tx = {"residency_jurisdiction": "US-TX"}
    de = {"residency_jurisdiction": "EU-DE"}
    assert autopilot.request_kind(ca) == "ccpa"
    assert autopilot.request_kind(tx) == "generic"      # never claim CCPA for a non-CA resident
    assert autopilot.request_kind(de) == "gdpr"
    assert autopilot.request_kind({}) == "generic"
    # broker restriction can force DOWN to generic but never upgrade
    assert autopilot.request_kind(tx, allowed=["ccpa", "generic"]) == "generic"
    assert autopilot.request_kind(ca, allowed=["generic"]) == "generic"
    assert autopilot.request_kind(ca, allowed=["ccpa", "generic"]) == "ccpa"






# --- human-task digest ------------------------------------------------------------------------

def test_human_tasks_digest_markdown():
    with temp_env():
        sid = "sub_test01"
        ledger.transition(sid, "mylife", "found", found=True)
        ledger.transition(sid, "mylife", "human_task_queued",
                          human_task_reason="gov ID demanded")
        ledger.transition(sid, "fastpeoplesearch", "blocked")
        md = report.human_tasks_markdown(sid)
        assert "gov ID demanded" in md
        assert "Withhold" in md
        assert "fastpeoplesearch" in md.lower()
        # empty ledger -> explicitly says nothing is needed
        assert "Nothing needs a human" in report.human_tasks_markdown("sub_other")


# --- CA data broker registry (coverage breadth: DROP + email lane) ---------------------------

def _registry_csv():
    """Mimic the CA registry CSV: junk row 0, label row 1 (with the real NBSP), data rows."""
    import csv as _csv
    import io as _io
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["", "junk header the site hides", "", "", "", ""])
    w.writerow(["Data broker\xa0name:", "Doing Business As (DBA), if applicable:",
                "Data broker primary website:", "Data broker primary contact email address:",
                "Data broker's primary website that contains details on how consumers can exercise "
                "their CA Consumer Privacy Act rights, including how to delete their personal information:",
                "The data broker or any of its subsidiaries is regulated by the federal Fair Credit "
                "Reporting Act (FCRA):"])
    w.writerow(["Acme Data LLC", "AcmeDBA", "https://acme.example",
                "privacy@acme.example", "https://acme.example/ccpa", "No"])
    w.writerow(["Credit Bureau Co", "", "https://cbc.example",
                "privacy@cbc.example", "https://cbc.example/rights", "Yes"])
    return buf.getvalue()












# --- hardening: locking / rate-limit / retry / idempotency / freshness / metrics ------------

def test_storage_lock_mutual_exclusion_and_stale_break():
    with temp_env() as data:
        target = data / "x.json"
        with storage.locked(target):                       # hold the lock
            try:
                with storage.locked(target, timeout=0.2):  # second acquire must time out
                    raise AssertionError("second acquire should have timed out")
            except TimeoutError:
                pass
        with storage.locked(target, timeout=0.2):          # released -> acquires fine
            pass
        # a stale lock (old mtime) from a crashed writer gets broken
        lock = target.with_name(target.name + ".lock")
        lock.write_text("999999")
        old = _time.time() - 120
        os.utime(lock, (old, old))
        with storage.locked(target, timeout=0.2, stale=30):
            pass




class _FlakySMTP:
    attempts = 0

    def __init__(self, host, port, timeout=None):
        pass

    def __enter__(self):
        _FlakySMTP.attempts += 1
        if _FlakySMTP.attempts < 3:
            raise _smtplib.SMTPServerDisconnected("transient")
        return self

    def __exit__(self, *a):
        return False

    def ehlo(self):
        pass

    def starttls(self):
        pass

    def login(self, u, p):
        pass

    def send_message(self, m):
        _FlakySMTP.sent = m


class _AuthFailSMTP(_FlakySMTP):
    def __enter__(self):
        return self

    def login(self, u, p):
        raise _smtplib.SMTPAuthenticationError(535, b"bad creds")






def _run(argv) -> dict:
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        pdd.main(argv)
    return _json.loads(buf.getvalue())




def test_show_reads_back_case_state_and_evidence():
    with temp_env():
        sid = _run(["intake", "--full-name", "Jane Q. Public",
                    "--email", "jane@example.com", "--consent"])["subject_id"]
        _run(["record", sid, "radaris", "found", "--found", "true",
              "--evidence", '{"listing_urls": ["https://radaris.com/p/x"]}'])
        shown = _run(["show", sid, "radaris"])
        assert shown["broker"] == "radaris" and shown["state"] == "found"
        assert shown["found"] is True
        assert shown["evidence"].get("listing_urls") == ["https://radaris.com/p/x"]
        # Unknown case returns a fresh (new) case, not an error.
        empty = _run(["show", sid, "not_a_broker"])
        assert empty["state"] == "new" and empty["evidence"] == {}












def test_report_metrics_removal_rate_and_overdue():
    with temp_env():
        sid = "sub_test01"
        for st in ("found", "submitted", "awaiting_processing", "confirmed_removed"):
            ledger.transition(sid, "a", st, **({"found": True} if st == "found" else {}))
        ledger.transition(sid, "b", "found", found=True)                        # open
        for st in ("found", "submitted", "awaiting_processing"):
            ledger.transition(sid, "c", st, **({"found": True} if st == "found" else {}))
        led = ledger.load(sid)
        led["c"]["next_recheck_at"] = "2000-01-01T00:00:00Z"                    # force overdue
        ledger.save(sid, led)
        m = report.metrics(sid)
        assert m["confirmed_removed"] == 1
        assert m["open_needs_action"] >= 1 and m["in_flight_claimed"] >= 1
        assert m["overdue_rechecks"] >= 1 and 0 < m["removal_rate"] <= 1


if __name__ == "__main__":
    failures = []
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
