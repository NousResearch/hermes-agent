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

import autopilot
import contextlib as _ctx
import io as _io
import json as _json
import time as _time

import badbool
import brokers
import config
import dossier
import emailer
import pdd
import legal
import ledger
import paths
import report
import storage
import tiers
import vectors

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
    assert config.browser_clears_captcha({"browser_backend": "local"}) is False
    assert config.browser_clears_captcha({"browser_backend": "auto"}, env={}) is False
    assert config.browser_clears_captcha({"browser_backend": "auto"}, env={"BROWSERBASE_API_KEY": "x"}) is True


def test_legacy_agent_browser_backend_loads_and_resaves_as_local():
    """Configs saved before agent-browser was removed must keep loading and saving."""
    with temp_env():
        storage.write_json(paths.config_path(), {"browser_backend": "agent-browser"})
        cfg = config.load_config()
        assert cfg["browser_backend"] == "local"
        config.save_config(cfg)
        assert config.load_config()["browser_backend"] == "local"


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


# --- spam-call lane: phone-led intake, focus phone, DNC one-shot, unsearchable brokers ------

def test_phone_led_intake_without_name():
    with temp_env():
        out = _run(["intake", "--phone", "+1-415-555-0137", "--email", "t@example.com", "--consent"])
        assert out["phones"] == 1 and out["names"] == []
        assert any("full_name" in w for w in out["warnings"])
        # intake with no identifier at all is refused
        try:
            _run(["intake", "--consent"])
        except SystemExit as e:
            assert "at least one identifier" in str(e)   # not just any argparse exit
        else:
            raise AssertionError("intake without any identifier must be refused")


def test_focus_phone_drives_disclosure_and_vectors():
    with temp_env():
        out = _run(["intake", "--full-name", "Jane Q. Public", "--email", "jane@example.com",
                    "--phone", "+1-415-555-0137", "--focus-phone", "+1-510-555-0199", "--consent"])
        sid = out["subject_id"]
        assert out["phones"] == 2 and out["focus_phone"] == "+1-510-555-0199"
        d = dossier.load(sid)
        assert d["preferences"]["focus_phone"] == "+1-510-555-0199"
        # the focus number is what a phone-asking opt-out form gets, not phones[0]
        assert dossier.select_disclosure(d, ["phone"])["phone"] == "+1-510-555-0199"
        sv = vectors.search_vectors(d, {"search": {"by": ["phone"]}})
        assert {v["query"]["phone"] for v in sv} == {"+1-415-555-0137", "+1-510-555-0199"}


def test_dnc_one_shot_surfaced_once_then_filed():
    with temp_env():
        sid = _run(["intake", "--full-name", "Jane Q. Public", "--email", "jane@example.com",
                    "--phone", "+1-415-555-0137", "--focus-phone", "+1-415-555-0137",
                    "--consent"])["subject_id"]
        d = dossier.load(sid)
        d["residency_jurisdiction"] = "US-CA"
        dossier.save(d)
        out = _run(["dnc", sid])
        assert out["eligible"] is True
        assert out["dnc_url"] == "https://www.donotcall.gov/register.html"
        assert "72 hours" in " ".join(out["steps"])
        assert f"dnc {sid} --declined" in " ".join(out["steps"])
        q = autopilot.next_actions(dossier.load(sid), [], _auto_cfg(), {}, env={})
        act = next(a for a in q["actions"] if a["type"] == "dnc_register")
        assert act["confirm_first"] is True   # a new registry disclosure always asks
        assert act["declined"]
        filed = _run(["dnc", sid, "--filed"])
        assert "dnc_filed_at" in filed
        q2 = autopilot.next_actions(dossier.load(sid), [], _auto_cfg(), {}, env={})
        assert not any(a["type"] == "dnc_register" for a in q2["actions"])
        # non-US residency -> never surfaced
        d2 = dossier.load(sid)
        d2["residency_jurisdiction"] = "EU-DE"
        dossier.save(d2)
        q3 = autopilot.next_actions(dossier.load(sid), [], _auto_cfg(), {}, env={})
        assert not any(a["type"] == "dnc_register" for a in q3["actions"])
        assert _run(["dnc", sid])["eligible"] is False


def test_unsearchable_group_and_fanout_skip_for_nameless_subject():
    with temp_env():
        sid = _run(["intake", "--phone", "+1-415-555-0137", "--consent"])["subject_id"]
        d = dossier.load(sid)
        bl = [_mini_broker("namesite"),
              {**_mini_broker("phonesite"), "search": {"by": ["phone"]}}]
        bp = tiers.batch_plan(d, bl, config.DEFAULT_CONFIG, {})
        assert bp["counts"]["unsearchable"] == 1
        assert bp["counts"]["unscanned"] == 1
        assert [r["broker_id"] for r in bp["groups"]["unsearchable"]] == ["namesite"]
        assert [r["broker_id"] for r in bp["groups"]["unscanned"]] == ["phonesite"]
        assert any("no runnable search vector" in t for t in bp["next_actions"])
        q = autopilot.next_actions(d, bl, _auto_cfg(), {}, env={})
        scan = next(a for a in q["actions"] if a["type"] == "scan_inline")
        assert scan["broker_ids"] == ["phonesite"]
        # fanout batches only brokers this subject can actually be searched on
        fan = _run(["fanout", sid])
        ids = [b for batch in fan["batches"] for b in batch["brokers"]]
        assert fan["skipped_unsearchable"] == 3   # 22 curated - 5 defunct - 14 live phone-capable
        for bid in ids:
            assert "phone" in (brokers.get(bid).get("search") or {}).get("by", ["name"])



# --- review fixes: nameless emails, dnc scope/decline, defunct brokers, focus-phone edges ---

def test_email_requests_refuse_nameless_subject():
    """A nameless subject must never render/send a legal email that says 'None'."""
    with temp_env():
        sid = _run(["intake", "--phone", "+1-415-555-0137", "--email", "t@example.com",
                    "--consent"])["subject_id"]
        try:
            _run(["render-email", sid, "spokeo", "--kind", "ccpa_indirect",
                  "--listing", "https://www.spokeo.com/x"])
        except SystemExit as e:
            assert "full name" in str(e)
        else:
            raise AssertionError("render-email must refuse a nameless legal request")
        # send-email needs a real email mode to reach the identity gate (not draft_only)
        config.save_config({**config.DEFAULT_CONFIG, "email_mode": "browser"})
        try:
            _run(["send-email", sid, "spokeo", "--kind", "generic",
                  "--listing", "https://www.spokeo.com/x"])
        except SystemExit as e:
            assert "full name" in str(e)   # sending needs the name even for generic
        else:
            raise AssertionError("send-email must refuse a nameless subject")
        # generic drafts stay renderable (operator-edited signature) and never say "None"
        out = _run(["render-email", sid, "spokeo", "--kind", "generic",
                    "--listing", "https://www.spokeo.com/x"])
        draft_text = Path(out["draft"]).read_text(encoding="utf-8")
        assert "None" not in draft_text and "[your full name]" in draft_text


def test_email_request_omits_name_identifier_when_nameless():
    with temp_env():
        sid = _run(["intake", "--phone", "+1-415-555-0137", "--consent"])["subject_id"]
        d = dossier.load(sid)
        fields, _disclosed = pdd._email_request(d, brokers.get("spokeo"), "ccpa_indirect",
                                                 ["https://www.spokeo.com/x"], None)
        assert "None" not in " ".join(fields["my_identifiers"])


def test_optout_email_lane_requires_identity_in_next():
    """The autonomous email lane digests (never queues) for a nameless subject."""
    with temp_env():
        mailer = _mini_broker("mailer")
        mailer["optout"] = {**mailer["optout"], "method": "email", "email": "privacy@mailer.example"}
        led = {"mailer": {"state": "found"}}
        sid = _run(["intake", "--phone", "+1-415-555-0137", "--consent"])["subject_id"]
        q = autopilot.next_actions(dossier.load(sid), [mailer], _auto_cfg(email_mode="browser"),
                                   led, env={})
        assert not any(a["type"] == "optout_email_send" for a in q["actions"])
        assert "full name" in _json.dumps(q["human_digest"])
        sid2 = _run(["intake", "--full-name", "Jane Q. Public", "--email", "jane@example.com",
                     "--consent"])["subject_id"]
        q2 = autopilot.next_actions(dossier.load(sid2), [mailer], _auto_cfg(email_mode="browser"),
                                    led, env={})
        assert any(a["type"] == "optout_email_send" for a in q2["actions"])


def test_dnc_register_needs_phone_led_run_and_can_be_declined():
    with temp_env():
        # name-led subject: no dnc action, and `next` must terminate
        sid = _run(["intake", "--full-name", "Jane Q. Public", "--email", "jane@example.com",
                    "--consent"])["subject_id"]
        _age = autopilot.cache_age_days
        autopilot.cache_age_days = lambda: 0.0   # fresh broker cache: no refresh action
        try:
            q = autopilot.next_actions(dossier.load(sid), [], _auto_cfg(), {}, env={})
        finally:
            autopilot.cache_age_days = _age
        assert not any(a["type"] == "dnc_register" for a in q["actions"])
        assert q["done_for_now"] is True   # a name-led run must terminate, not wedge on DNC
        # phone-led subject: surfaced once with confirm_first, then --declined clears it
        sid2 = _run(["intake", "--phone", "+1-415-555-0137", "--focus-phone", "+1-415-555-0137",
                     "--consent"])["subject_id"]
        q2 = autopilot.next_actions(dossier.load(sid2), [], _auto_cfg(), {}, env={})
        assert next(a for a in q2["actions"] if a["type"] == "dnc_register")["confirm_first"] is True
        dec = _run(["dnc", sid2, "--declined"])
        assert "dnc_declined_at" in dec
        q3 = autopilot.next_actions(dossier.load(sid2), [], _auto_cfg(), {}, env={})
        assert not any(a["type"] == "dnc_register" for a in q3["actions"])
        # a non-NANP number is not registrable even on a phone-led run
        sid3 = _run(["intake", "--phone", "+44 20 7946 0958", "--focus-phone", "+44 20 7946 0958",
                     "--consent"])["subject_id"]
        q4 = autopilot.next_actions(dossier.load(sid3), [], _auto_cfg(), {}, env={})
        assert not any(a["type"] == "dnc_register" for a in q4["actions"])
        assert _run(["dnc", sid3])["eligible"] is False


def test_defunct_brokers_excluded_from_all_planning():
    with temp_env():
        d = _consenting()
        dead = _mini_broker("deadbroker")
        dead["defunct"] = {"since": "2026-08-27", "evidence": "domain seized"}
        deadrest = _mini_broker("deadrest")
        deadrest["defunct"] = {"since": "2026-08-27", "evidence": "NXDOMAIN"}
        live = _mini_broker("livebroker")
        bl = [dead, deadrest, live]
        assert [a["broker_id"] for a in tiers.plan(d, bl, config.DEFAULT_CONFIG)] == ["livebroker"]
        bp = tiers.batch_plan(d, bl, config.DEFAULT_CONFIG, {})
        assert bp["defunct"] == ["deadbroker", "deadrest"]
        assert not any(r["broker_id"].startswith("dead") for g in bp["groups"].values() for r in g)
        runnable, unsearchable = tiers.runnable_for(d, bl)
        assert [b["id"] for b in runnable + unsearchable] == ["livebroker"]
        assert all("deadbroker" not in batch and "deadrest" not in batch
                   for batch in tiers.fanout(bl)["batches"])
        # open cases of dead brokers: no re-checks, one close-out digest entry instead
        led = {"deadbroker": {"broker_id": "deadbroker", "state": "submitted"},
               "deadrest": {"broker_id": "deadrest", "state": "confirmed_removed",
                            "next_recheck_at": "2020-01-01T00:00:00Z"},
               "livebroker": {"broker_id": "livebroker", "state": "confirmed_removed",
                               "next_recheck_at": "2020-01-01T00:00:00Z"}}
        q = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
        kinds = [(a["type"], a.get("broker_id")) for a in q["actions"]]
        assert ("verify_removal", "deadrest") not in kinds   # never re-scan a corpse
        assert ("verify_removal", "livebroker") in kinds     # live brokers still recheck
        close = [t for t in q["human_digest"] if "deadbroker" in _json.dumps(t)]
        assert close and "not_found" in _json.dumps(close)
        # a dead broker parked as a human task gets the close-out only, not a manual step too
        led["deadbroker"] = {"broker_id": "deadbroker", "state": "human_task_queued",
                             "human_task_reason": "needs fax"}
        q2 = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
        parked = [t for t in q2["human_digest"] if "deadbroker" in _json.dumps(t)]
        assert len(parked) == 1 and "not_found" in _json.dumps(parked)
        # blocked / indirect_exposure dead cases are hidden from planning, so they close out too
        for st in ("blocked", "indirect_exposure"):
            led["deadbroker"] = {"broker_id": "deadbroker", "state": st}
            q3 = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
            assert [t for t in q3["human_digest"]
                    if "deadbroker" in _json.dumps(t) and "broker is defunct" in _json.dumps(t)], st


def test_full_autonomy_note_honors_confirm_first():
    # dnc_register always carries confirm_first; the full-mode note must not tell the agent
    # to skip that pause.
    with temp_env():
        sid = _run(["intake", "--full-name", "Jane Q. Public", "--email", "jane@example.com",
                    "--focus-phone", "+1-415-555-0137", "--consent"])["subject_id"]
        q = autopilot.next_actions(dossier.load(sid), [], _auto_cfg(), {}, env={})
        assert q["autonomy"] == "full"
        assert [a["confirm_first"] for a in q["actions"] if a["type"] == "dnc_register"] == [True]
        assert "EXCEPT on an action with confirm_first=true" in q["note"]


def test_defunct_cases_never_recheck_or_wake_even_on_priority_runs():
    # radaris is curated, crucial and defunct; a --priority high run filters it out of the
    # broker list, but its ledger case must still be treated as dead.
    with temp_env():
        d = _consenting()
        bl = brokers.by_priority("high")
        assert "radaris" not in {b["id"] for b in bl} and brokers.get("radaris").get("defunct")
        led = {"radaris": {"broker_id": "radaris", "state": "confirmed_removed",
                           "next_recheck_at": "2020-01-01T00:00:00Z"}}
        q = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
        assert not [a for a in q["actions"] if a.get("broker_id") == "radaris"]
        led["radaris"]["next_recheck_at"] = "2099-01-01T00:00:00Z"
        assert autopilot.next_actions(d, bl, _auto_cfg(), led, env={})["next_wake_at"] is None
        led["radaris"] = {"broker_id": "radaris", "state": "awaiting_processing"}
        q2 = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
        assert [t for t in q2["human_digest"]
                if "radaris" in _json.dumps(t) and "broker is defunct" in _json.dumps(t)]


def test_ledger_not_found_close_out_only_for_defunct_brokers():
    with temp_env():
        sid = _run(["intake", "--full-name", "Jane Q. Public", "--email", "jane@example.com",
                    "--consent"])["subject_id"]
        assert brokers.get("radaris").get("defunct") and not brokers.get("usphonebook").get("defunct")
        for bid in ("radaris", "usphonebook"):
            _run(["record", sid, bid, "found", "--found", f"https://{bid}.example/x"])
            assert _run(["record", sid, bid, "submitted"])["next_recheck_at"]
        # a LIVE broker's in-flight request cannot be closed as not_found ...
        try:
            _run(["record", sid, "usphonebook", "not_found"])
        except SystemExit as e:
            assert "illegal transition 'submitted' -> 'not_found'" in str(e.code)
        else:
            raise AssertionError("live broker submitted -> not_found must be refused")
        # ... a defunct one can, and the closed case schedules no further wake
        out = _run(["record", sid, "radaris", "not_found", "--reason", "broker defunct"])
        assert out["state"] == "not_found" and out["next_recheck_at"] is None
    # the defunct flag widens only in-flight -> not_found
    assert not ledger.can_transition("confirmed_removed", "not_found", broker_defunct=True)
    assert not ledger.can_transition("submitted", "found", broker_defunct=True)


def test_focus_phone_edge_cases():
    with temp_env():
        # focus-phone alone satisfies the identifier requirement
        out = _run(["intake", "--focus-phone", "+1-415-555-0137", "--consent"])
        assert out["phones"] == 1 and out["focus_phone"] == "+1-415-555-0137"
        # a differently-formatted duplicate is ONE number, not two vectors
        out2 = _run(["intake", "--phone", "555-123-4567", "--focus-phone", "5551234567",
                     "--consent"])
        assert out2["phones"] == 1
        # --json intake honors --focus-phone too (was silently ignored)
        import tempfile as _tf
        with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            _json.dump({"identity": {"full_name": "Json Subject", "emails": ["j@example.com"]},
                        "consent": {"authorized": True, "method": "self"},
                        "residency_jurisdiction": "US"}, f)
            jpath = f.name
        out3 = _run(["intake", "--json", jpath, "--focus-phone", "+1-510-555-0199"])
        assert out3["focus_phone"] == "+1-510-555-0199"
        d = dossier.load(out3["subject_id"])
        assert d["preferences"]["focus_phone"] == "+1-510-555-0199"
        assert "+1-510-555-0199" in d["identity"]["phones"]


def test_synthesize_steps_keep_gate_quirks_within_cap():
    r = {"optout_url": "https://x.example/optout",
         "optout_quirks": ["q1", "q2", "q3", "q4 GATE", "q5"]}
    steps = tiers.synthesize_steps(r)
    assert any("q4 GATE" in s for s in steps)        # the 4th quirk (a gate finding) survives
    assert any("+1 more quirk" in s for s in steps)  # truncation is disclosed, never silent


if __name__ == "__main__":
    failures = []
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
