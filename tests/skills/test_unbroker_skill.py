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

def test_config_defaults_are_easiest():
    with temp_env():
        cfg = config.load_config()
        assert cfg["email_mode"] == "draft_only"
        assert cfg["browser_backend"] == "auto"
        assert cfg["tracker_backend"] == "local-json"
        assert cfg["encryption"] == "none"


def test_config_roundtrip_and_validation():
    with temp_env():
        config.save_config({"email_mode": "programmatic"})
        assert config.load_config()["email_mode"] == "programmatic"
        try:
            config.save_config({"email_mode": "bogus"})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid email_mode should raise")


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

def test_encryption_off_writes_plaintext():
    with temp_env():
        d = _consenting()
        dossier.save(d)
        p = paths.dossier_path(d["subject_id"])
        assert p.exists() and not Path(str(p) + ".age").exists()


def test_encryption_age_round_trip():
    if not _AGE:
        return  # age not installed -> effectively skipped (keeps hermetic CI green)
    with temp_env():
        config.save_config({"encryption": "age"})
        crypto.ensure_identity()
        assert crypto.is_engaged()
        d = _consenting()
        dossier.save(d)
        plain = paths.dossier_path(d["subject_id"])
        enc = Path(str(plain) + ".age")
        assert enc.exists() and not plain.exists()          # only ciphertext on disk
        assert not enc.read_bytes().lstrip().startswith(b"{")  # not plaintext JSON
        assert dossier.load(d["subject_id"])["identity"]["full_name"] == "Jane Q. Public"


def test_encryption_keeps_config_and_audit_plaintext():
    if not _AGE:
        return
    with temp_env():
        config.save_config({"encryption": "age"})
        crypto.ensure_identity()
        # config.json must stay readable plaintext (crypto reads it to decide)
        assert config.load_config()["encryption"] == "age"
        assert not Path(str(paths.config_path()) + ".age").exists()
        # audit log holds field NAMES only, kept plaintext by design
        ledger.transition("sub_test01", "spokeo", "found", found=True)
        assert paths.audit_path("sub_test01").exists()


# --- broker DB ----------------------------------------------------------------

def test_seed_broker_db_loads_and_is_well_formed():
    everyone = brokers.load_all()
    assert len(everyone) >= 10
    ids = {b["id"] for b in everyone}
    assert {"spokeo", "whitepages", "mylife"} <= ids
    for b in everyone:
        assert b.get("id") and b.get("name") and b.get("priority") in {"crucial", "high", "standard", "long_tail"}
        assert (b.get("optout") or {}).get("method")


def test_clusters_expose_ownership():
    cl = brokers.clusters()
    assert "freepeopledirectory" in cl.get("spokeo", [])
    assert "peoplelooker" in cl.get("beenverified", [])


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


def test_email_verification_tier_shifts_with_mode():
    spokeo = brokers.get("spokeo")
    assert tiers.select_tier(spokeo, "draft_only") == "T2"
    assert tiers.select_tier(spokeo, "programmatic") == "T1"
    assert tiers.select_tier(spokeo, "alias") == "T1"


def test_captcha_tier_shifts_with_browser():
    tps = brokers.get("truepeoplesearch")
    assert tiers.select_tier(tps, "programmatic", browser_clears_captcha=False) == "T2"
    assert tiers.select_tier(tps, "programmatic", browser_clears_captcha=True) == "T1"


def test_hard_human_requirements_force_t3():
    assert tiers.select_tier(brokers.get("mylife")) == "T3"  # gov_id
    # thatsthem's opt-out is Cloudflare-Turnstile gated (captcha:true) -> T2 without a
    # captcha-clearing browser backend, T1 with one. (Corrected 2026-06-30 after the
    # live scan found the real form gated; the record previously mis-declared captcha:false.)
    assert tiers.select_tier(brokers.get("thatsthem")) == "T2"
    assert tiers.select_tier(brokers.get("thatsthem"), browser_clears_captcha=True) == "T1"


def test_plan_excludes_disallowed_fields():
    d = _consenting()
    actions = tiers.plan(d, brokers.load_all(), config.DEFAULT_CONFIG)
    for a in actions:
        assert "ssn" not in a["disclosure_fields"]
        assert "profile_url" not in a["disclosure_fields"]


def test_disclosure_maps_street_when_broker_requires_it():
    # thatsthem's opt-out form requires a street line; select_disclosure must surface it from
    # current_address.line1 (regression: 'street' was in broker inputs but unmapped, silently dropped).
    d = _consenting()
    d["identity"]["current_address"]["line1"] = "123 Main St"
    out = dossier.select_disclosure(d, ["full_name", "street", "city", "state", "postal"])
    assert out["street"] == "123 Main St"
    # and when there is no street on file, it is simply omitted (never a blank/placeholder)
    d2 = _consenting()
    out2 = dossier.select_disclosure(d2, ["full_name", "street", "city"])
    assert "street" not in out2


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


def test_batch_plan_collapses_ownership_clusters():
    # a parent that is being acted on (found/submitted/...) covers its children -> child dropped
    d = _consenting()
    bl = [_mini_broker("parent", owns=["kid"]), _mini_broker("kid")]
    ledger = {"parent": {"state": "found"}, "kid": {"state": "found"}}
    bp = tiers.batch_plan(d, bl, config.DEFAULT_CONFIG, ledger)
    assert bp["cluster_savings"] == {"parent": ["kid"]}
    # the child must NOT also appear as its own actionable 'found' row
    found_ids = [r["broker_id"] for r in bp["groups"]["found"]]
    assert "parent" in found_ids and "kid" not in found_ids


def test_batch_plan_orders_found_parents_first():
    # found group must be sorted parents-first, most-children-first, standalone last.
    d = _consenting()
    bl = [_mini_broker("standalone"),
          _mini_broker("smallparent", owns=["c1"]),
          _mini_broker("bigparent", owns=["c1b", "c2b", "c3b"])]
    ledger = {"standalone": {"state": "found"}, "smallparent": {"state": "found"},
              "bigparent": {"state": "found"}}
    bp = tiers.batch_plan(d, bl, config.DEFAULT_CONFIG, ledger)
    order = [r["broker_id"] for r in bp["groups"]["found"]]
    assert order == ["bigparent", "smallparent", "standalone"]
    # PHASE 2 tip spells out the parents-first order and points at the playbook
    phase2 = [t for t in bp["next_actions"] if "PHASE 2" in t]
    assert phase2 and "PARENTS FIRST" in phase2[0] and "bigparent -> smallparent" in phase2[0]


def test_parent_playbook_has_bespoke_and_synthesised_steps():
    d = _consenting()
    bespoke = _mini_broker("bespokeparent", owns=["truthfinder", "ussearch"])
    # bespoke steps live IN the broker record (optout.playbook), not in code
    bespoke["optout"]["playbook"] = ["Step one from the record", "SUPPRESSION != DELETION warning"]
    bl = [bespoke,
          _mini_broker("newparent", owns=["k1", "k2"],
                       requires={"profile_url": True, "email_verification": True},
                       notes="synth note", quirks=["q1"]),
          _mini_broker("standalone")]
    ledger = {b["id"]: {"state": "found"} for b in bl}
    bp = tiers.batch_plan(d, bl, config.DEFAULT_CONFIG, ledger)
    pb = {p["broker_id"]: p for p in bp["parent_playbook"]}
    # standalone (no children) is NOT in the playbook
    assert "standalone" not in pb
    # bespoke recipe comes verbatim from the record's own playbook
    assert pb["bespokeparent"]["steps"] == bespoke["optout"]["playbook"]
    # synthesised recipe: newparent reflects its requires-flags + notes + quirks
    steps = " ".join(pb["newparent"]["steps"])
    assert "profile_url" in steps and "verification" in steps.lower()
    assert "synth note" in steps and "q1" in steps
    # ordering is stamped on each entry, parents-first
    assert [p["order"] for p in bp["parent_playbook"]] == [1, 2]


def test_batch_plan_phase_is_delete_when_all_scanned():
    d = _consenting()
    bl = [_mini_broker("aaa"), _mini_broker("bbb")]
    ledger = {"aaa": {"state": "confirmed_removed"}, "bbb": {"state": "not_found"}}
    bp = tiers.batch_plan(d, bl, config.DEFAULT_CONFIG, ledger)
    assert bp["phase"] == "delete"          # nothing unscanned
    assert bp["counts"]["unscanned"] == 0
    assert bp["counts"]["done"] == 1


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


def test_new_can_record_scan_outcome_directly():
    with temp_env():
        assert ledger.transition("sub_test01", "thatsthem", "found", found=True)["state"] == "found"
        assert ledger.transition("sub_test01", "radaris", "not_found")["state"] == "not_found"
        # a scan that is bot-blocked on the very first hit must be recordable as blocked directly
        # (no need to pass through 'searching' first) -- and not_found -> blocked when a re-scan is gated
        assert ledger.transition("sub_test01", "spokeo", "blocked")["state"] == "blocked"
        assert ledger.transition("sub_test01", "radaris", "blocked")["state"] == "blocked"
        # a blocked site later scanned via the operator's own (residential) browser resolves to a
        # real verdict, incl. not_found -- blocked -> not_found must be legal.
        assert ledger.transition("sub_test01", "spokeo", "not_found")["state"] == "not_found"


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


def test_ledger_illegal_transition_raises():
    with temp_env():
        try:
            ledger.transition("sub_test01", "spokeo", "confirmed_removed")  # new -> confirmed_removed
        except ValueError:
            pass
        else:
            raise AssertionError("illegal transition should raise")


def test_ledger_disclosure_log():
    with temp_env():
        ledger.log_disclosure("sub_test01", "spokeo", ["full_name", "contact_email"], "web_form")
        case = ledger.get_case("sub_test01", "spokeo")
        assert case["disclosure_log"][0]["fields"] == ["contact_email", "full_name"]


# --- dossier / consent / least-disclosure ------------------------------------

def test_consent_gate():
    assert dossier.is_authorized(_consenting()) is True
    nope = _consenting()
    nope["consent"] = {"authorized": False, "method": "self"}
    assert dossier.is_authorized(nope) is False
    try:
        dossier.require_authorized(nope)
    except PermissionError:
        pass
    else:
        raise AssertionError("require_authorized should raise for non-consenting subject")


def test_least_disclosure_selection():
    d = _consenting()
    got = dossier.select_disclosure(d, ["full_name", "contact_email", "profile_url", "ssn", "date_of_birth"])
    assert set(got) == {"full_name", "contact_email", "date_of_birth"}
    assert "ssn" not in got and "profile_url" not in got


def test_designated_contact_email_overrides_first():
    d = _consenting()
    d["identity"]["emails"] = ["first@x.com", "alias@x.com"]
    assert dossier.contact_email(d) == "first@x.com"
    d["preferences"]["contact_email_for_optouts"] = "alias@x.com"
    assert dossier.contact_email(d) == "alias@x.com"


# --- alternates / search vectors ---------------------------------------------

def test_all_names_and_locations_dedupe():
    d = _consenting()
    d["identity"]["also_known_as"] = ["Jane Public", "Jane Q. Public"]   # 2nd dups primary
    d["identity"]["prior_addresses"] = [{"city": "Berkeley", "state": "CA"}, {"city": "Oakland", "state": "CA"}]
    assert dossier.all_names(d) == ["Jane Q. Public", "Jane Public"]
    assert [loc["city"] for loc in dossier.all_locations(d)] == ["Oakland", "Berkeley"]  # current first, deduped


def test_search_vectors_fan_out_across_alternates():
    d = _consenting()
    d["identity"]["also_known_as"] = ["Jane Smith"]
    d["identity"]["prior_addresses"] = [{"city": "Berkeley", "state": "CA"}]
    d["identity"]["emails"] = ["a@x.com", "b@y.com"]
    d["identity"]["phones"] = ["+1-415-555-0137", "+1-510-555-0199"]
    broker = {"id": "x", "search": {"by": ["name", "phone", "email", "address"]}}
    v = vectors.search_vectors(d, broker)
    assert len([x for x in v if x["by"] == "name"]) == 4   # 2 names x 2 locations
    assert len([x for x in v if x["by"] == "phone"]) == 2
    assert len([x for x in v if x["by"] == "email"]) == 2
    assert len([x for x in v if x["by"] == "address"]) == 0  # no street line1 yet


def test_search_vectors_respect_broker_capabilities():
    d = _consenting()
    d["identity"]["emails"] = ["a@x.com"]
    v = vectors.search_vectors(d, {"id": "y", "search": {"by": ["name"]}})
    assert v and all(x["by"] == "name" for x in v)   # broker can't search email -> no email vectors


def test_search_vectors_address_needs_line1():
    d = _consenting()
    d["identity"]["current_address"] = {"line1": "123 Main St", "city": "Oakland", "state": "CA", "postal": "94601"}
    v = vectors.search_vectors(d, {"id": "z", "search": {"by": ["address"]}})
    assert len(v) == 1 and v[0]["by"] == "address" and v[0]["query"]["line1"] == "123 Main St"


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


def test_fanout_default_batch_size_is_five():
    # Field report: 8-broker batches time out; the default dropped to 5.
    g = tiers.fanout([{"id": f"b{i}"} for i in range(12)])
    assert all(len(b) <= 5 for b in g["batches"])
    assert g["batches"][0] == [f"b{i}" for i in range(5)]
    assert len(g["batches"]) == 3  # 5 + 5 + 2


# --- cdp (operator browser over the DevTools protocol) --------------------------------------

def test_cdp_launch_command_has_debug_flags():
    cmd = cdp.launch_command("/usr/bin/chrome", port=9333, profile=Path("/tmp/prof"))
    assert cmd[0] == "/usr/bin/chrome"
    assert "--remote-debugging-port=9333" in cmd
    assert "--user-data-dir=/tmp/prof" in cmd
    assert "--no-first-run" in cmd


def test_cdp_default_profile_uses_hermes_home():
    prev = os.environ.get("HERMES_HOME")
    with tempfile.TemporaryDirectory() as d:
        os.environ["HERMES_HOME"] = d
        try:
            assert cdp.default_profile() == Path(d) / "chrome-debug"
        finally:
            if prev is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = prev


def test_cdp_endpoint_status_parses_live_and_handles_down():
    orig = cdp._http_get
    cdp._http_get = lambda url, timeout: b'{"Browser":"Chrome/1.2","webSocketDebuggerUrl":"ws://x"}'
    try:
        st = cdp.endpoint_status(port=9222)
        assert st and st["Browser"] == "Chrome/1.2" and st["webSocketDebuggerUrl"] == "ws://x"
    finally:
        cdp._http_get = orig

    def _boom(url, timeout):
        raise ConnectionError("connection refused")
    cdp._http_get = _boom
    try:
        assert cdp.endpoint_status(port=9222) is None   # nothing listening -> None, never raises
    finally:
        cdp._http_get = orig


def test_cdp_find_browser_override():
    assert cdp.find_browser("/bin/sh") == "/bin/sh"                       # explicit path that exists
    assert cdp.find_browser("definitely-not-a-real-browser-xyz") is None  # bogus -> None (no crash)


def test_plan_surfaces_antibot():
    d = _consenting()
    broker = {"id": "tps", "optout": {"requires": {}}, "search": {"antibot": "datadome", "by": ["name"]}}
    actions = tiers.plan(d, [broker], config.DEFAULT_CONFIG)
    assert actions[0]["antibot"] == "datadome"


def test_plan_prewarns_when_dob_required_but_missing():
    # requires.dob gated broker (e.g. PeopleConnect guided-mode): warn up front, not mid-flow.
    broker = {"id": "intelius", "search": {"by": ["name"]},
              "optout": {"requires": {"dob": True, "email_verification": True}, "inputs": ["contact_email"]}}
    no_dob = _consenting()
    no_dob["identity"].pop("date_of_birth")
    warned = tiers.plan(no_dob, [broker], config.DEFAULT_CONFIG)[0]
    assert any("date_of_birth" in w for w in warned["needs_operator_input"])
    # A new requires key must not perturb tier selection.
    assert warned["tier"] == tiers.select_tier(
        {"optout": {"requires": {"email_verification": True}}}, "draft_only")
    with_dob = tiers.plan(_consenting(), [broker], config.DEFAULT_CONFIG)[0]
    assert with_dob["needs_operator_input"] == []


def test_plan_surfaces_optout_quirks_and_email():
    d = _consenting()
    broker = {"id": "radaris", "search": {"by": ["name"]},
              "optout": {"requires": {}, "email": "x@broker.test", "quirks": ["no profile URL -> email fallback"]}}
    a = tiers.plan(d, [broker], config.DEFAULT_CONFIG)[0]
    assert a["optout_email"] == "x@broker.test"
    assert a["optout_quirks"] == ["no profile URL -> email fallback"]


# --- legal / templates --------------------------------------------------------

def test_legal_render_keeps_missing_placeholders_literal():
    out = legal.render("emails/generic-optout.txt", {"broker_name": "Spokeo"})
    assert "Spokeo" in out
    assert "{full_name}" in out  # missing field left literal, never blank-injected


def test_render_optout_email_includes_listing_and_name():
    b = brokers.get("spokeo")
    out = legal.render_optout_email(b, {"full_name": "Jane Q. Public",
                                        "contact_email": "jane@example.com",
                                        "listing_urls": ["https://www.spokeo.com/jane"]})
    assert "Jane Q. Public" in out and "https://www.spokeo.com/jane" in out


def test_render_ccpa_indirect_request_names_only_own_identifiers():
    b = brokers.get("thatsthem")
    out = legal.render_request("ccpa_indirect", b, {
        "full_name": "Jane Q. Public",
        "contact_email": "jane@example.com",
        "my_identifiers": ["jane@example.com", 'the name "Jane Q. Public" where it appears as a relative'],
        "listing_urls": ["https://thatsthem.com/email/jane@example.com"],
    })
    # the request must frame this as the subject's OWN data on someone else's record
    assert "not the primary subject" in out
    assert "jane@example.com" in out
    assert "https://thatsthem.com/email/jane@example.com" in out
    # must NOT use the full-opt-out wording that claims the record is about the subject
    assert "DELETE all personal information you hold about me" not in out


# --- email verification-link extraction --------------------------------------

def test_extract_verification_link_prefers_broker_optout_link():
    body = ("Hello,\nClick https://www.spokeo.com/optout/confirm?token=abc to confirm.\n"
            "Unrelated: https://ads.example/promo\n")
    link = email_modes.extract_verification_link(body, brokers.get("spokeo"))
    assert link is not None and "spokeo.com" in link and "ads.example" not in link


def test_extract_verification_link_ignores_unrelated_only():
    assert email_modes.extract_verification_link("see https://example.com/news today") is None


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


def test_badbool_symbols_map_to_requirements_and_tiers():
    recs = {r["id"]: r for r in badbool.parse(BADBOOL_FIXTURE)}
    assert recs["mylife"]["optout"]["requires"]["phone_voice"] is True
    assert recs["mylife"]["optout"]["method"] == "phone"
    assert tiers.select_tier(recs["mylife"]) == "T3"
    assert recs["pimeyes"]["optout"]["requires"]["gov_id"] is True
    assert tiers.select_tier(recs["pimeyes"]) == "T3"


def test_badbool_merge_keeps_curated_and_adds_new():
    with temp_env():
        badbool.refresh(__import__("paths").brokers_cache_path(), markdown=BADBOOL_FIXTURE)
        merged = {b["id"]: b for b in brokers.load_all()}
        # curated record wins over the live one
        assert merged["beenverified"]["source"] == "BADBOOL"
        # a non-curated live record is added with auto confidence
        assert "pimeyes" in merged and merged["pimeyes"]["confidence"] == "auto"


# --- report -------------------------------------------------------------------

def test_status_counts_and_markdown():
    with temp_env():
        sid = "sub_test01"
        ledger.transition(sid, "spokeo", "searching")
        ledger.transition(sid, "spokeo", "found")
        ledger.transition(sid, "thatsthem", "searching")
        ledger.transition(sid, "thatsthem", "not_found")
        counts = report.status_counts(sid)
        assert counts.get("found") == 1 and counts.get("not_found") == 1
        md = report.render_markdown(sid)
        assert "status for" in md and "Count" in md


# --- autonomy: auto-configure ---------------------------------------------------------------

def test_autonomy_default_is_full_and_valid():
    with temp_env():
        assert config.load_config()["autonomy"] == "full"
        config.save_config({"autonomy": "assisted"})
        assert config.load_config()["autonomy"] == "assisted"
        try:
            config.save_config({"autonomy": "yolo"})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid autonomy should raise")


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

def test_emailer_settings_inference_and_floor():
    assert emailer.smtp_settings(env={}) is None
    assert emailer.imap_settings(env={}) is None
    env = {"EMAIL_ADDRESS": "a@gmail.com", "EMAIL_PASSWORD": "p"}
    assert emailer.smtp_settings(env)["host"] == "smtp.gmail.com"
    assert emailer.smtp_settings(env)["port"] == 587
    assert emailer.imap_settings(env)["host"] == "imap.gmail.com"
    assert emailer.imap_settings(env)["port"] == 993
    # unknown provider without an explicit host -> NOT configured (never guess blind)
    corp = {"EMAIL_ADDRESS": "a@corp.example", "EMAIL_PASSWORD": "p"}
    assert emailer.smtp_settings(corp) is None
    s = emailer.smtp_settings({**corp, "EMAIL_SMTP_HOST": "mail.corp.example",
                               "EMAIL_SMTP_PORT": "465"})
    assert (s["host"], s["port"]) == ("mail.corp.example", 465)


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


def test_emailer_send_requires_config_and_broker_address():
    broker = {"id": "x", "optout": {"email": "privacy@x.example"}}
    try:
        emailer.send(broker, "Subject: s\n\nb", env={})
    except RuntimeError:
        pass
    else:
        raise AssertionError("unconfigured SMTP must raise (draft fallback, not a crash)")
    try:
        emailer.send({"id": "y", "optout": {}}, "Subject: s\n\nb",
                     env={"EMAIL_ADDRESS": "a@gmail.com", "EMAIL_PASSWORD": "p"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("broker without a declared address must raise")


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


def test_browser_email_mode_is_autonomous_without_smtp_or_imap():
    with temp_env():
        assert config.save_config({"email_mode": "browser"})  # mode is valid + persists
        d = _consenting()
        d["residency_jurisdiction"] = "US-CA"
        mailer = _mini_broker("mailer")
        mailer["optout"]["method"] = "email"
        mailer["optout"]["email"] = "privacy@mailer.example"
        verifier = _mini_broker("verifier", requires={"email_verification": True})
        led = {"mailer": {"state": "found"},
               "verifier": {"broker_id": "verifier", "state": "submitted"}}
        # browser mode with NO EMAIL_* creds -> still fully autonomous (agent uses webmail)
        q = autopilot.next_actions(d, [mailer, verifier], _auto_cfg(email_mode="browser"), led, env={})
        sends = [a for a in q["actions"] if a["type"] == "optout_email_send"]
        assert sends and sends[0]["send_via"] == "browser" and sends[0]["to"] == "privacy@mailer.example"
        polls = [a for a in q["actions"] if a["type"] == "poll_verification"]
        assert polls and polls[0]["via"] == "browser"
        assert not q["human_digest"]        # browser mode needs no human for these


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

def test_verification_pending_to_awaiting_processing_is_legal():
    with temp_env():
        sid = "sub_test01"
        ledger.transition(sid, "intelius", "found", found=True)
        ledger.transition(sid, "intelius", "submitted")
        ledger.transition(sid, "intelius", "verification_pending")
        assert ledger.transition(sid, "intelius", "awaiting_processing")["state"] == "awaiting_processing"


def test_followup_stamps_and_due_queue():
    broker = {"optout": {"est_processing_days": 10}}
    d = {"preferences": {"rescan_interval_days": 30}}
    f_sub = ledger.followup_fields("submitted", broker, d)
    assert "next_recheck_at" in f_sub
    f_done = ledger.followup_fields("confirmed_removed", broker, d)
    assert "removal_confirmed_at" in f_done
    assert f_done["next_recheck_at"] > f_sub["next_recheck_at"]  # 30d rescan > 10d processing
    assert ledger.followup_fields("found", broker, d) == {}      # scan verdicts get no stamp
    led = {
        "a": {"broker_id": "a", "state": "awaiting_processing", "next_recheck_at": "2000-01-01T00:00:00Z"},
        "b": {"broker_id": "b", "state": "confirmed_removed", "next_recheck_at": "2999-01-01T00:00:00Z"},
    }
    assert [c["broker_id"] for c in ledger.due("sub_x", ledger=led)] == ["a"]


def test_badbool_auto_records_have_processing_estimate():
    recs = badbool.parse("## People Search Sites\n### Example\n[opt out](https://example.com/optout)\n")
    assert recs[0]["optout"]["est_processing_days"] == 14  # drives next_recheck_at for live records


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


def test_next_actions_fanout_above_threshold():
    with temp_env():
        d = _consenting()
        bl = [_mini_broker(f"b{i:02d}") for i in range(12)]
        q = autopilot.next_actions(d, bl, _auto_cfg(), {}, env={})
        assert any(a["type"] == "fanout_scan" for a in q["actions"])


def test_next_actions_routes_human_only_to_digest():
    with temp_env():
        d = _consenting()
        t3 = _mini_broker("faxer", requires={"fax": True})
        cb = _mini_broker("callbacker", requires={"phone_callback": True})
        led = {"faxer": {"state": "found"}, "callbacker": {"state": "found"}}
        q = autopilot.next_actions(d, [t3, cb], _auto_cfg(), led, env={})
        assert not any(a["type"].startswith("optout") for a in q["actions"])
        reasons = " ".join(t["reason"] for t in q["human_digest"])
        assert "human-only" in reasons and "phone-callback" in reasons


def test_next_actions_email_send_vs_draft_digest():
    with temp_env():
        d = _consenting()
        b = _mini_broker("mailer")
        b["optout"]["method"] = "email"
        b["optout"]["email"] = "privacy@mailer.example"
        led = {"mailer": {"state": "found"}}
        env = {"EMAIL_ADDRESS": "agent@gmail.com", "EMAIL_PASSWORD": "p"}
        q = autopilot.next_actions(d, [b], _auto_cfg(email_mode="programmatic"), led, env=env)
        assert any(a["type"] == "optout_email_send" for a in q["actions"])
        # draft mode: same case becomes a digest entry with the render command as agent prep
        q2 = autopilot.next_actions(d, [b], _auto_cfg(), led, env={})
        assert not any(a["type"] == "optout_email_send" for a in q2["actions"])
        assert any("render-email" in " ".join(t["agent_prep"]) for t in q2["human_digest"])


def test_next_actions_poll_verification_and_due_rechecks():
    with temp_env():
        d = _consenting()
        b = _mini_broker("verifier", requires={"email_verification": True})
        led = {
            "verifier": {"broker_id": "verifier", "state": "submitted"},
            "done1": {"broker_id": "done1", "state": "confirmed_removed",
                      "next_recheck_at": "2000-01-01T00:00:00Z"},
        }
        env = {"EMAIL_ADDRESS": "agent@gmail.com", "EMAIL_PASSWORD": "p"}
        q = autopilot.next_actions(d, [b, _mini_broker("done1")],
                                   _auto_cfg(email_mode="programmatic"), led, env=env)
        types = [a["type"] for a in q["actions"]]
        assert "poll_verification" in types and "verify_removal" in types
        # without IMAP, the verification click becomes a human digest entry instead
        q2 = autopilot.next_actions(d, [b], _auto_cfg(),
                                    {"verifier": {"broker_id": "verifier", "state": "submitted"}}, env={})
        assert not any(a["type"] == "poll_verification" for a in q2["actions"])
        assert any("verification email" in t["reason"] for t in q2["human_digest"])


def test_next_actions_blocked_stealth_or_operator_browser():
    with temp_env():
        d = _consenting()
        b = _mini_broker("gated")
        led = {"gated": {"state": "blocked"}}
        q = autopilot.next_actions(d, [b], _auto_cfg(), led, env={"BROWSERBASE_API_KEY": "bb"})
        assert any(a["type"] == "stealth_rescan" for a in q["actions"])
        q2 = autopilot.next_actions(d, [b], _auto_cfg(), led, env={})
        assert any("anti-bot" in t["reason"] for t in q2["human_digest"])


def test_assisted_mode_flags_confirm_first():
    with temp_env():
        d = _consenting()
        b = _mini_broker("solo")
        led = {"solo": {"state": "found"}}
        q = autopilot.next_actions(d, [b], _auto_cfg(autonomy="assisted"), led, env={})
        opt = [a for a in q["actions"] if a["type"] == "optout_web_form"]
        assert opt and all(a["confirm_first"] for a in opt)
        q2 = autopilot.next_actions(d, [b], _auto_cfg(), led, env={})
        assert all(not a["confirm_first"] for a in q2["actions"] if a["type"] == "optout_web_form")


def test_next_actions_refresh_then_done_flags():
    with temp_env():
        d = _consenting()
        bl = [_mini_broker("solo")]
        led = {"solo": {"state": "not_found"}}
        q = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
        assert any(a["type"] == "refresh_brokers" for a in q["actions"])  # no cache yet
        assert q["done_for_now"] is False
        storage.write_json(paths.brokers_cache_path(), [])  # fresh cache
        q2 = autopilot.next_actions(d, bl, _auto_cfg(), led, env={})
        assert q2["actions"] == []
        assert q2["done_for_now"] and q2["fully_done"]


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

def test_cluster_parents_have_playbook_and_deletion_lane():
    """Contract: every curated cluster parent must know EXACTLY how to remove the data.

    A parent record (owns children) must carry a non-empty field-verified optout.playbook
    and a structured deletion lane -- deletion beats suppression, and the knowledge lives
    in the record, not in code.
    """
    for b in brokers._load_curated():
        if not b.get("owns"):
            continue
        opt = b.get("optout") or {}
        bid = b["id"]
        assert opt.get("playbook"), f"{bid}: cluster parent missing optout.playbook"
        d = opt.get("deletion") or {}
        assert d.get("email") or d.get("via"), f"{bid}: cluster parent missing deletion lane"
        # every declared email must be a legal send-email recipient
        for addr in [opt.get("email"), d.get("email")]:
            if addr:
                assert addr in emailer.broker_addresses(b), f"{bid}: {addr} not sendable"


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


def test_deletion_prefer_flag_controls_autopilot_note():
    with temp_env():
        d = _consenting()
        pc = _mini_broker("pc", owns=["kid"])
        pc["optout"]["deletion"] = {"via": "in_flow", "prefer": False,
                                    "email": "privacy@pc.example", "notes": "delete undoes suppression"}
        q = autopilot.next_actions(d, [pc, _mini_broker("kid")], _auto_cfg(), {"pc": {"state": "found"}}, env={})
        act = next(a for a in q["actions"] if a.get("broker_id") == "pc" and a["type"] == "optout_web_form")
        assert "prefer_suppression" in act and "prefer_deletion" not in act
        dd = _mini_broker("dd")
        dd["optout"]["deletion"] = {"via": "email_followup", "email": "p@dd.example"}
        q2 = autopilot.next_actions(d, [dd], _auto_cfg(), {"dd": {"state": "found"}}, env={})
        act2 = next(a for a in q2["actions"] if a["type"] == "optout_web_form")
        assert "prefer_deletion" in act2 and "prefer_suppression" not in act2


def test_curated_whitepages_email_lane_is_autonomous():
    """The verified Whitepages pattern: privacyrequest@ bypasses the phone-callback tool."""
    b = brokers.get("whitepages")
    opt = b["optout"]
    assert opt["method"] == "email"
    assert opt["email"] == "privacyrequest@whitepages.com"
    assert opt["requires"]["phone_callback"] is False   # the callback is only the ALT tool
    # programmatic email -> fully automated (T1); draft mode -> needs a human for the verify loop
    assert tiers.select_tier(b, email_mode="programmatic") == "T1"
    assert tiers.select_tier(b, email_mode="draft_only") == "T2"


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


def test_email_lane_routing_and_rescue():
    with temp_env():
        d = _consenting()
        d["residency_jurisdiction"] = "US-CA"
        env = {"EMAIL_ADDRESS": "agent@gmail.com", "EMAIL_PASSWORD": "p"}

        # (a) primary email method -> email send action with residency-correct kind
        mailer = _mini_broker("mailer")
        mailer["optout"]["method"] = "email"
        mailer["optout"]["email"] = "privacy@mailer.example"
        # (b) RESCUE: T3 (gov_id) form but a deletion email exists (no via preference) ->
        # email lane instead of the human digest
        hard = _mini_broker("hardsite", requires={"gov_id": True})
        hard["optout"]["deletion"] = {"email": "privacy@hardsite.example",
                                      "kinds": ["ccpa", "generic"]}
        # (c) phone-callback form with deletion email -> email lane too
        cb = _mini_broker("callback2", requires={"phone_callback": True})
        cb["optout"]["deletion"] = {"email": "privacy@callback2.example"}
        led = {b: {"state": "found"} for b in ("mailer", "hardsite", "callback2")}
        q = autopilot.next_actions(d, [mailer, hard, cb],
                                   _auto_cfg(email_mode="programmatic"), led, env=env)
        sends = {a["broker_id"]: a for a in q["actions"] if a["type"] == "optout_email_send"}
        assert set(sends) == {"mailer", "hardsite", "callback2"}
        assert sends["mailer"]["kind"] == "ccpa"                     # CA resident
        assert sends["hardsite"]["to"] == "privacy@hardsite.example"
        assert "rescue" in sends["hardsite"]["why"]
        assert not q["human_digest"]                                 # nothing left for a human

        # without SMTP the same brokers fall back honestly: email draft digest / human digest
        q2 = autopilot.next_actions(d, [mailer, hard, cb], _auto_cfg(), led, env={})
        assert not any(a["type"] == "optout_email_send" for a in q2["actions"])
        assert len(q2["human_digest"]) == 3


def test_send_email_accepts_deletion_lane_recipient():
    env = {"EMAIL_ADDRESS": "agent@gmail.com", "EMAIL_PASSWORD": "p"}
    broker = {"id": "hardsite",
              "optout": {"deletion": {"email": "privacy@hardsite.example"}}}
    _FakeSMTP.sent = []
    out = emailer.send(broker, "Subject: Delete my data\n\nBody", env=env, _smtp_factory=_FakeSMTP)
    assert out["to"] == "privacy@hardsite.example"


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


def test_registry_parses_ca_csv():
    recs = registry.parse(_registry_csv())
    assert len(recs) == 2
    assert len({r["id"] for r in recs}) == 2                 # unique ids
    acme = next(r for r in recs if "acme" in r["id"])
    cbc = next(r for r in recs if "cbc" in r["id"] or "credit" in r["id"])
    assert acme["optout"]["method"] == "email"
    assert acme["optout"]["email"] == "privacy@acme.example"
    assert acme["optout"]["deletion"]["via"] == "drop"       # worked via DROP, not scanning
    assert acme["confidence"] == "registry"
    assert acme["category"] == "data_broker"
    assert acme["optout"]["fcra"] is False and cbc["optout"]["fcra"] is True


def test_registry_refresh_isolated_from_people_search():
    with temp_env():
        res = registry.refresh(paths.registry_cache_path(), csv_text=_registry_csv())
        assert res["parsed"] == 2 and res["fcra_regulated"] == 1
        reg_ids = {r["id"] for r in brokers.load_registry_cache()}
        assert len(reg_ids) == 2
        # CRITICAL: registry brokers must NOT leak into the people-search scan pipeline
        assert reg_ids.isdisjoint({b["id"] for b in brokers.load_all()})


def test_registry_multi_source_framework():
    # generic parser works for a non-CA state (proving multi-source, not CA-hardcoded)
    vt = registry.parse(_registry_csv(), jurisdiction="US-VT", has_drop=False)
    assert vt[0]["jurisdictions"] == ["US-VT"]
    assert vt[0]["source"] == "VT-registry"
    assert vt[0]["optout"]["deletion"]["via"] == "email"      # no DROP outside CA
    assert "no one-shot" in vt[0]["optout"]["deletion"]["notes"].lower()
    # VT/OR/TX are surfaced as portals with official URLs (not fabricated rows)
    ports = {p["jurisdiction"]: p for p in registry.portals()}
    assert set(ports) == {"US-VT", "US-OR", "US-TX"}
    assert all(p["url"].startswith("http") for p in ports.values())


def test_registry_refresh_all_ingests_csv_and_lists_portals():
    with temp_env():
        res = registry.refresh_all(paths.registry_cache_path(), fetched={"ca": _registry_csv()})
        assert res["total"] == 2
        assert res["sources"]["ca"]["parsed"] == 2 and res["sources"]["ca"]["added_after_dedupe"] == 2
        assert res["sources"]["vt"]["format"] == "portal"     # no bulk export, surfaced as portal
        assert len(res["portals"]) == 3
        assert len(brokers.load_registry_cache()) == 2


def test_next_surfaces_drop_for_ca_resident_only():
    with temp_env():
        registry.refresh(paths.registry_cache_path(), csv_text=_registry_csv())
        bl = [_mini_broker("solo")]

        ca = _consenting()
        ca["residency_jurisdiction"] = "US-CA"
        q = autopilot.next_actions(ca, bl, _auto_cfg(), {}, env={})
        assert any(a["type"] == "drop_submit" for a in q["actions"])
        assert q["coverage"]["registered_data_brokers"] == 2
        assert q["coverage"]["worked_via"] == "CA DROP one-shot"

        tx = _consenting()
        tx["residency_jurisdiction"] = "US-TX"
        q2 = autopilot.next_actions(tx, bl, _auto_cfg(), {}, env={})
        assert not any(a["type"] == "drop_submit" for a in q2["actions"])
        assert q2["coverage"]["worked_via"] == "targeted CCPA/GDPR email"

        ca["preferences"]["drop_filed_at"] = "2026-01-01T00:00:00Z"
        q3 = autopilot.next_actions(ca, bl, _auto_cfg(), {}, env={})
        assert not any(a["type"] == "drop_submit" for a in q3["actions"])


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


def test_email_rate_limit_paces_sends():
    with temp_env() as data:
        state = data / "rate.json"
        slept, now = [], [1000.0]
        emailer._respect_rate_limit(20, lambda s: slept.append(s), lambda: now[0], state)
        assert slept == []            # first send: nothing to wait for
        now[0] = 1005.0               # only 5s later
        emailer._respect_rate_limit(20, lambda s: slept.append(s), lambda: now[0], state)
        assert slept and abs(slept[0] - 15) < 0.01   # waited the remaining 15s of the 20s window


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


def test_email_send_retries_transient_then_succeeds():
    _FlakySMTP.attempts = 0
    env = {"EMAIL_ADDRESS": "agent@gmail.com", "EMAIL_PASSWORD": "p"}
    broker = {"id": "x", "optout": {"email": "privacy@x.example"}}
    out = emailer.send(broker, "Subject: s\n\nb", env=env, _smtp_factory=_FlakySMTP,
                       _sleep=lambda *_: None)
    assert out["attempts"] == 3 and "delivery_note" in out


def test_email_send_does_not_retry_permanent_error():
    env = {"EMAIL_ADDRESS": "agent@gmail.com", "EMAIL_PASSWORD": "p"}
    broker = {"id": "x", "optout": {"email": "privacy@x.example"}}
    try:
        emailer.send(broker, "Subject: s\n\nb", env=env, _smtp_factory=_AuthFailSMTP,
                     _sleep=lambda *_: None)
    except _smtplib.SMTPAuthenticationError:
        pass
    else:
        raise AssertionError("auth failure must raise immediately, not retry")


def _run(argv) -> dict:
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        pdd.main(argv)
    return _json.loads(buf.getvalue())


def test_send_email_is_idempotent_browser_mode():
    with temp_env():
        config.save_config({"email_mode": "browser"})
        sid = _run(["intake", "--full-name", "Jane Q. Public",
                    "--email", "jane@example.com", "--consent"])["subject_id"]
        _run(["record", sid, "radaris", "found", "--found", "true"])
        first = _run(["send-email", sid, "radaris", "--listing", "https://radaris.com/p/x"])
        assert first.get("state") == "submitted" and first.get("send_via") == "browser"
        again = _run(["send-email", sid, "radaris", "--listing", "https://radaris.com/p/x"])
        assert again.get("skipped") is True         # not re-sent


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


def test_dotenv_env_fills_missing_creds_and_shell_wins():
    prev_home = os.environ.get("HERMES_HOME")
    prev_key = os.environ.get("BROWSERBASE_API_KEY")
    with tempfile.TemporaryDirectory() as d:
        os.environ["HERMES_HOME"] = d
        (Path(d) / ".env").write_text(
            '# comment\nBROWSERBASE_API_KEY="from_dotenv"\nFIRECRAWL_API_KEY=fc_123\n', encoding="utf-8")
        try:
            os.environ.pop("BROWSERBASE_API_KEY", None)
            merged = config.dotenv_env()
            assert merged["BROWSERBASE_API_KEY"] == "from_dotenv"   # filled from .env
            assert merged["FIRECRAWL_API_KEY"] == "fc_123"          # quotes/comment handled
            os.environ["BROWSERBASE_API_KEY"] = "from_shell"
            assert config.dotenv_env()["BROWSERBASE_API_KEY"] == "from_shell"  # shell wins
        finally:
            for k, v in (("HERMES_HOME", prev_home), ("BROWSERBASE_API_KEY", prev_key)):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_cdp_cli_check_reports_not_running():
    orig = cdp.endpoint_status
    cdp.endpoint_status = lambda *a, **k: None
    try:
        out = _run(["cdp", "--check", "--port", "59981"])
        assert out["running"] is False and out["endpoint"].endswith(":59981")
    finally:
        cdp.endpoint_status = orig


def test_cdp_cli_detects_already_running_and_does_not_launch():
    # If a debug browser is already live, `cdp` must report it and NOT launch another.
    orig_status, orig_launch = cdp.endpoint_status, cdp.launch
    cdp.endpoint_status = lambda *a, **k: {"Browser": "Chrome/9", "webSocketDebuggerUrl": "ws://z"}

    def _no_launch(*a, **k):
        raise AssertionError("launch() must not be called when a browser is already live")
    cdp.launch = _no_launch
    try:
        out = _run(["cdp", "--port", "59982"])
        assert out["running"] is True and out["webSocketDebuggerUrl"] == "ws://z"
    finally:
        cdp.endpoint_status, cdp.launch = orig_status, orig_launch


def test_registry_candidate_urls_newest_first_with_floor():
    urls = registry.ca_candidate_urls(__import__("datetime").date(2027, 3, 1))
    assert urls[0].endswith("registry2027.csv") and urls[-1].endswith("registry2025.csv")
    assert registry.ca_candidate_urls(__import__("datetime").date(2024, 1, 1))[0].endswith("registry2025.csv")


def test_registry_and_badbool_warn_on_too_few():
    with temp_env():
        res = registry.refresh_all(paths.registry_cache_path(), fetched={"ca": _registry_csv()})
        assert "warning" in res["sources"]["ca"]            # 2 parsed < MIN_EXPECTED_CA
        md = "## People Search Sites\n### One\n[opt out](https://one.example/optout)\n"
        bres = badbool.refresh(paths.brokers_cache_path(), markdown=md)
        assert bres["parsed"] == 1 and "warning" in bres


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
