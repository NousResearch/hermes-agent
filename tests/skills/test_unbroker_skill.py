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
import dpa as dpa_mod   # noqa: E402
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


# --- GDPR / EU jurisdiction ----------------------------------------------------

def test_render_gdpr_erasure_includes_all_required_citations():
    """The GDPR erasure template must cite Art. 17, Art. 21, Charter Art. 8, and the
    30-day Art. 12(3) deadline. These are the legally-load-bearing references without
    which the request is weaker than a generic opt-out letter."""
    b = brokers.get("spokeo")
    out = legal.render_request("gdpr", b, {
        "full_name": "Jane Q. Public",
        "contact_email": "jane@example.com",
        "listing_urls": ["https://www.spokeo.com/jane"],
    })
    assert "Article 17" in out
    assert "Article 21" in out
    assert "Article 8" in out  # Charter of Fundamental Rights
    assert "30 days" in out or "Article 12(3)" in out  # statutory deadline


def test_render_gdpr_erasure_keeps_missing_placeholders_literal():
    """Missing fields must be left literal (never blank-injected). Same contract as
    the CCPA / generic templates; the contract lives in legal._SafeDict, not per-template."""
    b = brokers.get("spokeo")
    out = legal.render_request("gdpr", b, {"broker_name": "Spokeo"})
    assert "Spokeo" in out
    assert "{full_name}" in out  # missing field left literal


def test_render_gdpr_art21_only_omits_erasure_clause():
    """gdpr_art21_only is for brokers with a strong legitimate-interest claim where
    Art. 21 objection is the cleaner primary weapon. The template must NOT lead with
    erasure — it leads with objection and only mentions erasure as a fallback."""
    out = legal.render_request("gdpr_art21_only", {}, {
        "broker_name": "Some Broker",
        "full_name": "Jane",
        "contact_email": "jane@example.com",
        "listing_urls": ["https://example.com/jane"],
    })
    # the leading paragraph must be the objection
    first_para = out.split("\n\n", 1)[0]
    assert "Article 21" in first_para
    # Art. 17 erasure is a fallback here, mentioned later
    assert "Article 17" in out


def test_render_gdpr_indirect_deletion_names_only_own_identifiers():
    """Mirror of test_render_ccpa_indirect_request_names_only_own_identifiers — the GDPR
    indirect template must frame this as erasure of the subject's OWN data on someone
    else's record, not as erasure of the record itself."""
    b = brokers.get("thatsthem")
    out = legal.render_request("gdpr_indirect", b, {
        "full_name": "Jane Q. Public",
        "contact_email": "jane@example.com",
        "my_identifiers": ["jane@example.com", 'the name "Jane Q. Public" where it appears as a relative'],
        "listing_urls": ["https://thatsthem.com/email/jane@example.com"],
    })
    # must frame this as the subject's own data on someone else's record
    assert "NOT the primary subject" in out or "another individual" in out.lower()
    assert "jane@example.com" in out
    assert "https://thatsthem.com/email/jane@example.com" in out
    # must NOT use the full-erasure wording that claims the record is about the subject
    assert "erasure of all personal data relating to me that you currently process" not in out.lower() or "indirect" in out.lower()


def test_render_request_dispatch_supports_all_gdpr_kinds():
    """Behavior contract: every gdpr_* kind listed in the dispatch dict must render
    without KeyError. Catches typos in the template-name map at lint time."""
    b = brokers.get("spokeo")
    for kind in ("gdpr", "gdpr_art21_only", "gdpr_indirect"):
        out = legal.render_request(kind, b, {
            "full_name": "Jane Q. Public",
            "contact_email": "jane@example.com",
            "listing_urls": ["https://example.com/x"],
            "my_identifiers": ["jane@example.com"],
        })
        assert "Jane Q. Public" in out
        assert "Article 17" in out or "Article 21" in out


def test_legal_framework_us_codes_map_to_ccpa():
    """US residency codes must still map to the ccpa framework (no regression — existing
    US users must keep their CCPA / DROP pipeline)."""
    for code in ("US", "US-CA", "US-NY", "US-VT", "US-OR", "US-TX"):
        meta = dossier.legal_framework(code)
        assert meta["framework"].startswith("ccpa"), f"{code} -> {meta['framework']}"
        assert meta["default_request_kind"] == "ccpa"


def test_legal_framework_eu_codes_map_to_gdpr():
    """Every configured EU/EEA/UK residency must use a GDPR request kind."""
    for code in dossier.RESIDENCY_LEGAL_FRAMEWORK:
        if code == "UK" or code.startswith(("EU-", "EEA-")):
            meta = dossier.legal_framework(code)
            assert meta["framework"] in ("gdpr", "uk_gdpr"), f"{code} -> {meta['framework']}"
            assert meta["default_request_kind"] == "gdpr"


def test_legal_framework_generic_eu_falls_back_without_dpa():
    """The catch-all `EU` code (used when the subject didn't specify a member state)
    must still map to GDPR but with no specific DPA — the subject must declare one later."""
    meta = dossier.legal_framework("EU")
    assert meta["framework"] == "gdpr"
    assert meta["dpa"] is None
    assert meta["default_request_kind"] == "gdpr"


def test_legal_framework_unknown_residency_falls_back_to_generic():
    """An unknown residency code must NOT crash — the subject can still try with a generic
    right-to-delete request. Better than locking them out for a typo."""
    meta = dossier.legal_framework("ZZ")  # not a real code
    assert meta["framework"] == "generic"
    assert meta["default_request_kind"] == "generic"
    assert meta["dpa"] is None


def test_is_eu_residency_true_for_eu_and_uk():
    """Behavior contract: is_eu_residency() is the single source of truth for the
    'should this subject use the GDPR pipeline?' question."""
    assert dossier.is_eu_residency("EU-IT")
    assert dossier.is_eu_residency("EU-FR")
    assert dossier.is_eu_residency("UK")
    assert dossier.is_eu_residency("EU")
    assert dossier.is_eu_residency("EU-EEA")
    assert dossier.is_eu_residency("EEA-NO")


def test_is_eu_residency_false_for_us_and_unknown():
    """Mirror of the true case — US codes must NOT be classified as EU."""
    assert not dossier.is_eu_residency("US")
    assert not dossier.is_eu_residency("US-CA")
    assert not dossier.is_eu_residency("ZZ")


def test_config_default_jurisdiction_is_auto():
    """The new default_jurisdiction setting must default to 'auto' — the inference
    path — so existing installs pick up the new behaviour without manual config edits."""
    with temp_env():
        cfg = config.load_config()
        assert cfg.get("default_jurisdiction") == "auto"


def test_config_default_jurisdiction_validates_known_values():
    """save_config must accept the documented jurisdiction values; anything else fails
    loudly (better than silently writing a typo into the config file)."""
    with temp_env():
        for value in ("auto", "us", "eu", "uk", "generic"):
            cfg = dict(config.DEFAULT_CONFIG)
            cfg["default_jurisdiction"] = value
            config.save_config(cfg)  # must not raise


def test_config_default_jurisdiction_rejects_unknown_values():
    """An invalid jurisdiction must be rejected — typos in config.yaml should fail
    visibly, not silently disable the legal-framework pipeline."""
    with temp_env():
        cfg = dict(config.DEFAULT_CONFIG)
        cfg["default_jurisdiction"] = "antartica"  # not a valid code
        raised = False
        try:
            config.save_config(cfg)
        except ValueError:
            raised = True
        assert raised, "expected ValueError for unknown default_jurisdiction value"


# --- broker jurisdiction / EU coverage ----------------------------------------

def _eu_native_brokers():
    return [
        b for b in brokers.load_all()
        if "US" not in (b.get("jurisdictions") or [])
        and any(j == "UK" or j.startswith(("EU", "EEA"))
                for j in (b.get("jurisdictions") or []))
    ]


def test_eu_native_brokers_are_loaded():
    """The recursive broker loader must expose EU-native records at runtime."""
    assert _eu_native_brokers()


def test_us_brokers_have_gdpr_scope_field():
    """Every US broker record must declare its gdpr_scope explicitly (True OR False).
    Implicit 'false' via missing field would let a US broker that doesn't honor Art.17
    slip through the DPA-escalation planner and surprise the subject with a failed
    escalation."""
    for b in brokers.load_all():
        if "US" in (b.get("jurisdictions") or []) or not b.get("gdpr_scope") is None:
            assert "gdpr_scope" in b, f"broker {b['id']!r} missing gdpr_scope field"


def test_gdpr_scope_filter_returns_only_true_brokers():
    """gdpr_scope() must return ONLY brokers with gdpr_scope=True — used by the
    DPA-escalation planner to know which brokers can realistically be escalated."""
    scoped = brokers.gdpr_scope()
    expected = [b for b in brokers.load_all() if b.get("gdpr_scope") is True]
    assert scoped
    assert {b["id"] for b in scoped} == {b["id"] for b in expected}


def test_by_jurisdiction_returns_matching_brokers():
    """by_jurisdiction('EU-IT') must return exactly the brokers tagged EU-IT."""
    eu_it = brokers.by_jurisdiction("EU-IT")
    expected = [b for b in brokers.load_all() if "EU-IT" in (b.get("jurisdictions") or [])]
    assert eu_it
    assert {b["id"] for b in eu_it} == {b["id"] for b in expected}


def test_by_jurisdiction_eu_returns_eu_native_and_tagged_us_brokers():
    """by_jurisdiction('EU') — broader query — must return both EU-native brokers and
    US brokers that have EU in their jurisdictions (the full GDPR-eligible universe)."""
    broad = brokers.by_jurisdiction("EU")
    expected = [
        b for b in brokers.load_all()
        if any(code == "EU" or code.startswith("EU-")
               for code in (b.get("jurisdictions") or []))
    ]
    assert {b["id"] for b in broad} == {b["id"] for b in expected}
    assert any("US" in (b.get("jurisdictions") or []) for b in broad)
    assert any("US" not in (b.get("jurisdictions") or []) for b in broad)


def test_eu_brokers_have_gdpr_in_deletion_kinds():
    """Every EU-native broker must declare `gdpr` in its optout.deletion.kinds — the
    legal-request dispatcher in pdd.py uses this to pick the right template kind."""
    for b in _eu_native_brokers():
        kinds = (b.get("optout") or {}).get("deletion", {}).get("kinds", [])
        assert "gdpr" in kinds, f"{b['id']!r} missing 'gdpr' in deletion.kinds: {kinds}"


def test_eu_brokers_have_privacy_contact():
    """Every EU-native broker must have either an optout.email OR optout.deletion.email
    — sending an Art. 17 request to a broker with no privacy contact is impossible."""
    for b in _eu_native_brokers():
        optout = b.get("optout") or {}
        direct_email = optout.get("email")
        deletion_email = (optout.get("deletion") or {}).get("email")
        assert direct_email or deletion_email, \
            f"{b['id']!r} has no privacy email contact (optout.email={direct_email!r}, deletion.email={deletion_email!r})"


def test_eu_brokers_jurisdictions_are_eu_or_uk():
    """EU-native brokers must declare only GDPR jurisdictions in their list — mixing
    in 'US' would mean the EU subject's planner treats it as a US broker and won't
    surface the GDPR deletion path correctly."""
    for b in _eu_native_brokers():
        juris = b.get("jurisdictions") or []
        for j in juris:
            assert j == "UK" or j.startswith(("EU", "EEA")), \
                f"{b['id']!r} has non-EU jurisdiction {j!r}: {juris}"


def test_brokers_loader_is_idempotent():
    """Loading the broker DB twice must return identical results (no random ordering,
    no cache pollution). This guards against accidental non-determinism in subsequent
    pipeline stages."""
    a = brokers.load_all()
    b = brokers.load_all()
    assert [x["id"] for x in a] == [x["id"] for x in b]


# --- planner integration (autopilot.next_actions) -----------------------------

def _make_dossier(residency="EU-IT", name="Jane Q. Public"):
    """Build a minimal subject dossier for planner tests (no intake command)."""
    return {
        "subject_id": "sub_test",
        "consent": {"authorized": True, "method": "self"},
        "identity": {"full_name": name, "emails": ["jane@example.com"],
                     "current_address": {"city": "Milano", "state": "MI", "postal": "20121"}},
        "residency_jurisdiction": residency,
        "preferences": {},
    }


def test_request_kind_eu_residency_yields_gdpr():
    """request_kind() must delegate to dossier.legal_framework() for the gdpr mapping.
    Direct test of the EU-IT residency code."""
    d = _make_dossier(residency="EU-IT")
    assert autopilot.request_kind(d) == "gdpr"
    assert autopilot.request_kind(d, allowed=["gdpr", "generic"]) == "gdpr"
    # when broker doesn't accept gdpr, fall back to generic (NOT upgrade)
    assert autopilot.request_kind(d, allowed=["ccpa", "generic"]) == "generic"


def test_request_kind_us_residency_yields_ccpa():
    """US-CA residency yields ccpa (preserve existing behaviour)."""
    d = _make_dossier(residency="US-CA")
    assert autopilot.request_kind(d) == "ccpa"
    assert autopilot.request_kind(d, allowed=["ccpa", "gdpr"]) == "ccpa"


def test_request_kind_unknown_residency_yields_generic():
    """Unknown residency codes fall back to generic (matches dossier.legal_framework)."""
    d = _make_dossier(residency="ZZ")
    assert autopilot.request_kind(d) == "generic"


def test_dpa_escalation_threshold_constant_is_35_days():
    """The escalation threshold is 35 days (Art. 12(3)'s 30 days + 5-day grace).
    A regression that changes this number will silently change when subjects
    surface escalation actions to humans."""
    assert autopilot.DPA_ESCALATION_THRESHOLD_DAYS == 35


def test_next_actions_surfaces_dpa_escalation_after_35_days():
    """End-to-end: EU-IT subject with a 40-day-old submitted case to a gdpr_scope
    broker must produce a `dpa_escalate` action in next_actions output."""
    import datetime as _dt
    d = _make_dossier(residency="EU-IT")
    # Build a ledger with one case: submitted 40 days ago
    long_ago = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = {
        "spokeo": {
            "broker_id": "spokeo",
            "state": "submitted",
            "submitted_at": long_ago,
            "history": [{"at": long_ago, "to": "submitted"}],
        }
    }
    cfg = config.load_config()
    out = autopilot.next_actions(d, brokers.load_all(), cfg, ledger=ledger)
    escalate_actions = [a for a in out["actions"] if a.get("type") in ("dpa_escalate", "dpa_escalate_generic")]
    assert len(escalate_actions) >= 1
    a = escalate_actions[0]
    assert a["broker_id"] == "spokeo"
    assert a["dpa"] == "garante"
    assert a["age_days"] >= 35
    assert "pdd.py escalate" in a["command"]


def test_next_actions_does_not_escalate_young_cases():
    """Cases submitted <35 days ago must NOT produce a dpa_escalate action — the
    30-day Art. 12(3) clock hasn't elapsed yet."""
    import datetime as _dt
    d = _make_dossier(residency="EU-IT")
    recent = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = {
        "spokeo": {
            "broker_id": "spokeo",
            "state": "submitted",
            "submitted_at": recent,
            "history": [{"at": recent, "to": "submitted"}],
        }
    }
    cfg = config.load_config()
    out = autopilot.next_actions(d, brokers.load_all(), cfg, ledger=ledger)
    escalate_actions = [a for a in out["actions"] if a.get("type") in ("dpa_escalate", "dpa_escalate_generic")]
    assert escalate_actions == []


def test_next_actions_does_not_escalate_already_filed_dpa():
    """If the subject already filed a DPA complaint for the broker (recorded in
    preferences.dpa_complaint_filed_<bid>), next_actions must NOT re-surface the
    escalation — that would create an infinite escalation loop."""
    import datetime as _dt
    d = _make_dossier(residency="EU-IT")
    long_ago = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_iso = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = {
        "spokeo": {
            "broker_id": "spokeo",
            "state": "submitted",
            "submitted_at": long_ago,
            "history": [{"at": long_ago, "to": "submitted"}],
        }
    }
    d["preferences"]["dpa_complaint_filed_spokeo"] = now_iso  # already filed
    cfg = config.load_config()
    out = autopilot.next_actions(d, brokers.load_all(), cfg, ledger=ledger)
    escalate_actions = [a for a in out["actions"] if a.get("type") in ("dpa_escalate", "dpa_escalate_generic")]
    assert escalate_actions == []


def test_next_actions_does_not_escalate_non_gdpr_brokers():
    """A 40-day-old submitted case to a broker where gdpr_scope=False must NOT
    produce an escalation action — the broker doesn't honor Art.17, so escalating
    to a DPA makes no sense."""
    import datetime as _dt
    d = _make_dossier(residency="EU-IT")
    long_ago = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # familytreenow has gdpr_scope=False (notorious non-honorer)
    ledger = {
        "familytreenow": {
            "broker_id": "familytreenow",
            "state": "submitted",
            "submitted_at": long_ago,
            "history": [{"at": long_ago, "to": "submitted"}],
        }
    }
    cfg = config.load_config()
    out = autopilot.next_actions(d, brokers.load_all(), cfg, ledger=ledger)
    escalate_actions = [a for a in out["actions"] if a.get("type") in ("dpa_escalate", "dpa_escalate_generic")]
    assert escalate_actions == []


def test_next_actions_does_not_escalate_us_residency():
    """A US-CA subject with old submitted cases must NOT get dpa_escalate actions —
    the US has no DPA equivalent."""
    import datetime as _dt
    d = _make_dossier(residency="US-CA")
    long_ago = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = {
        "spokeo": {
            "broker_id": "spokeo",
            "state": "submitted",
            "submitted_at": long_ago,
            "history": [{"at": long_ago, "to": "submitted"}],
        }
    }
    cfg = config.load_config()
    out = autopilot.next_actions(d, brokers.load_all(), cfg, ledger=ledger)
    escalate_actions = [a for a in out["actions"] if a.get("type") in ("dpa_escalate", "dpa_escalate_generic")]
    assert escalate_actions == []


def test_next_actions_dpa_escalation_uses_spanish_adapter():
    """EU-ES subject with an old gdpr_scope broker case must produce a national
    AEPD escalation, not the generic fallback."""
    import datetime as _dt
    d = _make_dossier(residency="EU-ES")
    long_ago = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = {
        "spokeo": {
            "broker_id": "spokeo",
            "state": "submitted",
            "submitted_at": long_ago,
            "history": [{"at": long_ago, "to": "submitted"}],
        }
    }
    cfg = config.load_config()
    out = autopilot.next_actions(d, brokers.load_all(), cfg, ledger=ledger)
    escalate_actions = [a for a in out["actions"] if a.get("type") in ("dpa_escalate", "dpa_escalate_generic")]
    assert len(escalate_actions) >= 1
    a = escalate_actions[0]
    assert a["type"] == "dpa_escalate"
    assert a["dpa"] == "aepd"
    assert a["subject_residency"] == "EU-ES"


# --- DPA registry + escalation pipeline ---------------------------------------

def test_dpa_load_all_returns_curated_adapters():
    """The DPA loader must return all shipped adapters
    sorted by country. This is the smoke test: 'is the registry wired up?'"""
    adapters = dpa_mod.load_all()
    assert adapters
    countries = [a["country"] for a in adapters]
    assert countries == sorted(countries)


def test_dpa_get_resolves_known_adapter():
    """get() must find a shipped adapter by id, case-insensitive."""
    a = dpa_mod.get("garante")
    assert a is not None
    assert a["country"] == "IT"
    assert a["language"] == "it"
    # case-insensitive
    assert dpa_mod.get("GARANTE") == a


def test_dpa_get_returns_none_for_unknown_id():
    """get() must NOT crash on unknown DPA ids — return None."""
    assert dpa_mod.get("nonexistent_authority") is None


def test_dpa_adapters_satisfy_required_schema_fields():
    """Every shipped adapter must have the required fields per references/dpa/_schema.json.
    This is the contract test: missing fields = the loader will reject the adapter."""
    REQUIRED = {"id", "name", "country", "language", "web_form_url"}
    for adapter in dpa_mod.load_all():
        missing = REQUIRED - adapter.keys()
        assert not missing, f"{adapter.get('id')!r} missing required fields: {missing}"


def test_dpa_country_codes_are_iso_alpha2():
    """country field must be ISO 3166-1 alpha-2 (2 uppercase letters)."""
    import re
    for adapter in dpa_mod.load_all():
        country = adapter.get("country", "")
        assert re.fullmatch(r"[A-Z]{2}", country), \
            f"{adapter.get('id')!r} country {country!r} is not ISO 3166-1 alpha-2"


def test_dpa_web_form_urls_are_http_or_https():
    """web_form_url must be a real URL — pointing operators to broken pages is unacceptable."""
    import re
    for adapter in dpa_mod.load_all():
        url = adapter.get("web_form_url", "")
        assert re.match(r"^https?://", url), \
            f"{adapter.get('id')!r} web_form_url {url!r} is not a valid http(s) URL"


def test_dpa_dossier_id_matches_registry():
    """Every DPA referenced from dossier.RESIDENCY_LEGAL_FRAMEWORK must have a registered
    adapter. If a residency code points to a non-existent DPA, the subject will hit a
    confusing failure at complaint time."""
    referenced = {v["dpa"] for v in dossier.RESIDENCY_LEGAL_FRAMEWORK.values() if v.get("dpa")}
    registered = {a["id"] for a in dpa_mod.load_all()}
    missing = referenced - registered
    assert not missing, f"residency codes reference unregistered DPAs: {missing}"


def test_dpa_adapters_have_specific_template_or_explicit_generic_ok():
    """Every DPA adapter must either point at an existing complaint template or
    explicitly declare generic_ok=true. This prevents silent fallback to English generic
    text when a new national adapter is added."""
    missing = []
    for adapter in dpa_mod.load_all():
        rel = adapter.get("complaint_template") or f"dpa-complaints/{adapter['id']}.txt"
        template_path = paths.templates_dir() / rel
        if not template_path.exists() and not adapter.get("generic_ok"):
            missing.append(f"{adapter['id']} -> {rel}")
    assert not missing, f"DPA adapters missing template or generic_ok=true: {missing}"


def test_supported_eu_residencies_have_dpa_adapters():
    """Named EU country codes should be actionable. The generic catch-alls are the
    only GDPR residencies allowed to omit a DPA adapter."""
    allowed_generic = {"EU", "EU-EEA"}
    missing = []
    for code, meta in dossier.RESIDENCY_LEGAL_FRAMEWORK.items():
        if code in allowed_generic:
            continue
        if meta.get("framework") in ("gdpr", "uk_gdpr") and not meta.get("dpa"):
            missing.append(code)
    assert not missing, f"named EU/EEA/UK residencies missing DPA adapter: {missing}"


def test_dpa_for_residency_resolves_eu_codes():
    """for_residency() is the bridge between the dossier table and the DPA loader.
    Every EU residency code with a mapped DPA must resolve to that adapter."""
    for residency, meta in dossier.RESIDENCY_LEGAL_FRAMEWORK.items():
        expected_id = meta.get("dpa")
        if not expected_id:
            continue
        adapter = dpa_mod.for_residency(residency)
        assert adapter is not None, f"for_residency({residency!r}) returned None"
        assert adapter["id"] == expected_id, f"{residency} -> {adapter['id']}, expected {expected_id}"
        expected_country = "GB" if residency == "UK" else residency.rsplit("-", 1)[-1]
        assert adapter["country"] == expected_country


def test_dpa_for_residency_returns_none_for_generic_eu():
    """The catch-all `EU` code has no specific DPA — must return None (the loader
    surfaces this as an actionable error to the subject, not a crash)."""
    assert dpa_mod.for_residency("EU") is None


def test_dpa_for_residency_returns_none_for_us_residency():
    """US residency has no DPA equivalent — CCPA enforcement is complaint-based
    via the CA AG, not a national authority. for_residency returns None here."""
    assert dpa_mod.for_residency("US") is None
    assert dpa_mod.for_residency("US-CA") is None


def test_render_dpa_complaint_garante_uses_italian():
    """The Garante template is in Italian — a regression where the template fell back
    to the generic English version would silently produce a complaint that the Garante
    routes to a slower English-language queue."""
    out = legal.render_dpa_complaint("garante", {
        "full_name": "Jane Q. Public",
        "contact_email": "jane@example.com",
        "broker_name": "Spokeo",
        "request_date": "2026-07-01",
        "request_channel": "PEC",
        "current_address": "Via Roma 1, 20121 Milano MI",
    })
    assert "Regolamento (UE) 2016/679" in out  # Italian cite
    assert "articolo 77" in out.lower() or "art. 77" in out.lower()  # Italian Art. 77
    # the Italian template must NOT fall back to English fallback phrases
    assert "Article 77" not in out  # would only appear if the generic English template rendered


def test_render_dpa_complaint_cnil_uses_french():
    """Mirror of the Garante test for CNIL — French language template."""
    out = legal.render_dpa_complaint("cnil", {
        "full_name": "Jane Q. Public",
        "contact_email": "jane@example.com",
        "broker_name": "Spokeo",
        "request_date": "2026-07-01",
        "request_channel": "email",
        "current_address": "1 rue de la Paix, 75002 Paris",
    })
    # Normalise whitespace (templates split citations across newlines for readability)
    flat = " ".join(out.split())
    assert "Règlement (UE) 2016/679" in flat  # French cite (split across lines in template)
    assert "article 77" in out.lower()  # French Art. 77
    assert "Article 77" not in out  # the French template uses lowercase 'article'


def test_render_dpa_complaint_bfdi_uses_german():
    """BfDI template must be in German (not the English fallback)."""
    out = legal.render_dpa_complaint("bfdi", {
        "full_name": "Jane Q. Public",
        "contact_email": "jane@example.com",
        "broker_name": "Spokeo",
        "request_date": "2026-07-01",
        "request_channel": "E-Mail",
        "current_address": "Musterstraße 1, 10115 Berlin",
    })
    assert "Verordnung (EU) 2016/679" in out  # German cite
    assert "Art. 77" in out  # German Art. 77
    assert "Article 77" not in out


def test_render_new_dpa_templates_are_specific_not_generic():
    """The post-phase DPA adapters must render their own complaint packages, not the
    generic Article 77 fallback."""
    fields = {
        "full_name": "Jane Q. Public",
        "contact_email": "jane@example.com",
        "broker_name": "Spokeo",
        "request_date": "2026-07-01",
        "request_channel": "email",
        "current_address": "Example Street 1",
    }
    markers = {
        "aepd": ("Agencia Espanola de Proteccion de Datos", "articulo 77", "Ley Organica 3/2018"),
        "ap_nl": ("Autoriteit Persoonsgegevens", "artikel 77", "Uitvoeringswet AVG"),
        "apd_gba": ("Gegevensbeschermingsautoriteit", "artikel 77", "APD-GBA"),
        "cnpd_pt": ("Comissao Nacional de Protecao de Dados", "artigo 77", "Lei n.o 58/2019"),
        "datatilsynet_dk": ("Datatilsynet", "artikel 77", "databeskyttelsesforordningen"),
        "dpc_ie": ("Data Protection Commission", "Article 77", "Data Protection Act 2018"),
        "dsb_at": ("Oesterreichische Datenschutzbehoerde", "Art. 77", "§ 24"),
        "imy": ("Integritetsskyddsmyndigheten", "artikel 77", "dataskyddsförordningen"),
        "tietosuoja": ("tietosuojavaltuutetulle", "77 artiklan", "tietosuojalain"),
        "uodo": ("Do Prezesa Urzędu Ochrony Danych Osobowych", "art. 77", "RODO"),
        "aki_ee": ("Andmekaitse Inspektsioon", "Article 77", "AKI"),
        "anspdcp_ro": ("ANSPDCP", "Article 77", "Romania"),
        "azop_hr": ("AZOP", "Article 77", "Croatian"),
        "cnpd_lu": ("CNPD Luxembourg", "Article 77", "Commission nationale"),
        "cpdp_bg": ("CPDP Bulgaria", "Article 77", "Bulgarian"),
        "cpdp_cy": ("Cyprus Commissioner", "Article 77", "Greek filing"),
        "datenschutzstelle_li": ("Datenschutzstelle Liechtenstein", "Art. 77", "Art. 58 Abs. 2"),
        "datatilsynet_no": ("Datatilsynet Norway", "Article 77", "EEA"),
        "dvi_lv": ("Datu valsts inspekcija", "Article 77", "Latvia"),
        "hdpa_gr": ("Hellenic Data Protection Authority", "Article 77", "HDPA"),
        "idpc_mt": ("Malta IDPC", "Article 77", "Data Protection Act"),
        "ip_rs": ("Informacijski pooblascenec", "Article 77", "Slovenia"),
        "naih_hu": ("NAIH", "Article 77", "Hungarian"),
        "personuvernd_is": ("Personuvernd", "Article 77", "EEA"),
        "uoou_cz": ("UOOU Czechia", "Article 77", "Urad pro ochranu osobnich udaju"),
        "uoou_sk": ("Slovak UOOU", "Article 77", "Slovak"),
        "vdai_lt": ("VDAI Lithuania", "Article 77", "Lithuanian"),
    }
    for dpa_id, expected in markers.items():
        out = legal.render_dpa_complaint(dpa_id, fields)
        for marker in expected:
            assert marker in out, f"{dpa_id} template missing marker {marker!r}"
        assert "To the competent supervisory authority" not in out


def test_render_dpa_complaint_unknown_dpa_falls_back_to_generic():
    """render_dpa_complaint must never crash on an unknown DPA id — EU membership changes,
    DPAs get reorganised, and the complaint renderer must degrade gracefully to a generic
    template that the subject can hand-edit for their authority."""
    out = legal.render_dpa_complaint("nonexistent_authority_xyz", {
        "full_name": "Jane Q. Public",
        "contact_email": "jane@example.com",
        "broker_name": "Some Broker",
    })
    # the generic template (added in phase 2) or a default substitute must produce a string
    assert isinstance(out, str) and len(out) > 0


def test_escalate_command_renders_to_drafts_dir():
    """End-to-end: create an EU-IT subject, run escalate spokeo, verify the complaint
    file lands at the expected path with the broker + subject name interpolated."""
    with temp_env():
        d = dossier.create(
            identity={"full_name": "Jane Q. Public", "emails": ["jane@example.com"],
                      "current_address": {"city": "Milano", "state": "MI", "postal": "20121"}},
            consent={"authorized": True, "method": "self"},
            residency="EU-IT",
        )
        ns = pdd.build_parser().parse_args([
            "escalate", d["subject_id"], "spokeo",
            "--request-date", "2026-07-01",
            "--request-channel", "PEC",
        ])
        # cmd_escalate prints JSON to stdout and returns None; capture stdout instead
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            pdd.cmd_escalate(ns)
        import json
        out = json.loads(buf.getvalue())
        assert out["dpa"] == "garante"
        assert out["broker"] == "spokeo"
        assert out["subject"] == d["subject_id"]
        # the complaint file must exist and contain the broker name
        from pathlib import Path
        p = Path(out["complaint_path"])
        assert p.exists()
        text = p.read_text()
        assert "Spokeo" in text
        assert "Jane Q. Public" in text
        assert "2026-07-01" in text
        assert "PEC" in text


def test_render_email_writes_subject_scoped_drafts():
    """Multi-tenant safety: two subjects rendering the same broker/kind must not
    overwrite each other's draft under a global drafts directory."""
    with temp_env():
        first = dossier.create(
            identity={"full_name": "Jane Q. Public", "emails": ["jane@example.com"]},
            consent={"authorized": True, "method": "self"},
            residency="EU-IT",
        )
        second = dossier.create(
            identity={"full_name": "John Q. Public", "emails": ["john@example.com"]},
            consent={"authorized": True, "method": "self"},
            residency="EU-IT",
        )

        import io
        import json
        from contextlib import redirect_stdout

        outputs = []
        for subject, url in ((first, "https://example.invalid/jane"),
                             (second, "https://example.invalid/john")):
            ns = pdd.build_parser().parse_args([
                "render-email", subject["subject_id"], "spokeo",
                "--kind", "gdpr", "--listing", url,
            ])
            buf = io.StringIO()
            with redirect_stdout(buf):
                pdd.cmd_render_email(ns)
            outputs.append(json.loads(buf.getvalue()))

        paths_out = [Path(o["draft"]) for o in outputs]
        assert paths_out[0] != paths_out[1]
        assert first["subject_id"] in str(paths_out[0])
        assert second["subject_id"] in str(paths_out[1])
        assert "Jane Q. Public" in paths_out[0].read_text()
        assert "John Q. Public" in paths_out[1].read_text()


def test_render_gdpr_indirect_email_uses_ledger_evidence_identifiers():
    """A real EU indirect-exposure case should not render a blank "data of mine"
    block. When the scan evidence says the exposed datum is the subject's home
    address, render-email should use that dossier value automatically."""
    with temp_env():
        subject = dossier.create(
            identity={
                "full_name": "Jane Q. Public",
                "emails": ["jane@example.com"],
                "current_address": {
                    "line1": "123 Main St",
                    "postal": "20121",
                    "city": "Milano",
                    "state": "MI",
                },
            },
            consent={"authorized": True, "method": "self"},
            residency="EU-IT",
        )
        ledger.transition(
            subject["subject_id"], "paginebianche", "indirect_exposure",
            found=False,
            evidence={
                "listing_urls": ["https://www.paginebianche.it/ricerca?qs=Public&dv=Milano"],
                "exposed_fields": ["relative_full_name", "full_street_address", "landline"],
                "match_basis": "same household address under a relative record",
            },
        )

        import io
        import json
        from contextlib import redirect_stdout

        ns = pdd.build_parser().parse_args([
            "render-email", subject["subject_id"], "paginebianche",
            "--kind", "gdpr_indirect",
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            pdd.cmd_render_email(ns)
        out = json.loads(buf.getvalue())
        text = Path(out["draft"]).read_text()

        assert "my home address: 123 Main St 20121 Milano MI" in text
        assert "https://www.paginebianche.it/ricerca?qs=Public&dv=Milano" in text
        assert "relative_full_name" not in text
        assert "landline:" not in text


def test_escalate_command_refuses_us_residency():
    """A US subject must NOT be able to file an Art. 77 complaint — CCPA enforcement
    is via the CA AG, not a DPA. The command must refuse with a clear error."""
    with temp_env():
        d = dossier.create(
            identity={"full_name": "Jane Q. Public", "emails": ["jane@example.com"]},
            consent={"authorized": True, "method": "self"},
            residency="US-CA",
        )
        ns = pdd.build_parser().parse_args(["escalate", d["subject_id"], "spokeo"])
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            pdd.cmd_escalate(ns)
        import json
        out = json.loads(buf.getvalue())
        assert "error" in out
        assert "not EU/EEA/UK" in out["error"] or "not applicable" in out["error"]


def test_escalate_command_accepts_eea_residency():
    """Named EEA residencies (Norway/Iceland/Liechtenstein) are GDPR jurisdictions
    and must be accepted by `escalate`, not rejected by an EU-* prefix check."""
    with temp_env():
        d = dossier.create(
            identity={"full_name": "Jane Q. Public", "emails": ["jane@example.com"]},
            consent={"authorized": True, "method": "self"},
            residency="EEA-NO",
        )
        ns = pdd.build_parser().parse_args([
            "escalate", d["subject_id"], "spokeo", "--request-date", "2026-07-01",
        ])
        import io
        import json
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            pdd.cmd_escalate(ns)
        out = json.loads(buf.getvalue())
        assert out["dpa"] == "datatilsynet_no"
        assert out["dpa_name"] == "Datatilsynet"


def test_escalate_command_accepts_generic_dpa_override():
    """The documented --dpa generic escape hatch must actually render the generic
    complaint instead of being ignored."""
    with temp_env():
        d = dossier.create(
            identity={"full_name": "Jane Q. Public", "emails": ["jane@example.com"]},
            consent={"authorized": True, "method": "self"},
            residency="EU",
        )
        ns = pdd.build_parser().parse_args([
            "escalate", d["subject_id"], "spokeo", "--dpa", "generic",
            "--request-date", "2026-07-01",
        ])
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            pdd.cmd_escalate(ns)
        import json
        out = json.loads(buf.getvalue())
        assert out["dpa"] == "generic"
        assert out["dpa_name"] == "Generic EU/EEA supervisory authority"
        assert out["web_form_url"].endswith("/our-members_en")


def test_dpas_command_lists_and_resolves_residency():
    """`pdd.py dpas` is the EU analogue to `registry`: it shows the filing channel
    universe and resolves a subject residency to its national authority."""
    import io
    import json
    from contextlib import redirect_stdout

    ns = pdd.build_parser().parse_args(["dpas"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        pdd.cmd_dpas(ns)
    out = json.loads(buf.getvalue())
    assert out["dpa_adapters"] == len(out["adapters"])
    assert {a["id"] for a in out["adapters"]} == {a["id"] for a in dpa_mod.load_all()}

    ns = pdd.build_parser().parse_args(["dpas", "--residency", "EU-ES"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        pdd.cmd_dpas(ns)
    resolved = json.loads(buf.getvalue())
    assert resolved["dpa"]["id"] == "aepd"
    assert resolved["dpa"]["country"] == "ES"

    ns = pdd.build_parser().parse_args(["dpas", "--residency", "EEA-NO"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        pdd.cmd_dpas(ns)
    resolved = json.loads(buf.getvalue())
    assert resolved["dpa"]["id"] == "datatilsynet_no"
    assert resolved["dpa"]["country"] == "NO"


def test_escalate_command_refuses_subject_without_consent():
    """Mirror of other commands' consent gate — no consent = no escalation."""
    with temp_env():
        d = dossier.create(
            identity={"full_name": "Jane Q. Public", "emails": ["jane@example.com"]},
            consent={"authorized": False, "method": "self"},  # NOT authorized
            residency="EU-IT",
        )
        ns = pdd.build_parser().parse_args(["escalate", d["subject_id"], "spokeo"])
        # consent gate raises PermissionError rather than printing JSON; capture that
        raised = False
        try:
            pdd.cmd_escalate(ns)
        except PermissionError:
            raised = True
        assert raised, "expected PermissionError when subject is not authorized"


def test_escalate_command_records_ledger_state_on_file():
    """--file must transition the case to human_task_queued with a DPA reference,
    so the autonomous queue stops re-surfacing the broker.

    Realistic setup: the subject has already filed an Art. 17 request, the broker
    failed to respond within 30 days, and now the subject is escalating to the DPA.
    So the case starts in 'submitted' (broker had its chance, missed it).
    """
    with temp_env():
        d = dossier.create(
            identity={"full_name": "Jane Q. Public", "emails": ["jane@example.com"]},
            consent={"authorized": True, "method": "self"},
            residency="EU-IT",
        )
        # Simulate the realistic flow: broker was found, opt-out was submitted,
        # broker failed to respond. Only then is the DPA escalation appropriate.
        ledger.transition(d["subject_id"], "spokeo", "found",
                          found=True, evidence={"listing_url": "https://www.spokeo.com/jane"})
        ledger.transition(d["subject_id"], "spokeo", "submitted",
                          evidence={"sent_at": "2026-07-01", "channel": "PEC"})

        ns = pdd.build_parser().parse_args([
            "escalate", d["subject_id"], "spokeo", "--file",
        ])
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            pdd.cmd_escalate(ns)
        import json
        out = json.loads(buf.getvalue())
        assert "filed_at" in out
        # ledger must reflect the transition to human_task_queued
        case = ledger.load(d["subject_id"]).get("spokeo")
        assert case is not None
        assert case["state"] == "human_task_queued"
        assert case.get("dpa") == "garante"
        # subject dossier must record the complaint as filed
        d_loaded = dossier.load(d["subject_id"])
        assert d_loaded["preferences"]["dpa_complaint_filed_spokeo"] == out["filed_at"]
        queue = autopilot.next_actions(
            d_loaded, brokers.load_all(), config.load_config(),
            ledger=ledger.load(d["subject_id"]), env={},
        )
        digest = next(item for item in queue["human_digest"] if item["broker_id"] == "spokeo")
        assert digest["reason"] == "DPA complaint filed with Garante per la protezione dei dati personali"


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


def test_next_actions_uses_gdpr_indirect_for_eu_indirect_exposure():
    with temp_env():
        d = _consenting()
        d["residency_jurisdiction"] = "EU-IT"
        b = _mini_broker("paginebianche")
        b["optout"]["method"] = "email"
        b["optout"]["email"] = "privacy@paginebianche.it"
        led = {"paginebianche": {"state": "indirect_exposure"}}

        q = autopilot.next_actions(d, [b], _auto_cfg(), led, env={})
        prep = " ".join(" ".join(t["agent_prep"]) for t in q["human_digest"])
        assert "--kind gdpr_indirect" in prep
        assert "--kind ccpa_indirect" not in prep

        env = {"EMAIL_ADDRESS": "agent@gmail.com", "EMAIL_PASSWORD": "p"}
        q2 = autopilot.next_actions(d, [b], _auto_cfg(email_mode="programmatic"), led, env=env)
        action = next(a for a in q2["actions"] if a["type"] == "indirect_email_send")
        assert action["kind"] == "gdpr_indirect"
        assert "--kind gdpr_indirect" in action["command"]


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
