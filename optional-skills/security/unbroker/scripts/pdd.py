#!/usr/bin/env python3
"""unbroker - deterministic CLI helper.

The Hermes agent orchestrates scanning and opt-out submission with native tools
(`web_extract`, `browser_navigate`, email mechanisms). THIS CLI owns the
deterministic state: config, dossiers + consent, the broker DB, tier planning,
the ledger + audit log, draft/template rendering, and reports.

Run it through the `terminal` tool (it can read PII files under HERMES_HOME);
do NOT run it through `execute_code` (that sandbox scrubs env and redacts output).

Examples:
  python pdd.py setup
  python pdd.py intake --full-name "Jane Q. Public" --email jane@example.com \
      --city Oakland --state CA --residency US-CA --consent --consent-method self
  python pdd.py plan sub_xxxx --priority crucial
  python pdd.py record sub_xxxx spokeo found --found true \
      --evidence '{"listing_urls":["https://www.spokeo.com/..."]}'
  python pdd.py render-email sub_xxxx spokeo --listing https://www.spokeo.com/...
  python pdd.py status sub_xxxx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autopilot                   # noqa: E402
import badbool                      # noqa: E402
import cdp                          # noqa: E402
import brokers as brokers_mod      # noqa: E402
import config as config_mod        # noqa: E402
import crypto                       # noqa: E402
import dossier as dossier_mod      # noqa: E402
import email_modes                 # noqa: E402
import emailer                     # noqa: E402
import ledger as ledger_mod        # noqa: E402
import legal                       # noqa: E402
import paths as paths_mod          # noqa: E402
import registry                    # noqa: E402
import report as report_mod        # noqa: E402
import tiers                       # noqa: E402


def _out(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _require_subject(subject_id: str) -> dict:
    d = dossier_mod.load(subject_id)
    if not d:
        sys.exit(f"error: unknown subject {subject_id!r} (run `intake` first)")
    return d


def cmd_setup(args) -> None:
    if getattr(args, "auto", False):
        # Autonomous path: detect capabilities and pick the most autonomous valid config without
        # asking anyone. Read creds from $HERMES_HOME/.env too (the terminal shell doesn't export
        # them). Explicit flags still win below.
        cfg = config_mod.auto_configure(env=config_mod.dotenv_env())
    else:
        cfg = config_mod.load_config()
    for key in ("autonomy", "email_mode", "browser_backend", "tracker_backend", "encryption"):
        val = getattr(args, key)
        if val:
            cfg[key] = val
    if cfg.get("encryption") == "age":
        if not crypto.age_available():
            sys.exit("error: encryption=age requested but `age`/`age-keygen` not found. "
                     "Install age (e.g. `brew install age`) or use `--encryption none`.")
        crypto.ensure_identity()  # generate the key now so encryption is actually engaged
    path = config_mod.save_config(cfg)
    migrated = _migrate_subjects()  # rewrite existing dossiers/ledgers into the new at-rest format
    out = {
        "config_path": str(path),
        "config": cfg,
        "encryption_engaged": crypto.is_engaged(),
        "detected_upgrades": config_mod.detect_capabilities(),
        "migrated_subjects": migrated,
        "note": "Defaults are easiest-first (draft email, auto browser, local tracker, no encryption). "
                "Pass flags to opt into upgrades, then run `doctor` for a readiness summary.",
    }
    if cfg.get("encryption") == "age":
        out["age_identity"] = str(crypto.identity_path())
    _out(out)


def _migrate_subjects() -> int:
    """Re-save each subject's dossier + ledger so they match the current at-rest format."""
    sd = paths_mod.subjects_dir()
    if not sd.exists():
        return 0
    n = 0
    for child in sorted(sd.iterdir()):
        if not child.is_dir():
            continue
        sid = child.name
        d = dossier_mod.load(sid)
        if d is not None:
            dossier_mod.save(d)
            n += 1
        led = ledger_mod.load(sid)
        if led:
            ledger_mod.save(sid, led)
    return n


def _check_writable(path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def cmd_doctor(args) -> None:
    import platform

    cfg = config_mod.load_config()
    caps = config_mod.detect_capabilities(config_mod.dotenv_env())  # see creds in $HERMES_HOME/.env too
    data = paths_mod.data_dir()
    writable = _check_writable(data)
    curated = len(brokers_mod._load_curated())
    live = len(brokers_mod.load_live_cache())
    total = len(brokers_mod.load_all())

    L = ["unbroker - readiness check", "=" * 42,
         f"Python      : {platform.python_version()}",
         f"Data dir    : {data} ({'writable' if writable else 'NOT writable'})",
         f"Config      : autonomy={cfg.get('autonomy', 'full')} email={cfg['email_mode']} "
         f"browser={cfg['browser_backend']} "
         f"tracker={cfg['tracker_backend']} encryption={cfg['encryption']}",
         f"Brokers     : {total} available ({curated} curated + {live} live"
         + ("" if live else ", run `refresh-brokers` to expand to ~50") + ")",
         "", "Opt-in upgrades:"]
    rows = [
        ("Cloud browser (Browserbase) *RECOMMENDED*", caps["browserbase"],
         "default backend: clears soft CAPTCHAs (Turnstile/hCaptcha) -> more T1", "set BROWSERBASE_API_KEY"),
        ("Email auto (AgentMail)", caps["agentmail"],
         "send + auto-verify, per-broker aliases (Mode B/C)", "install agentmail skill / set AGENTMAIL_API_KEY"),
        ("Email send (CLI SMTP)", caps["smtp_send"],
         "`send-email` delivers opt-outs itself (Mode B)", "set EMAIL_ADDRESS / EMAIL_PASSWORD (+ EMAIL_SMTP_HOST)"),
        ("Verify-link poll (CLI IMAP)", caps["imap_read"],
         "`poll-verification` reads confirmation links itself", "set EMAIL_ADDRESS / EMAIL_PASSWORD (+ EMAIL_IMAP_HOST)"),
        ("Google Sheets tracker", caps["google_workspace"],
         "shared status dashboard", "set up the google-workspace skill"),
    ]
    for name, ok, enables, how in rows:
        L.append(f"  [{'ON ' if ok else 'off'}] {name:<28} {enables}")
        if not ok:
            L.append(f"        enable: {how}")

    # At-rest encryption: report TRUE engagement (configured + key present), not just binary presence.
    engaged = crypto.is_engaged()
    L.append(f"  [{'ON ' if engaged else 'off'}] {'At-rest encryption (age)':<28} "
             "encrypts dossiers + ledgers on disk")
    if engaged:
        L.append(f"        key: {crypto.identity_path()} (0600) - guards casual/backup/commit "
                 "exposure, NOT a full-HERMES_HOME read")
    elif cfg["encryption"] == "age":
        L.append("        WARNING: encryption=age is SET but NOT engaged (age binary or key missing);"
                 " dossiers would be PLAINTEXT")
    elif caps["age"]:
        L.append("        off - dossiers are plaintext (0600). enable: `setup --encryption age`")
    else:
        L.append("        off - dossiers are plaintext (0600). install `age` first to enable")

    L += ["", "Verdict:", "  Ready now in DRAFT mode (no setup needed): scan brokers, draft opt-out",
          "  emails for you to send, and track everything in the ledger."]
    if caps["browserbase"]:
        L.append("  Cloud browser ON (recommended default): soft/managed CAPTCHAs "
                 "(Turnstile/hCaptcha) clear automatically -> those brokers stay T1.")
    else:
        L.append("  No cloud browser: set BROWSERBASE_API_KEY (the recommended default) so soft "
                 "CAPTCHAs clear automatically; without it those brokers drop to T2 (human tasks).")
    if cfg["email_mode"] == "draft_only":
        L.append("  Email is draft-only: you send drafts + click verify links. For hands-off email "
                 "WITHOUT storing a password, run `setup --email-mode browser` (agent sends + opens "
                 "verify links via your logged-in webmail); or set EMAIL_* for SMTP/IMAP.")
    elif cfg["email_mode"] == "browser":
        L.append("  Email mode: browser (no password) - the agent sends opt-outs and opens verify "
                 "links via the operator's logged-in webmail. This needs Hermes pointed at the "
                 "operator's OWN Chrome over CDP (launch with --remote-debugging-port=9222 "
                 "--user-data-dir=~/.hermes/chrome-debug, signed into the webmail once); else it falls "
                 "back to drafts. Run `pdd.py cdp` to launch it (or `pdd.py cdp --print` for the command). "
                 "See methods.md 'Browser backends'.")
        cloud_scan = cfg.get("browser_backend") == "browserbase" or (
            cfg.get("browser_backend") == "auto" and caps.get("browserbase"))
        if cloud_scan:
            L.append("  NOTE: your scan backend is a cloud browser (Browserbase). It is great for "
                     "Phase-1 scanning but CANNOT be the browser that sends webmail (no inbox session) "
                     "and is itself Cloudflare/DataDome-gated on session-bound gates (e.g. PeopleConnect). "
                     "For Phase-2 email/verify, launch the operator's Chrome over CDP: `pdd.py cdp`.")
    if not crypto.is_engaged():
        L.append("  Storage: dossiers are PLAINTEXT JSON (0600 under HERMES_HOME). "
                 "Run `setup --encryption age` for at-rest encryption.")
    if not live:
        L.append("  Next: run `refresh-brokers` to load the full broker list.")

    # Freshness: warn when cached lists / curated mechanics are going stale (silent broker rot).
    import time as _time
    STALE_CACHE_DAYS, STALE_VERIFY_DAYS = 30, 180

    def _age_days(p) -> float | None:
        try:
            return (_time.time() - p.stat().st_mtime) / 86400.0
        except OSError:
            return None

    fresh = []
    for label, p in [("BADBOOL", paths_mod.brokers_cache_path()),
                     ("CA registry", paths_mod.registry_cache_path())]:
        age = _age_days(p)
        if age is None:
            fresh.append(f"{label}: not pulled")
        elif age > STALE_CACHE_DAYS:
            fresh.append(f"{label}: {age:.0f}d old (stale, re-pull)")
    stale_curated = documented = 0
    for b in brokers_mod._load_curated():
        conf = b.get("confidence")
        lv = b.get("last_verified")
        if conf == "documented" or not lv:
            documented += 1
            continue
        try:
            if (_time.time() - _time.mktime(_time.strptime(lv, "%Y-%m-%d"))) / 86400.0 > STALE_VERIFY_DAYS:
                stale_curated += 1
        except (ValueError, TypeError):
            pass
    if fresh:
        L.append("  Freshness: " + "; ".join(fresh) + " (run `refresh-brokers`).")
    if stale_curated or documented:
        L.append(f"  Freshness: {stale_curated} curated broker(s) last-verified >{STALE_VERIFY_DAYS}d ago; "
                 f"{documented} documented broker(s) awaiting first-use verification.")
    print("\n".join(L))


def cmd_cdp(args) -> None:
    """Launch (or detect) the operator's Chrome over CDP for Phase-2 browser + webmail work.

    A cloud browser cannot send the operator's webmail or clear session-bound gates; this points
    Hermes at the operator's real Chrome on a dedicated debug profile (see methods.md).
    """
    import shlex
    import time

    port = args.port
    profile = Path(args.profile).expanduser() if args.profile else cdp.default_profile()

    live = cdp.endpoint_status(port)
    if live:
        _out({"running": True, "endpoint": f"127.0.0.1:{port}",
              "browser": live.get("Browser"),
              "webSocketDebuggerUrl": live.get("webSocketDebuggerUrl"),
              "note": "a debuggable browser is already listening; point Hermes's browser tools at "
                      f"127.0.0.1:{port} and make sure the operator's webmail is signed in in THAT browser."})
        return

    if getattr(args, "check", False):
        _out({"running": False, "endpoint": f"127.0.0.1:{port}",
              "note": f"no debuggable browser here yet; run `pdd.py cdp --port {port}` (no --check) to launch one."})
        return

    browser = cdp.find_browser(args.browser)
    if not browser:
        _out({"running": False, "error": "no Chrome/Chromium-family browser found",
              "fix": "install Google Chrome, or pass --browser /path/to/chrome (or a command on PATH)"})
        return

    cmd = cdp.launch_command(browser, port, profile)
    if getattr(args, "print_only", False):
        _out({"running": False, "browser": browser, "profile": str(profile), "command": cmd,
              "shell": " ".join(shlex.quote(c) for c in cmd),
              "note": "run this yourself to launch the debug browser, then sign into your webmail once."})
        return

    pid = cdp.launch(browser, port, profile)
    live = None
    for _ in range(20):  # give Chrome a few seconds to open the debug port
        live = cdp.endpoint_status(port)
        if live:
            break
        time.sleep(0.5)
    _out({"running": bool(live), "launched_pid": pid, "browser": browser,
          "profile": str(profile), "endpoint": f"127.0.0.1:{port}",
          "webSocketDebuggerUrl": (live or {}).get("webSocketDebuggerUrl"),
          "next": ([f"point Hermes's browser tools at 127.0.0.1:{port} (CDP)",
                    "in the launched browser, sign into the operator's webmail ONCE (dedicated debug profile)",
                    "then run email/verify flows in browser mode -- they use this logged-in session"]
                   if live else
                   ["browser launched but the debug port has not answered yet; give it a few seconds, then "
                    f"re-run `pdd.py cdp --check --port {port}`"])})


def cmd_intake(args) -> None:
    if args.json:
        data = json.loads(Path(args.json).read_text(encoding="utf-8"))
        identity = data["identity"]
        consent = data.get("consent", {})
        residency = data.get("residency_jurisdiction", "US")
        prefs = data.get("preferences")
    else:
        if not args.full_name:
            sys.exit("error: --full-name (or --json) is required")
        identity = {"full_name": args.full_name, "emails": args.email or [], "phones": args.phone or []}
        if args.alias:
            identity["also_known_as"] = args.alias
        if args.dob:
            identity["date_of_birth"] = args.dob
        addr = {k: v for k, v in {"line1": args.street, "city": args.city,
                                  "state": args.state, "postal": args.postal}.items() if v}
        if addr:
            identity["current_address"] = addr
        priors = []
        for loc in args.prior_location or []:
            parts = [p.strip() for p in loc.split(",") if p.strip()]
            if not parts:
                continue
            entry = {"city": parts[0]}
            if len(parts) > 1:
                entry["state"] = parts[1]
            if len(parts) > 2:
                entry["postal"] = parts[2]
            priors.append(entry)
        if priors:
            identity["prior_addresses"] = priors
        cfg = config_mod.load_config()
        consent = {"authorized": bool(args.consent), "method": args.consent_method, "recorded_at": dossier_mod.now()}
        residency = args.residency or "US"
        prefs = {
            "email_mode": args.email_mode or cfg["email_mode"],
            "rescan_interval_days": cfg["default_rescan_interval_days"],
        }
        if args.contact_email:
            prefs["contact_email_for_optouts"] = args.contact_email
    d = dossier_mod.create(identity, consent, residency, prefs)
    _out({"subject_id": d["subject_id"], "authorized": dossier_mod.is_authorized(d),
          "residency": residency, "email_mode": (prefs or {}).get("email_mode"),
          "names": dossier_mod.all_names(d),
          "emails": len(d["identity"].get("emails") or []),
          "phones": len(d["identity"].get("phones") or []),
          "addresses": len(dossier_mod.all_addresses(d))})


def cmd_brokers(args) -> None:
    bl = brokers_mod.by_priority(*(args.priority or [])) if args.priority else brokers_mod.load_all()
    _out([
        {"id": b.get("id"), "name": b.get("name"), "priority": b.get("priority"),
         "method": (b.get("optout") or {}).get("method"), "owns": b.get("owns") or [],
         "source": b.get("source"), "confidence": b.get("confidence", "curated")}
        for b in bl
    ])


def cmd_refresh_brokers(args) -> None:
    res = badbool.refresh(paths_mod.brokers_cache_path())
    curated_ids = {b["id"] for b in brokers_mod._load_curated()}
    new = [b["id"] for b in brokers_mod.load_live_cache() if b["id"] not in curated_ids]
    out = {**res, "curated": len(curated_ids), "new_from_live": len(new),
           "people_search_total": len(brokers_mod.load_all()),
           "note": "Live records have confidence=auto; verify their opt-out URL before acting."}
    if not getattr(args, "no_registry", False):
        try:
            reg = registry.refresh_all(paths_mod.registry_cache_path())
            out["registry"] = {"total": reg["total"], "sources": reg["sources"],
                               "portals": reg["portals"],
                               "note": "Coverage lane worked via the CA DROP one-shot + CCPA email, "
                                       "not the people-search scan. VT/OR/TX are search portals (no "
                                       "bulk export); CA is the superset. See `drop` and `registry`."}
        except Exception as exc:  # noqa: BLE001 - registry pull is best-effort
            out["registry_error"] = str(exc)
    _out(out)


def cmd_registry(args) -> None:
    recs = brokers_mod.load_registry_cache()
    if not recs:
        _out({"registered_brokers": 0,
              "note": "registry empty - run `refresh-brokers` (pulls the CA Data Broker Registry)"})
        return
    fcra = sum(1 for r in recs if (r.get("optout") or {}).get("fcra"))
    out = {"registered_brokers": len(recs), "fcra_regulated": fcra,
           "source": "CA Data Broker Registry (CPPA, 2025)", "drop_url": registry.DROP_URL,
           "other_state_portals": registry.portals()}
    if args.search:
        q = args.search.lower()
        hits = [r for r in recs if q in (r.get("name") or "").lower()
                or q in (r.get("id") or "") or q in ((r.get("optout") or {}).get("email") or "").lower()]
        out["matches"] = [{"id": r["id"], "name": r["name"],
                           "email": (r.get("optout") or {}).get("email"),
                           "url": (r.get("optout") or {}).get("url"),
                           "fcra": (r.get("optout") or {}).get("fcra")} for r in hits[:args.limit]]
        out["match_count"] = len(hits)
    _out(out)


def cmd_drop(args) -> None:
    """The one-shot legal lever: CA DROP deletes from ALL registered brokers at once."""
    d = _require_subject(args.subject)
    dossier_mod.require_authorized(d)
    reg = brokers_mod.load_registry_cache()
    res = (d.get("residency_jurisdiction") or "US").upper()
    eligible = res.startswith("US-CA")
    if args.filed:
        prefs = d.setdefault("preferences", {})
        prefs["drop_filed_at"] = dossier_mod.now()
        dossier_mod.save(d)
        _out({"subject": args.subject, "drop_filed_at": prefs["drop_filed_at"],
              "note": "recorded; `next` will stop surfacing the DROP one-shot"})
        return
    _out({
        "subject": args.subject,
        "eligible": eligible,
        "residency": res,
        "drop_url": registry.DROP_URL,
        "covers_registered_brokers": len(reg),
        "steps": ([
            "Go to privacy.ca.gov/drop and create/verify a DROP account (CA resident).",
            "Submit ONE deletion request; it applies to EVERY registered data broker "
            f"({len(reg)} in the current registry). Brokers must process starting 2026-08-01.",
            "After filing, run `drop <subject> --filed` so the loop stops re-surfacing it.",
        ] if eligible else [
            "DROP is a California mechanism; this subject's residency is not US-CA.",
            "Parity path for non-CA: work the people-search sites via `next`, and send targeted "
            "CCPA/GDPR deletion emails to registry brokers that hold this person's data "
            "(`registry --search`, then `send-email`).",
        ]),
        "note": "DROP is the highest-leverage removal: one request covers the whole registry.",
    })


def cmd_plan(args) -> None:
    d = _require_subject(args.subject)
    dossier_mod.require_authorized(d)
    cfg = config_mod.load_config()
    bl = brokers_mod.by_priority(*(args.priority or [])) if args.priority else brokers_mod.load_all()
    bcc = config_mod.browser_clears_captcha(cfg)
    if getattr(args, "batch", False):
        _out(tiers.batch_plan(d, bl, cfg, ledger_mod.load(args.subject), bcc))
    else:
        _out(tiers.plan(d, bl, cfg, bcc))


def cmd_fanout(args) -> None:
    d = _require_subject(args.subject)
    dossier_mod.require_authorized(d)
    bl = brokers_mod.by_priority(*(args.priority or [])) if args.priority else brokers_mod.load_all()
    grouping = tiers.fanout(bl, batch_size=args.size)
    mode = "scan AND opt-out (operator authorized submissions)" if args.optout \
        else "READ-ONLY scan (submit nothing; reconnaissance only)"
    batches = []
    for i, ids in enumerate(grouping["batches"], 1):
        brief = (
            f"You are scan worker {i} of {len(grouping['batches'])} for the `unbroker` skill. First "
            f"load the `unbroker` skill and read its references/methods.md. Use the `web` toolset "
            f"(web_search `site:` + web_extract), NOT `browser` (browser navigation is heavy and times "
            f"out). Subject id: {args.subject}. Handle ONLY these brokers: {', '.join(ids)}. "
            f"For EACH broker: read references/brokers/<id>.json; run EVERY search vector from "
            f"`pdd.py plan {args.subject}` (filtered to your brokers); build URLs from search.url_patterns "
            f"and heed url_format_quirks; a 404 is INCONCLUSIVE (rebuild/try the on-site search box), not "
            f"not_found. ECONOMY: at most ~3 web calls per broker; the moment a page shows antibot "
            f"(Cloudflare 'just a moment'/DataDome) or hangs, record `blocked` and move on -- do NOT "
            f"retry-loop. Confirm the SUBJECT vs namesakes/relatives by ADDRESS/DOB before recording "
            f"`found` (ignore SEO-templated page titles/intro that just echo the query -- require a real "
            f"result card; a public property/address record with no displayed personal NAME is "
            f"not_found, not found). Record each outcome via `pdd.py record {args.subject} <broker> "
            f"<found|not_found|indirect_exposure|blocked> --found <bool> --evidence '{{\"listing_urls\":[...]}}'`. "
            f"Mode: {mode}. Broker JSON files are READ-ONLY for you -- do NOT edit them; if you discover "
            f"a URL/quirk, put it in your report for the parent to fold in. Return a concise structured "
            f"per-broker report."
        )
        batches.append({"batch": i, "brokers": ids, "brief": brief})
    _out({
        "subject": args.subject,
        "broker_count": grouping["broker_count"],
        "batch_size": grouping["batch_size"],
        "should_fanout": grouping["should_fanout"],
        "batch_count": len(batches),
        "batches": batches,
        "instruction": (
            "If should_fanout is true you MUST spawn ONE delegate_task subagent per batch IN PARALLEL, "
            "passing each batch's `brief`; do not scan all brokers yourself sequentially. Wait for every "
            "report, consolidate, then proceed to opt-outs. If false, just scan the brokers inline."
        ),
    })


def cmd_record(args) -> None:
    d = _require_subject(args.subject)
    dossier_mod.require_authorized(d)
    broker = brokers_mod.get(args.broker)
    # Auto-stamp follow-up scheduling (next_recheck_at / removal_confirmed_at) so the
    # autonomous loop knows when to come back without anyone remembering to set it.
    fields = ledger_mod.followup_fields(args.state, broker, d)
    if args.found is not None:
        fields["found"] = args.found
    if args.evidence:
        fields["evidence"] = json.loads(args.evidence)
    if args.reason:
        fields["human_task_reason"] = args.reason
    case = ledger_mod.transition(args.subject, args.broker, args.state, **fields)
    if args.disclosed:
        ledger_mod.log_disclosure(args.subject, args.broker, args.disclosed, args.channel or "unknown")
    _out({"broker": args.broker, "state": case["state"],
          "next_recheck_at": case.get("next_recheck_at")})


def _email_request(d: dict, b: dict, kind: str, listings, identifiers) -> tuple[dict, list[str]]:
    """Least-disclosure (fields, disclosed_names) for an opt-out/legal email of KIND.

    A removal letter must self-identify. Name + a contact email are already known to the
    broker (the name is displayed on the very listing being removed), so not extra exposure.
    """
    fields = dossier_mod.select_disclosure(d, (b.get("optout") or {}).get("inputs", []))
    ident = d.get("identity", {})
    if ident.get("full_name"):
        fields.setdefault("full_name", ident["full_name"])
    fields.setdefault("contact_email", dossier_mod.contact_email(d) or "")
    if listings:
        fields["listing_urls"] = listings
    if kind == "ccpa_indirect":
        # Indirect exposure: name ONLY the subject's own identifiers to scrub from a third party's
        # record. Default to the contact email + the subject's name-as-relative if none specified.
        # The indirect template renders ONLY these placeholders; do not over-report disclosure with
        # unrelated dossier fields (phone/street/postal) that select_disclosure happened to populate.
        ids = list(identifiers or [])
        if not ids:
            ids = [contact for contact in [dossier_mod.contact_email(d)] if contact]
            ids.append(f'the name "{ident.get("full_name")}" where it appears as a relative/associated person')
        fields = {
            "full_name": fields.get("full_name"),
            "contact_email": fields.get("contact_email"),
            "listing_urls": fields.get("listing_urls"),
            "my_identifiers": ids,
        }
        return fields, ["contact_email", "full_name", "my_identifiers"]
    return fields, sorted(fields.keys())


def cmd_render_email(args) -> None:
    d = _require_subject(args.subject)
    dossier_mod.require_authorized(d)
    b = brokers_mod.get(args.broker)
    if not b:
        sys.exit(f"error: unknown broker {args.broker!r}")
    kind = getattr(args, "kind", "generic") or "generic"
    fields, disclosed = _email_request(d, b, kind, args.listing, getattr(args, "identifier", None))
    if kind == "generic":
        draft = email_modes.render_draft(b, fields)
    else:
        draft = email_modes.render_request_draft(b, fields, kind=kind)
    ledger_mod.log_disclosure(args.subject, args.broker, list(disclosed), f"email_draft:{kind}")
    _out({"draft": str(draft), "kind": kind, "disclosed_fields": disclosed})


def cmd_send_email(args) -> None:
    """Mode B: render AND deliver the opt-out/legal request - no human in the loop.

    Sends ONLY to an address the broker record itself declares (emailer enforces it),
    then records the ledger transition + disclosure and auto-stamps the recheck date.
    """
    d = _require_subject(args.subject)
    dossier_mod.require_authorized(d)
    b = brokers_mod.get(args.broker)
    if not b:
        sys.exit(f"error: unknown broker {args.broker!r}")
    cfg = config_mod.load_config()
    mode = cfg.get("email_mode")
    if mode not in ("programmatic", "alias", "browser"):
        sys.exit("error: email_mode is draft_only; run `setup --email-mode browser` (no password; "
                 "sends via your logged-in webmail) or `--email-mode programmatic`, or use "
                 "`render-email` and send it yourself")
    if not args.listing:
        sys.exit("error: --listing <confirmed-url> is required (verify-before-disclose: never "
                 "email a broker about an unconfirmed listing)")
    # Idempotency: don't re-send if this case is already submitted/beyond (prevents duplicate
    # requests when an action is retried). --force overrides.
    _POST_SUBMIT = {"submitted", "verification_pending", "awaiting_processing", "confirmed_removed"}
    current = ledger_mod.get_case(args.subject, args.broker).get("state")
    if current in _POST_SUBMIT and not getattr(args, "force", False):
        _out({"skipped": True, "broker": args.broker, "state": current,
              "note": "already submitted; not re-sending (idempotent). Use --force to re-send."})
        return
    kind = getattr(args, "kind", "generic") or "generic"
    fields, disclosed = _email_request(d, b, kind, args.listing, getattr(args, "identifier", None))
    body = legal.render_optout_email(b, fields) if kind == "generic" else legal.render_request(kind, b, fields)

    if mode == "browser":
        # No network / no credentials: hand the agent a recipient-locked payload to send in the
        # operator's webmail via browser_* tools. State still records deterministically here.
        payload = emailer.browser_send_payload(b, body, to=args.to)
        ledger_mod.log_disclosure(args.subject, args.broker, list(disclosed), f"email_browser:{kind}")
        case = ledger_mod.transition(args.subject, args.broker, "submitted",
                                     **ledger_mod.followup_fields("submitted", b, d))
        _out({"send_via": "browser", "compose": payload, "kind": kind, "disclosed_fields": disclosed,
              "state": case["state"], "next_recheck_at": case.get("next_recheck_at"),
              "instruction": "In the operator's logged-in webmail, compose a NEW email to compose.to "
                             "with compose.subject/body EXACTLY (disclose nothing beyond it) and send "
                             "it via browser_* tools. Then use `verify-link` on any confirmation reply.",
              "note": "recipient is locked to the broker's declared address"})
        return

    result = emailer.send(b, body, to=args.to,
                          min_interval=float(cfg.get("email_min_interval_seconds", 0) or 0))
    ledger_mod.log_disclosure(args.subject, args.broker, list(disclosed), f"email_sent:{kind}")
    case = ledger_mod.transition(args.subject, args.broker, "submitted",
                                 **ledger_mod.followup_fields("submitted", b, d))
    _out({"sent": result, "send_via": "smtp", "kind": kind, "disclosed_fields": disclosed,
          "state": case["state"], "next_recheck_at": case.get("next_recheck_at"),
          "note": "if this broker verifies by email, `poll-verification` will pick up the link"})


def cmd_verify_link(args) -> None:
    """Extract a broker's verification link from email text the agent read in webmail (browser mode).

    IMAP-free counterpart to `poll-verification`: the agent opens the broker's confirmation email
    in the operator's webmail, pastes the body here, and gets the anti-phishing-scored link back.
    """
    _require_subject(args.subject)
    b = brokers_mod.get(args.broker)
    if not b:
        sys.exit(f"error: unknown broker {args.broker!r}")
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not text:
        sys.exit("error: provide --text '<email body>' (or --file) from the broker's confirmation email")
    link = email_modes.extract_verification_link(text, b)
    _out({"broker": args.broker, "verification_link": link,
          "next": ("browser_navigate the link IN THE SAME browser (sessions are browser-bound), "
                   f"complete the flow, then `record {args.subject} {args.broker} awaiting_processing`"
                   if link else
                   "no broker/opt-out-scoped link found in that text; confirm you opened the right email")})


def cmd_poll_verification(args) -> None:
    """Poll the inbox for brokers' verification links (Mode B) - replaces the human click-chase.

    For each in-flight case (submitted / verification_pending with email_verification),
    extract the broker's link (anti-phishing scored). A found link auto-advances
    submitted -> verification_pending (the email HAS arrived); the agent must then OPEN
    the link in its own browser (sessions are browser-bound) and record the next state.
    """
    d = _require_subject(args.subject)
    dossier_mod.require_authorized(d)
    led = ledger_mod.load(args.subject)
    targets = []
    for bid, case in sorted(led.items()):
        if args.broker and bid != args.broker:
            continue
        if case.get("state") not in ("submitted", "verification_pending"):
            continue
        b = brokers_mod.get(bid)
        if b and (((b.get("optout") or {}).get("requires")) or {}).get("email_verification"):
            targets.append((bid, case, b))
    if not targets:
        _out({"subject": args.subject, "results": [],
              "note": "no in-flight cases awaiting email verification"})
        return
    results = []
    for bid, case, b in targets:
        hit = emailer.find_verification_link(b, since_days=args.since_days)
        if hit:
            if case.get("state") == "submitted":
                ledger_mod.transition(args.subject, bid, "verification_pending",
                                      **ledger_mod.followup_fields("verification_pending", b, d))
            results.append({"broker": bid, "verification_link": hit["link"],
                            "email_from": hit.get("from"), "email_subject": hit.get("subject"),
                            "next": f"browser_navigate the link IN THE AGENT'S OWN BROWSER, complete "
                                    f"the flow, then `record {args.subject} {bid} awaiting_processing` "
                                    f"(or confirmed_removed only after a verifying re-scan)"})
        else:
            results.append({"broker": bid, "verification_link": None,
                            "next": "no matching email yet; poll again later (next_recheck_at is set)"})
    _out({"subject": args.subject, "results": results})


def cmd_next(args) -> None:
    d = _require_subject(args.subject)
    dossier_mod.require_authorized(d)
    cfg = config_mod.load_config()
    bl = brokers_mod.by_priority(*(args.priority or [])) if args.priority else brokers_mod.load_all()
    _out(autopilot.next_actions(d, bl, cfg, ledger_mod.load(args.subject)))


def cmd_tasks(args) -> None:
    _require_subject(args.subject)
    print(report_mod.human_tasks_markdown(args.subject))


def cmd_due(args) -> None:
    _require_subject(args.subject)
    cases = ledger_mod.due(args.subject)
    _out({"subject": args.subject, "due_count": len(cases),
          "cases": [{"broker_id": c.get("broker_id"), "state": c.get("state"),
                     "next_recheck_at": c.get("next_recheck_at")} for c in cases],
          "note": "run `next` for the concrete follow-up action per case"})


def cmd_show(args) -> None:
    """Read a case's recorded state + evidence (so the parent can re-verify a subagent's `found`
    without re-deriving listing URLs)."""
    _require_subject(args.subject)
    case = ledger_mod.get_case(args.subject, args.broker)
    _out({"broker": args.broker, "state": case.get("state"), "found": case.get("found"),
          "evidence": case.get("evidence") or {},
          "disclosure_log": case.get("disclosure_log") or [],
          "next_recheck_at": case.get("next_recheck_at"),
          "human_task_reason": case.get("human_task_reason"),
          "history": case.get("history") or []})


def cmd_status(args) -> None:
    _require_subject(args.subject)
    print(report_mod.render_markdown(args.subject))


def cmd_report(args) -> None:
    _require_subject(args.subject)
    if args.sheets:
        _out(report_mod.sheets_rows(args.subject))
    else:
        print(report_mod.render_markdown(args.subject))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdd", description="unbroker helper CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="write install config (easiest-first defaults; --auto = most autonomous)")
    s.add_argument("--auto", action="store_true",
                   help="detect capabilities and pick the most autonomous valid config (no questions)")
    s.add_argument("--autonomy", dest="autonomy", choices=sorted(config_mod.VALID["autonomy"]))
    s.add_argument("--email-mode", dest="email_mode", choices=sorted(config_mod.VALID["email_mode"]))
    s.add_argument("--browser-backend", dest="browser_backend", choices=sorted(config_mod.VALID["browser_backend"]))
    s.add_argument("--tracker-backend", dest="tracker_backend", choices=sorted(config_mod.VALID["tracker_backend"]))
    s.add_argument("--encryption", dest="encryption", choices=sorted(config_mod.VALID["encryption"]))
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("doctor", help="readiness check: config, brokers, available upgrades")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("cdp",
                       help="launch/detect the operator's Chrome over CDP (Phase-2 browser + webmail)")
    s.add_argument("--port", type=int, default=cdp.DEFAULT_PORT, help="remote debugging port (default 9222)")
    s.add_argument("--profile",
                   help="user-data-dir (default: $HERMES_HOME/chrome-debug, a dedicated debug profile)")
    s.add_argument("--browser", help="path to (or PATH name of) a Chrome/Chromium/Brave/Edge binary")
    s.add_argument("--check", action="store_true",
                   help="only report whether a debug browser is live; do not launch")
    s.add_argument("--print", dest="print_only", action="store_true",
                   help="print the launch command instead of launching it (run it yourself)")
    s.set_defaults(func=cmd_cdp)

    s = sub.add_parser("intake", help="create a subject dossier (records consent)")
    s.add_argument("--json", help="path to a dossier JSON file (overrides flags)")
    s.add_argument("--full-name")
    s.add_argument("--alias", action="append", metavar="NAME",
                   help="other name the subject is listed under (maiden/married/nickname); repeatable")
    s.add_argument("--email", action="append", metavar="EMAIL", help="repeatable")
    s.add_argument("--phone", action="append", metavar="PHONE", help="repeatable")
    s.add_argument("--street", help="current street line1 (enables reverse-address search)")
    s.add_argument("--city")
    s.add_argument("--state")
    s.add_argument("--postal")
    s.add_argument("--prior-location", dest="prior_location", action="append", metavar="City,ST",
                   help="a past city/state (or City,ST,ZIP); repeatable")
    s.add_argument("--dob", help="date of birth YYYY-MM-DD (only used if a broker requires it)")
    s.add_argument("--contact-email", dest="contact_email",
                   help="which email to use for opt-out correspondence (default: first)")
    s.add_argument("--residency", help="e.g. US, US-CA")
    s.add_argument("--consent", action="store_true", help="subject authorizes removal on their behalf")
    s.add_argument("--consent-method", default="self", choices=["self", "written_authorization", "poa"])
    s.add_argument("--email-mode", dest="email_mode", choices=sorted(config_mod.VALID["email_mode"]))
    s.set_defaults(func=cmd_intake)

    s = sub.add_parser("brokers", help="list the broker database (curated + live)")
    s.add_argument("--priority", action="append", choices=["crucial", "high", "standard", "long_tail"])
    s.set_defaults(func=cmd_brokers)

    s = sub.add_parser("refresh-brokers",
                       help="pull the latest BADBOOL people-search list + the CA data broker registry")
    s.add_argument("--no-registry", dest="no_registry", action="store_true",
                   help="skip the CA registry pull (BADBOOL people-search only)")
    s.set_defaults(func=cmd_refresh_brokers)

    s = sub.add_parser("registry",
                       help="CA Data Broker Registry coverage (hundreds of brokers; DROP/email lane)")
    s.add_argument("--search", help="find registered brokers by name / id / email substring")
    s.add_argument("--limit", type=int, default=25, help="max matches to print (default 25)")
    s.set_defaults(func=cmd_registry)

    s = sub.add_parser("drop",
                       help="CA DROP one-shot: delete from ALL registered brokers in one request")
    s.add_argument("subject")
    s.add_argument("--filed", action="store_true", help="mark DROP as filed (stops `next` surfacing it)")
    s.set_defaults(func=cmd_drop)

    s = sub.add_parser("plan", help="compute per-broker tier + next action for a subject")
    s.add_argument("subject")
    s.add_argument("--priority", action="append", choices=["crucial", "high", "standard", "long_tail"])
    s.add_argument("--batch", action="store_true",
                   help="phase-oriented batch view: overlays ledger state, groups by next action "
                        "(unscanned/found/indirect/blocked/in_progress/done), collapses ownership clusters")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("fanout", help="batch brokers into parallel delegate_task subagents (large runs)")
    s.add_argument("subject")
    s.add_argument("--priority", action="append", choices=["crucial", "high", "standard", "long_tail"])
    s.add_argument("--size", type=int, default=5, help="brokers per subagent batch (default 5; 8+ times out)")
    s.add_argument("--optout", action="store_true",
                   help="brief authorizes opt-out submission (default: read-only scan)")
    s.set_defaults(func=cmd_fanout)

    s = sub.add_parser("record", help="record a ledger state transition after an agent action")
    s.add_argument("subject")
    s.add_argument("broker")
    s.add_argument("state", choices=ledger_mod.STATES)
    s.add_argument("--found", type=lambda v: v.strip().lower() in ("1", "true", "yes", "y"))
    s.add_argument("--evidence", help="JSON object stored as case.evidence")
    s.add_argument("--disclosed", action="append", metavar="FIELD", help="field name disclosed")
    s.add_argument("--channel", help="disclosure channel, e.g. web_form / email")
    s.add_argument("--reason", help="for human_task_queued: why a human is needed (shown in `tasks`)")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("next", help="autonomous action queue: exactly what to do right now")
    s.add_argument("subject")
    s.add_argument("--priority", action="append", choices=["crucial", "high", "standard", "long_tail"])
    s.set_defaults(func=cmd_next)

    s = sub.add_parser("send-email", help="Mode B: render AND send the opt-out/legal request (records it)")
    s.add_argument("subject")
    s.add_argument("broker")
    s.add_argument("--listing", action="append", metavar="URL", required=False,
                   help="confirmed listing URL (required: verify-before-disclose)")
    s.add_argument("--kind", choices=["generic", "ccpa", "ccpa_agent", "ccpa_indirect", "gdpr"],
                   default="generic")
    s.add_argument("--identifier", action="append", metavar="ID",
                   help="(ccpa_indirect only) a specific own-identifier to remove; repeatable")
    s.add_argument("--to", help="override recipient (must be an address the broker record declares)")
    s.add_argument("--force", action="store_true", help="re-send even if already submitted (default: idempotent skip)")
    s.set_defaults(func=cmd_send_email)

    s = sub.add_parser("poll-verification",
                       help="Mode B (IMAP): poll the inbox for brokers' verification links (anti-phishing scored)")
    s.add_argument("subject")
    s.add_argument("--broker", help="only this broker (default: every in-flight verification case)")
    s.add_argument("--since-days", dest="since_days", type=int, default=3)
    s.set_defaults(func=cmd_poll_verification)

    s = sub.add_parser("verify-link",
                       help="browser mode: extract a broker's verification link from pasted webmail text")
    s.add_argument("subject")
    s.add_argument("broker")
    s.add_argument("--text", help="the confirmation email body (read from the operator's webmail)")
    s.add_argument("--file", help="path to a file with the email body (alternative to --text)")
    s.set_defaults(func=cmd_verify_link)

    s = sub.add_parser("tasks", help="ONE consolidated human-task digest (present at end of run)")
    s.add_argument("subject")
    s.set_defaults(func=cmd_tasks)

    s = sub.add_parser("show", help="read a case's state + evidence (for parent re-verification)")
    s.add_argument("subject")
    s.add_argument("broker")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("due", help="cases whose recheck window has arrived (cron re-scan queue)")
    s.add_argument("subject")
    s.set_defaults(func=cmd_due)

    s = sub.add_parser("render-email", help="render a Mode-A opt-out / legal-request draft (least-disclosure)")
    s.add_argument("subject")
    s.add_argument("broker")
    s.add_argument("--listing", action="append", metavar="URL", help="confirmed listing URL")
    s.add_argument("--kind", choices=["generic", "ccpa", "ccpa_agent", "ccpa_indirect", "gdpr"],
                   default="generic",
                   help="request type. 'ccpa_indirect' = delete MY identifiers from a third party's "
                        "record (indirect exposure); default 'generic' opt-out.")
    s.add_argument("--identifier", action="append", metavar="ID",
                   help="(ccpa_indirect only) a specific own-identifier to request removal of "
                        "(e.g. an email or phone). Repeatable. Defaults to the contact email + "
                        "name-as-relative if omitted.")
    s.set_defaults(func=cmd_render_email)

    s = sub.add_parser("status", help="print a Markdown status report")
    s.add_argument("subject")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("report", help="status report (default) or --sheets rows")
    s.add_argument("subject")
    s.add_argument("--sheets", action="store_true", help="emit Google Sheets rows as JSON")
    s.set_defaults(func=cmd_report)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (PermissionError, ValueError, RuntimeError, FileNotFoundError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
