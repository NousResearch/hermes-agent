"""The Control Centre, driven in a browser against a real control plane.

Not a unit test and deliberately not run by pytest: it needs a serving control plane, a
real tenant bundle on disk and a Chromium build, and it asserts against the *files the
runtime ended up with* rather than against mocks. That is the point — every failure it has
caught so far was invisible to the unit tests, because each was in the wiring between a
form and a route rather than in either one.

Run it:

    python -m nova apply <bundle>
    python -m nova serve <bundle> --host 127.0.0.1 --port 19200 &
    python tests/browser/control_centre_e2e.py <runtime-home> <bundle>

Exits non-zero on the first failed assertion group, and prints a pass/fail line per check.
""" 
import json, os, sys
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

BASE = os.environ.get("NOVA_E2E_BASE", "http://127.0.0.1:19200")
HOME = Path(sys.argv[1]); BUNDLE = Path(sys.argv[2])
CHROME = os.environ.get("NOVA_E2E_CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
ok, fail = [], []

def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail and not cond else ""))

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1500, "height": 1100})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    def goto(frag):
        # Navigating to the hash already in the address bar is a no-op, so component state
        # (the open tab) would survive what the test intends as a fresh load. Blank first.
        page.goto("about:blank")
        page.goto(f"{BASE}/#{frag}", wait_until="networkidle")
        page.wait_for_timeout(350)

    # ---- CREATE ----------------------------------------------------------
    print("\n# Create agent")
    goto("/agents")
    page.get_by_role("button", name="New agent").click()
    page.wait_for_selector("#new-agent-id")
    page.fill("#new-agent-id", "night-ops")
    page.fill("#new-agent-name", "Night Ops")
    page.fill("#new-agent-desc", "Overnight inventory sweeps.")
    page.get_by_role("button", name="Next").click()
    page.fill("#new-agent-soul", "You are Night Ops. Be terse and escalate anything over 5%.")
    page.get_by_role("button", name="Next").click()          # -> Model
    page.get_by_role("button", name="Next").click()          # -> Capabilities
    page.wait_for_selector("text=Toolsets")
    page.get_by_role("button", name="read_inventory", exact=True).click()
    page.get_by_role("button", name="web", exact=True).first.click()
    page.get_by_role("button", name="Next").click()          # -> Channels
    page.get_by_role("button", name="Next").click()          # -> Review
    page.wait_for_selector("text=Nothing has been created yet")
    check("review step shows the draft", "Night Ops" in page.content())
    page.get_by_role("button", name="Create agent").click()
    page.wait_for_selector("text=Applied — the runtime has it now", timeout=25000)
    check("create reports applied", True)
    # The result is shown before navigating: a "saved but not applied" outcome is exactly
    # the one somebody needs to read, and navigating on the response would hide it.
    page.wait_for_selector("text=now exists in this tenant", timeout=5000)
    check("create shows the outcome before navigating", True)
    page.get_by_role("button", name="Open night-ops").click()
    page.wait_for_timeout(500)

    # backend truth
    agents = json.loads(page.evaluate(
        "fetch('/platform/v1/agents').then(r=>r.text())"))
    check("agent is in the API list", any(a["id"] == "night-ops" for a in agents["agents"]))
    check("agent file on disk", (BUNDLE/"agents"/"night-ops.yaml").is_file())
    check("persona on disk", (BUNDLE/"prompts"/"night-ops.md").is_file())
    check("profile materialised", (HOME/"profiles"/"night-ops").is_dir())

    # ---- EDIT + PERSIST ---------------------------------------------------
    print("\n# Edit description, save, refresh")
    goto("/agents/night-ops")
    page.wait_for_selector("#desc-night-ops")
    page.fill("#desc-night-ops", "Edited from the Control Centre.")
    save = page.get_by_role("button", name="Save", exact=True).first
    save.click()
    page.wait_for_selector("text=Applied — the runtime has it now", timeout=25000)
    goto("/agents/night-ops")
    page.wait_for_selector("#desc-night-ops")
    check("description persisted across a reload",
          page.input_value("#desc-night-ops") == "Edited from the Control Centre.")

    print("\n# Save is disabled while clean")
    check("save disabled with no changes",
          page.get_by_role("button", name="Save", exact=True).first.is_disabled())

    # ---- SOUL (existing flow must still work) -----------------------------
    print("\n# Soul (regression)")
    page.get_by_role("tab", name="Soul").click()
    page.wait_for_selector("textarea#soul-night-ops")
    page.fill("textarea#soul-night-ops", "You are Night Ops. MARKER-P2")
    page.get_by_role("button", name="Save and apply").click()
    page.wait_for_selector("text=Applied — the agent is using it now", timeout=25000)
    check("soul reaches the runtime",
          "MARKER-P2" in (HOME/"profiles"/"night-ops"/"SOUL.md").read_text())

    # ---- CAPABILITIES -----------------------------------------------------
    print("\n# Capabilities")
    goto("/agents/night-ops")
    page.get_by_role("tab", name="Capabilities").click()
    page.wait_for_selector("text=Permissions")
    page.get_by_role("button", name="contact_suppliers", exact=True).click()
    page.get_by_role("button", name="Save permissions").click()
    page.wait_for_selector("text=Applied — the runtime has it now", timeout=25000)
    cfg = json.loads(page.evaluate("fetch('/platform/v1/agents/night-ops/config').then(r=>r.text())"))
    check("permission persisted", "contact_suppliers" in cfg["fields"]["permissions"])
    check("toolset from create persisted", "web" in (cfg["fields"]["tools"].get("toolsets") or []))

    # ---- DUPLICATE --------------------------------------------------------
    print("\n# Duplicate")
    goto("/agents/night-ops")
    page.get_by_role("button", name="Duplicate", exact=True).click()
    page.wait_for_selector("#dup-id-night-ops")
    page.fill("#dup-id-night-ops", "night-ops-eu")
    page.fill("#dup-name-night-ops", "Night Ops EU")
    page.get_by_role("button", name="Create the copy").click()
    page.wait_for_selector("text=Applied — the runtime has it now", timeout=25000)
    agents = json.loads(page.evaluate("fetch('/platform/v1/agents').then(r=>r.text())"))
    dup = next((a for a in agents["agents"] if a["id"] == "night-ops-eu"), None)
    check("duplicate exists", dup is not None)
    check("duplicate starts disabled", bool(dup) and dup.get("enabled") is False)
    check("duplicate carried the persona",
          "MARKER-P2" in (BUNDLE/"prompts"/"night-ops-eu.md").read_text())

    # ---- ARCHIVE / RESTORE ------------------------------------------------
    print("\n# Archive and restore")
    goto("/agents/night-ops")
    page.get_by_role("button", name="Archive", exact=True).click()
    page.wait_for_selector("text=Applied — the runtime has it now", timeout=25000)
    agents = json.loads(page.evaluate("fetch('/platform/v1/agents').then(r=>r.text())"))
    check("archived", next(a for a in agents["agents"] if a["id"]=="night-ops")["enabled"] is False)
    goto("/agents/night-ops")
    page.wait_for_selector("text=Disabled")
    page.get_by_role("button", name="Restore", exact=True).click()
    page.wait_for_selector("text=Applied — the runtime has it now", timeout=25000)
    agents = json.loads(page.evaluate("fetch('/platform/v1/agents').then(r=>r.text())"))
    check("restored", next(a for a in agents["agents"] if a["id"]=="night-ops")["enabled"] is True)

    # ---- SCHEDULES --------------------------------------------------------
    #
    # These need an automation the tenant has already declared. The shipped example bundle
    # declares none, so this section is skipped rather than failed when it is absent —
    # a missing fixture is not a broken control plane, and reporting it as one would train
    # everybody to ignore a red line.
    print("\n# Schedules")
    goto("/agents/operations")
    page.get_by_role("tab", name="Schedules").click()
    page.wait_for_timeout(1500)
    schedules_present = "Nightly sweep" in page.content()
    if not schedules_present:
        print("  SKIP  no 'Nightly sweep' automation in this bundle — schedule checks skipped")
    if schedules_present:
        check("agent's schedule is listed", True)
        check("scheduler liveness is stated", "Nothing is running these schedules" in page.content())
        page.get_by_role("button", name="Edit").first.click()
        page.wait_for_selector("input[id^='sch-name-']")
        name_input = page.locator("input[id^='sch-name-']").first
        name_input.fill("Nightly sweep (edited)")
        page.get_by_role("button", name="Update schedule").click()
        page.wait_for_timeout(1500)
        goto("/agents/operations"); page.get_by_role("tab", name="Schedules").click()
        page.wait_for_selector("text=Nightly sweep (edited)", timeout=15000)
        check("schedule edit persisted across a reload", True)
        page.get_by_role("button", name="Pause").first.click()
        page.wait_for_selector("text=Paused", timeout=15000)
        check("schedule paused", True)

    # ---- FAILURE CASES ----------------------------------------------------
    print("\n# Failure handling")
    goto("/agents/new")
    page.fill("#new-agent-id", "night-ops")
    check("duplicate id rejected in the form", "already exists" in page.content())
    page.fill("#new-agent-id", "Bad Id!")
    check("malformed id rejected in the form", "Lowercase letters" in page.content())

    goto("/agents/does-not-exist")
    page.wait_for_selector("text=No agent")
    check("unknown agent route explains itself", True)

    bad = json.loads(page.evaluate("""
      fetch('/platform/v1/agents/operations/update', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({fields:{permissions:['nope']}})}).then(r=>r.text())"""))
    check("invalid permission is a 400 naming the field",
          bad.get("error", {}).get("status") == 400 and "nope" in bad["error"]["message"])

    badsch = json.loads(page.evaluate("""
      fetch('/platform/v1/automations/x/decide', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({action:'update', updates:{schedule:'not a schedule'}})}).then(r=>r.text())"""))
    # 404 because the id is fake; the schedule grammar is checked against a real one below.
    check("unknown schedule is a 404", badsch.get("error", {}).get("status") == 404)

    # The grammar check needs a real automation to aim at. Skipped, not failed, when the
    # tenant has declared none — see the SCHEDULES section above.
    declared = json.loads(page.evaluate(
        "fetch('/platform/v1/automations').then(r=>r.text())"))["automations"]
    if declared:
        real_id = declared[0]["automation_id"]
        badgrammar = json.loads(page.evaluate(f"""
          fetch('/platform/v1/automations/{real_id}/decide', {{method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{action:'update', updates:{{schedule:'not a schedule'}}}})}}).then(r=>r.text())"""))
        check("invalid schedule refused before it reaches the runtime",
              badgrammar.get("error", {}).get("status") == 400
              and "not valid" in badgrammar["error"]["message"])
    else:
        print("  SKIP  no declared automation to aim the schedule-grammar check at")

    # ---- DELETE -----------------------------------------------------------
    print("\n# Delete (confirmation required)")
    goto("/agents/night-ops-eu")
    page.get_by_role("button", name="Delete…").click()
    page.wait_for_selector("#confirm-night-ops-eu")
    del_btn = page.get_by_role("button", name="Delete night-ops-eu")
    check("delete blocked until the id is typed", del_btn.is_disabled())
    page.fill("#confirm-night-ops-eu", "night-ops-eu")
    del_btn.click()
    page.wait_for_timeout(2500)
    agents = json.loads(page.evaluate("fetch('/platform/v1/agents').then(r=>r.text())"))
    check("agent deleted", not any(a["id"] == "night-ops-eu" for a in agents["agents"]))
    # The duplicate started disabled and apply does not materialise a disabled agent, so it
    # never had a profile. Delete night-ops instead, which does, to test the property that
    # matters: a config edit must not destroy conversation history or a .env NOVA never wrote.
    (HOME/"profiles"/"night-ops"/".env").write_text("SECRET=kept\n")
    goto("/agents/night-ops")
    page.get_by_role("button", name="Delete…").click()
    page.wait_for_selector("#confirm-night-ops")
    page.fill("#confirm-night-ops", "night-ops")
    page.get_by_role("button", name="Delete night-ops", exact=True).click()
    page.wait_for_timeout(2500)
    agents = json.loads(page.evaluate("fetch('/platform/v1/agents').then(r=>r.text())"))
    check("materialised agent deleted", not any(a["id"] == "night-ops" for a in agents["agents"]))
    check("runtime profile left alone", (HOME/"profiles"/"night-ops").is_dir())
    check("the agent's own .env survived",
          (HOME/"profiles"/"night-ops"/".env").read_text() == "SECRET=kept\n")

    # ---- SCHEDULES SCREEN: edit + the link back to the agent ---------------
    print("\n# Schedules screen")
    if not schedules_present:
        print("  SKIP  no declared automation in this bundle — schedules screen checks skipped")
    goto("/automations")
    if schedules_present:
        page.wait_for_selector("text=Nightly sweep (edited)", timeout=15000)
        page.get_by_role("button", name="Edit Nightly sweep (edited)").click()
        page.wait_for_selector("text=Update schedule")
        check("schedule edit panel opens on the schedules screen", True)
        page.get_by_role("button", name="Open agent operations").first.click()
        page.wait_for_timeout(600)
        check("a schedule links back to its agent", "/agents/operations" in page.url)
    else:
        # The screen must still render, with an empty state rather than a crash. That part
        # needs no fixture and is the half worth keeping when there is nothing scheduled.
        page.wait_for_timeout(1200)
        check("the schedules screen renders with nothing scheduled",
              "Automations" in page.content() or "automation" in page.content().lower())

    page.screenshot(path=str(HOME.parent/"p2-final.png"))
    expected = ("failed to load resource",)   # the browser logging our deliberate 4xx probes
    real = [
        e for e in errs
        if "favicon" not in e.lower() and not any(x in e.lower() for x in expected)
    ]
    check("no console errors", not real, str(real[:2]))
    b.close()

print(f"\n=== {len(ok)} passed, {len(fail)} failed ===")
if fail:
    print("failed:", fail)
    sys.exit(1)
