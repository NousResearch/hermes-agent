"""DOM inspection script for grok.com mode dropdown.

Opens grok.com using the same persistent profile as PatchrightGrokClient
(no separate login), takes screenshots, and dumps the DOM around the mode
dropdown so we can write reliable selectors for the actual UI clicker.

Output:
  /tmp/grok_dropdown_closed.png    — full page with dropdown closed
  /tmp/grok_dropdown_open.png      — full page with dropdown opened
  /tmp/grok_dropdown_dom.html      — full page HTML
  /tmp/grok_dropdown_findings.txt  — extracted candidate selectors

Run:
  cd ~/Projects/hermes-agent && python3 tests/grok/debug_mode_dropdown.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.patchright_client import _get_profile_dir, _GROK_URL


# Mode labels we'd want to identify in the dropdown menu
TARGET_LABELS = ["Auto", "Fast", "Expert", "Grok 4.3 (beta)", "Heavy"]

# Heuristic: where the "current mode" trigger button might be in DOM
# We try multiple strategies because grok.com uses React with random class names
TRIGGER_CANDIDATE_SELECTORS = [
    'button:has-text("Auto")',
    'button:has-text("Fast")',
    'button:has-text("Expert")',
    'button:has-text("Heavy")',
    '[role="combobox"]',
    '[aria-haspopup="menu"]',
    '[aria-haspopup="listbox"]',
    '[aria-expanded]',
    '[data-testid*="model"]',
    '[data-testid*="mode"]',
]


def main():
    from patchright.sync_api import sync_playwright

    profile_dir = _get_profile_dir()
    if not profile_dir.exists():
        print(f"[ERROR] Profile dir not found at {profile_dir}")
        print("        Run a regular hermes session first to create + log in.")
        sys.exit(1)

    out_dir = Path("/tmp")
    closed_png = out_dir / "grok_dropdown_closed.png"
    open_png = out_dir / "grok_dropdown_open.png"
    dom_html = out_dir / "grok_dropdown_dom.html"
    findings = out_dir / "grok_dropdown_findings.txt"

    print(f"[1] Launching Patchright with profile {profile_dir}")
    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=False,  # show window so user can see what's happening
        viewport={"width": 1280, "height": 900},
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()

    print(f"[2] Navigating to {_GROK_URL}")
    page.goto(_GROK_URL, wait_until="networkidle")
    time.sleep(3)  # let any post-load animations settle

    # Dismiss cookie banner if present
    try:
        btn = page.query_selector("#onetrust-accept-btn-handler")
        if btn:
            btn.click()
            time.sleep(1)
    except Exception:
        pass

    # Verify we're past login
    content = page.content()
    if "Sign in" in content and "Ask anything" not in content:
        print("[ERROR] Not logged in — please log in at grok.com first")
        ctx.close()
        pw.stop()
        sys.exit(1)

    print(f"[3] Screenshot: dropdown closed → {closed_png}")
    page.screenshot(path=str(closed_png), full_page=False)

    print("[4] Probing trigger candidate selectors...")
    notes = []
    notes.append("=== TRIGGER CANDIDATES (looking for the mode-selector button) ===\n")

    found_trigger = None
    for sel in TRIGGER_CANDIDATE_SELECTORS:
        try:
            elements = page.query_selector_all(sel)
            for el in elements:
                try:
                    text = (el.inner_text() or "").strip()
                    tag = el.evaluate("e => e.tagName")
                    aria_haspopup = el.get_attribute("aria-haspopup")
                    aria_expanded = el.get_attribute("aria-expanded")
                    role = el.get_attribute("role")
                    cls = (el.get_attribute("class") or "")[:120]
                    testid = el.get_attribute("data-testid")
                    line = (
                        f"  selector='{sel}'  tag={tag}  text='{text[:60]}'\n"
                        f"    aria-haspopup={aria_haspopup!r}  aria-expanded={aria_expanded!r}  role={role!r}\n"
                        f"    data-testid={testid!r}  class='{cls}...'\n"
                    )
                    notes.append(line)
                    # Keep the first one that looks like a mode button
                    if found_trigger is None and any(
                        label in text for label in TARGET_LABELS
                    ):
                        found_trigger = (sel, el, text)
                        notes.append(f"    ^^^ LIKELY TRIGGER (text matches mode label)\n")
                except Exception as e:
                    notes.append(f"  selector='{sel}'  [error inspecting: {e}]\n")
        except Exception as e:
            notes.append(f"  selector='{sel}'  [query failed: {e}]\n")

    if not found_trigger:
        notes.append(
            "\n[!] No trigger matched a mode label by text. Manual inspection "
            "needed — open /tmp/grok_dropdown_dom.html and search for 'Auto'.\n"
        )
    else:
        sel, _, text = found_trigger
        notes.append(f"\n[OK] Best trigger guess: selector='{sel}' currently shows '{text}'\n")

    # Try to open the dropdown if we found a trigger
    if found_trigger:
        sel, el, _ = found_trigger
        print(f"[5] Clicking trigger to open dropdown: {sel}")
        try:
            el.click()
            time.sleep(1.0)

            print(f"[6] Screenshot: dropdown open → {open_png}")
            page.screenshot(path=str(open_png), full_page=False)

            notes.append("\n=== DROPDOWN MENU ITEMS (looking for mode options) ===\n")
            for label in TARGET_LABELS:
                # Try several strategies for finding the menu item
                strategies = [
                    f'text="{label}"',
                    f'[role="menuitem"]:has-text("{label}")',
                    f'[role="option"]:has-text("{label}")',
                    f'button:has-text("{label}")',
                    f'div:has-text("{label}")',
                ]
                for strat in strategies:
                    try:
                        item = page.query_selector(strat)
                        if item:
                            tag = item.evaluate("e => e.tagName")
                            role = item.get_attribute("role")
                            cls = (item.get_attribute("class") or "")[:100]
                            testid = item.get_attribute("data-testid")
                            notes.append(
                                f"  '{label}' found via: {strat}\n"
                                f"    tag={tag}  role={role!r}  data-testid={testid!r}\n"
                                f"    class='{cls}...'\n"
                            )
                            break
                    except Exception:
                        continue
                else:
                    notes.append(f"  '{label}' NOT FOUND via any strategy\n")
        except Exception as e:
            notes.append(f"\n[!] Could not click trigger: {e}\n")
    else:
        print("[5] Skipping dropdown-open phase (no trigger identified)")

    print(f"[7] Saving full DOM → {dom_html}")
    dom_html.write_text(page.content())

    print(f"[8] Saving findings → {findings}")
    findings.write_text("".join(notes))

    print("\n=== DONE ===")
    print(f"  Closed screenshot: {closed_png}")
    print(f"  Open screenshot:   {open_png}")
    print(f"  Full DOM HTML:     {dom_html}")
    print(f"  Findings text:     {findings}")
    print("\nReview the screenshots + findings.txt to identify the right selectors.")
    print("Press Ctrl+C in this terminal to close the browser when done inspecting.\n")

    # Keep browser open for 30s so user can interact / inspect
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass

    ctx.close()
    pw.stop()


if __name__ == "__main__":
    main()
