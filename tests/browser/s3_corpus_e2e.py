"""A corpus mirrored from S3, in a real browser.

There is no bucket here, and that is the point of two of these checks: a sync that cannot
reach its bucket must come back as a legible sentence rather than a spinner or a stack
trace, and the corpus must be *unchanged* afterwards.

The rest is about the refusal. A mirrored corpus is read-only from NOVA's side, because an
upload into one survives exactly until the next sync deletes it. So the upload control must
not be on the screen at all — a disabled button that explains itself after the click is a
worse design than a control that was never offered.

Run it:

    python -m nova serve <bundle-with-an-origin> --host 127.0.0.1 --port 19301 &
    python tests/browser/s3_corpus_e2e.py <source-id>
"""

import os
import sys

from playwright.sync_api import sync_playwright

BASE = os.environ.get("NOVA_E2E_BASE", "http://127.0.0.1:19301")
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "company-handbook"
LOCAL = sys.argv[2] if len(sys.argv) > 2 else "product-docs"
SOURCE_TITLE = sys.argv[3] if len(sys.argv) > 3 else "Company Handbook"
LOCAL_TITLE = sys.argv[4] if len(sys.argv) > 4 else "Product Documentation"
ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail and not cond else ""))


with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path=os.environ.get(
            "NOVA_E2E_CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
        ),
        args=["--no-sandbox"],
    )
    page = b.new_page(viewport={"width": 1500, "height": 1150})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)

    def open_corpus(title):
        """Open one corpus card. The control is labelled "Documents" on every card, so it
        has to be reached through the card that names the corpus."""
        card = page.locator("div", has_text=title).filter(
            has=page.get_by_role("button", name="Documents")
        ).last
        card.get_by_role("button", name="Documents").click()
        page.wait_for_timeout(900)

    page.goto(f"{BASE}/#/knowledge", wait_until="networkidle")
    page.wait_for_timeout(600)

    print("\n# A mirrored corpus")
    open_corpus(SOURCE_TITLE)
    content = page.content()
    check("the screen says where the documents come from", "Mirrored from" in content)
    check("the bucket and prefix are named", "s3://" in content)
    check("pruning behaviour is stated rather than assumed",
          "no longer has are deleted here" in content or "left in place" in content)
    check("Sync replaces Upload for a mirror",
          page.get_by_role("button", name="Sync from the bucket").count() == 1)
    check("no upload control is offered on a mirror",
          "Upload a document" not in content)

    print("\n# A sync that cannot reach its bucket")
    page.get_by_role("button", name="Sync from the bucket").click()
    page.wait_for_selector("text=The bucket could not be read", timeout=60000)
    check("the failure is a sentence, not a spinner", True)
    check("it says the corpus is unchanged", "unchanged" in page.content())

    print("\n# The API refuses the writes the screen does not offer")
    status = page.evaluate(
        "async () => (await fetch('/platform/v1/knowledge/" + SOURCE + "/upload', {"
        "method: 'POST', headers: {'content-type': 'application/json'},"
        "body: JSON.stringify({filename: 'x.md', data: 'IyBY'})})).status"
    )
    check("uploading into a mirror is refused by the backend too", status == 409, str(status))
    status = page.evaluate(
        "async () => (await fetch('/platform/v1/knowledge/" + SOURCE + "/remove', {"
        "method: 'POST', headers: {'content-type': 'application/json'},"
        "body: JSON.stringify({name: 'refunds.md'})})).status"
    )
    check("removing from a mirror is refused by the backend too", status == 409, str(status))

    print("\n# A local corpus is untouched by any of this")
    page.goto(f"{BASE}/#/knowledge", wait_until="networkidle")
    page.wait_for_timeout(600)
    open_corpus(LOCAL_TITLE)
    local = page.content()
    check("a local corpus still offers Upload", "Upload a document" in local)
    check("a local corpus is not called a mirror", "Mirrored from" not in local)
    status = page.evaluate(
        "async () => (await fetch('/platform/v1/knowledge/" + LOCAL + "/sync', {"
        "method: 'POST', headers: {'content-type': 'application/json'}, body: '{}'})).status"
    )
    check("syncing a local corpus is refused and explained", status == 409, str(status))

    print("\n# Page errors")
    real = [e for e in errs if "Failed to load resource" not in e]
    check("no uncaught errors in the browser", not real, "; ".join(real[:3]))
    b.close()

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    for name in fail:
        print("  FAILED: " + name)
    sys.exit(1)
