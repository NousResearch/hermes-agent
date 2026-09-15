"""Uploading a document to a corpus, in the browser, checked on disk and in the index.

The check that matters is the last one: the uploaded text is found by the same search path
an agent uses. Everything before it is plumbing if that fails.

Run it:

    python -m nova apply <bundle>
    python -m nova serve <bundle> --host 127.0.0.1 --port 19500 &
    python tests/browser/knowledge_upload_e2e.py <bundle> <runtime-home>
"""

import json, os, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = os.environ.get("NOVA_E2E_BASE", "http://127.0.0.1:19500")
CHROME = os.environ.get("NOVA_E2E_CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
BUNDLE, HOME = Path(sys.argv[1]), Path(sys.argv[2])
CORPUS = BUNDLE / "knowledge" / "handbook"
ok, fail = [], []

def check(n, c, d=""):
    (ok if c else fail).append(n)
    print(("  PASS  " if c else "  FAIL  ") + n + (f"  {d}" if d and not c else ""))

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1500, "height": 1150})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    def goto(frag):
        page.goto("about:blank"); page.goto(f"{BASE}/#{frag}", wait_until="networkidle")
        page.wait_for_timeout(400)

    print("\n# Corpus documents")
    goto("/knowledge")
    page.wait_for_selector("text=Company Handbook", timeout=10000)
    page.get_by_role("button", name="Documents").first.click()
    page.wait_for_selector("text=Upload a document", timeout=10000)
    check("the document panel opens from a corpus card", True)
    check("what the corpus accepts is stated", "**/*.md" in page.content())
    check("existing documents are listed", "refunds.md" in page.content())

    print("\n# Upload")
    upload = BUNDLE.parent / "Quokka Returns (v3).md"
    upload.write_text("# Returns\n\nA distinctive phrase: quokka refunds within 14 days.\n")
    page.set_input_files("input[type=file]", str(upload))
    page.wait_for_selector("text=Stored", timeout=25000)
    check("upload reports what was stored", True)

    stored = CORPUS / "Quokka-Returns-v3.md"
    check("the filename was rebuilt safely", stored.is_file(), sorted(p.name for p in CORPUS.iterdir()))
    check("the contents landed intact", "quokka refunds" in stored.read_text())

    listed = json.loads(page.evaluate(
        "fetch('/platform/v1/knowledge/company-handbook/documents').then(r=>r.text())"))
    check("the index picked it up", listed["indexed"]["documents"] >= 3, str(listed["indexed"]))
    check("it appears in the document list",
          "Quokka-Returns-v3.md" in [d["name"] for d in listed["documents"]])

    print("\n# Searchable — the point of all of it")
    # Search through the index directly rather than the UI: the assertion is that an agent
    # could find it, and the agent's path is the index.
    import subprocess
    probe = subprocess.run([sys.executable, "-c", f"""
import sys
sys.path.insert(0, {str(Path.cwd())!r})
from nova.knowledge import KnowledgeIndex
with KnowledgeIndex.open({str(HOME / 'nova-knowledge.db')!r}, create=False) as ix:
    hits = ix.search('quokka', source_ids=['company-handbook'], limit=5)
print(len(hits))
"""], capture_output=True, text=True)
    check("the uploaded text is searchable", probe.stdout.strip().isdigit() and int(probe.stdout.strip()) > 0,
          probe.stdout + probe.stderr[-200:])

    print("\n# Refusals")
    bad = BUNDLE.parent / "payload.exe"
    bad.write_bytes(b"MZ\x90\x00")
    page.set_input_files("input[type=file]", str(bad))
    page.wait_for_selector("text=Nothing was stored", timeout=20000)
    check("a type the corpus does not accept is refused", True)
    check("the refusal names what is accepted", "accepts" in page.content().lower())
    check("nothing was written", not (CORPUS / "payload.exe").exists())

    print("\n# Remove")
    goto("/knowledge")
    page.get_by_role("button", name="Documents").first.click()
    page.wait_for_selector("text=Quokka-Returns-v3.md", timeout=10000)
    page.get_by_role("button", name="Remove Quokka-Returns-v3.md").click()
    page.get_by_role("button", name="Remove Quokka-Returns-v3.md", exact=False).last.click()
    page.wait_for_selector("text=Removed", timeout=20000)
    check("removed from disk", not stored.exists())
    listed = json.loads(page.evaluate(
        "fetch('/platform/v1/knowledge/company-handbook/documents').then(r=>r.text())"))
    check("removed from the index too",
          "Quokka-Returns-v3.md" not in [d["name"] for d in listed["documents"]])

    page.screenshot(path=str(HOME.parent / "corpus.png"))
    real = [e for e in errs if "favicon" not in e.lower() and "failed to load resource" not in e.lower()]
    check("no console errors", not real, str(real[:2]))
    b.close()

print(f"\n=== {len(ok)} passed, {len(fail)} failed ===")
sys.exit(1 if fail else 0)
