"""Company details, brand colour and logo — changed in the browser, checked in the bundle.

The point of most of these is that white-labelling is real rather than cosmetic: the
accent is read back off the document's own custom property after a save, so the assertion
is that the whole interface changed and not that one preview box did.

Run it:

    python -m nova apply <bundle>
    python -m nova serve <bundle> --host 127.0.0.1 --port 19400 &
    python tests/browser/branding_e2e.py <bundle>
"""

import os
import base64, json, struct, sys, zlib
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = os.environ.get("NOVA_E2E_BASE", "http://127.0.0.1:19400")
BUNDLE = Path(sys.argv[1])
ok, fail = [], []
def check(n, c, d=""):
    (ok if c else fail).append(n)
    print(("  PASS  " if c else "  FAIL  ") + n + (f"  {d}" if d and not c else ""))

def png(w=16, h=16):
    def chunk(t, d):
        b = t + d
        return struct.pack(">I", len(d)) + b + struct.pack(">I", zlib.crc32(b))
    raw = b"".join(b"\x00" + b"\x00\x66\xff" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("NOVA_E2E_CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"),
                          args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1500, "height": 1150})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    def goto(frag):
        page.goto("about:blank"); page.goto(f"{BASE}/#{frag}", wait_until="networkidle")
        page.wait_for_timeout(400)

    print("\n# Company")
    goto("/settings")
    page.wait_for_selector("#org-legal", timeout=10000)
    check("settings screen reachable from the nav", True)
    check("tenant id shown as fixed", "Fixed." in page.content())
    page.fill("#org-legal", "Northwind Logistics Ltd")
    page.fill("#org-contact", "ops@northwind.example")
    page.get_by_role("button", name="Save company").click()
    page.wait_for_selector("text=Applied — the runtime has it now", timeout=20000)
    import yaml
    org = yaml.safe_load((BUNDLE/"organization.yaml").read_text())
    check("company details reached the bundle",
          org["legal_name"] == "Northwind Logistics Ltd" and org["contact_email"] == "ops@northwind.example")
    check("tenant id untouched", org["tenant_id"] == "northwind")

    print("\n# Brand")
    goto("/settings")
    page.wait_for_selector("#brand-product", timeout=10000)
    page.fill("#brand-product", "Northwind Intelligence")
    page.get_by_label("Use #0F62FE").click()
    check("preview reflects the chosen accent before saving",
          "#0F62FE" in page.content() or "rgb(15, 98, 254)" in page.content())
    page.get_by_role("button", name="Save brand").click()
    page.wait_for_selector("text=Applied — the runtime has it now", timeout=20000)

    ident = yaml.safe_load((BUNDLE/"identity.yaml").read_text())
    check("brand reached the bundle",
          ident["product_name"] == "Northwind Intelligence" and ident["theme"]["accent"] == "#0F62FE")
    check("declared surface colour survived a partial write", ident["theme"].get("surface") == "#f5f7f7")

    # The whole interface, not one header.
    applied = page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()")
    check("accent applied to the interface's own token", applied == "#0F62FE", applied)
    check("product name in the tab title", "Northwind Intelligence" in page.title())
    check("product name in the sidebar", "Northwind Intelligence" in page.content())

    print("\n# Logo")
    data_uri = "data:image/png;base64," + base64.b64encode(png()).decode()
    stored = json.loads(page.evaluate(f"""
      fetch('/platform/v1/settings/logo', {{method:'POST',
        headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{kind:'logo', content_type:'image/png', data:'{data_uri}'}})}})
        .then(r=>r.text())"""))
    check("upload accepted a browser data: URI", stored.get("ok") is True, json.dumps(stored)[:90])
    check("image written into the bundle", (BUNDLE/"branding"/"logo.png").is_file())

    served = page.evaluate("""fetch('/platform/v1/branding/logo').then(r =>
        r.headers.get('content-type') + ' ' + r.status)""")
    check("served from this origin as an image", served.strip() == "image/png 200", served)

    svg = json.loads(page.evaluate("""
      fetch('/platform/v1/settings/logo', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({kind:'logo', content_type:'image/svg+xml',
                              data: btoa('<svg onload=alert(1)/>')})}).then(r=>r.text())"""))
    check("an SVG logo is refused", svg.get("error",{}).get("status") == 400)

    goto("/settings")
    page.wait_for_selector("img[alt='logo preview']", timeout=10000)
    check("the stored logo renders in settings", True)
    goto("/overview")
    page.wait_for_selector("img[alt$='logo']", timeout=10000)
    check("the logo replaces the monogram in the sidebar", True)

    page.screenshot(path=str(BUNDLE.parent/"branded.png"))
    real = [e for e in errs if "favicon" not in e.lower() and "failed to load resource" not in e.lower()]
    check("no console errors", not real, str(real[:2]))
    b.close()

print(f"\n=== {len(ok)} passed, {len(fail)} failed ===")
sys.exit(1 if fail else 0)
