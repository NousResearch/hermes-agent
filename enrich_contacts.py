#!/usr/bin/env python3
"""Contact enrichment via CloakBrowser Google search."""
import time
import sys
from cloakbrowser import launch

companies = [
    ("MaintainX", "customer education manager", "Customer Education Manager"),
    ("Instrumentl", "customer enablement manager", "Customer Enablement Manager"),
    ("Klaviyo", "customer education specialist", "Senior Customer Education Specialist"),
]

for company, role_keyword, role_title in companies:
    print(f"\n=== {company} ===")
    query = f"{role_keyword} {company}"
    try:
        browser = launch(headless=True, stealth_args=True)
        page = browser.new_page()
        page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)
        page.evaluate(f"""(q) => {{
            const input = document.querySelector('textarea[name="q"], input[name="q"]');
            if (input) {{ input.value = "{query}"; input.closest('form').submit(); }}
        }}""")
        time.sleep(3)
        links = page.evaluate("""() => {
            const items = [];
            document.querySelectorAll("a[jsname='UWckNb'], a.zReHs").forEach(a => {
                const h3 = a.querySelector('h3');
                if (h3 && a.href) items.push({title: h3.innerText.trim(), url: a.href});
            });
            return items;
        }""")
        for l in links[:10]:
            print(f"  {l['title']}")
            print(f"    {l['url']}")
        browser.close()
    except Exception as e:
        print(f"  ERROR: {e}")