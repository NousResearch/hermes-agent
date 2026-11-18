#!/usr/bin/env python3
"""Search a company for contacts via CloakBrowser."""
import time, sys
from cloakbrowser import launch

company = sys.argv[1]
role = sys.argv[2] if len(sys.argv) > 2 else "customer education"

query = f"{role} {company} linkedin"

try:
    browser = launch(headless=True, stealth_args=True)
    page = browser.new_page()
    page.goto('https://www.google.com', wait_until='domcontentloaded', timeout=20000)
    time.sleep(1)

    # Type query into the search box
    page.evaluate(f'document.querySelector(\'textarea[name="q"]\').value = `{query}`')
    time.sleep(0.5)

    # Submit the form
    page.evaluate(f'document.querySelector(\'form[action="/search"]\').submit()')
    time.sleep(4)

    # Extract results
    results = page.evaluate("""
        Array.from(document.querySelectorAll('a[jsname="UWckNb"], a.zReHs')).map(a => {
            const h3 = a.querySelector('h3');
            return h3 ? {title: h3.innerText.trim(), url: a.href} : null;
        }).filter(Boolean).slice(0, 8)
    """)
    
    for r in results:
        print(f"TITLE: {r['title']}")
        print(f"URL:   {r['url']}")
        print()
    
    browser.close()
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")