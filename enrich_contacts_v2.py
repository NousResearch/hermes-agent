#!/usr/bin/env python3
"""Contact enrichment v2 - search for specific people at target companies."""
import time
import sys
from cloakbrowser import launch

queries = [
    ('MaintainX', '"head of learning" OR "customer education manager" MaintainX linkedin'),
    ('MaintainX', '"VP customer success" OR "VP customer experience" MaintainX'),
    ('Instrumentl', '"customer enablement" OR "customer education" Instrumentl linkedin'),
    ('Klaviyo', '"head of learning" OR "customer education" Klaviyo linkedin'),
]

for company, query in queries:
    print(f'\n=== {company} === Query: {query}')
    try:
        browser = launch(headless=True, stealth_args=True)
        page = browser.new_page()
        page.goto('https://www.google.com', wait_until='domcontentloaded', timeout=15000)
        # Use the correct pattern: homepage -> type -> submit
        page.evaluate("""() => {
            const input = document.querySelector('textarea[name="q"], input[name="q"]');
            if (input) {
                input.value = arguments[0];
                input.closest('form').submit();
            }
        }""", query)
        time.sleep(4)
        # Extract all links and snippets
        results = page.evaluate("""() => {
            const items = [];
            document.querySelectorAll("a[jsname='UWckNb'], a.zReHs").forEach(a => {
                const h3 = a.querySelector('h3');
                if (h3 && a.href) {
                    items.push({title: h3.innerText.trim(), url: a.href});
                }
            });
            return items;
        }""")
        for r in results[:6]:
            print(f'  TITLE: {r["title"]}')
            print(f'  URL:   {r["url"]}')
        browser.close()
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}')