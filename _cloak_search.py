#!/usr/bin/env python3
"""CloakBrowser job posting search for Built LMS competitive intel."""
import sys, json

# Use the Python 3.9 install where cloakbrowser lives
try:
    from cloakbrowser import launch
except ImportError:
    print("FAIL: cloakbrowser not importable", file=sys.stderr)
    sys.exit(1)

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "customer education manager job"

browser = launch(headless=True)
page = browser.new_page()
url = f"https://www.google.com/search?q={query}&num=20&hl=en&gl=us"

# Navigate
page.goto(url, wait_until="domcontentloaded")

# Wait a moment for results to render
page.wait_for_timeout(3000)

# Get the body text
body_text = page.inner_text("body")

# Extract search results area specifically
results = {}
try:
    # Try to get the main results
    main = page.inner_text("#search")
    results["main"] = main[:5000]
except:
    pass

# Also grab the full page title and visible text
results["title"] = page.title()
results["body_excerpt"] = body_text[:8000]

page.close()
browser.close()

print(json.dumps(results, indent=2))