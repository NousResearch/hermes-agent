import urllib.request, urllib.parse

url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote('AI news latest')
req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15) as r:
    html = r.read().decode()

# Show sections around <a href> tags
import re
# Show context around each href
for m in re.finditer(r'<a[^>]*href="([^"]+)"[^>]*>', html):
    start = max(0, m.start()-100)
    end = min(len(html), m.end()+150)
    snippet = html[start:end]
    print('---')
    print(snippet)
    print()
