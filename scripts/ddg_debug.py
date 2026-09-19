import urllib.request, urllib.parse, re

url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote('AI news latest')
req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15) as r:
    html = r.read().decode()

# Mostrar estrutura dos resultados
import re
# DDG HTML results have structure like:
# <a class="result__a" href="...">Title<span class="result__a">...</span></a>
# <a class="result__url" href="...">domain.com</a>

# Find all a tags with title and url classes
results_a = re.findall(r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
print(f'result__a links: {len(results_a)}')
for href, text in results_a[:8]:
    clean_text = re.sub(r'<[^>]+>', '', text).strip()
    print(f'  [{clean_text[:70]}] -> {href[:90]}')

results_url = re.findall(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
print(f'\nresult__url links: {len(results_url)}')
for href, text in results_url[:8]:
    print(f'  {text.strip()[:50]} -> {href[:90]}')
