import os, json, sys
report_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), '.hermes', 'runbooks', 'token-health', 'report.html')
raw = os.environ.get('TOKEN_HEALTH_JSON','')
if not raw:
    sys.exit(1)
try:
    data = json.loads(raw)
except Exception as e:
    print(f'json parse failed: {e}')
    sys.exit(1)
overall = str(data.get('overall','unknown'))
expired = data.get('expired_count',0)
warnings = data.get('warnings_count',0)
print(f"{overall}\nexpired={expired} warnings={warnings}")
rows = []
for acc in data.get('accounts', []):
    rows.append((acc.get('provider',''), acc.get('email',''), acc.get('status',''), acc.get('detail','').replace('\n',' ')))
html = f'''<!DOCTYPE html>
<html lang="en" data-color-scheme="dark">
<head><meta charset="utf-8"><title>Token Health Report</title>
<style>
body {{ background:#11100f;color:#f5f5f4;font-family:system-ui;padding:2rem; }}
th,td {{ padding:0.5rem 1rem;text-align:left;border-bottom:1px solid #2c2a28; }}
</style></head><body>
<h1>Token Health Report</h1>
<p>Overall: <strong>{overall}</strong></p>
<p>Expired: {expired} — Warnings: {warnings}</p>
<table><tr><th>Provider</th><th>Email</th><th>Status</th><th>Detail</th></tr>
'''
for provider,email,status,detail in rows:
    html += f'<tr><td>{provider}</td><td>{email}</td><td>{status}</td><td>{detail}</td></tr>\n'
html += '</table></body></html>'
os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
with open(report_path, 'w') as f:
    f.write(html)
print(f'MEDIA:{report_path}')
