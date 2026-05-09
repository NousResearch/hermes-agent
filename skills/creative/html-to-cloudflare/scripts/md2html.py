#!/usr/bin/env python3
"""Convert a directory of markdown files to HTML wiki pages, with client-side auth.

Usage:
  python3 md2html.py WIKI_DIR [--dest DIR] [--cookie-name NAME] [--cookie-value VAL]

Auth flow:
  - login.html written as index.html (entry point)
  - Each wiki page gets inline auth-check <script>: redirects to login if cookie missing
  - Cookie set by login page on password success

Cookie defaults: wiki_auth=GW2026 (must match login.html SECRET)
"""
import os, re, sys, argparse

# ─── Config (change here to rotate the password) ──────────────────────────────
COOKIE_NAME = 'wiki_auth'
COOKIE_VALUE = 'GW2026'

# Nav: (href, label)
NAV_ITEMS = [
    ('/wiki/',                       'Home'),
    ('/wiki/entities/gordon-rouse',  'Gordon Rouse'),
    ('/wiki/entities/kla',           'KLA'),
    ('/wiki/concepts/ventura-relocation', 'Ventura'),
    ('/wiki/log',                    'Log'),
]

# ─── Markdown → HTML ───────────────────────────────────────────────────────────
def md_to_html(text):
    lines = text.split('\n')
    out, in_code = [], False

    for line in lines:
        if line.strip().startswith('```'):
            if not in_code: out.append('<pre><code>')
            else: out.append('</code></pre>')
            in_code = not in_code
            continue
        if in_code:
            out.append(line); continue

        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            out.append(f'<h{len(m.group(1))}>{m.group(2)}</h{len(m.group(1))}>'); continue

        if '|' in line and line.strip().startswith('|'):
            if set(line.strip().replace('-','').replace(':','').replace(' ','')) <= {'|'}:
                continue
            cells = [c.strip() for c in line.split('|') if c.strip()]
            out.append('<table><tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr></table>'); continue

        m = re.match(r'^[-*+]\s+(.*)', line)
        if m:
            out.append(f'<li>{m.group(1)}</li>'); continue

        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        line = re.sub(r'\*(.+?)\*',      r'<em>\1</em>',         line)
        line = re.sub(r'`(.+?)`',        r'<code>\1</code>',    line)
        line = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', line)
        line = re.sub(r'\[\[([^\]]+)\]\]',
                      lambda m: f'<a href="{m.group(1).replace(" ","-")}.html">{m.group(1)}</a>', line)

        if line.strip() in ('---','***','___'):
            out.append('<hr>')
        elif line.strip():
            out.append(f'<p>{line}</p>')

    return '\n'.join(out)

# ─── Auth check (injected into every wiki page) ───────────────────────────────
AUTH_CHECK = f"""
<script>
(function(){{
  var c=document.cookie.match(/{COOKIE_NAME}=([^;]+)/);
  if(!c||c[1]!=='{COOKIE_VALUE}'){{
    window.location.href='/wiki/login.html?dst='+encodeURIComponent(window.location.pathname);
  }}
}})();
</script>"""

# ─── CSS (dark GitHub theme) ───────────────────────────────────────────────────
CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; background: #0d1117; color: #e6edf3; line-height: 1.7; }
a { color: #58a6ff; text-decoration: none; } a:hover { text-decoration: underline; }
.topbar { background: #161b22; border-bottom: 1px solid #30363d; padding: 0 24px; position: sticky; top: 0; z-index: 10; }
.topbar nav { max-width: 900px; margin: 0 auto; display: flex; align-items: center; gap: 4px; height: 52px; }
.topbar ul { display: flex; list-style: none; gap: 4px; }
.topbar a { display: block; padding: 6px 14px; border-radius: 6px; color: #8b949e; font-size: 14px; }
.topbar a:hover { color: #e6edf3; text-decoration: none; background: #21262d; }
.topbar a.active { color: #e6edf3; background: #30363d; }
.topbar .wiki-label { margin-left: auto; color: #30363d; font-size: 13px; }
.content { max-width: 900px; margin: 0 auto; padding: 40px 24px; }
h1 { font-size: 28px; font-weight: 700; margin-bottom: 24px; border-bottom: 1px solid #30363d; padding-bottom: 16px; color: #58a6ff; }
h2 { font-size: 20px; font-weight: 600; margin: 28px 0 12px; }
h3 { font-size: 16px; font-weight: 600; margin: 20px 0 10px; color: #c9d1d9; }
p { margin: 0 0 14px; }
table { border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 14px; display: block; overflow-x: auto; }
td,th { border: 1px solid #30363d; padding: 10px 14px; text-align: left; }
th { background: #161b22; color: #8b949e; font-weight: 600; }
tr:nth-child(even) td { background: #161b22; }
code { background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 1px 6px; font-size: 13px; font-family: monospace; }
pre { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; overflow-x: auto; margin: 16px 0; }
pre code { border: none; padding: 0; background: none; }
ul { margin: 0 0 14px 24px; } li { margin: 4px 0; }
hr { border: none; border-top: 1px solid #30363d; margin: 28px 0; }
blockquote { border-left: 3px solid #58a6ff; padding-left: 16px; color: #8b949e; margin: 16px 0; }
"""

def nav_html(active):
    return '\n'.join(
        f'<li><a href="{href}" class="{"active" if active==href or (href!="/wiki/" and active.startswith(href)) else ""}">{label}</a></li>'
        for href, label in NAV_ITEMS
    )

def build_page(title, body, path):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} — Gordon's Wiki</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="topbar"><nav><ul>{nav_html('/'+path)}</ul><span class="wiki-label">Gordon's Wiki</span></nav></div>
  <div class="content">{body}</div>
  {AUTH_CHECK}
</body>
</html>'''

def convert(md_path, dest, rel):
    with open(md_path) as f:
        content = f.read()
    # Strip YAML frontmatter
    if content.startswith('---'):
        end = content.find('\n---', 3)
        if end != -1:
            content = content[end+4:]

    title = 'Wiki'
    for line in content.split('\n'):
        m = re.match(r'^#\s+(.*)', line)
        if m:
            title = m.group(1); break

    html = md_to_html(content)

    # Special routing: skip index.md (login is the index), route SCHEMA/log
    if rel == 'index.md':
        return
    rel = rel.replace('SCHEMA.md','schema.html').replace('log.md','log.html').replace('.md','.html')

    out = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(build_page(title, html, rel))
    print(f'  {out}')

LOGIN = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login — Gordon's Wiki</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; display: flex; align-items: center; justify-content: center; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 16px; padding: 40px; width: 100%; max-width: 400px; text-align: center; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; color: #e6edf3; }}
    p {{ font-size: 14px; color: #8b949e; margin-bottom: 32px; }}
    input {{ width: 100%; padding: 12px 16px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; color: #e6edf3; font-size: 16px; text-align: center; outline: none; margin-bottom: 16px; box-sizing: border-box; }}
    input:focus {{ border-color: #58a6ff; }}
    button {{ width: 100%; padding: 12px; background: #1f6feb; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; }}
    button:hover {{ background: #388bfd; }}
    .error {{ color: #f85149; font-size: 13px; margin-top: 12px; min-height: 20px; }}
    .hint {{ font-size: 12px; color: #484f58; margin-top: 24px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Gordon's Wiki</h1>
    <p>Enter your password to continue</p>
    <input type="password" id="pw" placeholder="Password" onkeydown="if(event.key==='Enter')login()" />
    <button onclick="login()">Sign In</button>
    <div class="error" id="err"></div>
    <div class="hint">Contact Gordon for access</div>
  </div>
  <script>
    var SECRET = '{COOKIE_VALUE}';
    function login() {{
      var pw = document.getElementById('pw').value;
      if (pw === SECRET) {{
        document.cookie = 'wiki_auth=' + SECRET + '; path=/wiki; max-age=31536000; SameSite=Strict';
        window.location.href = new URLSearchParams(window.location.search).get('dst') || '/wiki/';
      }} else {{
        document.getElementById('err').textContent = 'Incorrect password';
        document.getElementById('pw').value = '';
      }}
    }}
  </script>
</body>
</html>'''

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('wiki_dir')
    ap.add_argument('--dest', default='/opt/data/hermes-pages-repo/wiki')
    args = ap.parse_args()

    dest = args.dest

    # login.html as entry point
    with open(os.path.join(dest, 'index.html'), 'w') as f:
        f.write(LOGIN)
    print(f'  {dest}/index.html (login entry)')

    for root, _, files in os.walk(args.wiki_dir):
        for fn in files:
            if fn.endswith('.md'):
                convert(os.path.join(root, fn), dest, os.path.relpath(os.path.join(root, fn), args.wiki_dir))