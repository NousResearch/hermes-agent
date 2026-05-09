#!/usr/bin/env python3
"""Convert a directory of markdown files to HTML, preserving structure.
Replicate with modifications for new wiki builds."""

import os, re

def md_to_html(text):
    lines = text.split('\n')
    html_lines = []
    in_code = False
    
    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if not in_code:
                html_lines.append('<pre><code>')
                in_code = True
            else:
                html_lines.append('</code></pre>')
                in_code = False
            continue
        if in_code:
            html_lines.append(line)
            continue
        
        # Headers
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            html_lines.append(f'<h{len(m.group(1))}>{m.group(2)}</h{len(m.group(1))}>')
            continue
        
        # Table
        if '|' in line and line.strip().startswith('|'):
            if line.strip() == '|' or all(c in '-: |' for c in re.sub(r'[^-|: ]', '', line)):
                continue  # separator row
            cells = [c.strip() for c in line.split('|') if c.strip()]
            html_lines.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            continue
        
        # Bold/italic/code/links
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        line = re.sub(r'\*(.+?)\*', r'<em>\1</em>', line)
        line = re.sub(r'`(.+?)`', r'<code>\1</code>', line)
        line = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', line)
        line = re.sub(r'\[\[([^\]]+)\]\]', r'<a href="\1.html">\1</a>', line)
        
        if line.strip():
            html_lines.append(f'<p>{line}</p>')
    return '\n'.join(html_lines)

def build_page(title, content_html, active_page=''):
    nav_items = [
        ('index', 'Home'),
        ('entities/gordon-rouse', 'Gordon Rouse'),
        ('entities/kla', 'KLA'),
        ('concepts/ventura-relocation', 'Ventura Relocation'),
    ]
    nav_html = '\n'.join(
        f'<li><a href="{href}" class="{"active" if href == active_page else ""}">{label}</a></li>'
        for href, label in nav_items
    )
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} — Gordon's Wiki</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; background: #0d1117; color: #e6edf3; line-height: 1.7; }}
    a {{ color: #58a6ff; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
    .topbar {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 0 24px; position: sticky; top: 0; z-index: 10; }}
    .topbar nav {{ max-width: 900px; margin: 0 auto; display: flex; align-items: center; gap: 4px; height: 52px; }}
    .topbar li {{ list-style: none; }}
    .topbar a {{ display: block; padding: 6px 14px; border-radius: 6px; color: #8b949e; font-size: 14px; }}
    .topbar a:hover {{ color: #e6edf3; text-decoration: none; background: #21262d; }}
    .topbar a.active {{ color: #e6edf3; background: #30363d; }}
    .topbar .wiki-label {{ margin-left: auto; color: #30363d; font-size: 13px; }}
    .content {{ max-width: 900px; margin: 0 auto; padding: 40px 24px; }}
    h1 {{ font-size: 32px; font-weight: 700; margin-bottom: 32px; border-bottom: 1px solid #30363d; padding-bottom: 16px; color: #58a6ff; }}
    h2 {{ font-size: 22px; font-weight: 600; margin: 32px 0 16px; }}
    h3 {{ font-size: 18px; font-weight: 600; margin: 24px 0 12px; color: #c9d1d9; }}
    p {{ margin: 0 0 16px; }}
    table {{ border-collapse: collapse; margin: 16px 0; width: 100%; }}
    td, th {{ border: 1px solid #30363d; padding: 10px 14px; text-align: left; font-size: 14px; }}
    th {{ background: #161b22; color: #8b949e; font-weight: 600; }}
    tr:nth-child(even) td {{ background: #161b22; }}
    code {{ background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 1px 6px; font-size: 13px; font-family: monospace; }}
    pre {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; overflow-x: auto; margin: 16px 0; }}
    pre code {{ border: none; padding: 0; background: none; }}
    ul, ol {{ margin: 0 0 16px 24px; }} li {{ margin: 4px 0; }}
    hr {{ border: none; border-top: 1px solid #30363d; margin: 32px 0; }}
  </style>
</head>
<body>
  <div class="topbar">
    <nav><ul>{nav_html}</ul><span class="wiki-label">Gordon's Wiki</span></nav>
  </div>
  <div class="content">{content_html}</div>
</body>
</html>'''

def convert_file(md_path, wiki_dir, out_dir):
    with open(md_path, encoding='utf-8') as f:
        content = f.read()
    # Strip frontmatter
    if content.startswith('---'):
        end = content.find('\n---', 3)
        if end != -1:
            content = content[end+4:]
    # Extract title
    title = 'Wiki'
    for line in content.split('\n'):
        m = re.match(r'^#\s+(.*)', line)
        if m:
            title = m.group(1)
            break
    html = md_to_html(content)
    rel = os.path.relpath(md_path, wiki_dir)
    if rel == 'index.md': out = os.path.join(out_dir, 'index.html')
    elif rel == 'log.md': out = os.path.join(out_dir, 'log.html')
    elif rel == 'SCHEMA.md': out = os.path.join(out_dir, 'schema.html')
    else:
        sub, name = os.path.dirname(rel), os.path.splitext(os.path.basename(rel))[0]
        out = os.path.join(out_dir, sub, name) if sub else os.path.join(out_dir, f'{name}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(build_page(title, html))
    print(f'  wrote {out}')

def main(wiki_dir, out_dir):
    for root, dirs, files in os.walk(wiki_dir):
        for fn in files:
            if fn.endswith('.md'):
                convert_file(os.path.join(root, fn), wiki_dir, out_dir)

if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else '/opt/data/wiki',
         sys.argv[2] if len(sys.argv) > 2 else '/opt/data/hermes-pages-repo/wiki')