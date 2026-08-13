#!/usr/bin/env python3
"""Generate personal brand drafts and produce HTML report for Discord delivery."""
import sys
import os
sys.path.insert(0, '/home/kensei/repos/KenseiAgent/content_engine')

from llm_generate import generate_drafts_llm
from config import BRANDS, DB_PATH
from topics import get_topics
from database import insert_draft, init_db
import sqlite3
import uuid
from datetime import datetime, UTC

# Init DB
init_db()

# Clear old drafts from today
conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM drafts WHERE date(created_at) = date('now')")
conn.commit()
conn.close()

all_drafts = []
rate_limit_events = []
generation_failures = []

for brand in ['sahil_twitter', 'sahil_linkedin']:
    print(f'\n=== {brand} ===')
    brand_config = BRANDS.get(brand, {})
    brand_platforms = brand_config.get('platforms', [])
    platform = brand_platforms[0] if brand_platforms else 'twitter'
    
    topics = get_topics(brand, count=6)
    print(f'Topics found: {len(topics)}')
    
    if not topics:
        print(f'No topics for {brand}, skipping')
        generation_failures.append(f'{brand}: no topics available')
        continue
    
    # Generate up to 3 drafts per brand
    drafts = generate_drafts_llm(brand, topics[:3], platform=platform)
    print(f'Drafts generated: {len(drafts)}')
    
    for d in drafts:
        body = d.get('body_text', '')
        # Check if this was a fallback template
        is_fallback = ('Every AI tool' in body[:50] or 'The PMs who win' in body[:50] or 'Vibe coding' in body[:50])
        if is_fallback:
            rate_limit_events.append(f'{brand}: fallback template used (LLM 401)')
        
        # Extract fields with defaults for required columns
        draft_brand = d.get('brand') or brand
        draft_platform = d.get('platform') or platform
        draft_pillar = d.get('content_pillar') or d.get('pillar') or 'Build-in-Public'
        draft_topic = d.get('topic') or ''
        draft_title = d.get('title') or draft_topic
        draft_body = body or '[No body text generated]'
        draft_content_type = d.get('content_type') or 'text'
        draft_visual = d.get('visual_description') or ''
        slop_audit = d.get('slop_audit', {})
        draft_slop_score = d.get('slop_score', slop_audit.get('slop_score', 0))
        draft_slop_issues = d.get('slop_issues', '; '.join(slop_audit.get('issues', [])))
        draft_source = d.get('source_provenance') or {}
        draft_rationale = d.get('editorial_rationale') or ''
        
        # Insert into DB with correct signature
        insert_draft(
            draft_id=d.get('id') or str(uuid.uuid4()),
            brand=draft_brand,
            platform=draft_platform,
            pillar=draft_pillar,
            topic=draft_topic,
            title=draft_title,
            body_text=draft_body,
            content_type=draft_content_type,
            visual_description=draft_visual,
            slop_score=draft_slop_score if isinstance(draft_slop_score, int) else 0,
            slop_issues=str(draft_slop_issues),
            source_provenance=draft_source,
            editorial_rationale=draft_rationale,
        )
        all_drafts.append(d)
        print(f"  Inserted: {draft_brand}/{draft_platform} - pillar={draft_pillar}")

print(f'\n=== SUMMARY ===')
print(f'Total drafts inserted: {len(all_drafts)}')
if rate_limit_events:
    print(f'Rate limit/fallback events: {len(rate_limit_events)}')
    for e in rate_limit_events:
        print(f'  - {e}')
if generation_failures:
    print(f'Generation failures: {len(generation_failures)}')
    for e in generation_failures:
        print(f'  - {e}')

# Now generate HTML report
today = datetime.now(UTC).strftime('%Y-%m-%d')
timestamp = datetime.now(UTC).strftime('%Y%m%d-%H%M')
output_dir = '/home/kensei/.hermes/runbooks/content-personal'
os.makedirs(output_dir, exist_ok=True)
html_path = os.path.join(output_dir, f'{today}.html')

# Fetch drafts from DB
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT * FROM drafts WHERE date(created_at) = date('now') ORDER BY brand, created_at"
).fetchall()
conn.close()

# Group by brand
by_brand = {}
for row in rows:
    brand = row['brand']
    if brand not in by_brand:
        by_brand[brand] = []
    by_brand[brand].append(dict(row))

# Generate dark-mode HTML
html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Personal Brand Content — {today}</title>
    <style>
        :root {{
            --bg: #11100f;
            --card: #1c1a18;
            --card-alt: #2c2a28;
            --text: #f5f5f4;
            --muted: #a8a29e;
            --accent: #fbbf24;
            --border: #34302c;
            --twitter: #1da1f2;
            --linkedin: #0a66c2;
        }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 40px 20px;
        }}
        h1 {{
            color: var(--accent);
            border-bottom: 2px solid var(--border);
            padding-bottom: 10px;
        }}
        h2 {{
            color: var(--accent);
            margin-top: 40px;
        }}
        .brand-section {{
            margin-bottom: 40px;
        }}
        .draft-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        .draft-card:hover {{
            background: var(--card-alt);
        }}
        .meta {{
            color: var(--muted);
            font-size: 0.85em;
            margin-bottom: 10px;
        }}
        .pillar {{
            display: inline-block;
            background: var(--accent);
            color: var(--bg);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        .body-text {{
            line-height: 1.6;
            white-space: pre-wrap;
        }}
        .warning {{
            background: #7c2d12;
            border: 1px solid #ea580c;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .platform-twitter {{ border-left: 4px solid var(--twitter); }}
        .platform-linkedin {{ border-left: 4px solid var(--linkedin); }}
    </style>
</head>
<body>
    <h1>📝 Personal Brand Content — {datetime.now(UTC).strftime('%d/%m/%Y')}</h1>
    
    <div class="meta">Generated: {datetime.now(UTC).strftime('%d/%m/%Y %H:%M:%S')} UTC</div>
'''

if rate_limit_events:
    html_content += f'''
    <div class="warning">
        <strong>⚠️ Rate Limit Events:</strong> LLM endpoints returned 401. Fallback templates used.
        <ul>
            {''.join(f'<li>{e}</li>' for e in rate_limit_events)}
        </ul>
    </div>
'''

for brand, drafts in by_brand.items():
    platform_class = 'platform-twitter' if 'twitter' in brand else 'platform-linkedin'
    html_content += f'''
    <div class="brand-section {platform_class}">
        <h2>{brand.replace('_', ' ').title()} — {len(drafts)} drafts</h2>
'''
    for i, draft in enumerate(drafts, 1):
        html_content += f'''
        <div class="draft-card" id="draft-{draft['id']}">
            <div class="meta">
                <span class="pillar">{draft['pillar']}</span>
                · Platform: {draft['platform']}
                · ID: {draft['id'][:8]}...
            </div>
            <div class="body-text">{draft['body_text']}</div>
        </div>
'''
    html_content += '    </div>\n'

html_content += '''
</body>
</html>
'''

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f'\nHTML report written: {html_path}')
print(f'File size: {os.path.getsize(html_path)} bytes')

# Output Discord summary
print(f'\n=== DISCORD SUMMARY ===')
print(f'📝 Personal Brand Content — {datetime.now(UTC).strftime("%d/%m/%Y")}')
print(f'{len(all_drafts)} drafts · {len(by_brand.get("sahil_twitter", []))} Twitter · {len(by_brand.get("sahil_linkedin", []))} LinkedIn')
if rate_limit_events:
    print(f'⚠️ {len(rate_limit_events)} fallback templates (LLM 401)')
print(f'\nMEDIA:{html_path}')
