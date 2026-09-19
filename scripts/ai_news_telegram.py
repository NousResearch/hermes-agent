#!/usr/bin/env python3
"""
AI News Daily 8h — Script standalone (no_agent).
Busca feeds RSS de notícias de IA, seleciona 5 mais relevantes e envia ao Telegram.
Rodar como cron job com no_agent=True para evitar timeout no delivery.
"""
import os
import sys
import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

ENV_PATH = os.path.expanduser("~/.hermes/.env")
CACHE_DIR = os.path.expanduser("~/.hermes/cron/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def load_env(path):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                out[k.strip()] = v.strip()
    return out

# Feeds RSS confiáveis sobre IA/tech
FEEDS = [
    ("TechCrunch AI", "https://techcrunch.com/feed/?s=ai"),
    ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
    ("Wired AI", "https://www.wired.com/feed/tag/ai/latest/rss"),
    ("BBC Tech", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("HuggingFace Blog", "https://huggingface.co/blog/feed.xml"),
]

# Palavras-chave de IA para filtrar feeds gerais
AI_KEYWORDS = [
    'artificial intelligence', ' ai ', 'machine learning', 'gpt', 'llm',
    'openai', 'anthropic', 'google deepmind', 'gemini', 'claude',
    'neural', 'deep learning', 'llm', 'modelo de linguagem',
    'xai', 'grok', 'chatgpt', 'midjourney', 'diffusion',
]

BRASIL_TZ = timezone(timedelta(hours=-3))

def fetch_feed(url, timeout=10):
    """Fetch and parse an RSS/Atom feed, return list of (title, link, date_str, source)."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; HermesAgent/1.0)',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode('utf-8', errors='replace')
        # Detect if Atom or RSS
        if '<feed xmlns' in data or '<feed ' in data:
            root = ET.fromstring(data)
            entries = root.findall('.//entry')
            is_atom = True
        else:
            root = ET.fromstring(data)
            entries = root.findall('.//item')
            is_atom = False
    except Exception as e:
        print(f"  Feed error ({url[:50]}): {type(e).__name__}: {str(e)[:60]}", file=sys.stderr)
        return []

    items = []
    for entry in entries[:15]:
        if is_atom:
            title = entry.findtext('title') or ''
            # link may be in <link href="..."> or <link>text</link>
            link_el = entry.find('link')
            if link_el is not None:
                if link_el.get('href'):
                    link = link_el.get('href')
                else:
                    link = (link_el.text or '').strip()
            else:
                link = ''
            date_el = entry.find('published') or entry.find('updated')
            date_str = (date_el.text or '').strip() if date_el is not None else ''
        else:
            title = entry.findtext('title') or ''
            link = entry.findtext('link') or ''
            date_el = entry.find('pubDate') or entry.find('dc:date')
            date_str = date_el.text.strip() if date_el is not None else ''

        if not title or not link:
            continue
        title = title.strip()
        link = link.strip()
        if not title or not link.startswith('http'):
            continue
        items.append((title, link, date_str, ''))
    return items

def is_recent(date_str, max_age_hours=48):
    """Check if a date string indicates a recent article."""
    if not date_str:
        return True  # If no date, assume recent
    try:
        # Try common date formats
        date_str = date_str.strip()
        for fmt in [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                now = datetime.now(BRASIL_TZ)
                age = now - dt
                return age <= timedelta(hours=max_age_hours)
            except ValueError:
                continue
    except Exception:
        pass
    return True

def score_relevance(title, source):
    """Score how relevant an article is to AI. Higher = more relevant."""
    title_lower = title.lower()
    score = 0
    for kw in AI_KEYWORDS:
        if kw in title_lower:
            if kw in ['artificial intelligence', 'machine learning', 'google deepmind']:
                score += 3
            elif kw in ['openai', 'anthropic', 'gemini', 'claude', 'chatgpt', 'grok', 'xai']:
                score += 2
            else:
                score += 1
    # Prefer shorter, punchier titles (likely news, not essays)
    if len(title) < 80:
        score += 1
    if len(title) > 150:
        score -= 1
    return score

def dedup(items):
    """Deduplicate by link, keeping highest-scored version."""
    seen = {}
    for title, link, date_str, source in items:
        key = link.lower().rstrip('/')
        if key in seen:
            if score_relevance(title, source) > score_relevance(seen[key][0], seen[key][2]):
                seen[key] = (title, link, date_str, source)
        else:
            seen[key] = (title, link, date_str, source)
    return list(seen.values())

def main():
    env = load_env(ENV_PATH)
    BOT_TOKEN = env.get('TELEGRAM_BOT_TOKEN', '')
    if not BOT_TOKEN:
        print("ERRO: TELEGRAM_BOT_TOKEN não encontrado em ~/.hermes/.env")
        sys.exit(1)

    CHAT_ID = env.get('CHAT_ID', env.get('HERMES_SESSION_CHAT_ID', ''))
    if not CHAT_ID:
        print("ERRO: CHAT_ID não configurado")
        sys.exit(1)

    now = datetime.now(BRASIL_TZ)
    print(f"📰 AI News Daily — {now.strftime('%d/%m/%Y %H:%M')}")
    print(f"🔍 Buscando feeds RSS de IA...")

    all_items = []
    for name, url in FEEDS:
        items = fetch_feed(url)
        # Filter: keep recent items with AI relevance
        recent = [(t, l, d, name) for t, l, d, _ in items if is_recent(d)]
        all_items.extend(recent)
        print(f"  {name}: {len(items)} itens totais, {len(recent)} recentes")

    # Score and rank
    scored = [(score_relevance(t, s), t, l, d, s) for t, l, d, s in all_items]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = dedup([(t, l, d, s) for _, t, l, d, s in scored])

    # Re-score after dedup
    top.sort(key=lambda x: score_relevance(x[0], x[3]), reverse=True)

    # Filter again for actual AI relevance (must have at least one keyword match)
    top = [item for item in top if score_relevance(item[0], item[3]) >= 2]

    if not top:
        print("\n⚠ Nenhuma notícia de IA relevante encontrada nos feeds.")
        return

    selected = top[:5]
    print(f"\n📊 {len(selected)} notícias selecionadas:")

    # Build message
    header = f"📰 <b>IA nas últimas 24h</b> — {now.strftime('%d/%m/%Y %H:%M')} BRT"
    lines = [header, ""]
    for i, (title, link, date_str, source) in enumerate(selected, 1):
        lines.append(f"{i}. {title} — Fonte: {source} | <a href=\"{link}\">{link}</a>")

    msg = "\n".join(lines)
    print(f"\n📤 Enviando para Telegram (chat_id={CHAT_ID})...")

    # Send via Telegram Bot API
    data = urllib.parse.urlencode({
        'chat_id': CHAT_ID,
        'text': msg,
        'parse_mode': 'HTML',
        'disable_web_page_preview': 'true',
    }).encode('utf-8')

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data=data,
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            if body.get('ok'):
                print(f"✅ Enviado com sucesso! msg_id={body['result']['message_id']}")
            else:
                print(f"❌ Erro do Telegram: {body.get('description', 'unknown')}")
    except Exception as e:
        print(f"❌ Falha ao enviar: {type(e).__name__}: {e}")

    # Save to cache for audit
    cache_file = os.path.join(CACHE_DIR, f"ai_news_{now.strftime('%Y%m%d_%H%M')}.json")
    with open(cache_file, 'w') as f:
        json.dump({
            'timestamp': now.isoformat(),
            'chat_id': CHAT_ID,
            'articles': [{'title': t, 'link': l, 'source': s} for t, l, _, s in selected],
        }, f, indent=2, ensure_ascii=False)
    print(f"📁 Cache salvo: {cache_file}")

if __name__ == '__main__':
    main()
