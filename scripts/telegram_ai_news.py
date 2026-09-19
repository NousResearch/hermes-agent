#!/usr/bin/env python3
"""
AI News Daily 8h — standalone script (no_agent mode).
Busca notícias de IA via web_search, formata bullets e envia ao Telegram.
"""
import os
import sys
import time
import json
import urllib.request
import urllib.parse

ENV_PATH = os.path.expanduser("~/.hermes/.env")

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

def main():
    env = load_env(ENV_PATH)
    BOT_TOKEN = env.get('TELEGRAM_BOT_TOKEN', '')
    CHAT_ID = env.get('CHAT_ID', env.get('HERMES_SESSION_CHAT_ID', ''))
    if not BOT_TOKEN:
        print("ERRO: TELEGRAM_BOT_TOKEN não encontrado")
        sys.exit(1)
    if not CHAT_ID:
        print("ERRO: CHAT_ID não configurado")
        sys.exit(1)

    print("🔍 Buscando notícias de IA...")

    # Use hermes_tools if available
    try:
        from hermes_tools import web_search
        HAS_WEB_SEARCH = True
    except ImportError:
        HAS_WEB_SEARCH = False
        print("AVISO: hermes_tools.web_search não disponível, tentando DuckDuckGo...")

    results = []

    if HAS_WEB_SEARCH:
        queries = [
            'AI news artificial intelligence latest today',
            'AI breakthrough this week 2026',
            'openai anthropic google AI news September 2026',
        ]
        for q in queries:
            try:
                data = web_search(q, limit=6)
                for item in data.get('data', {}).get('web', []):
                    url = item.get('url', '')
                    title = item.get('title', '')
                    if url and title and url not in [r[1] for r in results]:
                        results.append((title, url))
                print(f"  Query '{q[:45]}...' → {len(results)} resultados totais")
                time.sleep(1)
            except Exception as e:
                print(f"  web_search error: {e}")
    else:
        # Fallback: DuckDuckGo
        import urllib.request as ur
        for q in ['AI news latest', 'AI breakthrough 2026', 'openai anthropic google AI']:
            try:
                url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(q)
                req = ur.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with ur.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode()
                import re
                for href, text in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html):
                    if 'duckduckgo.com' not in href and len(text.strip()) > 20:
                        results.append((text.strip(), href))
            except Exception as e:
                print(f"  DDG error: {e}")
            time.sleep(1)

    # Dedupe
    seen = set()
    unique = []
    for t, u in results:
        if u not in seen:
            seen.add(u)
            unique.append((t, u))
    results = unique[:5]

    if not results:
        print("Nenhuma notícia encontrada.")
        return

    print(f"\n📊 {len(results)} notícias encontradas")

    # Build message in Markdown (Telegram supports MarkdownV2, but simple Markdown is fine)
    now = time.strftime('%d/%m/%Y %H:%M')
    lines = [f"📰 **IA nas últimas 24h**  ({now})", ""]
    for title, url in results:
        # Escape for MarkdownV2 if needed, but simple markdown works
        lines.append(f"• <b>{title}</b> — <a href=\"{url}\">{url}</a>")
    msg = "\n".join(lines)

    print(f"\n📤 Enviando para Telegram (chat_id={CHAT_ID})...")
    data = urllib.parse.urlencode({
        'chat_id': CHAT_ID,
        'text': msg,
        'parse_mode': 'HTML',
    }).encode()
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
                print(f"❌ Erro da API: {body.get('description')}")
    except Exception as e:
        print(f"❌ Falha no envio: {e}")

if __name__ == '__main__':
    main()
