#!/usr/bin/env python3
"""
Step 2: read the latest cached news file and send to Telegram.
Run as a script-only cron job (no_agent=True) — avoids the delivery timeout.
"""
import os
import json
import urllib.request
import urllib.parse
import time

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
        return
    if not CHAT_ID:
        print("ERRO: CHAT_ID não configurado")
        return

    news_dir = os.path.expanduser("~/.hermes/cron/output/465a0c5c71e2")
    if not os.path.isdir(news_dir):
        print(f"Diretório não encontrado: {news_dir}")
        return

    files = sorted(os.listdir(news_dir), reverse=True)
    if not files:
        print("Nenhum arquivo de output encontrado.")
        return

    latest = os.path.join(news_dir, files[0])
    with open(latest, encoding='utf-8') as f:
        content = f.read()

    # Extract the actual news bullets (everything after "## Response")
    start = content.find("## Response")
    if start == -1:
        start = 0
    news_section = content[start:]

    # Send
    data = urllib.parse.urlencode({
        'chat_id': CHAT_ID,
        'text': news_section[:4096],  # Telegram limit
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
                print(f"✅ Enviado! msg_id={body['result']['message_id']}")
            else:
                print(f"❌ Erro: {body.get('description')}")
    except Exception as e:
        print(f"❌ Falha: {e}")

if __name__ == '__main__':
    main()
