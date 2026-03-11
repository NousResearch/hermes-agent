#!/usr/bin/env python3
"""
telegram_send_photo.py
Send a local image file as a Telegram photo using the Bot API.
Reads credentials from environment: BOT_TOKEN and CHAT_ID.

Usage:
  python3 telegram_send_photo.py <image_path> [caption]
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import requests
requests.packages.urllib3.disable_warnings()

def send_photo(image_path, caption="", chat_id=None, bot_token=None):
    if not bot_token:
        raise RuntimeError("bot_token argument required (or pass TELEGRAM_BOT_TOKEN)")
    if not chat_id:
        raise RuntimeError("chat_id argument required")

    path = os.path.expanduser(image_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    with open(path, "rb") as f:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": (os.path.basename(path), f, "image/png")},
            timeout=30,
            verify=False,
        )

    data = r.json()
    if r.status_code == 200 and data.get("ok"):
        print(f"Photo sent (message_id={data['result']['message_id']})")
        return True
    else:
        print(f"Failed: {r.status_code} — {data.get('description', 'unknown')}")
        return False
