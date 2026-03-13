#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Brifing Gönderici / Telegram Brief Sender
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hermes Agent Hackathon 2026 — Turkish Locale Skill Pack

PNG brifing kartını veya metin mesajını Telegram kanalına gönderir.
Telegram Bot API kullanır — sadece requests kütüphanesi gerekli.

Ortam Değişkenleri / Environment Variables:
    TELEGRAM_BOT_TOKEN       — Telegram Bot API token (@BotFather'dan alınır)
    TELEGRAM_HOME_CHANNEL    — Hedef kanal/chat ID (ör: -1001234567890 veya @kanal_adi)

Kullanım / Usage:
    python telegram_send.py --image ~/turkish_brief.png --caption "Günlük Bülten"
    python telegram_send.py --text "BIST100: 9.245 ▲ +1,23%"
    python telegram_send.py --image brief.png                    # Varsayılan caption

Bağımlılıklar / Dependencies:
    pip install requests
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("\033[91m✗ 'requests' kütüphanesi gerekli / required\033[0m")
    print("  pip install requests")
    sys.exit(1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sabitler / Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"

# Varsayılan caption (resim gönderiminde)
DEFAULT_CAPTION = "🌅 Günlük Brifing — Hermes Agent 🇹🇷"

# Telegram API limitleri
MAX_CAPTION_LENGTH = 1024       # sendPhoto caption limiti
MAX_MESSAGE_LENGTH = 4096       # sendMessage metin limiti
MAX_PHOTO_SIZE = 10 * 1024 * 1024  # 10 MB foto limiti

# ANSI renkleri
class C:
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ortam Değişkenleri / Environment Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_config():
    """
    Ortam değişkenlerinden Telegram yapılandırmasını okur.
    Eksik değişken varsa açıklayıcı hata mesajı verir.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    channel = os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()

    errors = []
    if not token:
        errors.append("TELEGRAM_BOT_TOKEN tanımlı değil")
    if not channel:
        errors.append("TELEGRAM_HOME_CHANNEL tanımlı değil")

    if errors:
        print(f"\n  {C.RED}{C.BOLD}✗ Yapılandırma hatası / Configuration error{C.RESET}\n")
        for err in errors:
            print(f"  {C.RED}• {err}{C.RESET}")
        print(f"""
  {C.DIM}Gerekli ortam değişkenleri / Required environment variables:{C.RESET}

    export TELEGRAM_BOT_TOKEN="123456:ABCdef..."
    export TELEGRAM_HOME_CHANNEL="-1001234567890"

  {C.DIM}Bot token almak için: @BotFather → /newbot
  Chat ID bulmak için:  @userinfobot veya @getmyid_bot{C.RESET}
""")
        sys.exit(1)

    return token, channel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Telegram API Fonksiyonları
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def send_photo(token, chat_id, image_path, caption=None):
    """
    Telegram'a fotoğraf gönderir (sendPhoto API).

    Args:
        token:      Bot API token
        chat_id:    Hedef kanal/chat ID
        image_path: PNG/JPG dosya yolu
        caption:    Fotoğraf altı yazısı (opsiyonel, maks 1024 karakter)

    Returns:
        (bool, dict): (başarı durumu, API yanıtı)
    """
    url = f"{TELEGRAM_API_BASE.format(token=token)}/sendPhoto"

    # Dosya kontrolü
    path = Path(image_path)
    if not path.exists():
        return False, {"error": f"Dosya bulunamadı: {image_path}"}

    file_size = path.stat().st_size
    if file_size > MAX_PHOTO_SIZE:
        size_mb = file_size / (1024 * 1024)
        return False, {"error": f"Dosya çok büyük: {size_mb:.1f}MB (limit: 10MB)"}

    # Caption uzunluk kontrolü
    if caption and len(caption) > MAX_CAPTION_LENGTH:
        caption = caption[:MAX_CAPTION_LENGTH - 3] + "..."

    # API isteği
    payload = {
        "chat_id": chat_id,
        "parse_mode": "HTML",
    }
    if caption:
        payload["caption"] = caption

    try:
        with open(path, "rb") as photo:
            files = {"photo": (path.name, photo, "image/png")}
            resp = requests.post(url, data=payload, files=files, timeout=30)

        result = resp.json()

        if result.get("ok"):
            return True, result
        else:
            return False, result

    except requests.exceptions.Timeout:
        return False, {"error": "Zaman aşımı (30s) — dosya çok büyük olabilir"}
    except requests.exceptions.ConnectionError:
        return False, {"error": "Bağlantı hatası — internet bağlantınızı kontrol edin"}
    except Exception as e:
        return False, {"error": f"Beklenmeyen hata: {str(e)}"}


def send_message(token, chat_id, text):
    """
    Telegram'a metin mesajı gönderir (sendMessage API).
    PNG gönderilemezse yedek olarak kullanılır.

    Args:
        token:    Bot API token
        chat_id:  Hedef kanal/chat ID
        text:     Gönderilecek metin (maks 4096 karakter)

    Returns:
        (bool, dict): (başarı durumu, API yanıtı)
    """
    url = f"{TELEGRAM_API_BASE.format(token=token)}/sendMessage"

    # Metin uzunluk kontrolü
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH - 20] + "\n\n[...kesik/truncated]"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        result = resp.json()

        if result.get("ok"):
            return True, result
        else:
            return False, result

    except requests.exceptions.Timeout:
        return False, {"error": "Zaman aşımı (15s)"}
    except requests.exceptions.ConnectionError:
        return False, {"error": "Bağlantı hatası"}
    except Exception as e:
        return False, {"error": f"Beklenmeyen hata: {str(e)}"}


def verify_bot(token):
    """
    Bot token'ını doğrular (getMe API).
    Bağlantı ve yetkilendirmeyi kontrol eder.
    """
    url = f"{TELEGRAM_API_BASE.format(token=token)}/getMe"
    try:
        resp = requests.get(url, timeout=10)
        result = resp.json()
        if result.get("ok"):
            bot = result["result"]
            return True, bot.get("username", "unknown")
        return False, result.get("description", "Bilinmeyen hata")
    except Exception as e:
        return False, str(e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ana Program / Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    parser = argparse.ArgumentParser(
        description="Telegram Brifing Gönderici / Telegram Brief Sender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler / Examples:
  %(prog)s --image ~/turkish_brief.png                     PNG kart gönder
  %(prog)s --image brief.png --caption "Sabah Bülteni"     Özel caption ile
  %(prog)s --text "BIST100: 9.245 ▲ +1,23%%"               Metin mesajı gönder
  %(prog)s --image brief.png --channel @my_channel         Farklı kanala gönder
  %(prog)s --verify                                         Bot bağlantısını test et

Ortam Değişkenleri / Environment Variables:
  TELEGRAM_BOT_TOKEN       Bot API token (@BotFather'dan)
  TELEGRAM_HOME_CHANNEL    Varsayılan hedef chat ID

Hermes Agent Hackathon 2026 — Turkish Locale Skill Pack 🇹🇷
        """
    )

    # Gönderim modu
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--image", "-i",
        type=str,
        help="Gönderilecek PNG/JPG dosya yolu"
    )
    group.add_argument(
        "--text", "-t",
        type=str,
        help="Gönderilecek metin mesajı"
    )
    group.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Bot bağlantısını doğrula (mesaj göndermez)"
    )

    # Opsiyonel ayarlar
    parser.add_argument(
        "--caption", "-c",
        type=str,
        default=DEFAULT_CAPTION,
        help=f"Fotoğraf caption'ı (varsayılan: '{DEFAULT_CAPTION}')"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Hedef kanal/chat ID (varsayılan: TELEGRAM_HOME_CHANNEL ortam değişkeni)"
    )
    parser.add_argument(
        "--fallback-text",
        type=str,
        default=None,
        help="PNG başarısız olursa gönderilecek yedek metin"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Sessiz mod — ekrana çıktı verme"
    )

    return parser.parse_args()


def log(msg, silent=False):
    """Koşullu konsol çıktısı."""
    if not silent:
        print(msg)


def main():
    args = parse_args()

    # Yapılandırma
    token, default_channel = get_config()
    channel = args.channel or default_channel

    # ── Doğrulama modu ──
    if args.verify:
        log(f"\n  {C.CYAN}🔍 Bot doğrulanıyor...{C.RESET}", args.silent)
        ok, info = verify_bot(token)
        if ok:
            log(f"  {C.GREEN}✅ Bot aktif: @{info}{C.RESET}", args.silent)
            log(f"  {C.DIM}Hedef kanal: {channel}{C.RESET}", args.silent)
        else:
            log(f"  {C.RED}✗ Bot doğrulama başarısız: {info}{C.RESET}", args.silent)
            sys.exit(1)
        log("", args.silent)
        return

    # ── Argüman kontrolü ──
    if not args.image and not args.text:
        print(f"\n  {C.RED}✗ --image veya --text parametresi gerekli{C.RESET}")
        print(f"  {C.DIM}Kullanım: {sys.argv[0]} --image dosya.png{C.RESET}")
        print(f"  {C.DIM}Yardım:   {sys.argv[0]} --help{C.RESET}\n")
        sys.exit(1)

    # ── Resim gönderim modu ──
    if args.image:
        image_path = os.path.expanduser(args.image)
        log(f"\n  {C.CYAN}📤 Fotoğraf gönderiliyor...{C.RESET}", args.silent)
        log(f"  {C.DIM}Dosya:  {image_path}{C.RESET}", args.silent)
        log(f"  {C.DIM}Kanal:  {channel}{C.RESET}", args.silent)

        ok, result = send_photo(token, channel, image_path, args.caption)

        if ok:
            msg_id = result.get("result", {}).get("message_id", "?")
            log(f"  {C.GREEN}✅ Fotoğraf başarıyla gönderildi! (msg_id: {msg_id}){C.RESET}", args.silent)
            log("", args.silent)
            return

        # Fotoğraf başarısız — yedek metin dene
        error = result.get("error") or result.get("description", "Bilinmeyen hata")
        log(f"  {C.YELLOW}⚠ Fotoğraf gönderilemedi: {error}{C.RESET}", args.silent)

        if args.fallback_text:
            log(f"  {C.CYAN}📝 Yedek metin gönderiliyor...{C.RESET}", args.silent)
            ok, result = send_message(token, channel, args.fallback_text)
            if ok:
                log(f"  {C.GREEN}✅ Yedek metin gönderildi!{C.RESET}", args.silent)
                log("", args.silent)
                return
            else:
                fallback_error = result.get("error") or result.get("description", "Bilinmeyen hata")
                log(f"  {C.RED}✗ Yedek metin de gönderilemedi: {fallback_error}{C.RESET}", args.silent)
        else:
            log(f"  {C.DIM}İpucu: --fallback-text ile yedek metin belirleyebilirsiniz{C.RESET}", args.silent)

        log("", args.silent)
        sys.exit(1)

    # ── Metin gönderim modu ──
    if args.text:
        log(f"\n  {C.CYAN}📤 Metin gönderiliyor...{C.RESET}", args.silent)
        log(f"  {C.DIM}Kanal:  {channel}{C.RESET}", args.silent)
        log(f"  {C.DIM}Uzunluk: {len(args.text)} karakter{C.RESET}", args.silent)

        ok, result = send_message(token, channel, args.text)

        if ok:
            msg_id = result.get("result", {}).get("message_id", "?")
            log(f"  {C.GREEN}✅ Mesaj başarıyla gönderildi! (msg_id: {msg_id}){C.RESET}", args.silent)
        else:
            error = result.get("error") or result.get("description", "Bilinmeyen hata")
            log(f"  {C.RED}✗ Mesaj gönderilemedi: {error}{C.RESET}", args.silent)
            sys.exit(1)

        log("", args.silent)


if __name__ == "__main__":
    main()
