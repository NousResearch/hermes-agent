#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe Günlük Brifing Kartı Oluşturucu / Turkish Daily Brief Card Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hermes Agent Hackathon 2026 — Turkish Locale Skill Pack

Pillow ile 1200x900px PNG kart oluşturur. Gazete ön sayfası tarzında:
- Türk bayrağı renkleri (kırmızı #E30A17, beyaz)
- BIST100 özeti, USD/TRY kuru, 3 haber başlığı
- Modern/neon tasarım, Nous Research + Hermes branding
- Tarih Türkçe formatında

Kullanım / Usage:
    python turkish_brief_card.py                           # Varsayılan demo verisi
    python turkish_brief_card.py --output ~/brief.png      # Özel çıktı yolu
    python turkish_brief_card.py --data brief_data.json    # JSON verisi ile

Bağımlılıklar / Dependencies:
    pip install Pillow
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("\033[91m✗ 'Pillow' kütüphanesi gerekli / required\033[0m")
    print("  pip install Pillow")
    sys.exit(1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tasarım Sabitleri / Design Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Kart boyutları
CARD_WIDTH = 1200
CARD_HEIGHT = 900

# Türk bayrağı renkleri + modern palet
COLORS = {
    "tr_red":       (227, 10, 23),      # #E30A17 — Türk bayrağı kırmızısı
    "tr_red_dark":  (180, 8, 18),       # Koyu kırmızı
    "white":        (255, 255, 255),
    "bg_dark":      (15, 15, 25),       # Koyu arka plan
    "bg_card":      (22, 22, 38),       # Kart arka planı
    "bg_section":   (30, 30, 50),       # Bölüm arka planı
    "text_primary": (255, 255, 255),    # Ana metin
    "text_muted":   (140, 140, 165),    # Soluk metin
    "text_accent":  (255, 215, 0),      # Altın vurgu
    "neon_green":   (0, 255, 136),      # Yeşil (yükseliş)
    "neon_red":     (255, 60, 60),      # Kırmızı (düşüş)
    "neon_cyan":    (0, 210, 255),      # Cyan vurgu
    "neon_gold":    (255, 200, 50),     # Altın neon
    "divider":      (60, 60, 85),       # Çizgi rengi
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Türkçe Tarih / Turkish Date
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_turkish_date():
    """Türkçe formatlanmış tarih döndürür."""
    ist = timezone(timedelta(hours=3))
    now = datetime.now(ist)

    gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    aylar = ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

    return {
        "full": f"{now.day} {aylar[now.month]} {now.year}, {gunler[now.weekday()]}",
        "time": now.strftime("%H:%M"),
        "short": f"{now.day:02d}.{now.month:02d}.{now.year}",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Font Yönetimi / Font Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_font(size, bold=False):
    """
    Sisteme uygun font yükler.
    Sırasıyla: DejaVu Sans, Liberation Sans, Arial, varsayılan font
    Türkçe karakterleri (ç, ğ, ı, İ, ö, ş, ü) destekleyen font gerekli.
    """
    # Font arama yolları (Linux, macOS, Windows)
    font_paths = []

    if bold:
        font_paths = [
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/SFCompact.ttf",
            # Genel
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
    else:
        font_paths = [
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFCompact.ttf",
            # Genel
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue

    # Hiçbir font bulunamazsa varsayılan
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


# Font cache — her seferinde yeniden yüklememek için
_font_cache = {}

def font(size, bold=False):
    """Önbellekli font yükleme."""
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = load_font(size, bold)
    return _font_cache[key]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Çizim Yardımcıları / Drawing Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_rounded_rect(draw, xy, radius, fill):
    """Yuvarlatılmış dikdörtgen çizer."""
    x1, y1, x2, y2 = xy
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
    draw.pieslice([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=fill)
    draw.pieslice([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=fill)
    draw.pieslice([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=fill)
    draw.pieslice([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=fill)


def draw_glow_line(draw, y, width, color, thickness=2):
    """Neon efektli yatay çizgi çizer."""
    # Dış parıltı (blur efekti)
    glow_color = tuple(min(255, c + 40) for c in color) + (60,)
    for offset in range(3, 0, -1):
        alpha_line = tuple(min(255, c + 20) for c in color[:3]) + (20 * offset,)
        draw.line([(40, y - offset), (width - 40, y - offset)], fill=color, width=1)
        draw.line([(40, y + offset), (width - 40, y + offset)], fill=color, width=1)
    # Ana çizgi
    draw.line([(40, y), (width - 40, y)], fill=color, width=thickness)


def truncate_text(text, max_chars=60):
    """Metni belirli karakter sayısında keser."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1].rsplit(" ", 1)[0] + "…"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kart Oluşturma / Card Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_card(data, output_path):
    """
    Ana fonksiyon: PNG brifing kartı oluşturur.

    data yapısı:
    {
        "bist100": {"value": "9.245,67", "change": "+1,23%", "direction": "up"},
        "usd_try": {"value": "38,45", "change": "-0,15%", "direction": "down"},
        "eur_try": {"value": "41,24", "change": "+0,08%", "direction": "up"},
        "gold":    {"value": "3.245", "change": "+0,52%", "direction": "up"},
        "news": [
            {"title": "Haber başlığı 1", "source": "Hürriyet"},
            {"title": "Haber başlığı 2", "source": "Bloomberg HT"},
            {"title": "Haber başlığı 3", "source": "NTV"},
        ]
    }
    """
    tarih = get_turkish_date()

    # Ana canvas oluştur (RGBA — şeffaflık desteği)
    img = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), COLORS["bg_dark"])
    draw = ImageDraw.Draw(img)

    y = 0  # Dikey pozisyon takipçisi

    # ──────────────────────────────────────────────
    # Üst Kırmızı Bant — Türk Bayrağı Şeridi
    # ──────────────────────────────────────────────
    draw.rectangle([0, 0, CARD_WIDTH, 6], fill=COLORS["tr_red"])

    # ──────────────────────────────────────────────
    # Başlık Alanı
    # ──────────────────────────────────────────────
    y = 24

    # Nous Research + Hermes branding (sol üst)
    draw.text((50, y), "NOUS RESEARCH", fill=COLORS["text_muted"], font=font(13))
    draw.text((200, y), "×", fill=COLORS["divider"], font=font(13))
    draw.text((220, y), "HERMES AGENT", fill=COLORS["neon_cyan"], font=font(13, bold=True))

    # Saat (sağ üst)
    draw.text((CARD_WIDTH - 120, y), tarih["time"], fill=COLORS["text_muted"], font=font(14))

    y = 50

    # Ana başlık
    draw.text((50, y), "GÜNLÜK BRİFİNG", fill=COLORS["white"], font=font(38, bold=True))

    # Türk bayrağı emoji yerine küçük bayrak görseli
    # Kırmızı dikdörtgen + beyaz ay-yıldız (basitleştirilmiş)
    flag_x, flag_y = CARD_WIDTH - 130, y + 5
    draw.rectangle([flag_x, flag_y, flag_x + 72, flag_y + 42], fill=COLORS["tr_red"])
    # Hilal (basitleştirilmiş daire)
    draw.ellipse([flag_x + 20, flag_y + 8, flag_x + 46, flag_y + 34], fill=COLORS["white"])
    draw.ellipse([flag_x + 26, flag_y + 8, flag_x + 52, flag_y + 34], fill=COLORS["tr_red"])
    # Yıldız (basit nokta)
    draw.polygon([
        (flag_x + 48, flag_y + 21),
        (flag_x + 43, flag_y + 17),
        (flag_x + 44, flag_y + 23),
        (flag_x + 39, flag_y + 19),
        (flag_x + 45, flag_y + 19),
    ], fill=COLORS["white"])

    y = 98
    # Tarih satırı
    draw.text((50, y), tarih["full"], fill=COLORS["text_accent"], font=font(16))

    y = 125
    draw_glow_line(draw, y, CARD_WIDTH, COLORS["tr_red"], thickness=2)

    # ──────────────────────────────────────────────
    # Piyasa Verileri Bölümü
    # ──────────────────────────────────────────────
    y = 145

    draw.text((50, y), "📈  PİYASALAR", fill=COLORS["neon_cyan"], font=font(18, bold=True))
    y += 38

    # 4 piyasa kutusu yan yana
    markets = [
        ("BIST100",    data["bist100"]["value"],  data["bist100"]["change"],  data["bist100"]["direction"]),
        ("USD/TRY",    data["usd_try"]["value"],  data["usd_try"]["change"],  data["usd_try"]["direction"]),
        ("EUR/TRY",    data["eur_try"]["value"],  data["eur_try"]["change"],  data["eur_try"]["direction"]),
        ("GRAM ALTIN", data["gold"]["value"],     data["gold"]["change"],     data["gold"]["direction"]),
    ]

    box_width = 260
    box_height = 90
    gap = 18
    start_x = 50

    for i, (label, value, change, direction) in enumerate(markets):
        bx = start_x + i * (box_width + gap)

        # Kutu arka planı
        draw_rounded_rect(draw, (bx, y, bx + box_width, y + box_height), 10, COLORS["bg_section"])

        # Üst: etiket
        draw.text((bx + 15, y + 10), label, fill=COLORS["text_muted"], font=font(13))

        # Orta: değer (büyük)
        if label == "GRAM ALTIN":
            display_val = f"₺{value}"
        elif label == "BIST100":
            display_val = value
        else:
            display_val = f"₺{value}"
        draw.text((bx + 15, y + 30), display_val, fill=COLORS["white"], font=font(24, bold=True))

        # Alt: değişim
        if direction == "up":
            chg_color = COLORS["neon_green"]
            arrow = "▲"
        elif direction == "down":
            chg_color = COLORS["neon_red"]
            arrow = "▼"
        else:
            chg_color = COLORS["text_muted"]
            arrow = "─"

        draw.text((bx + 15, y + 62), f"{arrow} {change}", fill=chg_color, font=font(14, bold=True))

    y += box_height + 25
    draw_glow_line(draw, y, CARD_WIDTH, COLORS["divider"], thickness=1)

    # ──────────────────────────────────────────────
    # Haber Başlıkları Bölümü
    # ──────────────────────────────────────────────
    y += 18

    draw.text((50, y), "📰  GÜNDEM", fill=COLORS["neon_cyan"], font=font(18, bold=True))
    y += 38

    news = data.get("news", [])[:3]

    for i, item in enumerate(news):
        # Numara badge'i
        badge_x = 50
        draw_rounded_rect(draw, (badge_x, y, badge_x + 32, y + 32), 6, COLORS["tr_red"])
        draw.text((badge_x + 10, y + 5), str(i + 1), fill=COLORS["white"], font=font(16, bold=True))

        # Başlık
        title = truncate_text(item["title"], 72)
        draw.text((badge_x + 45, y + 2), title, fill=COLORS["white"], font=font(17))

        # Kaynak
        source = item.get("source", "")
        draw.text((badge_x + 45, y + 26), source, fill=COLORS["text_muted"], font=font(12))

        y += 58

    # Eğer 3'ten az haber varsa boşluk doldur
    if len(news) < 3:
        for i in range(len(news), 3):
            badge_x = 50
            draw_rounded_rect(draw, (badge_x, y, badge_x + 32, y + 32), 6, COLORS["bg_section"])
            draw.text((badge_x + 10, y + 5), str(i + 1), fill=COLORS["text_muted"], font=font(16, bold=True))
            draw.text((badge_x + 45, y + 6), "Haber bekleniyor...", fill=COLORS["text_muted"], font=font(16))
            y += 58

    # ──────────────────────────────────────────────
    # Alt Bilgi Bandı
    # ──────────────────────────────────────────────
    y = CARD_HEIGHT - 75
    draw_glow_line(draw, y, CARD_WIDTH, COLORS["divider"], thickness=1)

    y += 12

    # Sol: Branding
    draw.text(
        (50, y),
        "HERMES AGENT",
        fill=COLORS["neon_cyan"],
        font=font(14, bold=True),
    )
    draw.text(
        (50, y + 22),
        "Turkish Locale Skill Pack — Hackathon 2026",
        fill=COLORS["text_muted"],
        font=font(11),
    )

    # Sağ: Kaynak notu
    draw.text(
        (CARD_WIDTH - 350, y),
        "Veriler: Yahoo Finance, RSS kaynaklarından derlenmiştir",
        fill=COLORS["text_muted"],
        font=font(11),
    )
    draw.text(
        (CARD_WIDTH - 350, y + 18),
        f"Oluşturulma: {tarih['full']} {tarih['time']} (UTC+3)",
        fill=COLORS["text_muted"],
        font=font(11),
    )

    # Alt kırmızı bant
    draw.rectangle([0, CARD_HEIGHT - 6, CARD_WIDTH, CARD_HEIGHT], fill=COLORS["tr_red"])

    # ──────────────────────────────────────────────
    # Kaydet
    # ──────────────────────────────────────────────
    # RGBA → RGB (PNG şeffaflık sorunlarını önlemek için)
    img_rgb = Image.new("RGB", img.size, COLORS["bg_dark"])
    img_rgb.paste(img, mask=img.split()[3])

    # Çıktı dizinini oluştur
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    img_rgb.save(output_path, "PNG", quality=95)
    return output_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo Verisi / Demo Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_demo_data():
    """Test/demo için örnek veri seti döndürür."""
    return {
        "bist100": {"value": "9.245,67", "change": "+1,23%", "direction": "up"},
        "usd_try": {"value": "38,45", "change": "-0,15%", "direction": "down"},
        "eur_try": {"value": "41,24", "change": "+0,08%", "direction": "up"},
        "gold":    {"value": "3.245", "change": "+0,52%", "direction": "up"},
        "news": [
            {
                "title": "Merkez Bankası faiz kararını açıkladı: Politika faizi sabit tutuldu",
                "source": "Bloomberg HT",
            },
            {
                "title": "BIST100 endeksi güne yükselişle başladı, bankacılık sektörü öncü",
                "source": "Hürriyet",
            },
            {
                "title": "Türkiye-AB ticaret görüşmelerinde yeni dönem: Gümrük birliği güncellenecek",
                "source": "NTV",
            },
        ],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ana Program / Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    parser = argparse.ArgumentParser(
        description="Türkçe Günlük Brifing Kartı Oluşturucu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler / Examples:
  %(prog)s                                    Demo verisi ile kart oluştur
  %(prog)s --output ~/Desktop/brief.png       Özel çıktı yolu
  %(prog)s --data market_data.json            JSON dosyasından veri oku
  %(prog)s --demo                             Demo modu (varsayılan)

Hermes Agent Hackathon 2026 — Turkish Locale Skill Pack 🇹🇷
        """
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/turkish_brief.png"),
        help="Çıktı PNG dosyasının yolu (varsayılan: ~/turkish_brief.png)"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Veri JSON dosyası (piyasa + haber verisi)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Demo verisi ile çalıştır (varsayılan)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Veri kaynağını belirle
    if args.data:
        try:
            with open(args.data, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  📄 Veri dosyası yüklendi: {args.data}")
        except FileNotFoundError:
            print(f"\033[91m✗ Dosya bulunamadı: {args.data}\033[0m")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"\033[91m✗ JSON ayrıştırma hatası: {e}\033[0m")
            sys.exit(1)
    else:
        data = get_demo_data()
        print("  📊 Demo verisi kullanılıyor")

    # Kartı oluştur
    print(f"  🎨 Kart oluşturuluyor ({CARD_WIDTH}x{CARD_HEIGHT}px)...")

    try:
        output = generate_card(data, args.output)
        file_size = os.path.getsize(output)
        size_kb = file_size / 1024

        print(f"  ✅ Kart başarıyla oluşturuldu!")
        print(f"  📁 Dosya: {output}")
        print(f"  📐 Boyut: {CARD_WIDTH}x{CARD_HEIGHT}px ({size_kb:.1f} KB)")
        print()

    except Exception as e:
        print(f"\033[91m✗ Kart oluşturma hatası: {e}\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
