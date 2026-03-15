#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kripto Para Fiyat Takip Aracı / Crypto Price Tracker (TRY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hermes Agent Hackathon 2026 — Turkish Locale Skill Pack

CoinGecko ücretsiz API üzerinden kripto para verilerini çeker
ve ANSI renkli terminal tablosu olarak Türk Lirası cinsinden gösterir.

Kullanım / Usage:
    python bist100_prices.py                          # Varsayılan 5 coin (BTC,ETH,SOL,AVAX,DOT)
    python bist100_prices.py --coins BTC,ETH,SOL      # Belirli coinler
    python bist100_prices.py --json                    # JSON çıktı (pipeline için)
    python bist100_prices.py --sort price              # Fiyata göre sırala

Bağımlılıklar / Dependencies:
    pip install requests
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta

try:
    import requests
except ImportError:
    print("\033[91m✗ 'requests' kütüphanesi gerekli / required\033[0m")
    print("  pip install requests")
    sys.exit(1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sabitler / Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Desteklenen kripto paralar: sembol → CoinGecko ID eşlemesi
COIN_MAP = {
    "BTC":   "bitcoin",
    "ETH":   "ethereum",
    "SOL":   "solana",
    "AVAX":  "avalanche-2",
    "DOT":   "polkadot",
    "BNB":   "binancecoin",
    "XRP":   "ripple",
    "ADA":   "cardano",
    "DOGE":  "dogecoin",
    "MATIC": "matic-network",
    "LINK":  "chainlink",
    "ATOM":  "cosmos",
    "UNI":   "uniswap",
    "LTC":   "litecoin",
    "APT":   "aptos",
    "NEAR":  "near",
    "ARB":   "arbitrum",
    "OP":    "optimism",
    "FTM":   "fantom",
    "ALGO":  "algorand",
}

# Türkçe coin isimleri
COIN_NAMES = {
    "BTC":   "Bitcoin",
    "ETH":   "Ethereum",
    "SOL":   "Solana",
    "AVAX":  "Avalanche",
    "DOT":   "Polkadot",
    "BNB":   "BNB",
    "XRP":   "Ripple",
    "ADA":   "Cardano",
    "DOGE":  "Dogecoin",
    "MATIC": "Polygon",
    "LINK":  "Chainlink",
    "ATOM":  "Cosmos",
    "UNI":   "Uniswap",
    "LTC":   "Litecoin",
    "APT":   "Aptos",
    "NEAR":  "NEAR Protocol",
    "ARB":   "Arbitrum",
    "OP":    "Optimism",
    "FTM":   "Fantom",
    "ALGO":  "Algorand",
}

# Varsayılan coinler
DEFAULT_COINS = ["BTC", "ETH", "SOL", "AVAX", "DOT"]

# ANSI renk kodları
class C:
    """Terminal renkleri / Terminal colors"""
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"
    BG_RED  = "\033[41m"
    WHITE   = "\033[97m"
    MAGENTA = "\033[95m"

# CoinGecko API
COINGECKO_MARKETS_API = "https://api.coingecko.com/api/v3/coins/markets"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Yardımcı Fonksiyonlar / Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_turkish_number(value, decimals=2):
    """
    Sayıyı Türkçe formatına çevirir.
    1234567.89 → "1.234.567,89"
    """
    formatted = f"{value:,.{decimals}f}"
    # İngilizce format: 1,234,567.89 → Türkçe: 1.234.567,89
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted


def format_turkish_lira(value, decimals=2):
    """
    Sayıyı ₺ sembolü ile Türkçe formatına çevirir.
    1234567.89 → "₺1.234.567,89"
    """
    return f"₺{format_turkish_number(value, decimals)}"


def format_volume(vol):
    """
    Hacmi okunabilir formata çevirir.
    1500000 → "1,5M"  /  2300000000 → "2,3B"
    """
    if vol >= 1_000_000_000_000:
        return f"₺{vol / 1_000_000_000_000:.1f}T".replace(".", ",")
    elif vol >= 1_000_000_000:
        return f"₺{vol / 1_000_000_000:.1f}B".replace(".", ",")
    elif vol >= 1_000_000:
        return f"₺{vol / 1_000_000:.1f}M".replace(".", ",")
    elif vol >= 1_000:
        return f"₺{vol / 1_000:.1f}K".replace(".", ",")
    return f"₺{int(vol)}"


def format_market_cap(cap):
    """
    Piyasa değerini okunabilir formata çevirir.
    """
    if cap >= 1_000_000_000_000:
        return f"₺{cap / 1_000_000_000_000:.2f}T".replace(".", ",")
    elif cap >= 1_000_000_000:
        return f"₺{cap / 1_000_000_000:.1f}B".replace(".", ",")
    elif cap >= 1_000_000:
        return f"₺{cap / 1_000_000:.1f}M".replace(".", ",")
    return f"₺{int(cap)}"


def get_turkish_date():
    """Türkçe tarih formatı döndürür."""
    ist = timezone(timedelta(hours=3))
    now = datetime.now(ist)

    gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    aylar = ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

    gun_adi = gunler[now.weekday()]
    ay_adi = aylar[now.month]

    return f"{now.day} {ay_adi} {now.year}, {gun_adi} — {now.strftime('%H:%M')}"


def get_crypto_market_status():
    """
    Kripto piyasası 7/24 açıktır.
    Türkiye saatini gösterir.
    """
    ist = timezone(timedelta(hours=3))
    now = datetime.now(ist)
    return True, f"7/24 AÇIK — {now.strftime('%H:%M')} TSİ"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Veri Çekme / Data Fetching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_crypto_prices(coins):
    """
    CoinGecko /coins/markets API ile kripto fiyatlarını TRY cinsinden çeker.
    Tek istekle tüm coinleri alır — rate-limit sorunu yoktur.

    Returns:
        list[dict]: Her coin için {symbol, name, price, change_24h, change_pct_24h, volume, ...}
    """
    # Sembollerden CoinGecko ID'lerine çevir
    coin_ids = []
    symbol_order = []
    for sym in coins:
        sym_upper = sym.upper()
        cg_id = COIN_MAP.get(sym_upper)
        if cg_id:
            coin_ids.append(cg_id)
            symbol_order.append(sym_upper)
        else:
            print(f"  {C.YELLOW}⚠ Bilinmeyen coin: {sym_upper} (atlanıyor){C.RESET}", file=sys.stderr)

    if not coin_ids:
        return None

    print(f"  Veri çekiliyor... ({len(coin_ids)} coin)", end="", flush=True, file=sys.stderr)

    params = {
        "vs_currency": "try",
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "per_page": len(coin_ids),
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h",
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "HermesAgent/1.0",
    }

    try:
        resp = requests.get(COINGECKO_MARKETS_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            print(f"\r{' ' * 60}\r", end="", file=sys.stderr)
            return None

        # CoinGecko ID → sembol eşlemesi (ters çevirme)
        id_to_symbol = {v: k for k, v in COIN_MAP.items()}

        results = []
        for coin in data:
            cg_id = coin.get("id", "")
            symbol = id_to_symbol.get(cg_id, coin.get("symbol", "").upper())

            price = coin.get("current_price", 0) or 0
            change_24h = coin.get("price_change_24h", 0) or 0
            change_pct = coin.get("price_change_percentage_24h", 0) or 0
            volume = coin.get("total_volume", 0) or 0
            market_cap = coin.get("market_cap", 0) or 0
            high_24h = coin.get("high_24h", 0) or 0
            low_24h = coin.get("low_24h", 0) or 0
            ath = coin.get("ath", 0) or 0

            results.append({
                "symbol":      symbol,
                "name":        COIN_NAMES.get(symbol, coin.get("name", symbol)),
                "price":       price,
                "change":      change_24h,
                "change_pct":  change_pct,
                "volume":      volume,
                "market_cap":  market_cap,
                "high_24h":    high_24h,
                "low_24h":     low_24h,
                "ath":         ath,
                "currency":    "TRY",
            })

        print(f"\r{' ' * 60}\r", end="", file=sys.stderr)
        return results if results else None

    except requests.exceptions.HTTPError as e:
        print(f"\n  {C.RED}✗ CoinGecko API hatası: {e}{C.RESET}", file=sys.stderr)
        return None
    except requests.exceptions.ConnectionError:
        print(f"\n  {C.RED}✗ Bağlantı hatası — internet bağlantınızı kontrol edin.{C.RESET}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\n  {C.RED}✗ Beklenmeyen hata: {e}{C.RESET}", file=sys.stderr)
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tablo Formatları / Table Formatting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_table(coins):
    """
    ANSI renkli terminal tablosu oluşturur.
    Yeşil = yükseliş, Kırmızı = düşüş
    """
    if not coins:
        print(f"\n  {C.RED}✗ Veri alınamadı. Lütfen internet bağlantınızı kontrol edin.{C.RESET}")
        return

    # Türkçe tarih ve piyasa durumu
    tarih = get_turkish_date()
    acik, durum = get_crypto_market_status()
    durum_renk = C.GREEN if acik else C.RED

    # Başlık
    print()
    print(f"  {C.BOLD}{C.WHITE}₿ KRİPTO PARA FİYATLARI (TRY){C.RESET}")
    print(f"  {C.DIM}{tarih}{C.RESET}")
    print(f"  Piyasa: {durum_renk}{durum}{C.RESET}")
    print()

    # Fiyat ondalık basamak sayısını belirle
    def price_decimals(price):
        if price >= 100_000:
            return 0
        elif price >= 1_000:
            return 2
        elif price >= 1:
            return 2
        else:
            return 4

    # Tablo başlığı
    header = (
        f"  {C.BOLD}{C.CYAN}"
        f"{'SEMBOL':<8}"
        f"{'İSİM':<16}"
        f"{'FİYAT (₺)':>16}"
        f"{'24s DEĞİŞİM':>14}"
        f"{'%':>9}"
        f"{'HACİM (24s)':>13}"
        f"{'P.DEĞERİ':>12}"
        f"{C.RESET}"
    )
    print(header)
    print(f"  {C.DIM}{'━' * 86}{C.RESET}")

    # Kazanan/kaybeden sayaçları
    yukselenler = 0
    dusenler = 0
    degismeyenler = 0

    for coin in coins:
        pct = coin["change_pct"]

        # Renk seçimi
        if pct > 0:
            color = C.GREEN
            arrow = "▲"
            yukselenler += 1
        elif pct < 0:
            color = C.RED
            arrow = "▼"
            dusenler += 1
        else:
            color = C.YELLOW
            arrow = "─"
            degismeyenler += 1

        # Değerleri Türkçe formatla
        dec = price_decimals(coin["price"])
        fiyat = format_turkish_number(coin["price"], dec)
        degisim = f"{arrow} {format_turkish_number(abs(coin['change']), dec)}"
        yuzde = f"{'+' if pct >= 0 else ''}{format_turkish_number(pct)}%"
        hacim = format_volume(coin["volume"])
        piyasa_degeri = format_market_cap(coin["market_cap"])
        isim = coin["name"][:14]

        row = (
            f"  {C.BOLD}{C.MAGENTA}{coin['symbol']:<8}{C.RESET}"
            f"{C.DIM}{isim:<16}{C.RESET}"
            f"{color}{fiyat:>16}{C.RESET}"
            f"{color}{degisim:>14}{C.RESET}"
            f"{color}{C.BOLD}{yuzde:>9}{C.RESET}"
            f"{C.DIM}{hacim:>13}{C.RESET}"
            f"{C.DIM}{piyasa_degeri:>12}{C.RESET}"
        )
        print(row)

    # 24 saatlik düşük/yüksek alt satırı
    print(f"  {C.DIM}{'━' * 86}{C.RESET}")

    # Mini 24h aralık gösterimi
    for coin in coins:
        pct = coin["change_pct"]
        color = C.GREEN if pct > 0 else (C.RED if pct < 0 else C.YELLOW)
        dec = price_decimals(coin["price"])
        low = format_turkish_number(coin["low_24h"], dec)
        high = format_turkish_number(coin["high_24h"], dec)
        # Basit bar gösterici
        if coin["high_24h"] > coin["low_24h"]:
            pos = (coin["price"] - coin["low_24h"]) / (coin["high_24h"] - coin["low_24h"])
            pos = max(0, min(1, pos))
            bar_len = 16
            filled = int(pos * bar_len)
            bar = f"{'━' * filled}{color}●{C.DIM}{'━' * (bar_len - filled)}"
        else:
            bar = f"{'━' * 8}●{'━' * 8}"

        print(
            f"  {C.DIM}{coin['symbol']:<8}"
            f"  24s: ₺{low}  {bar}{C.RESET}  {C.DIM}₺{high}{C.RESET}"
        )

    print()

    # Özet
    print(
        f"  {C.GREEN}▲ {yukselenler} yükselen{C.RESET}  "
        f"{C.RED}▼ {dusenler} düşen{C.RESET}  "
        f"{C.YELLOW}─ {degismeyenler} değişmeyen{C.RESET}  "
        f"{C.DIM}│ Toplam {len(coins)} coin{C.RESET}"
    )
    print(f"  {C.DIM}Kaynak: CoinGecko API • Fiyatlar Türk Lirası (₺) cinsindendir{C.RESET}")
    print()


def render_json(coins):
    """JSON formatında çıktı (pipeline ve otomasyon için)."""
    ist = timezone(timedelta(hours=3))
    now = datetime.now(ist)

    output = {
        "meta": {
            "timestamp": now.isoformat(),
            "source": "CoinGecko",
            "currency": "TRY",
            "count": len(coins),
        },
        "coins": coins,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ana Program / Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(
        description="Kripto Para Fiyat Takibi / Crypto Price Tracker (TRY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler / Examples:
  %(prog)s                           Varsayılan 5 coin (BTC,ETH,SOL,AVAX,DOT)
  %(prog)s --coins BTC,ETH,SOL       Belirli coinler
  %(prog)s --coins BTC --json        JSON çıktı
  %(prog)s --sort price              Fiyata göre sırala

Desteklenen coinler / Supported coins:
  BTC ETH SOL AVAX DOT BNB XRP ADA DOGE MATIC
  LINK ATOM UNI LTC APT NEAR ARB OP FTM ALGO

Hermes Agent Hackathon 2026 — Turkish Locale Skill Pack 🇹🇷
        """
    )

    parser.add_argument(
        "--coins",
        type=str,
        default=None,
        help="Virgülle ayrılmış coin kodları (ör: BTC,ETH,SOL)"
    )
    parser.add_argument(
        "--sort",
        choices=["symbol", "price", "change", "volume", "market_cap"],
        default=None,
        help="Sıralama kriteri: symbol, price, change, volume, market_cap"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON formatında çıktı ver (otomasyon/pipeline için)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Coin listesini belirle
    if args.coins:
        coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    else:
        coins = DEFAULT_COINS[:]

    if not coins:
        print(f"{C.RED}✗ Coin kodu belirtilmedi.{C.RESET}")
        sys.exit(1)

    # Verileri çek
    results = fetch_crypto_prices(coins)

    if not results:
        print(f"\n  {C.RED}✗ Kripto para verisi alınamadı.{C.RESET}")
        print(f"  {C.DIM}İnternet bağlantınızı kontrol edin.{C.RESET}\n")
        sys.exit(1)

    # Sıralama
    if args.sort:
        sort_keys = {
            "symbol":     lambda s: s["symbol"],
            "price":      lambda s: s["price"],
            "change":     lambda s: s["change_pct"],
            "volume":     lambda s: s["volume"],
            "market_cap": lambda s: s["market_cap"],
        }
        reverse = args.sort in ("price", "change", "volume", "market_cap")
        results.sort(key=sort_keys[args.sort], reverse=reverse)

    # Çıktı
    if args.json:
        render_json(results)
    else:
        render_table(results)


if __name__ == "__main__":
    main()
