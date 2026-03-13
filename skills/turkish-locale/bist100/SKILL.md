---
name: bist100
version: 1.0.0
description: "Borsa İstanbul BIST100 endeksi, hisse takibi, döviz kurları (USD/TRY, EUR/TRY) ve altın fiyatları. / Borsa Istanbul BIST100 index, stock tracking, currency rates, and gold prices for the Turkish market."
tags: [turkish, finance, bist100, borsa, stocks, forex, gold, usd-try, eur-try]
author: Hermes Agent Hackathon 2026
triggers:
  - "BIST100", "BIST 100", "borsa", "Borsa İstanbul"
  - "hisse", "hisse senedi", stock ticker like "THYAO", "GARAN", "ASELS"
  - "dolar kuru", "euro kuru", "döviz", "USD/TRY", "EUR/TRY"
  - "altın fiyatı", "gram altın", "çeyrek altın"
  - "piyasa", "piyasalar nasıl", "borsa nasıl"
---

# 📈 BIST100 & Turkish Markets / Borsa İstanbul ve Türk Piyasaları

> Real-time tracking of Borsa Istanbul, currency rates, and commodities.
> Borsa İstanbul, döviz kurları ve emtia fiyatlarının gerçek zamanlı takibi.

---

## When To Use / Ne Zaman Kullanılır

- User asks about Turkish stock market or BIST100 index
- User queries a specific Turkish stock (e.g., THYAO, GARAN, SISE, ASELS)
- User asks about USD/TRY, EUR/TRY, or other currency pairs
- User asks about gold prices (gram altın, çeyrek altın, ons altın)
- Part of the daily brief pipeline (called by turkish-daily-brief)
- User says "piyasalar nasıl?", "borsa ne durumda?", "dolar kaç?"

---

## Market Reference / Piyasa Referansı

### Borsa İstanbul İşlem Saatleri
```
Sürekli İşlem:  10:00 - 18:00 (UTC+3)
Açılış Seansı:  09:40 - 10:00
Kapanış Seansı: 18:00 - 18:10
Tatil Günleri:  Cumartesi, Pazar, Resmi Tatiller
Zaman Dilimi:   Europe/Istanbul (UTC+3, yıl boyu)
```

### Major BIST100 Stocks / Önemli BIST100 Hisseleri
```
THYAO  — Türk Hava Yolları (Turkish Airlines)
GARAN  — Garanti BBVA Bankası
AKBNK  — Akbank
YKBNK  — Yapı Kredi Bankası
ISCTR  — İş Bankası
SISE   — Şişecam
ASELS  — ASELSAN (Savunma/Defence)
TUPRS  — Tüpraş (Enerji/Refinery)
EREGL  — Ereğli Demir Çelik
KCHOL  — Koç Holding
SAHOL  — Sabancı Holding
TAVHL  — TAV Havalimanları
BIMAS  — BİM Mağazaları
TCELL  — Turkcell
TOASO  — Tofaş Otomobil
FROTO  — Ford Otosan
PETKM  — Petkim
EKGYO  — Emlak Konut GYO
PGSUS  — Pegasus Havayolları
SASA   — SASA Polyester
```

### Currency Pairs / Döviz Çiftleri
```
USD/TRY  — Amerikan Doları / Türk Lirası
EUR/TRY  — Euro / Türk Lirası
GBP/TRY  — İngiliz Sterlini / Türk Lirası
JPY/TRY  — Japon Yeni / Türk Lirası
CHF/TRY  — İsviçre Frangı / Türk Lirası
```

### Gold & Commodities / Altın ve Emtialar
```
Gram Altın     — 1 gram 24 ayar altın (TRY)
Çeyrek Altın   — ~1.75 gram (TRY)
Yarım Altın    — ~3.50 gram (TRY)
Tam Altın      — ~7.00 gram (TRY)
Ons Altın      — XAU/USD (Troy ounce)
Gümüş          — XAG/USD
Brent Petrol   — Brent Crude (USD)
```

---

## Data Sources & Methods / Veri Kaynakları ve Yöntemler

### Method 1: Web Scraping (Primary / Ana Yöntem)

**Bloomberg HT — Most Reliable for Turkish Markets:**

```python
from hermes_tools import web_extract

# BIST100 Index
result = web_extract(["https://www.bloomberght.com/borsa/endeksler/bist-100"])
# Extract: Current value, change %, daily high/low, volume

# Individual Stock
result = web_extract(["https://www.bloomberght.com/borsa/hisseler/THYAO"])
# Extract: Price, change, volume, market cap

# Currency Rates
result = web_extract(["https://www.bloomberght.com/doviz"])
# Extract: USD/TRY, EUR/TRY, GBP/TRY

# Gold Prices
result = web_extract(["https://www.bloomberght.com/altin"])
# Extract: Gram, çeyrek, yarım, tam, ons
```

**Bigpara (Hürriyet Finance):**

```python
# BIST100
result = web_extract(["https://bigpara.hurriyet.com.tr/borsa/canli-borsa/"])

# Currency
result = web_extract(["https://bigpara.hurriyet.com.tr/doviz/"])

# Gold
result = web_extract(["https://bigpara.hurriyet.com.tr/altin/"])
```

**Investing.com TR:**

```python
# BIST100
result = web_extract(["https://tr.investing.com/indices/ise-100"])

# Comprehensive market overview
result = web_extract(["https://tr.investing.com/"])
```

### Method 2: Yahoo Finance API (Backup / Yedek Yöntem)

```python
from hermes_tools import terminal

# BIST100 Index
terminal("curl -s 'https://query1.finance.yahoo.com/v8/finance/chart/XU100.IS?interval=1d&range=1d'")

# Individual Stock (add .IS suffix for Borsa Istanbul)
terminal("curl -s 'https://query1.finance.yahoo.com/v8/finance/chart/THYAO.IS?interval=1d&range=1d'")

# USD/TRY
terminal("curl -s 'https://query1.finance.yahoo.com/v8/finance/chart/USDTRY=X?interval=1d&range=1d'")

# EUR/TRY
terminal("curl -s 'https://query1.finance.yahoo.com/v8/finance/chart/EURTRY=X?interval=1d&range=1d'")

# Gold (XAU/USD)
terminal("curl -s 'https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d&range=1d'")
```

**Yahoo Finance Ticker Reference for BIST:**
```
BIST100 Index:  XU100.IS
BIST30 Index:   XU030.IS
BIST Bank:      XBANK.IS
Stocks:         {TICKER}.IS  (e.g., THYAO.IS, GARAN.IS, ASELS.IS)
USD/TRY:        USDTRY=X
EUR/TRY:        EURTRY=X
```

### Method 3: DuckDuckGo Search (Fallback / Son Çare)

```python
from hermes_tools import web_search

# Quick price check when APIs are down
result = web_search("BIST100 bugün", limit=3)
result = web_search("dolar TL kuru bugün", limit=3)
result = web_search("gram altın fiyatı bugün", limit=3)
```

---

## Procedures / İşlem Adımları

### Procedure 1: Market Overview / Piyasa Özeti

**Trigger:** "Piyasalar nasıl?", "Borsa ne durumda?", "Market overview"

Steps:
1. Fetch BIST100 index value and daily change
2. Fetch USD/TRY and EUR/TRY rates
3. Fetch gram altın price
4. Check if market is open/closed based on current time (UTC+3)
5. Format as compact summary

**Output:**
```
📊 PİYASA ÖZETİ — 11 Mart 2026, 14:30

BIST100:    9,245.67  ▲ +1.23% (+112.45)
USD/TRY:    38.4521   ▼ -0.15%
EUR/TRY:    41.2380   ▲ +0.08%
Gram Altın: ₺3,245    ▲ +0.52%

📈 Borsa AÇIK — kapanışa 3s 30dk
```

### Procedure 2: Stock Lookup / Hisse Sorgulama

**Trigger:** "THYAO nasıl?", "THYAO hissesi", "Garanti hissesi ne durumda?"

Steps:
1. Identify stock ticker from user input
   - Direct ticker: "THYAO" → THYAO
   - Company name: "Türk Hava Yolları" → THYAO
   - Partial match: "Garanti" → GARAN
2. Fetch stock data from Bloomberg HT or Yahoo Finance
3. Extract: price, daily change %, volume, high/low, market cap
4. Format with Turkish number formatting (1.000,50)

**Ticker Resolution Table:**
```python
COMPANY_TICKERS = {
    "thy": "THYAO", "türk hava": "THYAO", "turkish airlines": "THYAO",
    "garanti": "GARAN", "akbank": "AKBNK", "yapı kredi": "YKBNK",
    "iş bankası": "ISCTR", "şişecam": "SISE", "aselsan": "ASELS",
    "tüpraş": "TUPRS", "ereğli": "EREGL", "koç": "KCHOL",
    "sabancı": "SAHOL", "tav": "TAVHL", "bim": "BIMAS",
    "turkcell": "TCELL", "tofaş": "TOASO", "ford otosan": "FROTO",
    "petkim": "PETKM", "pegasus": "PGSUS", "sasa": "SASA",
}
```

**Output:**
```
📈 THYAO — Türk Hava Yolları
━━━━━━━━━━━━━━━━━━━━━━━━━━

Fiyat:     ₺287,40    ▲ +2.15%
Değişim:   +₺6,05
Hacim:     45.2M
Gün İçi:   ₺282,00 - ₺289,50
P.Değeri:  ₺396,2 Milyar

🕐 Son güncelleme: 14:30
```

### Procedure 3: Currency Rates / Döviz Kurları

**Trigger:** "Dolar kaç?", "Euro kuru", "Döviz kurları"

Steps:
1. Detect which currencies are requested
2. Fetch from Bloomberg HT or Yahoo Finance
3. Include buy/sell spread if available
4. Show daily change direction and percentage

**Output:**
```
💱 DÖVİZ KURLARI — 11 Mart 2026

USD/TRY:  38,4521  ▼ -0,15%  (Alış: 38,42 / Satış: 38,48)
EUR/TRY:  41,2380  ▲ +0,08%  (Alış: 41,20 / Satış: 41,28)
GBP/TRY:  48,7650  ▲ +0,22%  (Alış: 48,72 / Satış: 48,81)

🕐 Son güncelleme: 14:30 • TCMB efektif kurları
```

### Procedure 4: Gold Prices / Altın Fiyatları

**Trigger:** "Altın kaç?", "Gram altın", "Çeyrek altın fiyatı"

Steps:
1. Fetch gold prices from Bloomberg HT or Bigpara
2. Include all popular variants (gram, çeyrek, yarım, tam, ons)
3. Show daily changes

**Output:**
```
🥇 ALTIN FİYATLARI — 11 Mart 2026

Gram Altın:    ₺3.245,00   ▲ +₺17  (+0,52%)
Çeyrek Altın:  ₺5.320,00   ▲ +₺28  (+0,53%)
Yarım Altın:   ₺10.640,00  ▲ +₺56  (+0,53%)
Tam Altın:     ₺21.280,00  ▲ +₺112 (+0,53%)
Ons Altın:     $2.185,40   ▲ +$8,20 (+0,38%)

🕐 Son güncelleme: 14:30
```

### Procedure 5: Portfolio Watchlist / Portföy Takip Listesi

**Trigger:** "Portföyümü takip et", "Şu hisseleri izle: THYAO, GARAN, ASELS"

Steps:
1. Parse ticker list from user input
2. Fetch current prices for all tickers
3. Format as compact watchlist table
4. Offer to schedule periodic updates

**Output:**
```
👁 TAKİP LİSTEN — 11 Mart 2026, 14:30

Hisse    Fiyat      Değişim    %
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THYAO    ₺287,40    +₺6,05    ▲ +2,15%
GARAN    ₺142,80    -₺1,20    ▼ -0,83%
ASELS    ₺84,50     +₺0,75    ▲ +0,90%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ Takip zamanlamak ister misin?
```

---

## Number Formatting / Sayı Formatlama

Turkish number format differs from English:
```python
def format_turkish_number(value, decimals=2):
    """Format number in Turkish style: 1.234.567,89"""
    if isinstance(value, str):
        value = float(value.replace('.', '').replace(',', '.'))
    formatted = f"{value:,.{decimals}f}"
    # Swap . and , for Turkish format
    formatted = formatted.replace(',', 'X').replace('.', ',').replace('X', '.')
    return formatted

def format_turkish_currency(value, symbol="₺"):
    """Format as Turkish currency."""
    return f"{symbol}{format_turkish_number(value)}"

# Examples:
# 1234567.89  →  "₺1.234.567,89"
# 38.4521     →  "38,4521"
# 0.0523      →  "%5,23" (percentage)
```

---

## Market Status Detection / Piyasa Durumu Tespiti

```python
from datetime import datetime, timezone, timedelta

def get_market_status():
    """Check if Borsa Istanbul is currently open."""
    ist = timezone(timedelta(hours=3))
    now = datetime.now(ist)
    
    # Weekend check
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return "KAPALI", "Hafta sonu — Pazartesi 10:00'da açılır"
    
    hour_min = now.hour * 60 + now.minute
    
    if hour_min < 580:  # Before 09:40
        return "KAPALI", "Açılış seansı 09:40'ta başlar"
    elif hour_min < 600:  # 09:40-10:00
        return "AÇILIŞ SEANSI", "Sürekli işlem 10:00'da başlar"
    elif hour_min < 1080:  # 10:00-18:00
        remaining = 1080 - hour_min
        h, m = divmod(remaining, 60)
        return "AÇIK", f"Kapanışa {h}s {m}dk"
    elif hour_min < 1090:  # 18:00-18:10
        return "KAPANIŞ SEANSI", "Gün sonu seansı"
    else:
        return "KAPALI", "Yarın 10:00'da açılır"
```

---

## Error Handling / Hata Yönetimi

- If Bloomberg HT fails → try Bigpara → try Yahoo Finance → try web_search
- If market is closed → show last closing values with note
- If ticker not found → suggest closest match from known tickers list
- If currency API fails → try TCMB XML (https://www.tcmb.gov.tr/kurlar/today.xml)
- Always show data timestamp so user knows freshness

---

## Pitfalls / Dikkat Edilecekler

1. **Market hours:** BIST operates 10:00-18:00 UTC+3, no DST — don't show "live" outside hours
2. **Yahoo suffix:** All BIST tickers need `.IS` suffix on Yahoo Finance
3. **Number format:** Turkish uses dot for thousands, comma for decimal — NEVER mix with English format
4. **Gold variants:** Çeyrek/yarım/tam altın are NOT simply multiples of gram due to crafting markup
5. **TCMB rates:** Central bank rates differ from market rates — specify which you're showing
6. **Weekend data:** Don't show stale Friday data as "current" on Monday — label it clearly
7. **Lira symbol:** Use ₺ (U+20BA), not TL or TRY in display
