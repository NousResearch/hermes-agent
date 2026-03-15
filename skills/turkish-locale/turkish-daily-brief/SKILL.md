---
name: turkish-daily-brief
version: 1.0.0
description: "Telegram veya Discord üzerinden otomatik günlük Türkçe brifing — haberler, piyasalar, hava durumu ve günün özeti. / Automated daily Turkish brief via Telegram or Discord — news, markets, weather, and day summary."
tags: [turkish, daily-brief, telegram, discord, automation, cronjob, morning-brief]
author: Hermes Agent Hackathon 2026
triggers:
  - "günlük brifing", "sabah brifing", "daily brief"
  - "her sabah haber gönder", "otomatik sabah özeti"
  - "Telegram'dan brifing", "günlük rapor"
  - user wants to schedule recurring Turkish news/market updates
---

# 🌅 Turkish Daily Brief / Türkçe Günlük Brifing

> Automated morning brief combining news, markets, and weather — delivered
> to Telegram, Discord, or CLI every day at the time you choose.
>
> Haberler, piyasalar ve hava durumunu birleştiren otomatik sabah brifing —
> her gün seçtiğin saatte Telegram, Discord veya CLI'ye teslim edilir.

---

## When To Use / Ne Zaman Kullanılır

- User wants to set up a recurring Turkish daily brief
- User asks "her sabah brifing gönder" or "schedule daily brief"
- User wants a one-time "günün özeti" (today's summary)
- User wants to customize what goes into their daily brief

---

## What's Included / İçerik

The daily brief assembles these sections:

```
┌─────────────────────────────────────────┐
│  🌅 GÜNAYDIN BRİFİNG — {tarih}         │
│                                          │
│  📰 1. GÜNDEM (Top 5 haberler)           │
│  📈 2. PİYASA (BIST100, döviz, altın)   │
│  🌤  3. HAVA DURUMU (kullanıcı şehri)    │
│  🗓  4. BUGÜN NE VAR (özel günler)       │
│  💡 5. GÜNÜN SÖZÜ (motivasyon)           │
│                                          │
└─────────────────────────────────────────┘
```

---

## Procedures / İşlem Adımları

### Procedure 1: One-Time Brief / Tek Seferlik Brifing

**Trigger:** "Bugünün özetini ver", "Günün brifingini hazırla"

Steps:
1. Load `turkish-news` skill → fetch top 5 headlines
2. Load `bist100` skill → fetch market overview (BIST100, USD/TRY, EUR/TRY, gram altın)
3. Fetch weather for user's city (default: İstanbul)
4. Check today's date for special occasions
5. Assemble and format the brief
6. Present to user

**Implementation:**

```python
from hermes_tools import web_search, web_extract, terminal
from datetime import datetime, timezone, timedelta

ist = timezone(timedelta(hours=3))
now = datetime.now(ist)
tarih = now.strftime("%d %B %Y, %A")

# Map English day/month names to Turkish
gun_map = {
    "Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba",
    "Thursday": "Perşembe", "Friday": "Cuma", "Saturday": "Cumartesi",
    "Sunday": "Pazar"
}
ay_map = {
    "January": "Ocak", "February": "Şubat", "March": "Mart",
    "April": "Nisan", "May": "Mayıs", "June": "Haziran",
    "July": "Temmuz", "August": "Ağustos", "September": "Eylül",
    "October": "Ekim", "November": "Kasım", "December": "Aralık"
}

for en, tr in gun_map.items():
    tarih = tarih.replace(en, tr)
for en, tr in ay_map.items():
    tarih = tarih.replace(en, tr)

# --- Section 1: News ---
# (Use turkish-news procedures to fetch top 5 headlines)

# --- Section 2: Markets ---
# (Use bist100 procedures to fetch market snapshot)

# --- Section 3: Weather ---
result = web_search(f"istanbul hava durumu bugün", limit=1)
# Parse temperature and conditions

# --- Section 4: Special Days ---
ozel_gunler = {
    (1, 1): "🎉 Yılbaşı",
    (23, 4): "🇹🇷 23 Nisan Ulusal Egemenlik ve Çocuk Bayramı",
    (1, 5): "💪 1 Mayıs Emek ve Dayanışma Günü",
    (19, 5): "🇹🇷 19 Mayıs Atatürk'ü Anma, Gençlik ve Spor Bayramı",
    (15, 7): "🇹🇷 15 Temmuz Demokrasi ve Milli Birlik Günü",
    (30, 8): "🇹🇷 30 Ağustos Zafer Bayramı",
    (29, 10): "🇹🇷 29 Ekim Cumhuriyet Bayramı",
    (10, 11): "🖤 10 Kasım Atatürk'ü Anma Günü",
    (14, 2): "❤️ Sevgililer Günü",
    (8, 3): "💜 Dünya Kadınlar Günü",
    (14, 3): "🧪 Pi Günü",
    (21, 3): "🌸 Nevruz",
}

# --- Section 5: Daily Quote ---
# Rotate from a curated list of Turkish motivational quotes
```

### Procedure 2: Schedule Recurring Brief / Tekrarlayan Brifing Zamanla

**Trigger:** "Her sabah 8'de brifing gönder", "Daily brief schedule"

Steps:
1. Ask user for delivery time (default: 08:00 Istanbul time)
2. Ask user for delivery channel (Telegram/Discord/CLI)
3. Ask user for city (weather, default: İstanbul)
4. Ask user for optional stock watchlist
5. Create self-contained cronjob prompt
6. Schedule with schedule_cronjob

**CRITICAL: The cronjob prompt must be 100% self-contained. The future agent 
has NO memory of this conversation. Include ALL URLs, procedures, and formatting.**

**Cronjob Setup:**

```python
# Convert Istanbul time to UTC cron (Istanbul is UTC+3, no DST)
# User wants 08:00 Istanbul → 05:00 UTC
istanbul_hour = 8  # user's desired hour
utc_hour = istanbul_hour - 3

schedule = f"0 {utc_hour} * * *"  # Every day at the specified UTC hour
```

**Self-Contained Cronjob Prompt Template:**

```
Sen Hermes Agent'sın. Türkçe günlük sabah brifing hazırla ve gönder.

GÖREV: Aşağıdaki bölümleri içeren bir brifing oluştur.

═══════════════════════════════════════
BÖLÜM 1: HABERLER
═══════════════════════════════════════
Şu RSS kaynaklarından en önemli 5 haberi çek:
- Hürriyet: https://www.hurriyet.com.tr/rss/anasayfa
- NTV: https://www.ntv.com.tr/son-dakika.rss
- Bloomberg HT: https://www.bloomberght.com/rss

Her RSS kaynağı için:
  curl -s -m 10 '{url}'
ile çek, XML'den başlıkları ayıkla, tekrar edenleri ele, en önemli 5 tanesini seç.
Her haberi 1 cümleyle özetle.

═══════════════════════════════════════
BÖLÜM 2: PİYASALAR
═══════════════════════════════════════
Şu verileri çek:
- BIST100: web_extract("https://bigpara.hurriyet.com.tr/borsa/canli-borsa/")
- Döviz: web_extract("https://bigpara.hurriyet.com.tr/doviz/")
- Altın: web_extract("https://bigpara.hurriyet.com.tr/altin/")

{watchlist_section}

Formatla:
  BIST100:    değer  ▲/▼ %değişim
  USD/TRY:    değer  ▲/▼ %değişim
  EUR/TRY:    değer  ▲/▼ %değişim
  Gram Altın: değer  ▲/▼ %değişim

═══════════════════════════════════════
BÖLÜM 3: HAVA DURUMU
═══════════════════════════════════════
web_search("{sehir} hava durumu bugün") ile hava durumunu bul.
Sıcaklık ve genel durum yaz.

═══════════════════════════════════════
FORMAT
═══════════════════════════════════════
Tüm çıktıyı şu formatta birleştir:

🌅 GÜNAYDIN BRİFİNG — {tarih}

📰 GÜNDEM
1. {haber_1}
2. {haber_2}
3. {haber_3}
4. {haber_4}
5. {haber_5}

📈 PİYASALAR
BIST100:    {değer}  {yön} {yüzde}
USD/TRY:    {değer}  {yön} {yüzde}
EUR/TRY:    {değer}  {yön} {yüzde}
Gram Altın: {değer}  {yön} {yüzde}
{watchlist}

🌤 HAVA — {şehir}
{sıcaklık}°C, {durum}

İyi günler! ☀️

NOT: Türkçe sayı formatı kullan (nokta binlik, virgül ondalık: 1.234,56).
Çıktı dili: Türkçe.
```

**Actual schedule_cronjob Call:**

```python
# Use Hermes Agent's schedule_cronjob tool:
schedule_cronjob(
    prompt=SELF_CONTAINED_PROMPT,  # The full template above, filled in
    schedule="0 5 * * *",          # 08:00 Istanbul = 05:00 UTC
    deliver="telegram",            # or "discord", "origin"
    name="turkish-daily-brief"
)
```

### Procedure 3: Customize Brief / Brifingi Özelleştir

**Trigger:** "Brifinge kripto ekle", "Hava durumunu çıkar", "Spor haberleri de olsun"

Available customizations:
- **Add sections:** kripto (CoinTelegraph TR), spor, teknoloji
- **Remove sections:** hava durumu, günün sözü
- **Change city:** İstanbul → Ankara, İzmir, Antalya, etc.
- **Add watchlist:** specific BIST tickers to track
- **Change time:** different delivery hour
- **Change language:** mixed TR/EN or full English

Steps:
1. List current cronjob with list_cronjobs
2. Remove existing turkish-daily-brief job
3. Modify the prompt template per user request
4. Reschedule with updated prompt

### Procedure 4: Ad-hoc Market + News Combo / Anlık Piyasa + Haber Kombini

**Trigger:** "Piyasalar ve haberler nasıl?", "Kısa güncel durum ver"

Steps:
1. Run both news fetch and market fetch in parallel
2. Combine into a mini-brief format
3. Present immediately (no scheduling)

---

## Delivery Channel Setup / Teslimat Kanalı Kurulumu

### Telegram Setup
```
The user's Telegram delivery is configured in Hermes Agent's config.
Use deliver="telegram" to send to the user's home channel.
For a specific chat: deliver="telegram:CHAT_ID"
```

### Discord Setup
```
Use deliver="discord" to send to user's home channel.
For a specific channel: deliver="discord:CHANNEL_ID"
```

### CLI (Local) Setup
```
Use deliver="local" to save briefs to ~/daily-briefs/
Files named: brief-YYYY-MM-DD.txt
```

---

## Special Days Database / Özel Günler Veritabanı

```python
TURKISH_SPECIAL_DAYS = {
    # Resmi Tatiller
    (1, 1):   ("🎉", "Yılbaşı"),
    (23, 4):  ("🇹🇷", "Ulusal Egemenlik ve Çocuk Bayramı"),
    (1, 5):   ("💪", "Emek ve Dayanışma Günü"),
    (19, 5):  ("🇹🇷", "Atatürk'ü Anma, Gençlik ve Spor Bayramı"),
    (15, 7):  ("🇹🇷", "Demokrasi ve Milli Birlik Günü"),
    (30, 8):  ("🇹🇷", "Zafer Bayramı"),
    (29, 10): ("🇹🇷", "Cumhuriyet Bayramı"),
    
    # Anma Günleri
    (10, 11): ("🖤", "Atatürk'ü Anma Günü — 09:05'te saygı duruşu"),
    (18, 3):  ("🌹", "Çanakkale Deniz Zaferi"),
    
    # Popüler Günler
    (14, 2):  ("❤️", "Sevgililer Günü"),
    (8, 3):   ("💜", "Dünya Kadınlar Günü"),
    (21, 3):  ("🌸", "Nevruz / Bahar Bayramı"),
    (1, 4):   ("😄", "Nisan Bir Şakası"),
    (5, 6):   ("🌍", "Dünya Çevre Günü"),
    (24, 11): ("👨‍🏫", "Öğretmenler Günü"),
    (31, 12): ("🎆", "Yılbaşı Gecesi"),
    
    # Note: Ramazan Bayramı and Kurban Bayramı dates change yearly
    # (based on Islamic lunar calendar) — check dynamically
}
```

---

## Motivational Quotes Pool / Motivasyon Sözleri Havuzu

```python
import random

GUNUN_SOZLERI = [
    "Başarı, cesaretin çocuğudur. — Atatürk",
    "Hayatta en hakiki mürşit ilimdir. — Atatürk",
    "Pes etme! Başlangıcın çok zor olabilir, sonun çok daha iyi olacak.",
    "Bugün zorlanabilirsin ama yarın güçlü olacaksın.",
    "Küçük adımlar, büyük yolculuklar başlatır.",
    "Her sabah yeni bir başlangıç, yeni bir fırsat.",
    "Yapamam deme, henüz yapamıyorum de.",
    "Hedefin yoksa, her rüzgar seni savurur. — Seneca",
    "En karanlık gece bile bir gün ağaracaktır.",
    "Bugün yarının dünüdür, iyi değerlendir.",
    "Olmaz denen her şey, birileri yapana kadardır.",
    "Sabır acıdır, meyvesi tatlıdır. — Aristoteles",
    "Düşmek ayıp değil, düştüğün yerde kalmak ayıp.",
    "Tek bildiğim hiçbir şey bilmediğimdir. — Sokrates",
]

def gunun_sozu():
    return random.choice(GUNUN_SOZLERI)
```

---

## Full Output Template / Tam Çıktı Şablonu

```
🌅 GÜNAYDIN BRİFİNG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 {gün}, {gün_adı} — {saat}
{ozel_gun_satiri}

📰 GÜNDEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. {haber_1} [{kaynak}]
2. {haber_2} [{kaynak}]
3. {haber_3} [{kaynak}]
4. {haber_4} [{kaynak}]
5. {haber_5} [{kaynak}]

📈 PİYASALAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BIST100:    {değer}    {▲/▼} {%değişim}
USD/TRY:    ₺{değer}   {▲/▼} {%değişim}
EUR/TRY:    ₺{değer}   {▲/▼} {%değişim}
Gram Altın: ₺{değer}   {▲/▼} {%değişim}

{hisse_takip_listesi}

🌤 HAVA DURUMU — {şehir}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sıcaklık}°C — {durum}
Hissedilen: {hissedilen}°C | Nem: %{nem}

💡 GÜNÜN SÖZÜ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"{soz}" — {yazar}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
İyi günler! ☀️  •  Hermes Agent 🇹🇷
```

---

## Error Handling / Hata Yönetimi

- If news fetch fails → still send brief with "Haberler alınamadı" note, include markets
- If market data fails → still send brief with "Piyasa verileri alınamadı", include news
- If weather fails → skip weather section, don't block the brief
- If ALL sections fail → send "Brifing bugün hazırlanamadı, lütfen manuel kontrol edin"
- Log all errors for debugging
- Retry each section up to 2 times before giving up

---

## Cronjob Management / Zamanlanmış Görev Yönetimi

```
# List active briefs
list_cronjobs()  →  look for name="turkish-daily-brief"

# Pause/resume
remove_cronjob(job_id)  →  removes the schedule
# Re-schedule with same or modified prompt

# One-shot test
schedule_cronjob(
    prompt=BRIEF_PROMPT,
    schedule="1m",           # Run once in 1 minute (test)
    deliver="telegram",
    name="turkish-daily-brief-test"
)
```

---

## Pitfalls / Dikkat Edilecekler

1. **Self-contained prompt:** Cronjob agent has ZERO memory — put EVERYTHING in the prompt
2. **Timezone:** Always convert Istanbul time to UTC for cron (Istanbul = UTC+3, year-round)
3. **Market hours:** Morning brief at 08:00 means markets haven't opened yet (10:00)
   → Show previous day's close with note, or schedule a market update at 10:30
4. **RSS reliability:** Some feeds go down — always have fallback sources
5. **Message length:** Telegram messages have 4096 char limit — keep brief concise
6. **Rate limits:** Don't schedule more than 1 brief per hour to avoid source blocking
7. **Weekend briefs:** Markets are closed Sat/Sun — adjust market section accordingly
8. **Religious holidays:** Ramazan/Kurban dates shift yearly — check dynamically
