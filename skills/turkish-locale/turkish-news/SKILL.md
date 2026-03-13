---
name: turkish-news
version: 1.0.0
description: "Türk haber kaynaklarından (Hürriyet, Sabah, Bloomberg HT, CoinTelegraph TR, NTV, AA) gerçek zamanlı haber toplama ve özetleme. / Real-time Turkish news aggregation and summarization from major Turkish media sources."
tags: [turkish, news, rss, hurriyet, sabah, bloomberg-ht, cointelegraph, ntv, aa]
author: Hermes Agent Hackathon 2026
triggers:
  - user asks for Turkish news
  - "türkiye haberleri", "gündem", "son dakika", "manşetler"
  - "bugünün haberleri", "haber özetle"
  - "Bloomberg HT", "kripto haberleri"
---

# 📰 Turkish News / Türk Haberleri

> Aggregate, filter, and summarize news from Turkey's top media sources.
> Türkiye'nin önde gelen medya kaynaklarından haberleri topla, filtrele ve özetle.

---

## When To Use / Ne Zaman Kullanılır

- User asks for Turkish news in any form
- User mentions specific Turkish news sources
- User wants news about Turkey, Turkish economy, Turkish politics
- User asks "gündem ne?", "son dakika var mı?", "bugün ne oldu?"
- Part of the daily brief pipeline (called by turkish-daily-brief)

---

## Sources & RSS Feeds / Kaynaklar ve RSS Beslemeleri

### Primary Sources / Ana Kaynaklar

#### 1. Hürriyet — Genel Haber
```
Main:      https://www.hurriyet.com.tr/rss/anasayfa
Gündem:    https://www.hurriyet.com.tr/rss/gundem
Ekonomi:   https://www.hurriyet.com.tr/rss/ekonomi
Dünya:     https://www.hurriyet.com.tr/rss/dunya
Teknoloji: https://www.hurriyet.com.tr/rss/teknoloji
Spor:      https://www.hurriyet.com.tr/rss/spor
```

#### 2. Sabah — Genel Haber
```
Main:      https://www.sabah.com.tr/rss/anasayfa.xml
Gündem:    https://www.sabah.com.tr/rss/gundem.xml
Ekonomi:   https://www.sabah.com.tr/rss/ekonomi.xml
Dünya:     https://www.sabah.com.tr/rss/dunya.xml
Teknoloji: https://www.sabah.com.tr/rss/teknoloji.xml
```

#### 3. Bloomberg HT — Finans & Ekonomi
```
Main:      https://www.bloomberght.com/rss
Piyasa:    https://www.bloomberght.com/rss/piyasa
Haberler:  https://www.bloomberght.com/rss/haberler
```

#### 4. CoinTelegraph TR — Kripto Para
```
Main:      https://tr.cointelegraph.com/rss
```

#### 5. NTV — Genel Haber
```
Main:      https://www.ntv.com.tr/son-dakika.rss
Gündem:    https://www.ntv.com.tr/turkiye.rss
Ekonomi:   https://www.ntv.com.tr/ekonomi.rss
Dünya:     https://www.ntv.com.tr/dunya.rss
Teknoloji: https://www.ntv.com.tr/teknoloji.rss
```

#### 6. Anadolu Ajansı (AA) — Haber Ajansı
```
Main:      https://www.aa.com.tr/tr/rss/default?cat=guncel
Ekonomi:   https://www.aa.com.tr/tr/rss/default?cat=ekonomi
Dünya:     https://www.aa.com.tr/tr/rss/default?cat=dunya
Spor:      https://www.aa.com.tr/tr/rss/default?cat=spor
```

### Supplementary Sources / Ek Kaynaklar

#### 7. TRT Haber — Kamu Yayıncılığı
```
Main:      https://www.trthaber.com/sondakika.rss
```

#### 8. Dünya Gazetesi — Ekonomi Odaklı
```
Main:      https://www.dunya.com/rss
```

---

## Procedures / İşlem Adımları

### Procedure 1: Quick News Summary / Hızlı Haber Özeti

**Trigger:** "Gündem ne?", "Haberleri özetle", "Bugün ne oldu?"

Steps:
1. Fetch RSS feeds from top 3 sources (Hürriyet, NTV, Bloomberg HT)
2. Parse XML to extract title, link, pubDate, description
3. Deduplicate by similarity (same story from multiple sources)
4. Sort by recency (newest first)
5. Take top 10 unique stories
6. Summarize each in 1-2 sentences in Turkish
7. Format as numbered list with source attribution

**Implementation:**

```python
from hermes_tools import web_extract, terminal
import json

# Fetch RSS feeds via curl (faster than web_extract for XML)
sources = {
    "Hürriyet": "https://www.hurriyet.com.tr/rss/anasayfa",
    "NTV": "https://www.ntv.com.tr/son-dakika.rss",
    "Bloomberg HT": "https://www.bloomberght.com/rss",
}

all_items = []
for name, url in sources.items():
    result = terminal(f"curl -s -m 10 '{url}' | python3 -c \"\nimport sys, xml.etree.ElementTree as ET\ntry:\n    tree = ET.parse(sys.stdin)\n    for item in tree.findall('.//item')[:10]:\n        title = item.findtext('title', '')\n        link = item.findtext('link', '')\n        desc = item.findtext('description', '')\n        date = item.findtext('pubDate', '')\n        print(f'{{\\\"title\\\": \\\"{title}\\\", \\\"link\\\": \\\"{link}\\\", \\\"source\\\": \\\"{name}\\\"}}')\nexcept: pass\n\"")
    # Parse results...

# Present to user as formatted Turkish summary
```

**Output Format:**

```
📰 GÜNÜN MANŞETLERİ — 11 Mart 2026

1. [Başlık] — kaynak
   Kısa özet...

2. [Başlık] — kaynak
   Kısa özet...

...

🕐 Son güncelleme: 08:30
```

### Procedure 2: Category-Specific News / Kategoriye Özel Haberler

**Trigger:** "Ekonomi haberleri", "Spor haberleri", "Kripto haberleri", "Dünya haberleri"

Steps:
1. Detect category from user input:
   - ekonomi/finans → Ekonomi RSS feeds
   - spor → Spor RSS feeds
   - kripto/bitcoin → CoinTelegraph TR
   - dünya/uluslararası → Dünya RSS feeds
   - teknoloji → Teknoloji RSS feeds
2. Fetch category-specific RSS from 2-3 sources
3. Parse, deduplicate, sort by recency
4. Take top 8 stories
5. Format with category header

**Category Mapping:**
```
ekonomi  → hurriyet/ekonomi + bloomberght/piyasa + sabah/ekonomi
spor     → hurriyet/spor + ntv/spor + aa/spor
kripto   → tr.cointelegraph.com/rss
dünya    → hurriyet/dunya + ntv/dunya + aa/dunya
teknoloji → hurriyet/teknoloji + ntv/teknoloji + sabah/teknoloji
gündem   → hurriyet/gundem + ntv/turkiye + aa/guncel
```

### Procedure 3: Deep Dive on a Story / Haberi Derinlemesine İncele

**Trigger:** User asks for details on a specific news story

Steps:
1. Use web_extract on the article URL
2. Extract full article text
3. Summarize key points in Turkish
4. Provide context if needed (related events, background)
5. Offer to find related stories from other sources

### Procedure 4: Source-Specific Fetch / Kaynağa Özel Çekme

**Trigger:** "Bloomberg HT'den haberleri getir", "Hürriyet manşetleri"

Steps:
1. Identify the requested source
2. Fetch all available RSS categories from that source
3. Parse and present with source branding
4. Use source-appropriate formatting (e.g., Bloomberg HT gets financial emphasis)

---

## RSS Parsing Helper / RSS Ayrıştırma Yardımcısı

Standard approach for all RSS fetches:

```python
import xml.etree.ElementTree as ET
from datetime import datetime

def parse_rss(xml_content, source_name, max_items=10):
    """Parse RSS XML and return structured items."""
    items = []
    try:
        root = ET.fromstring(xml_content)
        for item in root.findall('.//item')[:max_items]:
            items.append({
                'title': (item.findtext('title') or '').strip(),
                'link': (item.findtext('link') or '').strip(),
                'description': (item.findtext('description') or '').strip(),
                'pub_date': (item.findtext('pubDate') or '').strip(),
                'source': source_name,
            })
    except ET.ParseError:
        pass
    return items

def deduplicate(items, threshold=0.6):
    """Remove duplicate stories across sources by title similarity."""
    seen = []
    unique = []
    for item in items:
        title_words = set(item['title'].lower().split())
        is_dupe = False
        for seen_words in seen:
            overlap = len(title_words & seen_words) / max(len(title_words | seen_words), 1)
            if overlap > threshold:
                is_dupe = True
                break
        if not is_dupe:
            seen.append(title_words)
            unique.append(item)
    return unique
```

---

## Output Templates / Çıktı Şablonları

### Compact Format (for Telegram / CLI):
```
📰 TÜRKİYE GÜNDEMİ — {tarih}

1. {başlık} [{kaynak}]
2. {başlık} [{kaynak}]
3. {başlık} [{kaynak}]
...

🔗 Detay için numara söyle
```

### Detailed Format (for full response):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📰 {KATEGORI} HABERLERİ
  {tarih} • {saat}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▸ {başlık}
  {kaynak} • {zaman}
  {özet}
  🔗 {link}

▸ {başlık}
  {kaynak} • {zaman}
  {özet}
  🔗 {link}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Toplam {n} haber • Son güncelleme: {saat}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Error Handling / Hata Yönetimi

- If RSS feed is unreachable: skip source, note in output, try alternatives
- If all feeds fail: fall back to web_extract on source homepages
- If content is paywalled: extract available preview text, note "tam içerik ücretli"
- Rate limiting: wait 1s between requests, max 3 concurrent fetches
- Timeout: 10s per RSS feed, 15s per web_extract

---

## Pitfalls / Dikkat Edilecekler

1. **Encoding:** Turkish chars (ç, ğ, ı, İ, ö, ş, ü) must be preserved — always use UTF-8
2. **RSS availability:** Some sources may change/break RSS URLs — fall back to web scraping
3. **Deduplication:** Turkish news outlets often run same AA wire stories — always deduplicate
4. **Time zones:** All Turkish sources use UTC+3 (Europe/Istanbul) — normalize timestamps
5. **Content length:** RSS descriptions may be truncated — use web_extract for full articles
6. **Political sensitivity:** Present news factually without editorial commentary
