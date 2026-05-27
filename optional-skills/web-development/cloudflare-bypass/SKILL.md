---
name: cloudflare-bypass
description: Cloudflare bypass aracı — JS challenge'ını çözer, sayfayı temiz markdown olarak döndürür. Sıfır maliyet, sıfır API key.
version: 1.0.0
author: akifcankilic
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cloudflare, bypass, web-scraping, markdown, scraper]
    related_skills: [scrapling, web-extract]
prerequisites:
  commands: [python3]
  pip: [cloudscraper, html2text]
---

# Cloudflare Bypass

**TL;DR:** `pip install cloudscraper html2text` + `python3 cfget.py <url>` = Cloudflare'ı geç, temiz markdown al. 0₺, 0 API key, 0 Chromium.

## Ne işe yarar?

Cloudflare'in "Just a moment..." / "Checking your browser" koruması olan sitelerden içerik almanı sağlar. curl, wget, hatta headless Chromium'un geçemediği yerden cloudscraper geçer.

## Nasıl çalışır?

1. **cloudscraper** → Cloudflare JS challenge'ını çözer, sayfayı HTTP'den alır
2. **html2text** → Ham HTML'i temiz markdown'a çevirir
3. Çıktı direkt terminale basılır — pipe'la devam edebilirsin

> **Önemli:** cloudscraper JS çalıştırmaz. Sadece Cloudflare challenge'ını çözer, sayfanın server'dan gönderdiği HTML'i alır. Eğer site React/SPA gibi client-side render yapıyorsa (9gag gibi), içerik gelmez.

## Kurulum

```bash
pip install cloudscraper html2text
```

Eğer sisteminde PEP 668 koruması varsa (`--break-system-packages` gerekebilir):

```bash
pip install --break-system-packages cloudscraper html2text
```

## Kullanım

```bash
# Sayfayı al, markdown gör
python3 cfget.py https://eksisozluk.com/debe

# İlk 20 satır
python3 cfget.py https://example.com | head -20

# Dosyaya kaydet
python3 cfget.py https://example.com > sayfa.md
```

Script'in nerede durduğu önemli değil — tek dosya, bağımlılığı yok. İstersen `~/cfget.py`'ye koy, istersen `/usr/local/bin/cfget` yap.

## Sınırlamalar

- **Client-side render (SPA) sitelerde çalışmaz** — React, Angular, Vue ile yüklenen sayfalarda boş gelir
- **Login gereken yerlerde** form tabanlı giriş dener ama OAuth/CAPTCHA/2FA'yı geçemez
- **Rate limit** yok — peş peşe çok request atarsan IP ban yiyebilirsin, saygılı ol

## Test Edilen Siteler

| Site | Durum | Not |
|------|-------|-----|
| eksisozluk.com | ✅ | Server render, tüm içerik gelir |
| webtekno.com | ✅ | Teknoloji haberleri |
| donanimhaber.com | ✅ | Teknoloji haberleri |
| cnnturk.com | ✅ | Haber sitesi |
| fanatik.com.tr | ✅ | Spor haberleri |
| itch.io | ✅ | Oyun marketplace |
| 9gag.com | ❌ | SPA, JS render |

## cfget.py

Script'in kendisi tek dosya, ~25 satır:

```python
import sys, re, cloudscraper, html2text

url = sys.argv[1]
scraper = cloudscraper.create_scraper()
r = scraper.get(url, timeout=30)
r.raise_for_status()

converter = html2text.HTML2Text()
converter.ignore_images = True
converter.body_width = 0
converter.unicode_snob = True

md = converter.handle(r.text)
md = re.sub(r'\n{3,}', '\n\n', md)
print(md.strip())
```

---

*Vibe coding ile yapıldı. Sorun olursa PR aç, bakarız.*
