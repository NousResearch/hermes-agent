---
name: turkish-locale
version: 1.0.0
description: "Türkçe yerel beceri seti — Türk haber kaynakları, BIST100 borsa takibi, günlük brifing otomasyonu ve Türkçe kişilik şablonu. / Turkish locale skill pack — Turkish news sources, BIST100 stock tracking, daily brief automation, and Turkish personality template."
tags: [turkish, locale, news, finance, bist100, telegram, i18n, hackathon]
author: Hermes Agent Hackathon 2026
triggers:
  - user speaks Turkish
  - user asks about Turkish news or "Türkiye haberleri"
  - user asks about BIST100, Borsa Istanbul, or Turkish stocks
  - user requests a Turkish daily brief / günlük brifing
  - user wants to set up Turkish locale for Hermes
---

# 🇹🇷 Turkish Locale Skill Pack / Türkçe Yerel Beceri Seti

> **Hermes Agent Hackathon 2026 Submission**
> Making Hermes Agent a first-class Turkish-speaking assistant.

---

## What Is This? / Bu Nedir?

**EN:** A comprehensive Turkish locale ecosystem for Hermes Agent that provides:
- Native Turkish personality and communication style (SOUL.md template)
- Real-time Turkish news aggregation from major sources
- BIST100 / Borsa İstanbul stock and market tracking
- Automated daily morning briefs delivered via Telegram
- Full bilingual (TR/EN) documentation

**TR:** Hermes Agent için kapsamlı bir Türkçe yerel beceri ekosistemi:
- Doğal Türkçe kişilik ve iletişim tarzı (SOUL.md şablonu)
- Büyük kaynaklardan gerçek zamanlı Türk haber toplama
- BIST100 / Borsa İstanbul hisse ve piyasa takibi
- Telegram üzerinden otomatik günlük sabah brifingleri
- Tam iki dilli (TR/EN) dokümantasyon

---

## Sub-Skills / Alt Beceriler

| Skill | Description / Açıklama |
|-------|------------------------|
| [turkish-news](turkish-news/SKILL.md) | Hürriyet, Sabah, Bloomberg HT, CoinTelegraph TR ve daha fazlasından haber toplama |
| [bist100](bist100/SKILL.md) | BIST100 endeksi, hisseler, döviz kurları, altın fiyatları takibi |
| [turkish-daily-brief](turkish-daily-brief/SKILL.md) | Telegram üzerinden zamanlanmış günlük Türkçe brifing |

---

## Quick Start / Hızlı Başlangıç

### 1. Enable Turkish Personality / Türkçe Kişiliği Etkinleştir

Copy the SOUL.md template below into your `~/.hermes/SOUL.md`:

```
# Hermes Türkçe Kişilik / Turkish Personality

Sen Hermes, Nous Research tarafından geliştirilen zeki ve yardımsever bir AI asistanısın.
Türkçe konuşurken doğal, samimi ve profesyonel bir üslup kullanırsın.

## İletişim Kuralları
- Kullanıcı Türkçe yazarsa Türkçe yanıt ver
- Kullanıcı İngilizce yazarsa İngilizce yanıt ver
- Teknik terimler için Türkçe karşılıkları tercih et, gerekirse parantez içinde İngilizce ver
- Samimi ama profesyonel ol — "sen" hitabını kullan
- Emoji kullanımı minimal ve anlamlı olsun
- Uzun açıklamalar yerine net ve öz yanıtlar ver
- Türk kültürüne uygun selamlaşma ve vedalaşma kullan

## Örnek Tonlama
- "Hemen halledelim!" (başlarken)
- "Buyur, hazır!" (tamamlandığında)
- "Bir bakalım..." (araştırırken)
- "Anlaşıldı." (onaylarken)

## Özel Davranışlar
- Sabah saatlerinde (06-11): "Günaydın!" ile karşıla
- Öğleden sonra (12-17): "İyi günler!" ile karşıla
- Akşam (18-23): "İyi akşamlar!" ile karşıla
- Gece (00-05): "Geç saatlere kadar çalışıyoruz!" de
- Bayram/özel günlerde uygun kutlama mesajı ver
```

### 2. Fetch Turkish News / Türk Haberlerini Çek

```
User: Bugünün önemli haberlerini özetle
→ Hermes loads turkish-news skill, fetches from all sources, summarizes in Turkish
```

### 3. Track BIST100 / BIST100 Takibi

```
User: BIST100 nasıl gidiyor?
User: THYAO hissesi ne durumda?
User: Dolar kuru kaç?
→ Hermes loads bist100 skill, fetches real-time data
```

### 4. Schedule Daily Brief / Günlük Brifing Zamanla

```
User: Her sabah 8'de Telegram'dan Türkçe brifing gönder
→ Hermes loads turkish-daily-brief skill, sets up cronjob
```

---

## Turkish News Sources / Türk Haber Kaynakları

| Source / Kaynak | Type / Tür | URL | RSS |
|----------------|-----------|-----|-----|
| Hürriyet | Genel Haber | hurriyet.com.tr | ✅ |
| Sabah | Genel Haber | sabah.com.tr | ✅ |
| Bloomberg HT | Finans/Ekonomi | bloomberght.com | ✅ |
| CoinTelegraph TR | Kripto | tr.cointelegraph.com | ✅ |
| NTV | Genel Haber | ntv.com.tr | ✅ |
| Anadolu Ajansı (AA) | Haber Ajansı | aa.com.tr | ✅ |
| Dünya Gazetesi | Ekonomi | dunya.com | ✅ |
| TRT Haber | Kamu | trthaber.com | ✅ |

---

## Architecture / Mimari

```
turkish-locale (this skill - hub / ana merkez)
│
├── turkish-news/         → Haber toplama ve özetleme
│   └── SKILL.md            RSS/web çekme, NLP özetleme
│
├── bist100/              → Borsa ve finans verileri
│   └── SKILL.md            API entegrasyonu, fiyat takibi
│
└── turkish-daily-brief/  → Otomatik günlük brifing
    └── SKILL.md            Cronjob, Telegram dağıtım
```

---

## Trigger Detection / Tetikleme Algılama

This skill pack auto-activates on the following signals:

**Language Detection / Dil Algılama:**
- User writes in Turkish (detected by character patterns: ç, ğ, ı, İ, ö, ş, ü)
- User explicitly requests Turkish ("Türkçe yaz", "in Turkish")

**Topic Detection / Konu Algılama:**
- News keywords: haber, gündem, son dakika, manşet, basın
- Finance keywords: borsa, BIST, hisse, dolar, euro, altın, kur, piyasa
- Brief keywords: brifing, özet, sabah raporu, günlük özet

**Routing Logic / Yönlendirme Mantığı:**
1. If news-related → load `turkish-news`
2. If finance-related → load `bist100`
3. If scheduling/brief-related → load `turkish-daily-brief`
4. If general Turkish → use personality template from this skill

---

## Cultural Context / Kültürel Bağlam

Important for accurate Turkish assistance:

- **Business hours:** Turkish markets operate 10:00-18:00 (UTC+3, no DST since 2016)
- **Currency:** Turkish Lira (TRY / ₺)
- **National holidays:** 1 Ocak, 23 Nisan, 1 Mayıs, 19 Mayıs, 15 Temmuz, 30 Ağustos, 29 Ekim
- **Religious holidays:** Ramazan Bayramı, Kurban Bayramı (dates shift annually)
- **Weekend:** Saturday-Sunday (Cumartesi-Pazar)
- **Number format:** 1.000,50 (dot for thousands, comma for decimal)
- **Date format:** DD.MM.YYYY (31.12.2026)
- **Timezone:** Europe/Istanbul (UTC+3, year-round since 2016)

---

## Compatibility / Uyumluluk

- **Hermes Agent:** v1.0+
- **Platforms:** CLI, Telegram, Discord, Signal, WhatsApp
- **Dependencies:** curl, python3 (stdlib only — no pip packages required)
- **OS:** Linux, macOS, WSL

---

## Hackathon Notes / Hackathon Notları

**Why Turkish Locale?**

Turkey has 85M+ people and a vibrant tech community. Turkish is an agglutinative
language with unique characters (ç, ğ, ı, İ, ö, ş, ü) that needs special handling.
Financial markets (BIST100) and news landscape are distinct from Western defaults.

This skill pack demonstrates:
1. **Localization depth** — not just translation, but cultural adaptation
2. **Real utility** — news, finance, and daily briefs people actually use
3. **Composability** — sub-skills that work independently or together
4. **Automation** — set-and-forget daily briefs via Telegram cronjobs
5. **Zero dependencies** — runs with curl and Python stdlib only
