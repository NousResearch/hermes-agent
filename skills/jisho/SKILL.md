---
name: jisho
description: "Jisho.org üzerinden Japonca kelime arama, kanji bilgisi ve İngilizce çeviri. JMdict tabanlı, JLPT seviyesi ve gramer etiketli sonuçlar."
version: 1.0.0
metadata:
  hermes:
    tags: [japanese, dictionary, kanji, english, jisho, 日本語, translation, reading, furigana, jlpt]
    related_skills: [japanese-turkish-dict, jlpt-study]
---

# Jisho.org Sözlük Skill

[Jisho.org](https://jisho.org) Japonca öğrenenler için en kapsamlı çevrimiçi sözlüklerden biridir. JMdict ve JMnedict veritabanlarını kullanır. Bu skill üç ana arama modunu destekler: kelime arama, kanji bilgisi ve İngilizce'den Japonca'ya ters arama.

## API Endpointleri

```
# Kelime / ifade arama (Japonca, romaji veya İngilizce)
GET https://jisho.org/api/v1/search/words?keyword=<sorgu>

# Kanji detayı
GET https://jisho.org/api/v1/search/words?keyword=%23kanji%20<kanji>

# Özel etiketli arama örnekleri
GET https://jisho.org/api/v1/search/words?keyword=%23common%20<sorgu>     # sadece yaygın kelimeler
GET https://jisho.org/api/v1/search/words?keyword=%23jlpt-n5%20<sorgu>   # JLPT seviyesine göre filtre
```

API kimlik doğrulaması gerektirmez. Yanıt her zaman JSON formatındadır.

## Yanıt Yapısı

### Kelime Arama (`data[]`)

```json
{
  "slug": "水",
  "is_common": true,
  "tags": ["wanikani1"],
  "jlpt": ["jlpt-n5"],
  "japanese": [
    { "word": "水", "reading": "みず" },
    { "word": "お水", "reading": "おみず" }
  ],
  "senses": [
    {
      "english_definitions": ["water"],
      "parts_of_speech": ["Noun"],
      "tags": [],
      "info": [],
      "restrictions": [],
      "see_also": [],
      "antonyms": []
    }
  ],
  "attribution": { "jmdict": true, "jmnedict": false, "dbpedia": false }
}
```

Önemli alanlar:
| Alan | Açıklama |
|------|----------|
| `japanese[].word` | Kanji ile yazılış |
| `japanese[].reading` | Kana okunuşu |
| `is_common` | Günlük kullanımda yaygın mı |
| `jlpt[]` | JLPT seviyesi (`jlpt-n5` … `jlpt-n1`) |
| `senses[].english_definitions` | İngilizce anlamlar |
| `senses[].parts_of_speech` | Gramer kategorisi (Noun, Verb, Adjective…) |
| `senses[].info` | Ek not (argo, resmi, lehçe vb.) |
| `senses[].see_also` | İlgili kelimeler |

## Çalışma Modları

### 1. Kelime Arama

Kullanıcı bir Japonca kelime, kana, romaji veya İngilizce kelime sorduğunda:

1. `WebFetch` ile `https://jisho.org/api/v1/search/words?keyword=<sorgu>` çek
2. `data[]` dizisinden ilk 5 sonucu al
3. Aşağıdaki formatta sun

**Çıktı formatı:**
```
📖 "<sorgu>" — Jisho.org sonuçları

1. 水【みず】(Noun) ★ yaygın · N5
   → water
   → Bkz: お湯 (sıcak su)

2. 水【すい】(Prefix)
   → water (prefix in compounds)
```

Etiket açıklamaları:
- `★ yaygın` → `is_common: true`
- `N5`…`N1` → JLPT seviyesi (`jlpt[0]`'dan `jlpt-n5` → `N5`)
- Birden fazla okunuş varsa hepsini göster
- `info` alanı dolu ise parantez içinde belirt (örn. `(argo)`, `(resmi)`)

### 2. Kanji Bilgisi

Kullanıcı tek bir kanji hakkında detay istediğinde (`#kanji` etiketiyle arama):

1. `WebFetch` ile `https://jisho.org/api/v1/search/words?keyword=%23kanji%20<kanji>` çek
2. API, kanji için `data[]` içinde standart kelime girişlerini döndürür
3. Yanıt boş gelirse kendi bilginle temel okuyuş ve anlam bilgisini ver

**Çıktı formatı:**
```
🈶 水 (mizu / sui)

Anlamlar: water
On-yomi: スイ (sui)
Kun-yomi: みず (mizu)
JLPT: N5
Yaygın: evet

Örnek kelimeler:
  水分 (すいぶん) → moisture, water content
  水曜日 (すいようび) → Wednesday
  水泳 (すいえい) → swimming
```

Kanji API yanıtı kelime girişleri şeklinde geldiği için örnek kelimeleri listedeki `japanese[].word` + `japanese[].reading` + `senses[].english_definitions` üçlüsünden çıkar.

### 3. İngilizce → Japonca (Ters Arama)

İngilizce kelime girildiğinde API otomatik olarak İngilizce anlamlardan arama yapar; ek parametre gerekmez.

1. `WebFetch` ile `https://jisho.org/api/v1/search/words?keyword=water` gibi çek
2. Sonuçları Japonca→İngilizce formatında sun (mod 1 ile aynı format)
3. Birden fazla Japonca karşılık olabilir; hepsini listele

### 4. Etiket Filtreli Arama

Kullanıcı seviye veya yaygınlık filtresi istediğinde `%23` (URL-encoded `#`) ile etiket ekle:

| Kullanıcı isteği | Keyword |
|-----------------|---------|
| "sadece yaygın kelimeler" | `%23common%20<sorgu>` |
| "N5 kelimeleri listele" | `%23jlpt-n5%20<sorgu>` |
| "N3 kelimeleri" | `%23jlpt-n3%20<sorgu>` |
| "fiil olarak ara" | `%23verb%20<sorgu>` |
| "sıfat ara" | `%23adjective%20<sorgu>` |

## Sayfalama

Jisho API sayfalama parametresi sunmaz; tüm sonuçlar tek yanıtta gelir. Yanıtta genellikle 20+ giriş olabilir. Varsayılan olarak ilk 5 sonucu göster; kullanıcı "daha fazla" dediğinde sonraki 5'i sun.

## Tetikleyiciler

Bu skill şu durumlarda devreye girer:

- "Jisho'da … ara"
- "… Japoncası ne?"
- "… kanji'sini açıkla / hakkında bilgi ver"
- "… kelimesinin okunuşu nedir?"
- "[Japonca/romaji kelime] ne demek?" — `japanese-turkish-dict` yanıt veremezse bu skill'e düş
- Kullanıcı JLPT seviyesiyle birlikte kelime aradığında ("N4 kelimeleri arasında ara")
- Gramer kategorisi belirtilerek arama ("Japonca fiilleri içinde ara")

## Notlar

- API yanıtı `meta.status` ≠ 200 ise hata mesajı göster ve kullanıcıyı https://jisho.org adresine yönlendir.
- `data` dizisi boş gelirse: "Sonuç bulunamadı. Farklı bir yazım veya romaji deneyin" öner.
- Türkçe karşılık istendiğinde `japanese-turkish-dict` skill'i veya kendi bilgini kullan; Jisho yalnızca İngilizce tanım sağlar.
- Romaji desteği: API romaji girişi kabul eder (ör. `keyword=mizu`); kullanıcı romaji girerse doğrudan ilet.
- `see_also` dolu ise "Ayrıca bkz." satırı ekle.
- Birden fazla sense (anlam grubu) varsa numaralandırarak listele.
