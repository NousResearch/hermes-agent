---
name: jlpt-study
description: "JLPT N1-N5 seviyelerinde Japonca kelime, gramer ve kanji çalışması. Kelime listesi, gramer açıklamaları ve quiz modları."
version: 1.0.0
metadata:
  hermes:
    tags: [japanese, jlpt, vocabulary, grammar, kanji, japonca, study, quiz, N5, N4, N3, N2, N1]
    related_skills: [japanese-turkish-dict]
---

# JLPT Çalışma Asistanı

JLPT (Japanese Language Proficiency Test) N1–N5 seviyeleri için kelime, gramer ve kanji materyali sunar. Birincil kaynak `jlptstudy.net`; N1 ve N3 için Claude'un dahili JLPT bilgisi kullanılır.

## Seviyeler ve Kaynak Durumu

| Seviye | Kelime Listesi | Gramer Listesi | Kanji Listesi | Kaynak |
|--------|---------------|----------------|---------------|--------|
| N5 | ✓ ~700 kelime | ✓ ~125 nokta | ✓ ~100 kanji | jlptstudy.net |
| N4 | ✓ ~1500 kelime | ✓ mevcut | — | jlptstudy.net |
| N3 | — | — | — | Claude bilgisi |
| N2 | ✓ ~2280 kelime | — | ✓ mevcut | jlptstudy.net |
| N1 | — | — | — | Claude bilgisi |

## URL Kalıpları

```
# Kelime listeleri (HTML, doğrudan fetchlenebilir)
http://jlptstudy.net/N5/lists/n5_vocab-list.html
http://jlptstudy.net/N4/lists/n4_vocab-list.html
http://jlptstudy.net/N2/lists/n2_vocab-list.html

# Gramer listeleri
http://jlptstudy.net/N5/lists/n5_grammar-list.html
http://jlptstudy.net/N4/lists/n4_grammar-list.html

# Kanji listeleri
http://jlptstudy.net/N5/lists/n5_kanji-list.html
http://jlptstudy.net/N2/lists/n2_kanji-list.html
```

## Çalışma Modları

### 1. Kelime Listesi Görüntüleme

Kullanıcı bir seviye için kelime listesi istediğinde:

1. İlgili `vocab-list.html` URL'ini `WebFetch` ile çek
2. `prompt`: "Extract vocabulary entries. Return as table: number | kanji | reading | meaning. First [N] entries only."
3. Türkçe çeviri istenirse `japanese-turkish-dict` skill'i veya kendi bilginle destekle

**Çıktı formatı:**
```
📚 JLPT N5 Kelime Listesi (1–20)

#  | Kelime      | Okunuş    | Anlam
---|-------------|-----------|------------------------------
0  | ああ        | ああ      | "Ah!, Oh!"
1  | 会う        | あう      | buluşmak, görüşmek
2  | 青          | あお      | mavi
...

Devam için: "sonraki 20" veya "#40'tan itibaren göster"
```

Sayfalama için her seferinde 20–30 kelime sun; kullanıcı "devam" veya sayı belirtirse sonraki bloğu getir.

### 2. Gramer Çalışma

Kullanıcı gramer listesi istediğinde:

1. İlgili `grammar-list.html` URL'ini `WebFetch` ile çek
2. `prompt`: "Extract grammar points with pattern, meaning, and example sentence."
3. Her gramer noktasını açıkla; örneğin:
   - Yapı: `V+ます`
   - Anlam: fiil kibar biçimi
   - Örnek: `私は毎日本を読みます。` → "Her gün kitap okurum."

**N3 ve N1 için:** jlptstudy.net'te bu seviyeler mevcut değil. Kendi bilginle JLPT gramer listesi sun (örneğin N3 için: ～ために、～ように、～ばかり、～だけでなく, vb.).

### 3. Kanji Çalışma

1. İlgili `kanji-list.html` URL'ini `WebFetch` ile çek
2. Her kanji için: karakter, on-yomi, kun-yomi, Türkçe anlam, örnek kelimeler
3. Site sadece karakter listesi verir; anlam ve okuyuşları kendi bilginle ekle

**Çıktı formatı:**
```
🈶 JLPT N5 Kanji

一 (いち/ひと-) → bir; örnek: 一つ (ひとつ), 一日 (いちにち)
二 (に/ふた-) → iki; örnek: 二つ (ふたつ), 二月 (にがつ)
...
```

### 4. Kelime Quiz Modu

Kullanıcı quiz istediğinde:

1. Kelime listesini çek
2. Rastgele 10 kelime seç (ya da kullanıcının belirttiği sayıda)
3. Her soru için iki mod:
   - **JA→TR**: Japonca göster, Türkçe/İngilizce sor
   - **TR→JA**: Türkçe/İngilizce göster, Japonca sor
4. Cevabı al, doğru/yanlış bildir, doğruyu göster

**Quiz akışı:**
```
🎯 JLPT N5 Kelime Quiz (10 soru)

Soru 1/10:
  "明るい" ne anlama gelir?

> [kullanıcı cevaplar]

✅ Doğru! "明るい" (あかるい) = parlak, neşeli

Soru 2/10: ...
```

Hepsi bitince skor göster: `8/10 — Harika!`

### 5. Gramer Quiz Modu

1. Gramer listesini çek
2. Rastgele seçilen örnek cümleden boşluk doldurma soru üret
3. Örnek:

```
🎯 JLPT N5 Gramer Quiz

"私は毎日本を___ます。" → boşluğa ne gelir?
(İpucu: okuma eylemi)

> 読み

✅ Doğru! 読み+ます → 読みます (V+ます kibar form)
```

### 6. Günlük Çalışma (Daily Study)

Kullanıcı "bugünlük çalışma" veya "günlük X kelime" dediğinde:
- Listeden X adet kelime seç (varsayılan: 15)
- Önce kelime listesini sun (okuma+anlam)
- Ardından mini quiz yap (seçilen kelimelerden 5 soru)

## N1 ve N3 İçin Claude Bilgisi

jlptstudy.net N1 ve N3 içermez. Bu seviyeler için:

- **N3 kelime** (~3750 toplam JLPT kelimesinden N3 kısmı): Kendi bilginle üret; sık geçen N3 kelimeleri örnekle sun
- **N3 gramer** (~170 nokta): ～ために、～ように、～ばかり、～だけでなく、～ながら、～てしまう vb.
- **N1 kelime** (~10000 toplam JLPT içinden N1 kısmı): Kendi bilginle üret
- **N1 gramer** (~100+ ileri nokta): ～に際して、～を踏まえて、～いかんによらず vb.

Bu seviyeler için kullanıcıya kaynağın Claude bilgisi olduğunu belirt.

## Tetikleyiciler

Bu skill şu durumlarda devreye girer:

- "JLPT [seviye] [kelime/gramer/kanji] çalış / listele / göster"
- "N5 kelime listesi", "N4 gramer", "N2 kanji"
- "JLPT quiz", "kelime quiz N3"
- "Japonca sınava hazırlanıyorum"
- "[N5/N4/N3/N2/N1] sınava hazırlık"
- "JLPT [seviye] günlük çalışma"
- "今日のJLPT勉強" (bugünkü JLPT çalışması)

## Notlar

- Site 2010 öncesi sınav listelerini temel alır; resmi liste o tarihten beri yayınlanmamıştır, ancak içerik hâlâ geçerlidir.
- N3 ve N1 için `japanese-turkish-dict` skill'iyle çapraz kontrol önerilebilir.
- Uzun listeler için sayfalama yap; kullanıcıyı bilgi yüküyle bunaltma.
- Quiz soru sayısı varsayılan 10, kullanıcı değiştirebilir.
- Türkçe karşılıklar için kendi bilgini kullan; sözlük önerisi olarak `japanese-turkish-dict` skill'ini öner.
