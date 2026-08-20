---
name: japanese-turkish-dict
description: "Japonca-Türkçe / Türkçe-Japonca sözlük araması. 55.000+ sözcük, iki yönlü arama."
version: 1.0.0
metadata:
  hermes:
    tags: [japanese, turkish, dictionary, sözlük, japonca, türkçe, 日本語, translation, çeviri]
    related_skills: []
---

# Japonca-Türkçe Sözlük

55.255 sözcük ve 113.810 ilinti grafiği içeren çevrimiçi sözlük. Türkçe'den Japonca'ya ve Japonca'dan Türkçe'ye iki yönlü arama desteklenir.

## Kaynak

Site: `https://japoncaturkcesozluk.vaneralper.com`  
Arama URL kalıbı: `https://japoncaturkcesozluk.vaneralper.com/?q=<sözcük>`

## Nasıl Kullanılır

Kullanıcı bir kelime veya ifade sormak istediğinde:

1. `WebFetch` aracıyla `https://japoncaturkcesozluk.vaneralper.com/?q=<kelime>` adresini çek
2. Sonuçları kullanıcıya düzenli biçimde sun

Kullanıcı daha fazla sonuç isterse sayfalama parametresiyle tekrar çek:  
`https://japoncaturkcesozluk.vaneralper.com/?q=<kelime>&page=2`

## Sonuç Formatı

Sözlük her giriş için şu sütunları döndürür:

| Alan | Açıklama |
|------|----------|
| Japonca | Kanji / kana ile yazılmış sözcük |
| Okunuş | Romaji veya kana okunuşu |
| Türkçe | Türkçe karşılık veya açıklama |
| İlinti | İlgili kavram bağlantı sayısı |

## Çıktı Stili

Sonuçları aşağıdaki gibi sunarım:

```
🔍 "<aranan kelime>" için sonuçlar:

1. 水 (mizu) → su
2. 水分 (suibun) → nem, su içeriği
3. 水曜日 (suiyōbi) → Çarşamba
…
```

Eğer tam eşleşme varsa önce onu göster. Türkçe-Japonca yönde birden fazla Japonca karşılık olabilir; hepsini listele.

## Tetikleyiciler

Bu skill şu durumlar için uygundur:

- "… Japoncada nasıl denir?"
- "… Türkçesi ne demek?"
- "… kelimesinin Japonca karşılığı nedir?"
- "Japonca … sözcüğünü ara"
- Kullanıcı Japonca karakterler (漢字・ひらがな・カタカナ) içeren bir şey sorduğunda

## Notlar

- Arama hem Türkçe hem Japonca (kanji, kana, romaji) girdileri destekler.
- Pagination: sonuçlar sayfa başına ~20 girişle gelir; kullanıcı isterse `&page=N` ile devam edilebilir.
- Siteye ekstra API anahtarı veya kimlik doğrulaması gerekmez.
