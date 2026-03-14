# Whisper Language Support

99 languages grouped by transcription quality tier.

## Top-tier support (WER < 10%)

English (en), Spanish (es), French (fr), German (de), Italian (it),
Portuguese (pt), Dutch (nl), Polish (pl), Russian (ru), Japanese (ja),
Korean (ko), Chinese (zh)

## Good support (WER 10–20%)

Arabic (ar), Turkish (tr), Vietnamese (vi), Swedish (sv), Finnish (fi),
Czech (cs), Romanian (ro), Hungarian (hu), Danish (da), Norwegian (no),
Thai (th), Hebrew (he), Greek (el), Indonesian (id), Malay (ms)

## All 99 languages

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani,
Bashkir, Basque, Belarusian, Bengali, Bosnian, Breton, Bulgarian, Burmese,
Cantonese, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English,
Estonian, Faroese, Finnish, French, Galician, Georgian, German, Greek,
Gujarati, Haitian Creole, Hausa, Hawaiian, Hebrew, Hindi, Hungarian,
Icelandic, Indonesian, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer,
Korean, Lao, Latin, Latvian, Lingala, Lithuanian, Luxembourgish, Macedonian,
Malagasy, Malay, Malayalam, Maltese, Maori, Marathi, Moldavian, Mongolian,
Myanmar, Nepali, Norwegian, Nynorsk, Occitan, Pashto, Persian, Polish,
Portuguese, Punjabi, Pushto, Romanian, Russian, Sanskrit, Serbian, Shona,
Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili,
Swedish, Tagalog, Tajik, Tamil, Tatar, Telugu, Thai, Tibetan, Turkish,
Turkmen, Ukrainian, Urdu, Uzbek, Vietnamese, Welsh, Yiddish, Yoruba

## Model recommendations by tier

| Tier | Recommended model | Expected WER |
|------|-------------------|--------------|
| Top-tier (en, es, fr, de, …) | base or turbo | < 10% |
| Good (ar, tr, vi, …) | medium or large | 10–20% |
| Lower-resource | large | 20–30% |

## Tips

- **Specify the language** with `--language CODE` — skips auto-detection and
  improves both speed and accuracy.
- **Use an initial prompt** in the target language to guide recognition of
  domain-specific terms or proper nouns.
- **Use a larger model** for lower-resource languages where quality matters.
- Language codes follow ISO 639-1 (two-letter codes).
