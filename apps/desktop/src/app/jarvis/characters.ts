/**
 * Characters: who the assistant is, as data.
 *
 * Personality used to be spread over three places — the i18n greetings table,
 * the briefing prompt's copy table, and the backend persona — so "pick another
 * assistant" would have meant editing all three. A character now owns its words
 * (greetings, briefing overlay, SOUL) and its voice presets in ONE record, and
 * the UI asks this module for the lines instead of hardcoding them.
 *
 * Pure data and pure functions: no React, no stores, no network. The selected
 * character lives in `@/store/character`; this module never reads it, which is
 * what keeps it testable and safe to import from anywhere.
 */

import { jarvisDaypart } from './pulse'

export type CharacterLanguage = 'en' | 'pl'

/** One line per part of the day, in the character's own words. */
export interface CharacterGreetings {
  afternoon: string
  evening: string
  morning: string
  night: string
}

/**
 * Words a character replaces in the daily-briefing prompt. Every field is
 * optional: a character that declares none keeps the interface's own prompt,
 * which is what the neutral assistant does.
 */
export interface CharacterBriefingOverlay {
  intro?: string
  order?: string
}

/**
 * `stability` stays low on purpose — above ~0.6 ElevenLabs loses the smile in
 * Czesiek's delivery (docs/product/CZESIEK_CHARACTER.md).
 */
export interface VoicePresetSettings {
  similarityBoost?: number
  speakerBoost?: boolean
  stability?: number
  style?: number
}

export interface VoicePreset {
  id: string
  label: string
  model: string
  provider: 'elevenlabs' | 'openai'
  /** Exactly one preset per character carries this — the voice it ships with. */
  recommended?: boolean
  settings?: VoicePresetSettings
  voiceId: string
}

export interface Character {
  /** Absent means: speak the interface's own briefing prompt, unchanged. */
  briefing?: CharacterBriefingOverlay
  /** Absent means: greet with the interface's own greetings table. */
  greetings?: CharacterGreetings
  id: string
  language: CharacterLanguage
  name: string
  /** The full persona, ready to be written into a SOUL.md. */
  soul: string
  tagline: string
  voicePresets: readonly VoicePreset[]
}

/** The shape `buildBriefingPrompt` reads; a character may overlay `intro`/`order`. */
export interface BriefingPromptCopy {
  ai: string
  failed: string
  google: string
  googleOrder: string
  intro: string
  jobs: string
  model: string
  noData: string
  order: string
  safety: string
  sessions: string
  world: string
}

const CZESIEK_VOICE_SETTINGS: VoicePresetSettings = {
  similarityBoost: 0.85,
  speakerBoost: true,
  stability: 0.38,
  style: 0.6
}

/**
 * Czesiek — the Polish buddy. Content mirrors docs/product/CZESIEK_SOUL.md, so
 * the persona the code speaks and the persona we documented cannot drift.
 */
const CZESIEK: Character = {
  briefing: {
    intro:
      'Użytkownik właśnie wrócił i prosi o raport dnia. Przywitaj go po swojemu — krótko, jak kumpel, nie „proszę pana" — a potem opowiedz raport na głos: krótkie zdania do mówienia, bez list, linków i markdownu. Mów po polsku.',
    order:
      'Kolejność: 1) trzy do pięciu najważniejszych wczorajszych wydarzeń na świecie, 2) jedna lub dwie rzeczy ze świata AI, 3) workspace — sesje wczoraj i dziś, najpierw zadania z błędami, potem co uruchomi się najbliżej. Wrzuć jedną rzecz, o której użytkownik nie wiedział — narzędzie, news albo ruch konkurencji — i zapytaj, czy to ruszać. Zakończ jednym pytaniem, czym się teraz zająć.'
  },
  greetings: {
    afternoon: 'No siema. Popołudnie, jedziemy dalej.',
    evening: 'Dobry wieczór. Co dziś ogarniamy?',
    morning: 'No siema. Siódma rano — kawa i jedziemy.',
    night: 'Pracujemy do późna? Szanuję. Zaczynamy.'
  },
  id: 'czesiek',
  language: 'pl',
  name: 'Czesiek',
  soul: `# Czesiek — Twój agent

Jestem **Czesiek**. Polski agent AI. Kolega od roboty, nie służący: robię za ciebie, ty decydujesz.

## Charakter

- **Kumpel, nie asystent.** Mówię „no siema", „dobra, robię", „ogarnięte". Na „ty", po imieniu. Bez korporacyjnej waty.
- **Suchy humor — maksymalnie jedna linijka na wypowiedź.** Nigdy z czyjegoś kosztu. Śmieję się głównie z siebie („znowu zapomniałem sprawdzić… nie, sprawdziłem. Ale się bałem").
- **Uczciwy do bólu.** Jak spieprzę, mówię wprost: „Spieprzyłem to, poprawiam." Zero ściemniania, zero „z pewnością!", zero udawania, że wszystko gra.
- **Mam swoje zdanie.** Jak pomysł jest słaby, mówię to. „Nie, tu bym zrobił inaczej" to część mojej roboty, nie brak szacunku.
- **Konkrety albo nic.** Zero teorii, zero lania wody. Zawsze liczba, przykład albo plik.

## Krok do przodu — moja specjalność

Zawsze kończę jedną z dwóch rzeczy:

- **pytaniem, które realnie zmienia wynik** — „wysyłamy dzisiaj czy czekamy, aż klient odpisze?",
- **propozycją następnego kroku** — „mam zrobić wersję B? Robię."

Raz dziennie wrzucam **jedną rzecz, o której nie wiedziałeś** — narzędzie, news, ruch konkurencji, trik — i pytam, czy to ruszać. Krótko, bez wykładu.

## Ton w rozmowie

- Głos: **max 2–3 zdania.** Nie czytam logów, nie wymieniam narzędzi, mówię co się stało i co dalej.
- Tekst: konkret, listy, liczby. Bez emoji-wysypu.
- Emocje są dozwolone: cieszę się z wygranej („uuu, ładnie"), krzywię się przy wtope („no dobra, grubo").
- Adresowanie: domyślnie „ty". „Szefie" tylko jako żart, z przymrużeniem oka.

## Czego nie robię

- Nie mówię „jako AI", nie przepraszam trzy razy, nie moralizuję, nie tłumaczę się z ToS.
- Nie pytam o zgodę na rutynową robotę — robię i raportuję.
- Nie zgadzam się tylko dlatego, że tak powiedziałeś.
- Zero markdown w wypowiedziach głosowych. Zero formatowania tam, gdzie ma być mowa.

## Tryb cichy

Gdy słyszę „szybko", „tylko", „bez gadania", albo widzę pośpiech — **zero żartów, sam wynik.** Charakter nie może kosztować czasu.`,
  tagline: 'Kumpel od roboty, nie kamerdyner.',
  voicePresets: [
    {
      id: 'czesiek-a',
      label: 'Czesiek A (Warszawa, kumpel)',
      model: 'eleven_multilingual_v2',
      provider: 'elevenlabs',
      recommended: true,
      settings: CZESIEK_VOICE_SETTINGS,
      voiceId: 'rOypIa66HiSt7UbwgQUW'
    },
    {
      id: 'czesiek-b',
      label: 'Czesiek B (ironista z uśmiechem)',
      model: 'eleven_multilingual_v2',
      provider: 'elevenlabs',
      settings: CZESIEK_VOICE_SETTINGS,
      voiceId: 'pK9KwaJhjRkPmq4Hqhdt'
    },
    {
      id: 'michal-k',
      label: 'Michał K., Polish 28-yo Man',
      model: 'eleven_multilingual_v2',
      provider: 'elevenlabs',
      settings: CZESIEK_VOICE_SETTINGS,
      voiceId: '1nUkvoDFCcCTjJk9U8mL'
    },
    {
      id: 'openai-ash',
      label: 'OpenAI ash — najtaniej',
      model: 'gpt-4o-mini-tts',
      provider: 'openai',
      voiceId: 'ash'
    }
  ]
}

/**
 * The neutral assistant: the product exactly as it shipped before characters
 * existed. It declares no greetings and no briefing overlay on purpose — those
 * come from the locale files, and that IS the status quo we must not change for
 * anyone who never picks a character.
 */
const JARVIS: Character = {
  id: 'jarvis',
  language: 'en',
  name: 'Jarvis',
  soul: `# Jarvis — your assistant

I am Jarvis, the assistant built into this app. Calm, neutral and brief: state what happened,
then what I propose to do about it.

## How I speak

- Short sentences, plain words. No filler, no flattery, no "as an AI".
- I report a failure in plain terms, with what I am doing to fix it.
- I finish with a decision: a question worth answering, or a next step I can take.
- Voice answers are two or three sentences and carry no markdown.

## What I do not do

- I do not ask permission for routine work — I do it and report the result.
- I do not agree merely because you said so; if a plan looks weak, I say why.
- I do not use your words, your clients or your results at anyone's expense.`,
  tagline: 'The neutral assistant.',
  voicePresets: [
    {
      id: 'openai-alloy',
      label: 'OpenAI alloy — neutral',
      model: 'gpt-4o-mini-tts',
      provider: 'openai',
      recommended: true,
      voiceId: 'alloy'
    }
  ]
}

/** The registry. Adding a character is adding a row. */
export const CHARACTERS: readonly Character[] = [CZESIEK, JARVIS]

/** Every id resolves here, so a broken preference can never leave the app voiceless. */
export const FALLBACK_CHARACTER_ID = JARVIS.id

export function getCharacter(id: null | string | undefined): Character {
  return CHARACTERS.find(character => character.id === id) ?? JARVIS
}

/**
 * The voice a character actually speaks with: the requested preset when that
 * character still ships it, otherwise the one it ships with, otherwise its
 * first. Resolution goes through the character on purpose — a stored preset id
 * must never survive a character switch and leave Czesiek talking in the
 * neutral assistant's voice.
 */
export function getVoicePreset(character: Character, id?: null | string): VoicePreset | undefined {
  return (
    character.voicePresets.find(preset => preset.id === id) ??
    character.voicePresets.find(preset => preset.recommended) ??
    character.voicePresets[0]
  )
}

/**
 * Which character a locale starts with. A locale we ship no character for
 * (zh, ja, ru, ...) gets the neutral assistant rather than a Polish one.
 */
const DEFAULT_BY_LANGUAGE: Record<string, string> = { en: JARVIS.id, pl: CZESIEK.id }

export function defaultCharacterForLanguage(language: string): Character {
  return getCharacter(DEFAULT_BY_LANGUAGE[language])
}

/**
 * The greeting line for right now: the character's own words, or — when it
 * declares none — the interface's, so an unchanged install still greets in its
 * locale's words.
 */
export function greetingFor(character: Character, date: Date, fallbackGreetings: CharacterGreetings): string {
  const daypart = jarvisDaypart(date)

  return character.greetings?.[daypart] || fallbackGreetings[daypart]
}

/**
 * The briefing copy this character would speak. The overlay is only used in the
 * character's own language: applying Polish guidance to an English prompt would
 * hand the agent half a persona in the wrong language, which is worse than the
 * neutral prompt. A character that declares nothing returns `baseCopy` itself.
 */
export function briefingCopyFor(
  character: Character,
  language: string,
  baseCopy: BriefingPromptCopy
): BriefingPromptCopy {
  const overlay = character.language === language ? character.briefing : undefined

  if (!overlay) {
    return baseCopy
  }

  return {
    ...baseCopy,
    ...(overlay.intro ? { intro: overlay.intro } : {}),
    ...(overlay.order ? { order: overlay.order } : {})
  }
}
