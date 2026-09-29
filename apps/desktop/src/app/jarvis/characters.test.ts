import { afterEach, describe, expect, it } from 'vitest'

import { pl } from '@/i18n/pl'
import { $characterId, setCharacter } from '@/store/character'

import { buildBriefingPrompt } from './briefing'
import {
  briefingCopyFor,
  type BriefingPromptCopy,
  CHARACTERS,
  defaultCharacterForLanguage,
  FALLBACK_CHARACTER_ID,
  getCharacter,
  greetingFor
} from './characters'
import { jarvisDaypart } from './pulse'

const at = (hour: number) => new Date(2026, 8, 25, hour, 15)

/** The interface's own greetings — what a fresh install shows today. */
const FALLBACK_GREETINGS = pl.jarvisShell.home.greetings

const HOUR_BY_DAYPART: { daypart: keyof typeof FALLBACK_GREETINGS; hour: number }[] = [
  { daypart: 'morning', hour: 9 },
  { daypart: 'afternoon', hour: 14 },
  { daypart: 'evening', hour: 20 },
  { daypart: 'night', hour: 2 }
]

const BASE_COPY: BriefingPromptCopy = {
  ai: 'AI news',
  failed: 'Sources that did not answer',
  google: 'Google',
  googleOrder: 'BASE GOOGLE ORDER',
  intro: 'BASE INTRO',
  jobs: 'Scheduled jobs',
  model: 'Active model',
  noData: 'BASE NO DATA',
  order: 'BASE ORDER',
  safety: 'BASE SAFETY',
  sessions: 'Sessions',
  world: 'World news'
}

const neutral = () => getCharacter(FALLBACK_CHARACTER_ID)
const czesiek = () => getCharacter('czesiek')

afterEach(() => {
  setCharacter(FALLBACK_CHARACTER_ID)
})

describe('getCharacter', () => {
  it('resolves an id it does not ship to the neutral assistant, never to nothing', () => {
    for (const unknown of [undefined, null, '', 'jarvis', 'from_the_future']) {
      expect(getCharacter(unknown).id).toBe(FALLBACK_CHARACTER_ID)
    }
  })

  it('ships the Polish buddy under his own id', () => {
    expect(czesiek().id).toBe('czesiek')
    expect(czesiek().language).toBe('pl')
  })
})

describe('defaultCharacterForLanguage', () => {
  it('gives Polish a Polish assistant and every other locale the neutral one', () => {
    expect(defaultCharacterForLanguage('pl').id).toBe('czesiek')
    expect(defaultCharacterForLanguage('en').id).toBe(FALLBACK_CHARACTER_ID)
    // A locale we ship no character for must not import a Polish one.
    expect(defaultCharacterForLanguage('zh').id).toBe(FALLBACK_CHARACTER_ID)
  })
})

describe('greetingFor', () => {
  it('keeps the interface greeting for the neutral assistant, all day long', () => {
    for (let hour = 0; hour < 24; hour += 1) {
      const date = at(hour)

      expect(greetingFor(neutral(), date, FALLBACK_GREETINGS)).toBe(FALLBACK_GREETINGS[jarvisDaypart(date)])
    }
  })

  it('speaks the character words at each part of the day, and never twice the same', () => {
    const spoken = HOUR_BY_DAYPART.map(({ daypart, hour }) => ({
      daypart,
      line: greetingFor(czesiek(), at(hour), FALLBACK_GREETINGS)
    }))

    for (const { daypart, line } of spoken) {
      expect(line).toBe(czesiek().greetings?.[daypart])
      expect(line).not.toBe(FALLBACK_GREETINGS[daypart])
    }

    expect(new Set(spoken.map(entry => entry.line)).size).toBe(HOUR_BY_DAYPART.length)
  })

  it('falls back per part of the day, not for the whole character', () => {
    const wordless = { ...czesiek(), greetings: undefined }
    const night = HOUR_BY_DAYPART.find(({ daypart }) => daypart === 'night')?.hour ?? 2

    expect(greetingFor(wordless, at(night), FALLBACK_GREETINGS)).toBe(FALLBACK_GREETINGS.night)
  })
})

describe('briefingCopyFor', () => {
  it('hands back the interface copy untouched for a character that declares no overlay', () => {
    const copy = briefingCopyFor(neutral(), 'pl', BASE_COPY)

    expect(copy).toBe(BASE_COPY)
    expect(copy.intro).toBe('BASE INTRO')
  })

  it('overlays only the fields the character declares, in its own language only', () => {
    const overlayed = briefingCopyFor(czesiek(), 'pl', BASE_COPY)

    expect(overlayed.intro).not.toBe(BASE_COPY.intro)
    expect(overlayed.order).not.toBe(BASE_COPY.order)
    // Everything the character has no opinion about stays the interface's.
    expect(overlayed.safety).toBe(BASE_COPY.safety)
    expect(overlayed.noData).toBe(BASE_COPY.noData)

    // The overlay is Polish: an English prompt must not receive it.
    expect(briefingCopyFor(czesiek(), 'en', BASE_COPY)).toBe(BASE_COPY)
  })

  it('leaves the fields a partial overlay omits', () => {
    const partial = { ...czesiek(), briefing: { intro: 'CZESIEK INTRO' } }
    const overlayed = briefingCopyFor(partial, 'pl', BASE_COPY)

    expect(overlayed.intro).toBe('CZESIEK INTRO')
    expect(overlayed.order).toBe(BASE_COPY.order)
  })
})

describe('the registry', () => {
  it('gives every character an identity, a persona and a voice', () => {
    expect(new Set(CHARACTERS.map(character => character.id)).size).toBe(CHARACTERS.length)

    for (const character of CHARACTERS) {
      expect(character.name.trim().length).toBeGreaterThan(0)
      expect(character.tagline.trim().length).toBeGreaterThan(0)
      expect(character.soul.trim().length).toBeGreaterThan(0)
      expect(character.voicePresets.length).toBeGreaterThan(0)
      expect(character.voicePresets.filter(preset => preset.recommended).length).toBe(1)
      expect(new Set(character.voicePresets.map(preset => preset.id)).size).toBe(character.voicePresets.length)

      for (const preset of character.voicePresets) {
        expect(preset.label.trim().length).toBeGreaterThan(0)
        expect(preset.voiceId.trim().length).toBeGreaterThan(0)
        expect(preset.model.trim().length).toBeGreaterThan(0)
      }
    }
  })

  it('keeps the ElevenLabs delivery low-stability, where the smile lives', () => {
    const elevenlabs = CHARACTERS.flatMap(character =>
      character.voicePresets.filter(preset => preset.provider === 'elevenlabs')
    )

    expect(elevenlabs.length).toBeGreaterThan(0)

    for (const preset of elevenlabs) {
      expect(preset.settings?.stability).toBeLessThan(0.6)
      expect(preset.model).toBe('eleven_multilingual_v2')
    }
  })

  it('ships Czesiek with an alternative voice, not just one', () => {
    expect(czesiek().voicePresets.length).toBeGreaterThanOrEqual(3)
  })
})

describe('briefing wiring', () => {
  it('carries the selected character briefing voice into the prompt it builds', () => {
    const base = buildBriefingPrompt(null, 'pl', 'Raport dnia')
    const czesiekIntro = czesiek().briefing?.intro ?? ''

    expect(czesiekIntro.length).toBeGreaterThan(0)
    expect(base).not.toContain(czesiekIntro)

    setCharacter('czesiek')
    expect(buildBriefingPrompt(null, 'pl', 'Raport dnia')).toContain(czesiekIntro)

    setCharacter(FALLBACK_CHARACTER_ID)
    expect(buildBriefingPrompt(null, 'pl', 'Raport dnia')).toBe(base)
  })

  it('starts on the neutral assistant and only moves when the choice is persisted', () => {
    expect($characterId.get()).toBe(FALLBACK_CHARACTER_ID)

    setCharacter('czesiek')
    expect($characterId.get()).toBe('czesiek')

    // A nonsense choice is resolved before it is stored, so the app can never
    // come back voiceless on the next launch.
    setCharacter('from_the_future')
    expect($characterId.get()).toBe(FALLBACK_CHARACTER_ID)
  })
})
