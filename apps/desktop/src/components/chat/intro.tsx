import { type CSSProperties, useState } from 'react'

import { type IntroCopy, type Translations, useI18n } from '@/i18n'
import { en } from '@/i18n/en'
import { capitalize, normalize } from '@/lib/text'

import introCopyJsonl from './intro-copy.jsonl?raw'

type IntroCopyRecord = IntroCopy & {
  personality: string
}

type IntroTranslations = NonNullable<Translations['intro']>

export type IntroProps = {
  personality?: string
  seed?: number
}

const NEUTRAL_PERSONALITIES = new Set(['', 'default', 'none', 'neutral'])

function normalizeKey(value?: string): string {
  return normalize(value)
}

function titleize(value: string): string {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map(capitalize)
    .join(' ')
}

function isIntroCopyRecord(value: unknown): value is IntroCopyRecord {
  if (!value || typeof value !== 'object') {
    return false
  }

  const record = value as Record<string, unknown>

  return (
    typeof record.personality === 'string' &&
    typeof record.headline === 'string' &&
    typeof record.body === 'string' &&
    Boolean(record.personality.trim()) &&
    Boolean(record.headline.trim()) &&
    Boolean(record.body.trim())
  )
}

function parseIntroCopy(raw: string): Record<string, IntroCopy[]> {
  const byPersonality: Record<string, IntroCopy[]> = {}

  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim()

    if (!trimmed) {
      continue
    }

    try {
      const parsed: unknown = JSON.parse(trimmed)

      if (!isIntroCopyRecord(parsed)) {
        continue
      }

      const key = normalizeKey(parsed.personality)
      byPersonality[key] ??= []
      byPersonality[key].push({
        headline: parsed.headline.trim(),
        body: parsed.body.trim()
      })
    } catch {
      // Bad generated copy should not break the whole desktop app.
    }
  }

  return byPersonality
}

const INTRO_COPY_BY_PERSONALITY = parseIntroCopy(introCopyJsonl)
const DEFAULT_INTRO_TRANSLATIONS = en.intro!

function validCopy(copies: readonly IntroCopy[] | undefined): readonly IntroCopy[] | null {
  return copies?.length ? copies : null
}

function neutralCopy(
  localized: Record<string, readonly IntroCopy[]>,
  fallbackCopy: readonly IntroCopy[]
): readonly IntroCopy[] {
  return (
    validCopy(localized.none) ||
    validCopy(localized.default) ||
    validCopy(INTRO_COPY_BY_PERSONALITY.none) ||
    validCopy(INTRO_COPY_BY_PERSONALITY.default) ||
    fallbackCopy
  )
}

function localizedCopyForPersonality(
  localized: Record<string, readonly IntroCopy[]>,
  personalityKey: string
): readonly IntroCopy[] | null {
  if (NEUTRAL_PERSONALITIES.has(personalityKey)) {
    return validCopy(localized[personalityKey]) || validCopy(localized.none) || validCopy(localized.default)
  }

  return validCopy(localized[personalityKey])
}

function fallbackCopyForPersonality(
  personalityKey: string,
  fallbackTemplates: IntroTranslations['fallbackTemplates']
): readonly IntroCopy[] {
  if (NEUTRAL_PERSONALITIES.has(personalityKey)) {
    return []
  }

  const label = titleize(personalityKey)

  return [
    {
      headline: fallbackTemplates.modeOnHeadline(label),
      body: fallbackTemplates.modeOnBody
    },
    {
      headline: fallbackTemplates.needSeeHeadline(label),
      body: fallbackTemplates.needSeeBody
    },
    {
      headline: fallbackTemplates.readyHeadline(label),
      body: fallbackTemplates.readyBody
    },
    {
      headline: fallbackTemplates.tackleHeadline(label),
      body: fallbackTemplates.tackleBody
    },
    {
      headline: fallbackTemplates.beginHeadline,
      body: fallbackTemplates.beginBody(label)
    }
  ]
}

function pickCopy(copies: readonly IntroCopy[], seed = 0, fallbackCopy: readonly IntroCopy[]): IntroCopy {
  return copies[Math.abs(seed) % copies.length] || fallbackCopy[0]
}

const WORDMARK = 'HERMES AGENT'

function resolveCopy(
  personality: string | undefined,
  seed: number | undefined,
  translations: IntroTranslations
): IntroCopy {
  const personalityKey = normalizeKey(personality)
  const localizedCopy = localizedCopyForPersonality(translations.copy, personalityKey)

  if (localizedCopy) {
    return pickCopy(localizedCopy, seed, translations.fallbackCopy)
  }

  const copies = NEUTRAL_PERSONALITIES.has(personalityKey)
    ? neutralCopy(translations.copy, translations.fallbackCopy)
    : INTRO_COPY_BY_PERSONALITY[personalityKey] || fallbackCopyForPersonality(personalityKey, translations.fallbackTemplates)

  return pickCopy(copies, seed, translations.fallbackCopy)
}

export function Intro({ personality, seed }: IntroProps) {
  const { t } = useI18n()
  const [mountSeed] = useState(() => Math.floor(Math.random() * 100000))
  const copy = resolveCopy(personality, mountSeed + (seed ?? 0), t.intro ?? DEFAULT_INTRO_TRANSLATIONS)

  return (
    <div
      className="pointer-events-none flex w-full min-w-0 flex-col items-center justify-center px-0.5 py-6 text-center text-muted-foreground sm:px-6 lg:px-8"
      data-slot="aui_intro"
    >
      <div className="w-full min-w-0">
        <p
          aria-label={WORDMARK}
          className="fit-text mx-auto mb-1 w-[calc(100%-1rem)] font-['Collapse'] font-bold uppercase leading-[0.9] tracking-[0.08em] text-midground mix-blend-plus-lighter dark:text-foreground/90"
          style={{ '--fit-min': '2.75rem' } as CSSProperties}
        >
          <span>
            <span>{WORDMARK}</span>
          </span>
          <span aria-hidden="true">{WORDMARK}</span>
        </p>

        <p className="m-0 text-center leading-normal tracking-tight">{copy.body}</p>
      </div>
    </div>
  )
}
