import { type CSSProperties, useState } from 'react'

import { type Locale, useI18n } from '@/i18n'

import introCopyJsonl from './intro-copy.jsonl?raw'
import introCopyZhJsonl from './intro-copy.zh.jsonl?raw'

type IntroCopy = {
  headline: string
  body: string
}

type IntroCopyRecord = IntroCopy & {
  personality: string
}

export type IntroProps = {
  personality?: string
  seed?: number
}

const NEUTRAL_PERSONALITIES = new Set(['', 'default', 'none', 'neutral'])

const FALLBACK_COPY: IntroCopy[] = [
  {
    headline: 'What are we moving today?',
    body: "Send a bug, branch, plan, or rough idea. I'll inspect the repo and turn it into the next concrete step."
  },
  {
    headline: "What's on your mind?",
    body: "Bring the code, question, or stuck part. I'll read the room before making changes."
  },
  {
    headline: 'What should Hermes look at?',
    body: "Send the task, failing path, or half-formed plan. I'll help turn it into action."
  },
  {
    headline: 'Where should we start?',
    body: "Bring the problem, goal, or file. I'll inspect first and keep the next step concrete."
  },
  {
    headline: 'What needs attention?',
    body: "Send the context you have. I'll help sort it into a plan or a fix."
  }
]

function normalizeKey(value?: string): string {
  return (value || '').trim().toLowerCase()
}

function titleize(value: string): string {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
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

const INTRO_COPY_BY_LOCALE: Partial<Record<Locale, Record<string, IntroCopy[]>>> = {
  en: parseIntroCopy(introCopyJsonl),
  zh: parseIntroCopy(introCopyZhJsonl)
}

function copyByPersonality(locale: Locale): Record<string, IntroCopy[]> {
  return INTRO_COPY_BY_LOCALE[locale] ?? INTRO_COPY_BY_LOCALE.en ?? {}
}

function neutralCopy(locale: Locale): IntroCopy[] {
  const copy = copyByPersonality(locale)

  return copy.none || copy.default || FALLBACK_COPY
}

function fallbackCopyForPersonality(locale: Locale, personalityKey: string): IntroCopy[] {
  if (NEUTRAL_PERSONALITIES.has(personalityKey)) {
    return neutralCopy(locale)
  }

  const label = titleize(personalityKey)

  return [
    {
      headline: `${label} mode is on. What should we work on?`,
      body: "Send the task, file, or rough idea. I'll use your configured voice and keep the work grounded in this repo."
    },
    {
      headline: `What does ${label} Hermes need to see?`,
      body: "Bring the context or the stuck part. I'll adapt to your configured personality."
    },
    {
      headline: `${label} mode is ready.`,
      body: "Send the problem, file, or idea. I'll follow the personality you've configured."
    },
    {
      headline: `What should ${label} Hermes tackle?`,
      body: "Drop the task here. I'll keep the work grounded in the repo."
    },
    {
      headline: 'Where should we begin?',
      body: `Give me the context and I'll answer in ${label} mode.`
    }
  ]
}

function pickCopy(copies: IntroCopy[], seed = 0): IntroCopy {
  return copies[Math.abs(seed) % copies.length] || FALLBACK_COPY[0]
}

const WORDMARK = 'HERMES AGENT'

function resolveCopy(locale: Locale, personality?: string, seed?: number): IntroCopy {
  const personalityKey = normalizeKey(personality)
  const copy = copyByPersonality(locale)

  const copies = NEUTRAL_PERSONALITIES.has(personalityKey)
    ? copy[personalityKey] || neutralCopy(locale)
    : copy[personalityKey] || fallbackCopyForPersonality(locale, personalityKey)

  return pickCopy(copies, seed)
}

export function Intro({ personality, seed }: IntroProps) {
  const { locale } = useI18n()
  const [mountSeed] = useState(() => Math.floor(Math.random() * 100000))
  const copy = resolveCopy(locale, personality, mountSeed + (seed ?? 0))

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
