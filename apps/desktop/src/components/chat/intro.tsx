import { type CSSProperties, useState } from 'react'

import introCopyJsonl from './intro-copy.jsonl?raw'

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
    headline: 'Что будем делать сегодня?',
    body: 'Пришли баг, ветку, план или идею. Я изучу репозиторий и предложу следующий шаг.'
  },
  {
    headline: 'О чём думаете?',
    body: 'Поделитесь кодом, вопросом или проблемой. Я сначала разберусь, потом предложу решение.'
  },
  {
    headline: 'Что показать Hermes?',
    body: 'Отправь задачу, падающий тест или набросок плана. Я помогу превратить это в действие.'
  },
  {
    headline: 'С чего начнём?',
    body: 'Принеси проблему, цель или файл. Я сначала изучу, потом предложу конкретный следующий шаг.'
  },
  {
    headline: 'Что требует внимания?',
    body: 'Отправь всё, что есть. Я помогу разобраться и составить план или исправление.'
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

const INTRO_COPY_BY_PERSONALITY = parseIntroCopy(introCopyJsonl)

function neutralCopy(): IntroCopy[] {
  return INTRO_COPY_BY_PERSONALITY.none || INTRO_COPY_BY_PERSONALITY.default || FALLBACK_COPY
}

function fallbackCopyForPersonality(personalityKey: string): IntroCopy[] {
  if (NEUTRAL_PERSONALITIES.has(personalityKey)) {
    return neutralCopy()
  }

  const label = titleize(personalityKey)

  return [
    {
      headline: `Режим «${label}» включён. Что будем делать?`,
      body: 'Отправь задачу, файл или идею. Я буду работать в заданном стиле и держать фокус на репозитории.'
    },
    {
      headline: `Что показать Hermes в режиме «${label}»?`,
      body: 'Поделитесь контекстом или проблемой. Я адаптируюсь к выбранной персоналии.'
    },
    {
      headline: `Режим «${label}» готов к работе.`,
      body: 'Отправь проблему, файл или идею. Я буду следовать настроенной персоналии.'
    },
    {
      headline: `Что поставить на выполнение в режиме «${label}»?`,
      body: 'Опиши задачу здесь. Я буду работать в контексте репозитория.'
    },
    {
      headline: 'С чего начать?',
      body: `Дай мне контекст, и я отвечу в режиме «${label}».`
    }
  ]
}

function pickCopy(copies: IntroCopy[], seed = 0): IntroCopy {
  return copies[Math.abs(seed) % copies.length] || FALLBACK_COPY[0]
}

const WORDMARK = 'HERMES AGENT'

function resolveCopy(personality?: string, seed?: number): IntroCopy {
  const personalityKey = normalizeKey(personality)

  const copies = NEUTRAL_PERSONALITIES.has(personalityKey)
    ? INTRO_COPY_BY_PERSONALITY[personalityKey] || neutralCopy()
    : INTRO_COPY_BY_PERSONALITY[personalityKey] || fallbackCopyForPersonality(personalityKey)

  return pickCopy(copies, seed)
}

export function Intro({ personality, seed }: IntroProps) {
  const [mountSeed] = useState(() => Math.floor(Math.random() * 100000))
  const copy = resolveCopy(personality, mountSeed + (seed ?? 0))

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
