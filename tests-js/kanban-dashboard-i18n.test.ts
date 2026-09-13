import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const BUNDLE = path.join(REPO_ROOT, 'plugins', 'kanban', 'dashboard', 'dist', 'index.js')
const source = fs.readFileSync(BUNDLE, 'utf-8')

function extractFunction(name: string): string {
  const marker = `function ${name}(`
  const start = source.indexOf(marker)
  assert.ok(start >= 0, `missing ${name} in Kanban dashboard bundle`)
  const braceStart = source.indexOf('{', start)
  assert.ok(braceStart >= 0, `missing body for ${name}`)

  let depth = 0

  for (let i = braceStart; i < source.length; i += 1) {
    if (source[i] === '{') {depth += 1}

    if (source[i] === '}') {
      depth -= 1

      if (depth === 0) {return source.slice(start, i + 1)}
    }
  }

  throw new Error(`unterminated ${name} in Kanban dashboard bundle`)
}

type Tx = (
  t: unknown,
  path: string,
  fallback: string,
  vars?: Record<string, string | number>,
) => string

const catalog: Record<string, string> = {
  'taskCount.one': '{n} задача',
  'taskCount.few': '{n} задачи',
  'taskCount.many': '{n} задач',
  'taskCount.other': '{n} задачи',
  'eventKinds.created': 'создано',
  'eventPayloadKeys.status': 'статус',
  'eventPayloadKeys.assignee': 'исполнитель',
  none: 'нет',
}

const tx: Tx = (_t, key, fallback, vars) => {
  let value = catalog[key] ?? fallback

  for (const [name, replacement] of Object.entries(vars ?? {})) {
    value = value.replaceAll(`{${name}}`, String(replacement))
  }

  return value
}

function loadHelpers(intlValue: typeof Intl, timeAgoValue?: (ts: number) => string) {
  const names = [
    'fallbackPluralCategory',
    'formatTaskCount',
    'formatTimeAgo',
    'getEventKindLabel',
    'getEventPayloadKeyLabel',
    'formatEventPayload',
  ] as const

  const code = names.map(extractFunction).join('\n')

  const factory = new Function(
    'tx',
    'getColumnLabel',
    'Intl',
    'timeAgo',
    `${code}\nreturn { ${names.join(', ')} };`,
  ) as (
    txFn: Tx,
    getColumnLabel: (_t: unknown, status: string) => string,
    intl: typeof Intl,
    timeAgo?: (ts: number) => string,
  ) => Record<(typeof names)[number], (...args: any[]) => any>

  return factory(
    tx,
    (_t, status) => (status === 'ready' ? 'Готово к работе' : status),
    intlValue,
    timeAgoValue,
  )
}

test('Kanban task counts keep Russian plural categories without Intl.PluralRules', () => {
  const noPluralRules = {
    ...Intl,
    PluralRules: class {
      constructor() {
        throw new Error('PluralRules unavailable')
      }
    },
  } as unknown as typeof Intl

  const helpers = loadHelpers(noPluralRules)

  assert.equal(helpers.formatTaskCount({}, 'ru-RU', 1), '1 задача')
  assert.equal(helpers.formatTaskCount({}, 'ru-RU', 2), '2 задачи')
  assert.equal(helpers.formatTaskCount({}, 'ru-RU', 5), '5 задач')
  assert.equal(helpers.formatTaskCount({}, 'ru-RU', 11), '11 задач')
  assert.equal(helpers.formatTaskCount({}, 'ru-RU', 21), '21 задача')
  assert.equal(helpers.formatTaskCount({}, 'ru-RU', 22), '22 задачи')
  assert.equal(helpers.formatTaskCount({}, 'ru-RU', 25), '25 задач')
})

test('Kanban relative-time fallback is safe when Intl.RelativeTimeFormat is unavailable', () => {
  const noRelativeTime = {
    ...Intl,
    RelativeTimeFormat: class {
      constructor() {
        throw new Error('RelativeTimeFormat unavailable')
      }
    },
  } as unknown as typeof Intl

  const withoutLegacyFallback = loadHelpers(noRelativeTime)
  assert.equal(withoutLegacyFallback.formatTimeAgo('ru-RU', Date.now() / 1000 - 120), '')

  const withLegacyFallback = loadHelpers(noRelativeTime, () => 'legacy fallback')
  assert.equal(
    withLegacyFallback.formatTimeAgo('ru-RU', Date.now() / 1000 - 120),
    'legacy fallback',
  )
})

test('Kanban event helpers localize event names, payload keys, and status values', () => {
  const helpers = loadHelpers(Intl)

  assert.equal(helpers.getEventKindLabel({}, 'created'), 'создано')
  assert.equal(
    helpers.formatEventPayload({}, { status: 'ready', assignee: null }),
    'статус: Готово к работе · исполнитель: нет',
  )
})
