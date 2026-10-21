import type { ModelOptionProvider } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { accountResetMs, modelResetMs, usageWindows } from './provider-limit'

const NOW = Date.parse('2026-10-05T15:00:00Z')
const iso = (minutes: number) => new Date(NOW + minutes * 60_000).toISOString()

const provider = (limit: ModelOptionProvider['limit']): ModelOptionProvider => ({
  slug: 'anthropic',
  name: 'Anthropic',
  models: ['a', 'b'],
  limit
})

describe('provider limits', () => {
  it('reports an account-wide limit until its reset, then clears without a refetch', () => {
    const limited = provider({ scope: 'account', resets_at: iso(30) })

    expect(accountResetMs(limited, NOW)).toBe(NOW + 30 * 60_000)
    expect(modelResetMs(limited, 'a', NOW)).toBeNull()
    expect(accountResetMs(limited, NOW + 31 * 60_000)).toBeNull()
  })

  it('tags only the cooled-down models, each until its own reset', () => {
    const limited = provider({ scope: 'models', models: { a: iso(10) } })

    expect(accountResetMs(limited, NOW)).toBeNull()
    expect(modelResetMs(limited, 'a', NOW)).toBe(NOW + 10 * 60_000)
    expect(modelResetMs(limited, 'b', NOW)).toBeNull()
    expect(modelResetMs(limited, 'a', NOW + 11 * 60_000)).toBeNull()
  })

  it('leads with the usage window closest to the wall and drops windows that rolled over', () => {
    const windows = usageWindows(
      {
        ...provider(null),
        usage: {
          windows: [
            { label: 'Weekly', used_percent: 39, resets_at: iso(60 * 24) },
            { label: '5-hour', used_percent: 92.4, resets_at: iso(90) },
            { label: 'Opus week', used_percent: 99, resets_at: iso(-5) }
          ]
        }
      },
      NOW
    )

    expect(windows?.map(w => [w.label, w.remaining])).toEqual([
      ['5-hour', 8],
      ['Weekly', 61]
    ])
  })
})
