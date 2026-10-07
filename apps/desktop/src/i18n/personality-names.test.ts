import { describe, expect, it } from 'vitest'

import { BUILTIN_PERSONALITIES } from '@/lib/personalities'

import { en } from './en'
import { pt } from './pt'

describe('personality display names', () => {
  it('labels every built-in personality in English and Portuguese', () => {
    for (const locale of [en, pt]) {
      const names = locale.settings.config.personalityNames as Record<string, string>

      for (const id of BUILTIN_PERSONALITIES) {
        expect(names[id], id).toBeTruthy()
      }
    }
  })
})
