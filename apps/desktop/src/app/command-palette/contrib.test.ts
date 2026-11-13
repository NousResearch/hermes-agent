/**
 * Palette toggles register once at boot, before `display.language` loads, and
 * the palette reads `detail()` each time it builds its rows. The note has to
 * come out in the language active at that moment and follow the state the row
 * just flipped, not the English sampled at registration.
 */
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { setRuntimeI18nLocale } from '@/i18n'

import { PALETTE_AREA, type PaletteContribution, paletteToggle } from './contrib'

const ID = 'test.paletteToggle'

let dispose: (() => void) | undefined

afterEach(() => {
  dispose?.()
  dispose = undefined
  setRuntimeI18nLocale('en')
})

function registeredRow(): PaletteContribution {
  const row = registry.getArea(PALETTE_AREA).find(c => c.id === ID)

  if (!row) {
    throw new Error(`palette toggle ${ID} is not registered`)
  }

  return row.data as PaletteContribution
}

describe('paletteToggle', () => {
  it('reads an already-registered row in the live language and state', () => {
    let enabled = true

    setRuntimeI18nLocale('en')
    dispose = registry.register(
      paletteToggle({
        id: ID,
        label: 'Toggle test',
        get: () => enabled,
        set: next => {
          enabled = next
        }
      })
    )

    expect(registeredRow().detail?.()).toBe('on')

    // The language loads after the row registered.
    setRuntimeI18nLocale('ru')
    expect(registeredRow().detail?.()).toBe('вкл')

    // Running the row flips the state; the note follows without re-registering.
    registeredRow().run()
    expect(enabled).toBe(false)
    expect(registeredRow().detail?.()).toBe('выкл')

    setRuntimeI18nLocale('en')
    expect(registeredRow().detail?.()).toBe('off')
  })
})
