import { describe, expect, it } from 'vitest'

import { shouldHandleStopActiveRun } from './stop-active-run'

describe('shouldHandleStopActiveRun', () => {
  it('handles a busy primary composer', () => {
    expect(shouldHandleStopActiveRun(true, true, false)).toBe(true)
  })

  it('does not halt a busy secondary tile when the primary handles the chord', () => {
    expect(shouldHandleStopActiveRun(false, true, false)).toBe(false)
  })

  it('does not consume the stop chord while awaiting input', () => {
    expect(shouldHandleStopActiveRun(true, true, true)).toBe(false)
  })

  it('halts only the primary composer when mounted busy tiles share the event', () => {
    const halts = { primary: 0, secondary: 0 }

    const listeners = [
      { key: 'primary' as const, isPrimary: true },
      { key: 'secondary' as const, isPrimary: false }
    ].map(({ key, isPrimary }) => {
      const listener = (event: Event) => {
        if (!shouldHandleStopActiveRun(isPrimary, true, false)) {
          return
        }

        event.preventDefault()

        halts[key] += 1
      }

      window.addEventListener('hermes:stop-active-run', listener)

      return listener
    })

    const event = new CustomEvent('hermes:stop-active-run', { cancelable: true })
    window.dispatchEvent(event)

    listeners.forEach(listener => window.removeEventListener('hermes:stop-active-run', listener))
    expect(halts).toEqual({ primary: 1, secondary: 0 })
    expect(event.defaultPrevented).toBe(true)
  })
})