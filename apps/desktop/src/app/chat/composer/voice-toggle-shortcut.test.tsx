import { act, render } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { describe, expect, it } from 'vitest'

import { defaultBindings, KEYBIND_ACTIONS, keybindAction } from '@/lib/keybinds/actions'

const VOICE_TOGGLE_EVENT = 'hermes:voice-toggle-record'

describe('voice.toggleRecord registry entry', () => {
  it('is registered as a rebindable, unbound view action', () => {
    const action = keybindAction('voice.toggleRecord')

    expect(action).toBeDefined()
    expect(action?.category).toBe('view')
    // Shipped unbound so it can't collide with mod+b (view.toggleSidebar);
    // the user assigns a combo in the shortcuts panel.
    expect(action?.defaults).toEqual([])
    expect(KEYBIND_ACTIONS.some(entry => entry.id === 'voice.toggleRecord')).toBe(true)
    expect(defaultBindings()['voice.toggleRecord']).toEqual([])
  })
})

// Faithful mirror of index.tsx's voice-toggle wiring: the keybind handler in
// use-keybinds.ts dispatches a window CustomEvent, and the composer flips the
// same voiceConversationActive state the mic button owns. Driven through a
// REAL window event so it exercises the add/removeEventListener lifecycle
// rather than a direct setter call.
function Harness({ onState }: { onState: (active: boolean) => void }) {
  const [active, setActive] = useState(false)

  useEffect(() => {
    onState(active)
  }, [active, onState])

  useEffect(() => {
    const onToggle = () => setActive(value => !value)
    window.addEventListener(VOICE_TOGGLE_EVENT, onToggle)

    return () => window.removeEventListener(VOICE_TOGGLE_EVENT, onToggle)
  }, [])

  return null
}

describe('voice toggle event bridge', () => {
  it('toggles voice state on each dispatched event', () => {
    const states: boolean[] = []
    render(<Harness onState={active => states.push(active)} />)

    act(() => {
      window.dispatchEvent(new CustomEvent(VOICE_TOGGLE_EVENT))
    })
    expect(states.at(-1)).toBe(true)

    act(() => {
      window.dispatchEvent(new CustomEvent(VOICE_TOGGLE_EVENT))
    })
    expect(states.at(-1)).toBe(false)
  })

  it('stops responding after unmount', () => {
    const states: boolean[] = []
    const { unmount } = render(<Harness onState={active => states.push(active)} />)

    unmount()
    const countAfterUnmount = states.length

    act(() => {
      window.dispatchEvent(new CustomEvent(VOICE_TOGGLE_EVENT))
    })

    expect(states.length).toBe(countAfterUnmount)
  })
})
