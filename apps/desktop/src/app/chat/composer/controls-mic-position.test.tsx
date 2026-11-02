import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { I18nProvider } from '@/i18n'
import type { VoiceLiveStatus } from '@/lib/voice-live'
import { $voiceLiveStatus } from '@/store/voice-live'

import { ComposerControls } from './controls'

vi.mock('./model-pill', () => ({ ModelPill: () => null }))

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { active: false, enabled: false }
}

type Props = React.ComponentProps<typeof ComposerControls>

function renderRow(overrides: Partial<Props> = {}) {
  const { container } = render(
    <I18nProvider configClient={null} initialLocale="en">
      <ComposerControls
        autoSpeak={false}
        busy={false}
        busyAction="stop"
        canSubmit={true}
        conversation={{
          active: false,
          level: 0,
          muted: false,
          onEnd: vi.fn(),
          onStart: vi.fn(),
          onStopTurn: vi.fn(),
          onToggleMute: vi.fn(),
          status: 'idle'
        }}
        disabled={false}
        hasComposerPayload={false}
        onDictate={vi.fn()}
        onQueue={vi.fn()}
        onSteer={vi.fn()}
        onToggleAutoSpeak={vi.fn()}
        state={state}
        voiceStatus="idle"
        {...overrides}
      />
    </I18nProvider>
  )

  const row = container.firstElementChild as HTMLElement
  const children = Array.from(row.children)
  const mic = row.querySelector('[data-slot="fan-menu-anchor"]') as HTMLElement

  return { children, mic, row }
}

// The run states a composer moves through while the pointer is parked on the mic.
const STATES: Array<[string, Partial<Props>]> = [
  ['Start voice', { busy: false, hasComposerPayload: false }],
  ['Send', { busy: false, hasComposerPayload: true }],
  ['Stop', { busy: true, busyAction: 'stop', hasComposerPayload: false }]
]

afterEach(() => {
  cleanup()
  $voiceLiveStatus.set(null)
})

// The controls pack to the right edge, so the mic holds still only if everything to
// its right has the same width in every state (#147426: Start voice is a circle plus an
// engine chevron, Send / Stop a bare circle, so the mic moved ~20px and a click aimed at
// dictation started a voice conversation). jsdom has no layout, so this pins what makes
// the width constant: ONE element follows the mic and it reserves the same width in each
// state. The reserved width is a CSS class; its pixel value is not asserted.
describe('composer mic position (#147426)', () => {
  it('follows the mic with one slot that reserves the same width for Start voice, Send and Stop', () => {
    $voiceLiveStatus.set({ available: true, mode: 'chained', model: '', reason: null, voice: '' } as VoiceLiveStatus)

    const slots = STATES.map(([, overrides]) => {
      const { children, mic } = renderRow(overrides)
      const afterMic = children.slice(children.indexOf(mic) + 1)

      cleanup()

      return afterMic
    })

    for (const afterMic of slots) {
      expect(afterMic).toHaveLength(1)
      expect(afterMic[0]?.className).toContain('min-w-')
    }

    expect(new Set(slots.map(([slot]) => slot?.className)).size).toBe(1)
  })

  it('keeps the mid-turn Steer and Queue buttons left of the mic', () => {
    const { children, mic, row } = renderRow({ busy: true, busyAction: 'interrupt', hasComposerPayload: true })

    for (const label of ['Steer the current run', 'Queue message']) {
      const button = row.querySelector(`[aria-label="${label}"]`)

      expect(button).not.toBeNull()
      expect(children.indexOf(children.find(child => child.contains(button!))!)).toBeLessThan(children.indexOf(mic))
    }
  })
})
