import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { I18nProvider } from '@/i18n'
import { applyWakeStartResult, applyWakeStatus, resetWakeWordState } from '@/store/wake-word'

import { ComposerControls } from './controls'

vi.mock('./model-pill', () => ({ ModelPill: () => null }))

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { active: false, enabled: false }
}

function renderControls(overrides: Partial<React.ComponentProps<typeof ComposerControls>> = {}) {
  return render(
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
        hasComposerPayload={true}
        onDictate={vi.fn()}
        onQueue={vi.fn()}
        onToggleAutoSpeak={vi.fn()}
        state={state}
        voiceStatus="idle"
        {...overrides}
      />
    </I18nProvider>
  )
}

async function expectShortcutTooltip(label: string, shortcut: string) {
  fireEvent.pointerMove(screen.getByLabelText(label), { pointerType: 'mouse' })

  const tooltip = await screen.findByRole('tooltip')

  expect(tooltip.textContent).toContain(label)
  expect(tooltip.textContent).toContain(shortcut)
}

afterEach(() => {
  cleanup()
})

describe('ComposerControls shortcut tooltips', () => {
  it('shows Enter for Send', async () => {
    renderControls()

    await expectShortcutTooltip('Send', '↵')
  })

  it('shows Enter for Steer', async () => {
    renderControls({ busy: true, busyAction: 'steer' })

    await expectShortcutTooltip('Steer the current run', '↵')
  })

  it('shows Ctrl+Enter for Queue', async () => {
    renderControls({ busy: true, busyAction: 'queue' })

    await expectShortcutTooltip('Queue message', 'Ctrl+↵')
  })
})

describe('wake-word ear visibility', () => {
  afterEach(() => {
    resetWakeWordState()
  })

  it('stays mounted during a busy agent turn', () => {
    applyWakeStatus({ available: true, enabled: true, listening: true, phrase: 'hey hermes' })
    renderControls({ busy: true, busyAction: 'stop' })

    expect(screen.getByLabelText('Wake word: "hey hermes" — listening')).toBeTruthy()
  })

  it('stays mounted (enabled in config) even when a start was refused', () => {
    applyWakeStatus({ available: true, enabled: true, listening: false, phrase: 'hey hermes' })
    // Transient refusal marks available false but enabled keeps it mounted.
    applyWakeStartResult({ hint: 'mic busy', reason: 'unavailable', started: false })
    renderControls()

    expect(screen.getByLabelText('Wake word: "hey hermes" — off')).toBeTruthy()
  })

  it('hides only when unavailable AND not enabled in config', () => {
    applyWakeStatus({ available: false, enabled: false, listening: false, phrase: 'hey hermes' })
    renderControls()

    expect(screen.queryByLabelText(/Wake word/)).toBeNull()
  })

  it('shows a disabled paused ear inside the voice-conversation pill', () => {
    applyWakeStatus({ available: true, enabled: true, listening: true, phrase: 'hey hermes' })
    renderControls({
      conversation: {
        active: true,
        level: 0,
        muted: false,
        onEnd: vi.fn(),
        onStart: vi.fn(),
        onStopTurn: vi.fn(),
        onToggleMute: vi.fn(),
        status: 'listening'
      }
    })

    const ear = screen.getByLabelText('Wake word: "hey hermes" — paused during voice chat')
    expect((ear as HTMLButtonElement).disabled).toBe(true)
  })
})
