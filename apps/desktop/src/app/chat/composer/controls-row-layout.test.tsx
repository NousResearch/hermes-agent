import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { I18nProvider } from '@/i18n'

import { ComposerControls } from './controls'

vi.mock('./model-pill', () => ({ ModelPill: () => null }))

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { active: false, enabled: false }
}

function controlsRow() {
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
        hasComposerPayload={true}
        minimal
        onDictate={vi.fn()}
        onQueue={vi.fn()}
        onToggleAutoSpeak={vi.fn()}
        state={state}
        voiceStatus="idle"
      />
    </I18nProvider>
  )

  return container.firstElementChild as HTMLElement
}

afterEach(() => {
  cleanup()
  delete window.document.documentElement.dataset.hermesDesktopHost
})

// Native Desktop lays out the minimal controls row as on main: packed from the
// start, on one line.
it('keeps the minimal controls row on one line, packed from the start, in native Desktop', () => {
  const row = controlsRow()

  expect(row.className).not.toContain('justify-end')
  expect(row.style.flexWrap).toBe('')
})

// The browser host's touch-size controls can outgrow a minimal row: they pack
// to the end and wrap onto a second line instead of clipping Send.
it('packs the minimal controls row to the end and lets it wrap in browser-hosted Desktop', () => {
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  const row = controlsRow()

  expect(row.className).toContain('justify-end')
  expect(row.style.flexWrap).toBe('wrap')
})
