import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { I18nProvider } from '@/i18n'
import type { SetupBeatToolResult } from '@/store/setup-beat'

import { SetupBeatTool } from './setup-beat-tool'

const view: SessionView = { ...PRIMARY_SESSION_VIEW, $runtimeId: atom('beat-session'), $storedId: atom('stored-beat') }

function renderRow(result: SetupBeatToolResult | undefined) {
  // SAFETY: SetupBeatTool reads only result (and the session's requests when result is absent).
  const row = { args: {}, result, toolCallId: 'call-1', toolName: 'setup_beat' } as ToolCallMessagePartProps

  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SessionViewProvider value={view}>
        <SetupBeatTool {...row} />
      </SessionViewProvider>
    </I18nProvider>
  )
}

describe('SetupBeatTool', () => {
  afterEach(() => {
    cleanup()
  })

  it("draws the words typed while the card waited as the user's bubble", () => {
    renderRow({ beat: 'theme', said: 'look around and suggest one for me :D' })

    expect(
      screen.getByText('look around and suggest one for me :D').closest('[data-slot="typed-answer"]')
    ).not.toBeNull()
  })

  it('draws nothing for a refusal meant for the model', () => {
    const { container } = renderRow({ error: 'Apply the layout with set_theme; its result settles this beat.' })

    expect(container.innerHTML).toBe('')
  })

  it('says a card answer was not accepted, and a row left without a card once the turn ended was not answered', () => {
    renderRow({ beat: 'layout', error: 'The card sent an answer this beat does not accept.' })
    expect(screen.getByText('Answer not accepted')).not.toBeNull()
    cleanup()

    renderRow(undefined)
    expect(screen.getByText('No answer')).not.toBeNull()
  })
})
