import { cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { I18nProvider } from '@/i18n'
import { $activeSessionId } from '@/store/session'
import { $subagentsBySession, clearSessionSubagents, upsertSubagent } from '@/store/subagents'
import { stubResizeObserver } from '@/test/jsdom'

import { DelegateTool } from './delegate'

stubResizeObserver()

const SESSION = 'sess-delegate'

function view(): SessionView {
  return {
    ...({} as SessionView),
    $runtimeId: atom<null | string>(SESSION),
    kind: 'primary'
  }
}

function renderCard() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SessionViewProvider value={view()}>
        <DelegateTool args={{ tasks: [{ goal: 'Inspect visualOutcome' }] }} result={undefined} toolCallId="call-7" />
      </SessionViewProvider>
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  clearSessionSubagents(SESSION)
  $subagentsBySession.set({})
  $activeSessionId.set(null)
})

describe('delegate card fade', () => {
  // Scaffold opacity opens a stacking context. The activity ticker is a
  // transformed reel clipped to one line. When the mark sat on the wrapper
  // around both, Chromium stopped clipping the reel and every old activity
  // line painted through the current one at 0.67 (or worse, nested). The
  // mark belongs on the goal row. The ticker stays outside it.
  it('does not put the activity ticker inside a scaffold fade', () => {
    $activeSessionId.set(SESSION)
    upsertSubagent(
      SESSION,
      {
        goal: 'Inspect visualOutcome',
        status: 'running',
        subagent_id: 'delegate-tool:call-7:0',
        task_index: 0,
        text: 'Thinking about the catalog'
      },
      true,
      'subagent.thinking'
    )
    upsertSubagent(
      SESSION,
      {
        subagent_id: 'delegate-tool:call-7:0',
        task_index: 0,
        tool_name: 'terminal',
        preview: 'from pathlib import Path'
      },
      false,
      'subagent.tool'
    )

    const { container } = renderCard()
    const ticker = container.querySelector('[data-tool-ticker]')
    const marked = container.querySelector('[data-conversation-scaffold]')

    expect(ticker).not.toBeNull()
    expect(marked).not.toBeNull()
    expect(ticker?.closest('[data-conversation-scaffold]')).toBeNull()
    expect(container.querySelector('[data-delegate-card]')?.hasAttribute('data-conversation-scaffold')).toBe(false)
  })
})
