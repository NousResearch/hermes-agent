import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { en } from '@/i18n/en'
import { $subagentsBySession } from '@/store/subagents'

import { DelegateTool } from './delegate'

const SESSION_ID = 'session-1'

/** Minimal view: only the runtime id the card reads to slice the store. */
function sessionView(): SessionView {
  return {
    ...({} as SessionView),
    $runtimeId: atom<null | string>(SESSION_ID),
    kind: 'primary'
  }
}

function renderCard(result: unknown) {
  render(
    <SessionViewProvider value={sessionView()}>
      <DelegateTool args={{ tasks: [{ goal: 'Research Cursor' }] }} result={result} toolCallId="call-1" />
    </SessionViewProvider>
  )
}

const copy = en.assistant.tool

afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
})

describe('delegate card status glyph', () => {
  // The regression: every settled non-`failed` result used to collapse to
  // lifecycle `completed` and render a green "Done" check — so a child that
  // returned unverified or half-finished work read as an accepted task.
  it('does not label a completed lifecycle with an unverified outcome as done', () => {
    renderCard({ results: [{ status: 'completed', outcome: 'unverified', summary: 'here is what I found' }] })

    expect(screen.getByLabelText(copy.statusUnverified)).toBeTruthy()
    expect(screen.queryByLabelText(copy.statusDone)).toBeNull()
  })

  it('marks a partial outcome as partial, not done', () => {
    renderCard({ results: [{ status: 'completed', outcome: 'partial', summary: 'got halfway' }] })

    expect(screen.getByLabelText(copy.statusPartial)).toBeTruthy()
    expect(screen.queryByLabelText(copy.statusDone)).toBeNull()
  })

  it('still marks a proven failure as an error, whichever side proves it', () => {
    renderCard({ results: [{ status: 'completed', outcome: 'failed', summary: '' }] })

    expect(screen.getByLabelText(copy.statusError)).toBeTruthy()
  })

  it('does not read a lifecycle-only result envelope as success either', () => {
    // An envelope predating the `outcome` field says the loop ended and nothing
    // more. Absence of a logical result is not evidence of one.
    renderCard({ results: [{ status: 'completed', summary: 'legacy backend' }] })

    expect(screen.getByLabelText(copy.statusUnverified)).toBeTruthy()
    expect(screen.queryByLabelText(copy.statusDone)).toBeNull()
  })
})
