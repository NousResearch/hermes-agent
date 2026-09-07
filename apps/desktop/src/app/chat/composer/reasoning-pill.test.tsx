import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { setDefaultReasoningEffort } from '@/store/session'

import { ReasoningPill } from './reasoning-pill'
import type { ChatBarState } from './types'

const modelState = (over: Partial<ChatBarState['model']> = {}): ChatBarState['model'] => ({
  canSwitch: true,
  model: 'gpt-6',
  provider: 'openai',
  reasoningMenuContent: <div />,
  supportsReasoning: true,
  ...over
})

afterEach(() => {
  cleanup()
  setDefaultReasoningEffort('')
})

// The pill must show the effort the runtime will actually use: the session's
// explicit choice, else the profile's configured default — never a hardcoded
// medium when the profile inherited something else.
describe('ReasoningPill effort resolution', () => {
  it('shows the profile default when the session has no explicit effort', () => {
    setDefaultReasoningEffort('high')

    render(<ReasoningPill disabled={false} model={modelState()} />)

    expect(screen.getByText('High')).toBeTruthy()
  })

  it('shows the session effort over the profile default', () => {
    setDefaultReasoningEffort('high')

    render(<ReasoningPill disabled={false} model={modelState({ reasoningEffort: 'low' })} />)

    expect(screen.getByText('Low')).toBeTruthy()
  })

  it('falls back to the built-in default when nothing is configured', () => {
    render(<ReasoningPill disabled={false} model={modelState()} />)

    expect(screen.getByText('Med')).toBeTruthy()
  })
})
