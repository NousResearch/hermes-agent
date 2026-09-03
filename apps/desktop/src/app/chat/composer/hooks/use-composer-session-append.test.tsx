import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { appendComposerToSessionDraft, clearSessionDraft, mainComposerScope } from '@/store/composer'

import { useComposerDraft } from './use-composer-draft'
import type { QueueEditState } from '../composer-utils'

const mockComposerApi = { setText: vi.fn() }

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => mockComposerApi }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } }),
  useComposerRuntime: () => ({
    getState: () => ({ text: '' }),
    subscribe: () => () => undefined
  })
}))

function Probe({ sessionKey }: { sessionKey: string }) {
  useComposerDraft({
    activeQueueSessionKey: sessionKey,
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null as QueueEditState | null },
    sessionId: sessionKey
  })
  return null
}

describe('useComposerDraft session-scoped external append', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    clearSessionDraft('session-a')
    clearSessionDraft('session-b')
    mockComposerApi.setText.mockClear()
  })

  it('appends into the matching mounted composer without replacing existing draft', () => {
    render(<Probe sessionKey="session-a" />)
    act(() => appendComposerToSessionDraft('session-a', 'existing input', []))
    mockComposerApi.setText.mockClear()
    act(() => appendComposerToSessionDraft('session-a', 'region annotation', []))
    expect(mockComposerApi.setText).toHaveBeenLastCalledWith('existing input\n\nregion annotation')
  })

  it('does not repaint a composer mounted for another session', () => {
    render(<Probe sessionKey="session-b" />)
    mockComposerApi.setText.mockClear()
    act(() => appendComposerToSessionDraft('session-a', 'must not enter B', []))
    expect(mockComposerApi.setText).not.toHaveBeenCalled()
  })
})
