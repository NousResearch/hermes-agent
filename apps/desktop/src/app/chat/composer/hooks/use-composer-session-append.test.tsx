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
    act(() => appendComposerToSessionDraft('session-a', '已有输入', []))
    mockComposerApi.setText.mockClear()
    act(() => appendComposerToSessionDraft('session-a', '区域标注', []))
    expect(mockComposerApi.setText).toHaveBeenLastCalledWith('已有输入\n\n区域标注')
  })

  it('does not repaint a composer mounted for another session', () => {
    render(<Probe sessionKey="session-b" />)
    mockComposerApi.setText.mockClear()
    act(() => appendComposerToSessionDraft('session-a', '不应进入 B', []))
    expect(mockComposerApi.setText).not.toHaveBeenCalled()
  })
})
