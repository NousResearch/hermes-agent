import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { composerHost } from '@/sdk/composer'
import {
  clearSessionDraft,
  freshDraftScope,
  mainComposerScope,
  patchMainComposerAttachmentOccurrence
} from '@/store/composer'
import { $selectedStoredSessionId } from '@/store/session'

import { useComposerActions } from '../hooks/use-composer-actions'

import { markActiveComposer } from './focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from './scope'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

const png = new Blob(
  [
    Uint8Array.from(
      atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGOQi1gAAAGuARekSxIOAAAAAElFTkSuQmCC'),
      char => char.charCodeAt(0)
    )
  ],
  { type: 'image/png' }
)

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: 'Attach' },
  voice: { active: false, enabled: false }
}

const submit = vi.fn(async () => true)
const originalBridge = window.hermesDesktop

function Composer({
  sessionKey,
  sessionId,
  target
}: {
  sessionKey: string | null
  sessionId?: string
  target: string
}) {
  const scope = {
    add: mainComposerScope.add,
    remove: mainComposerScope.remove,
    target,
    update: mainComposerScope.update,
    updateIfCurrent: patchMainComposerAttachmentOccurrence
  }

  const actions = useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn(), scope })

  return (
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target }}>
      <ComposerSurfaceProvider value="test-surface">
        <div data-composer-surface-id="test-surface" data-composer-target={target}>
          <ChatBar
            busy={false}
            disabled={false}
            gateway={null}
            onAttachImageBlob={actions.attachImageBlob}
            onCancel={vi.fn()}
            onSubmit={submit}
            queueSessionKey={sessionKey}
            sessionId={sessionId}
            state={state}
          />
        </div>
      </ComposerSurfaceProvider>
    </ComposerScopeProvider>
  )
}

function Harness({
  sessionKey = null,
  sessionId,
  target = 'main'
}: {
  sessionKey?: string | null
  sessionId?: string
  target?: string
}) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <Composer sessionId={sessionId} sessionKey={sessionKey} target={target} />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

const mockStaging = (saveImageBuffer: NonNullable<typeof window.hermesDesktop>['saveImageBuffer']) => {
  const trashPath = vi.fn(async (_path: string) => true)
  window.hermesDesktop = { ...originalBridge, saveImageBuffer, trashPath }
  vi.stubGlobal(
    'createImageBitmap',
    vi.fn(async () => ({ width: 1, height: 1, close: vi.fn() }))
  )
  vi.stubGlobal(
    'URL',
    class extends URL {
      static createObjectURL = () => 'blob:annotation'
      static revokeObjectURL = () => undefined
    }
  )
  markActiveComposer('main')

  return trashPath
}

it('does not mistake retained disconnected-session IDs for the mounted fresh draft', () => {
  $selectedStoredSessionId.set('other-chat')
  render(<Harness sessionId="other-chat" sessionKey={freshDraftScope()} />)

  expect(composerHost.captureDraft('other-chat')).toBeNull()
  expect(composerHost.captureDraft('new')).not.toBeNull()
})

it('keeps a split draft pinned across an unrelated rerender that replaces the attach callback', async () => {
  let finish!: (path: string) => void

  const stage = vi.fn(
    () =>
      new Promise<string>(resolve => {
        finish = resolve
      })
  )

  mockStaging(stage)
  const { rerender } = render(<Harness sessionKey="other-chat" target="tile:other-chat" />)
  const address = composerHost.captureDraft('other-chat')
  expect(address).not.toBeNull()
  const uploading = composerHost.attachImage(address!, png)
  await waitFor(() => expect(stage).toHaveBeenCalledOnce())
  rerender(<Harness sessionKey="other-chat" target="tile:other-chat" />)
  await act(async () => {
    finish('/staged/tile.png')
    expect(await uploading).toBe(true)
  })
})

it('requires PNG decoding to succeed before staging', async () => {
  const stage = vi.fn(async () => '/staged/corrupt.png')
  mockStaging(stage)
  vi.stubGlobal('createImageBitmap', vi.fn().mockRejectedValue(new Error('decode failed')))
  render(<Harness />)

  expect(await composerHost.attachImage(composerHost.captureDraft('new')!, png)).toBe(false)
  expect(stage).not.toHaveBeenCalled()
})

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  clearSessionDraft(freshDraftScope())
  clearSessionDraft('other-chat')
  $selectedStoredSessionId.set(null)
  window.hermesDesktop = originalBridge
  vi.unstubAllGlobals()
  submit.mockClear()
})

it('acknowledges an image added to the same fresh draft without editing or submitting text', async () => {
  const stage = vi.fn(async () => '/staged/sketch.png')
  mockStaging(stage)
  const { getByText } = render(<Harness />)
  const address = composerHost.captureDraft('new')
  expect(address).not.toBeNull()
  await act(async () => {
    expect(await composerHost.setDraft('new', 'Keep this text')).toBe(true)
    expect(await composerHost.attachImage(address!, png, { filename: 'sketch.png' })).toBe(true)
  })
  expect(stage).toHaveBeenCalledExactlyOnceWith(new Uint8Array(await png.arrayBuffer()), '.png', 'sketch.png')
  expect(getByText('sketch.png')).toBeTruthy()
  expect(await composerHost.getDraft('new')).toBe('Keep this text')
  expect(submit).not.toHaveBeenCalled()
})

it('rejects an in-flight image when the composer changes drafts during staging', async () => {
  let finish!: (path: string) => void

  const stage = vi.fn(
    () =>
      new Promise<string>(resolve => {
        finish = resolve
      })
  )

  const trash = mockStaging(stage)
  const { rerender } = render(<Harness />)
  const address = composerHost.captureDraft()
  expect(address).not.toBeNull()
  const uploading = composerHost.attachImage(address!, png)
  await waitFor(() => expect(stage).toHaveBeenCalledOnce())
  act(() => {
    $selectedStoredSessionId.set('other-chat')
    rerender(<Harness sessionKey="other-chat" />)
  })
  await act(async () => {
    finish('/staged/late.png')
    expect(await uploading).toBe(false)
  })
  expect(trash).toHaveBeenCalledExactlyOnceWith('/staged/late.png')
  expect(mainComposerScope.$attachments.get()).toEqual([])
  expect(composerHost.captureDraft('new')).toBeNull()
  expect(composerHost.captureDraft('other-chat')).not.toBeNull()
  expect(await composerHost.attachImage(address!, png)).toBe(false)
  expect(stage).toHaveBeenCalledOnce()
})

it('invalidates the old handle after a batched round trip without disabling the current draft', async () => {
  const stage = vi.fn(async () => '/staged/new.png')
  mockStaging(stage)
  render(<Harness />)
  const address = composerHost.captureDraft('new')!

  act(() => {
    $selectedStoredSessionId.set('other-chat')
    expect(composerHost.captureDraft('other-chat')).toBeNull()
    $selectedStoredSessionId.set(null)
  })

  expect(await composerHost.attachImage(address, png)).toBe(false)
  expect(stage).not.toHaveBeenCalled()
  await act(async () => {
    const current = composerHost.captureDraft('new')
    expect(current).not.toBeNull()
    expect(await composerHost.attachImage(current!, png)).toBe(true)
  })
})

it('rejects non-image bytes with a misleading image MIME before staging', async () => {
  const stage = vi.fn(async () => '/staged/wrong.png')
  mockStaging(stage)
  render(<Harness />)
  const address = composerHost.captureDraft('new')!
  const notAnImage = new Blob(['This is not an image.'], { type: 'image/png' })

  expect(await composerHost.attachImage(address, notAnImage)).toBe(false)
  expect(stage).not.toHaveBeenCalled()
})
