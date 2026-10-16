import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import { type PropsWithChildren } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatBarProps } from '@/app/chat/composer/types'
import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'

import { type ComposerTarget, requestComposerSubmitConfirmed } from './focus'
import { useComposerSubmit } from './hooks/use-composer-submit'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from './scope'

interface HarnessOptions {
  busy?: boolean
  inputDisabled?: boolean
  onSubmit?: ChatBarProps['onSubmit']
  paneVisible?: boolean
  scopeTarget?: ComposerTarget
  sessionKey?: string | null
  surfaceId: string
  draft?: string
}

function renderConfirmedComposer({
  busy = false,
  draft = '',
  inputDisabled = false,
  onSubmit = vi.fn(async () => true),
  paneVisible = true,
  scopeTarget = 'main',
  sessionKey = 'stored-session',
  surfaceId
}: HarnessOptions) {
  const draftRef = { current: draft }
  const draftScopeRef = { current: sessionKey }
  const editorRef = { current: window.document.createElement('div') }

  const clearDraft = vi.fn(() => {
    draftRef.current = ''
  })

  const loadIntoComposer = vi.fn()
  const stashAt = vi.fn()
  const onSteer = vi.fn(async () => true)
  const onSteerHidden = vi.fn(async () => true)

  const Wrapper = ({ children }: PropsWithChildren) => (
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: scopeTarget }}>
      <ComposerSurfaceProvider value={surfaceId}>
        <PaneVisibleContext.Provider value={paneVisible}>
          <div
            data-composer-surface-id={surfaceId}
            data-composer-target={scopeTarget}
            data-pane-hidden={paneVisible ? undefined : ''}
          >
            {children}
          </div>
        </PaneVisibleContext.Provider>
      </ComposerSurfaceProvider>
    </ComposerScopeProvider>
  )

  renderHook(
    () =>
      useComposerSubmit({
        activeQueueSessionKey: sessionKey,
        activeQueueSessionKeyRef: { current: sessionKey },
        attachments: [],
        busy,
        clearDraft,
        disabled: false,
        draftScopeRef,
        draftRef,
        drainNextQueued: vi.fn(async () => false),
        editorRef,
        exitQueuedEdit: vi.fn(() => false),
        focusInput: vi.fn(),
        inputDisabled,
        loadIntoComposer,
        onCancel: vi.fn(),
        onSteer,
        onSteerHidden,
        onSubmit,
        queueCurrentDraft: vi.fn(() => true),
        queueEdit: null,
        queuedPrompts: [],
        sessionId: `runtime-${surfaceId}`,
        setComposerText: vi.fn(),
        stashAt
      }),
    { wrapper: Wrapper }
  )

  return { clearDraft, draftRef, loadIntoComposer, onSteer, onSteerHidden, onSubmit, stashAt }
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('message-owned composer submission', () => {
  it('returns false when no composer claims the exact visible surface', async () => {
    render(<div data-composer-surface-id="orphan" data-composer-target="main" />)

    await expect(requestComposerSubmitConfirmed('interaction', 'main', 'orphan')).resolves.toBe(false)
  })

  it('does not accept a visible but unclaimed sibling surface', async () => {
    const composer = renderConfirmedComposer({ surfaceId: 'owned-surface' })
    render(<div data-composer-surface-id="other-surface" data-composer-target="main" />)

    await expect(requestComposerSubmitConfirmed('interaction', 'main', 'other-surface')).resolves.toBe(false)
    expect(composer.onSubmit).not.toHaveBeenCalled()
  })

  it.each([
    ['disabled', { inputDisabled: true }],
    ['busy', { busy: true }]
  ] as const)('fails closed when the exact composer is %s', async (_condition, options) => {
    const composer = renderConfirmedComposer({ surfaceId: 'guarded-surface', ...options })

    await expect(requestComposerSubmitConfirmed('interaction', 'main', 'guarded-surface')).resolves.toBe(false)
    expect(composer.onSubmit).not.toHaveBeenCalled()
    expect(composer.onSteer).not.toHaveBeenCalled()
    expect(composer.onSteerHidden).not.toHaveBeenCalled()
  })

  it('does not accept a surface after its pane is hidden', async () => {
    const composer = renderConfirmedComposer({ paneVisible: false, surfaceId: 'hidden-surface' })

    await expect(requestComposerSubmitConfirmed('interaction', 'main', 'hidden-surface')).resolves.toBe(false)
    expect(composer.onSubmit).not.toHaveBeenCalled()
  })

  it('routes a tile request only to that exact tile and never to the main composer', async () => {
    const main = renderConfirmedComposer({ scopeTarget: 'main', sessionKey: 'main-session', surfaceId: 'main-surface' })

    const tile = renderConfirmedComposer({
      scopeTarget: 'tile:origin',
      sessionKey: 'tile-session',
      surfaceId: 'origin-surface'
    })

    let accepted = false
    await act(async () => {
      accepted = await requestComposerSubmitConfirmed('interaction', 'tile:origin', 'origin-surface')
    })

    expect(accepted).toBe(true)
    expect(tile.onSubmit).toHaveBeenCalledExactlyOnceWith(
      'interaction',
      expect.objectContaining({ composerScope: 'tile-session', submission_id: expect.any(String) })
    )
    expect(main.onSubmit).not.toHaveBeenCalled()
  })

  it('waits for the originating gateway submit result before reporting acceptance', async () => {
    let resolveSubmit!: (accepted: boolean) => void

    const gatewaySubmit = new Promise<boolean>(resolve => {
      resolveSubmit = resolve
    })

    const onSubmit = vi.fn(() => gatewaySubmit)

    const composer = renderConfirmedComposer({
      draft: 'unfinished user draft',
      onSubmit,
      sessionKey: 'origin-session',
      surfaceId: 'origin-surface'
    })

    let settled = false

    const result = requestComposerSubmitConfirmed('interaction', 'main', 'origin-surface').then(value => {
      settled = true

      return value
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledExactlyOnceWith(
        'interaction',
        expect.objectContaining({ composerScope: 'origin-session', submission_id: expect.any(String) })
      )
    )
    expect(settled).toBe(false)

    await act(async () => resolveSubmit(true))
    await expect(result).resolves.toBe(true)
    expect(composer.draftRef.current).toBe('unfinished user draft')
    expect(composer.clearDraft).not.toHaveBeenCalled()
  })

  it('reports gateway rejection and leaves the unsent composer draft untouched', async () => {
    const onSubmit = vi.fn(async () => false)

    const composer = renderConfirmedComposer({
      draft: 'keep this unsent text',
      onSubmit,
      surfaceId: 'origin-surface'
    })

    await expect(requestComposerSubmitConfirmed('interaction', 'main', 'origin-surface')).resolves.toBe(false)
    expect(composer.draftRef.current).toBe('keep this unsent text')
    expect(composer.clearDraft).not.toHaveBeenCalled()
    expect(composer.loadIntoComposer).not.toHaveBeenCalled()
    expect(composer.stashAt).not.toHaveBeenCalled()
  })

  it('reports a rejected submit promise without clearing the original draft', async () => {
    const onSubmit = vi.fn(async () => {
      throw new Error('gateway rejected the request')
    })

    const composer = renderConfirmedComposer({
      draft: 'keep this on transport error',
      onSubmit,
      surfaceId: 'origin-surface'
    })

    await expect(requestComposerSubmitConfirmed('interaction', 'main', 'origin-surface')).resolves.toBe(false)
    expect(composer.draftRef.current).toBe('keep this on transport error')
    expect(composer.clearDraft).not.toHaveBeenCalled()
    expect(composer.loadIntoComposer).not.toHaveBeenCalled()
  })

  it('reports a synchronous submit throw without leaving the acknowledgement pending', async () => {
    const onSubmit = vi.fn(() => {
      throw new Error('synchronous gateway failure')
    })

    const composer = renderConfirmedComposer({
      draft: 'keep this after a synchronous failure',
      onSubmit,
      surfaceId: 'origin-surface'
    })

    await expect(requestComposerSubmitConfirmed('interaction', 'main', 'origin-surface')).resolves.toBe(false)
    expect(onSubmit).toHaveBeenCalledExactlyOnceWith(
      'interaction',
      expect.objectContaining({ composerScope: 'stored-session', submission_id: expect.any(String) })
    )
    expect(composer.draftRef.current).toBe('keep this after a synchronous failure')
    expect(composer.clearDraft).not.toHaveBeenCalled()
    expect(composer.loadIntoComposer).not.toHaveBeenCalled()
  })
})
