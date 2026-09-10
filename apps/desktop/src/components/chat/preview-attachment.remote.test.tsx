import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { $previewTabs, $previewTarget } from '@/store/preview'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import type { SessionOwnerRoute } from '@/store/session-request-router'
import { clearAllSessionStates } from '@/store/session-states'

import { PreviewAttachment } from './preview-attachment'

const saveGatewayFile = vi.fn(async () => ({ saved: true }))

function view(
  storedId: string,
  ownerRoute: SessionOwnerRoute = { connectionId: 'remote-gateway', mode: 'remote', profile: 'assistant' }
): SessionView {
  return {
    kind: 'primary',
    $awaitingResponse: atom(false),
    $busy: atom(false),
    $cwd: atom('/srv/work'),
    $fast: atom(false),
    $lastVisibleIsUser: atom(false),
    $messages: atom([]),
    $messagesEmpty: atom(true),
    $model: atom(''),
    $ownerRoute: atom(ownerRoute),
    $provider: atom(''),
    $reasoningEffort: atom(''),
    $runtimeId: atom(null),
    $storedId: atom(storedId),
    $turnStartedAt: atom(null)
  }
}

describe('PreviewAttachment remote session routing', () => {
  beforeEach(() => {
    saveGatewayFile.mockClear()
    vi.stubGlobal('hermesDesktop', { saveGatewayFile })
    $previewTabs.set([])
    $connection.set({ connectionId: 'local-device', mode: 'local', profile: 'default' } as never)
    setSessionOwnerHint('remote-session', {
      connectionId: 'remote-gateway',
      mode: 'remote',
      profile: 'assistant'
    })
  })

  afterEach(() => {
    cleanup()
    clearAllSessionStates()
    _resetSessionOwnerHintsForTests()
    $connection.set(null)
    vi.unstubAllGlobals()
  })

  it('downloads through the backend and profile that own the transcript', async () => {
    render(
      <SessionViewProvider value={view('remote-session')}>
        <PreviewAttachment target="/srv/media/movie.mp4" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Download' }))

    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: 'remote-gateway',
        path: '/srv/media/movie.mp4',
        profile: 'assistant',
        sessionId: 'remote-session',
        suggestedName: 'movie.mp4'
      })
    )
  })

  it('toggles a local-owner preview using its persisted raw source', async () => {
    render(
      <SessionViewProvider
        value={view('local-session', {
          connectionId: 'local-device',
          mode: 'local',
          profile: 'default'
        })}
      >
        <PreviewAttachment target="/tmp/report.md" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    await waitFor(() => expect($previewTarget.get()?.source).toBe('/tmp/report.md'))

    fireEvent.click(screen.getByRole('button', { name: 'Hide' }))
    await waitFor(() => expect($previewTabs.get()).toHaveLength(0))
  })

  it('previews remote video through the transcript owner stream', async () => {
    render(
      <SessionViewProvider value={view('remote-session')}>
        <PreviewAttachment target="/srv/media/movie.mp4" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))

    await waitFor(() =>
      expect($previewTarget.get()).toMatchObject({
        kind: 'url',
        label: 'movie.mp4',
        source: 'gateway:remote-gateway:assistant::remote-session:%2Fsrv%2Fmedia%2Fmovie.mp4',
        url: 'hermes-media://remote/%2Fsrv%2Fmedia%2Fmovie.mp4?connectionId=remote-gateway&profile=assistant&sessionId=remote-session'
      })
    )
  })

  it('loads a remote image through the transcript owner instead of the foreground', async () => {
    render(
      <SessionViewProvider value={view('remote-session')}>
        <PreviewAttachment target="/srv/media/frame.png" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))

    await waitFor(() => expect($previewTarget.get()?.previewKind).toBe('image'))
    expect($previewTarget.get()).toMatchObject({
      ownerRoute: {
        connectionId: 'remote-gateway',
        mode: 'remote',
        profile: 'assistant'
      },
      previewKind: 'image',
      source: 'gateway:remote-gateway:assistant::remote-session:%2Fsrv%2Fmedia%2Fframe.png',
      transient: true
    })
  })

  it('keeps the Desktop route profile separate from the backend target profile', async () => {
    render(
      <SessionViewProvider
        value={view('remote-session', {
          connectionId: 'remote-gateway',
          mode: 'remote',
          profile: 'desktop-alias',
          targetProfile: 'backend-profile'
        })}
      >
        <PreviewAttachment target="/srv/media/movie.mp4" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Download' }))

    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: 'remote-gateway',
        path: '/srv/media/movie.mp4',
        profile: 'desktop-alias',
        sessionId: 'remote-session',
        suggestedName: 'movie.mp4',
        targetProfile: 'backend-profile'
      })
    )
  })

  it('does not treat the same path on two gateways as the same preview', async () => {
    const first = render(
      <SessionViewProvider value={view('remote-session')}>
        <PreviewAttachment target="/srv/media/movie.mp4" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    await waitFor(() =>
      expect($previewTarget.get()?.source).toBe(
        'gateway:remote-gateway:assistant::remote-session:%2Fsrv%2Fmedia%2Fmovie.mp4'
      )
    )
    first.unmount()

    render(
      <SessionViewProvider
        value={view('remote-session', {
          connectionId: 'second-gateway',
          mode: 'remote',
          profile: 'assistant'
        })}
      >
        <PreviewAttachment target="/srv/media/movie.mp4" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    await waitFor(() =>
      expect($previewTarget.get()?.source).toBe(
        'gateway:second-gateway:assistant::remote-session:%2Fsrv%2Fmedia%2Fmovie.mp4'
      )
    )
  })

  it('does not treat the same path in two sessions as the same preview', async () => {
    const first = render(
      <SessionViewProvider value={view('remote-session')}>
        <PreviewAttachment target="/srv/media/movie.mp4" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    await waitFor(() =>
      expect($previewTarget.get()?.source).toBe(
        'gateway:remote-gateway:assistant::remote-session:%2Fsrv%2Fmedia%2Fmovie.mp4'
      )
    )
    first.unmount()

    render(
      <SessionViewProvider value={view('second-session')}>
        <PreviewAttachment target="/srv/media/movie.mp4" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    await waitFor(() =>
      expect($previewTarget.get()?.source).toBe(
        'gateway:remote-gateway:assistant::second-session:%2Fsrv%2Fmedia%2Fmovie.mp4'
      )
    )
  })

  it('uses the view runtime owner when stored session ids collide across gateways', async () => {
    setSessionOwnerHint('remote-session', {
      connectionId: 'other-gateway',
      mode: 'remote',
      profile: 'assistant'
    })
    render(
      <SessionViewProvider value={view('remote-session')}>
        <PreviewAttachment target="/srv/media/movie.mp4" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Download' }))

    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: 'remote-gateway',
        path: '/srv/media/movie.mp4',
        profile: 'assistant',
        sessionId: 'remote-session',
        suggestedName: 'movie.mp4'
      })
    )
  })
})
