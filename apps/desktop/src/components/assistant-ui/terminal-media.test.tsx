import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { setSessionOwnerResolver } from '@/api/client'
import { artifactImageSrc, collectArtifactsForSession } from '@/app/artifacts/artifact-utils'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { SessionViewProvider } from '@/app/chat/session-view'
import { $previewTabs } from '@/store/preview'
import { $connection } from '@/store/session'

import { MarkdownImage, MessageTextContent } from './markdown-text'

const original = window.hermesDesktop
const path = '/workspace/result.png'
const bytes = 'data:image/png;base64,dGVybWluYWw='

afterEach(() => {
  cleanup()
  $previewTabs.set([])
  window.hermesDesktop = original
  $connection.set(null)
  setSessionOwnerResolver(null)
  vi.restoreAllMocks()
})

it.each(['local', 'remote'])(
  'a %s Hermes chat reads terminal images through its owner, never a same-path host file',
  async mode => {
    const api = vi.fn(async () => ({ dataUrl: bytes }))
    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,SE9TVA==')
    window.hermesDesktop = { api, readFileDataUrl } as never
    $connection.set({ connectionId: 'foreground', mode, profile: 'foreground' } as never)
    const connectionId = mode === 'local' ? 'local' : 'ssh-owner'
    setSessionOwnerResolver(id =>
      id === 'stored-terminal' ? { connectionId, profile: 'bot', targetProfile: 'backend-bot' } : null
    )
    const view = { $runtimeId: atom('runtime-terminal'), $storedId: atom('stored-terminal') }
    render(
      <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId, profile: 'bot' }}>
        <SessionViewProvider value={view as never}>
          <MarkdownImage alt="terminal result" src={path} />
        </SessionViewProvider>
      </ComposerScopeProvider>
    )
    expect((await screen.findByRole('img', { name: 'terminal result' })).getAttribute('src')).toBe(bytes)
    expect(api).toHaveBeenCalledWith({
      connectionId,
      profile: 'backend-bot',
      path: '/api/fs/read-data-url?path=%2Fworkspace%2Fresult.png&session_id=stored-terminal'
    })
    expect(readFileDataUrl).not.toHaveBeenCalled()
  }
)

it.each(['gateway-B', 'gateway-A'])(
  'does not borrow the equal session ID owned by another profile on %s',
  async resolverConnection => {
    const api = vi.fn(async () => ({ dataUrl: bytes }))
    window.hermesDesktop = { api } as never
    $connection.set({ connectionId: 'gateway-B', mode: 'remote', profile: 'profile-B' } as never)
    setSessionOwnerResolver(() => ({ connectionId: resolverConnection, profile: 'profile-B' }))
    const view = { $runtimeId: atom(null), $storedId: atom('shared-session') }
    render(
      <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId: 'gateway-A', profile: 'profile-A' }}>
        <SessionViewProvider value={view as never}>
          <MarkdownImage alt="captured result" src={path} />
        </SessionViewProvider>
      </ComposerScopeProvider>
    )
    await screen.findByRole('img', { name: 'captured result' })
    expect(api).toHaveBeenCalledWith({
      connectionId: 'gateway-A', profile: 'profile-A',
      path: '/api/fs/read-data-url?path=%2Fworkspace%2Fresult.png&session_id=shared-session'
    })
  }
)

it.each(['gateway-B', 'gateway-A'])(
  'keeps a document composer owner when an equal session resolves on %s',
  async resolverConnection => {
    const api = vi.fn(async () => ({ text: 'captured document', path: '/workspace/note.md' }))
    const saveGatewayFile = vi.fn(async () => ({ saved: false }))
    window.hermesDesktop = { api, saveGatewayFile } as never
    $connection.set({ connectionId: 'gateway-B', mode: 'remote', profile: 'profile-B' } as never)
    setSessionOwnerResolver(() => ({ connectionId: resolverConnection, profile: 'profile-B' }))
    const view = { $runtimeId: atom(null), $storedId: atom('shared-session'), $cwd: atom('/workspace') }
    render(
      <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, connectionId: 'gateway-A', profile: 'profile-A' }}>
        <SessionViewProvider value={view as never}>
          <MessageTextContent text="MEDIA:/workspace/note.md" />
        </SessionViewProvider>
      </ComposerScopeProvider>
    )
    fireEvent.click(await screen.findByRole('button', { name: 'Download' }))
    await waitFor(() => expect(saveGatewayFile).toHaveBeenCalledWith(expect.objectContaining({
      connectionId: 'gateway-A', profile: 'profile-A', sessionId: 'shared-session'
    })))
    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    await waitFor(() => expect(api).toHaveBeenCalledWith({
      connectionId: 'gateway-A', profile: 'profile-A',
      path: '/api/fs/read-text?path=%2Fworkspace%2Fnote.md&session_id=shared-session'
    }))
  }
)

it('keeps artifact indexing lazy and preserves the image session/profile at display admission', async () => {
  const api = vi.fn(async () => ({ dataUrl: bytes }))
  window.hermesDesktop = { api, readFileDataUrl: vi.fn() } as never
  $connection.set({ connectionId: 'other', mode: 'local', profile: 'other' } as never)

  const [artifact] = collectArtifactsForSession({ id: 'artifact-session', profile: 'artifact-profile' } as never, [
    { role: 'assistant', content: `MEDIA:${path}` }
  ])

  expect(artifact.value).toBe(path)
  expect(api).not.toHaveBeenCalled()
  await expect(
    artifactImageSrc(
      artifact.value,
      { connectionId: 'artifact-gateway', profile: artifact.profile },
      artifact.sessionId
    )
  ).resolves.toBe(bytes)
  expect(api).toHaveBeenCalledWith({
    connectionId: 'artifact-gateway',
    profile: 'artifact-profile',
    path: '/api/fs/read-data-url?path=%2Fworkspace%2Fresult.png&session_id=artifact-session'
  })
})

it.each([
  ['~/.hermes/profiles/bot/USER.md', '~/.hermes/profiles/bot/USER.md'],
  ['~\\.hermes\\profiles\\bot\\USER.md', '~\\.hermes\\profiles\\bot\\USER.md'],
  ['/workspace/note.md', '/workspace/note.md'],
  ['./note.md', '/workspace/note.md']
])('a delivered terminal document %s previews consistently with its gateway download', async (path, previewPath) => {
  const api = vi.fn(async () => ({ text: 'terminal document', path: '/workspace/note.md' }))
  const saveGatewayFile = vi.fn(async () => ({ saved: true }))

  const normalizePreviewTarget = vi.fn(async () => ({
    kind: 'file',
    path: '/workspace/note.md',
    previewKind: 'missing'
  }))

  window.hermesDesktop = { api, normalizePreviewTarget, saveGatewayFile } as never
  $connection.set({ connectionId: 'local', mode: 'local', profile: 'foreground' } as never)
  setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'bot' }))
  const view = { $runtimeId: atom(null), $storedId: atom('document-session'), $cwd: atom('/workspace') }
  render(
    <SessionViewProvider value={view as never}>
      <MessageTextContent text={`MEDIA:${path}`} />
    </SessionViewProvider>
  )
  fireEvent.click(await screen.findByRole('button', { name: 'Download' }))
  await waitFor(() =>
    expect(saveGatewayFile).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'local',
        profile: 'bot',
        sessionId: 'document-session',
        path
      })
    )
  )
  expect(normalizePreviewTarget).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
  await waitFor(() =>
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'local',
        profile: 'bot',
        path: `/api/fs/read-text?path=${encodeURIComponent(previewPath)}&session_id=document-session`
      })
    )
  )
  expect(normalizePreviewTarget).not.toHaveBeenCalled()
})
