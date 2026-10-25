import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { LocalFilePreview } from '@/app/chat/right-rail/preview-file'
import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { desktopFsCacheKey } from '@/lib/desktop-fs'
import { $previewTabs, openPreview, type PreviewTarget } from '@/store/preview'
import { $connection } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { PreviewAttachment } from './preview-attachment'

const owner = { connectionId: 'backend-b', profile: 'writer' }
const originalDesktop = window.hermesDesktop
const api = vi.fn()
let backendText = ''
const saveGatewayFile = vi.fn(async () => ({ saved: true }))

vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre>
}))

beforeEach(() => {
  vi.clearAllMocks()
  backendText = '# Backend B\n\n[linked note](./linked.md)'
  $previewTabs.set([])
  $connection.set({ connectionId: 'backend-a', mode: 'remote', profile: 'default' } as never)
  setApiRequestConnection('backend-a')
  setApiRequestProfile('default')
  $sessionTiles.set([{ ownerRoute: owner, storedSessionId: 'session-b' }] as never)
  window.hermesDesktop = { api, saveGatewayFile, normalizePreviewTarget: vi.fn() } as never
  api.mockImplementation(async request => {
    expect(request).toMatchObject(owner)

    if (request.path.startsWith('/api/fs/read-text?')) {
      return { text: backendText, binary: false, byteSize: 45 }
    }

    if (request.path.startsWith('/api/fs/read-data-url?')) {
      return { dataUrl: 'data:application/pdf;base64,JVBERi0xLjc=' }
    }

    if (request.path.startsWith('/api/fs/git-root?')) {
      return { root: null }
    }

    if (request.path === '/api/fs/write-text') {
      return { path: '/workspace/report.md' }
    }

    throw new Error(`Unexpected request: ${request.path}`)
  })
})

afterEach(() => {
  cleanup()
  window.hermesDesktop = originalDesktop
  $connection.set(null)
  $sessionTiles.set([])
  $previewTabs.set([])
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

it.each(['pdf', 'md'])(
  'downloads and reads a %s attachment from its tile backend while another backend is active',
  async extension => {
    const file = `/workspace/report.${extension}`

    const view: SessionView = {
      ...PRIMARY_SESSION_VIEW,
      $cwd: atom('/workspace'),
      $storedId: atom('session-b'),
      $runtimeId: atom('runtime-b'),
      kind: 'tile'
    }

    const mounted = render(
      <SessionViewProvider value={view}>
        <PreviewAttachment target={file} />
      </SessionViewProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Download' }))
    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith(
        expect.objectContaining({
          ...owner,
          sessionId: 'session-b',
          path: file
        })
      )
    )
    fireEvent.click(screen.getByRole('button', { name: 'Open preview' }))
    await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
    const target = $previewTabs.get()[0]!.target
    expect(target.fileOwner).toEqual(owner)
    expect(window.hermesDesktop!.normalizePreviewTarget).not.toHaveBeenCalled()
    mounted.unmount()
    const createObjectURL = vi.fn(() => 'blob:backend-b-pdf')
    vi.spyOn(URL, 'createObjectURL').mockImplementation(createObjectURL)
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    render(<LocalFilePreview reloadKey={0} target={target} />)

    if (extension === 'pdf') {
      await waitFor(() => expect(screen.getByTitle('report.pdf').getAttribute('src')).toBe('blob:backend-b-pdf'))
      expect(createObjectURL).toHaveBeenCalledWith(expect.objectContaining({ type: 'application/pdf' }))
    } else {
      await screen.findByRole('heading', { name: 'Backend B' })
      fireEvent.click(screen.getByRole('link', { name: 'linked note' }))
      await waitFor(() =>
        expect(
          $previewTabs
            .get()
            .some(
              tab =>
                tab.target.path === '/workspace/linked.md' && tab.target.fileOwner?.connectionId === owner.connectionId
            )
        ).toBe(true)
      )
    }
  }
)

it('keeps same-path preview tabs and filesystem caches separate for two backends', () => {
  const target: PreviewTarget = {
    kind: 'file',
    label: 'report.md',
    path: '/workspace/report.md',
    previewKind: 'text',
    source: '/workspace/report.md',
    url: 'file:///workspace/report.md',
    fileOwner: owner
  }

  openPreview(target)
  openPreview({ ...target, fileOwner: { ...owner, connectionId: 'backend-a' } })
  expect($previewTabs.get()).toHaveLength(2)
  expect(desktopFsCacheKey(null, owner)).not.toBe(desktopFsCacheKey(null, { ...owner, connectionId: 'backend-a' }))
  expect(desktopFsCacheKey($connection.get(), owner)).toBe(desktopFsCacheKey(null, owner))
})

it('checks conflicts and saves Markdown on the retained backend owner', async () => {
  const descriptors = Object.getOwnPropertyDescriptors(Range.prototype)
  Object.defineProperties(Range.prototype, {
    getClientRects: { configurable: true, value: () => [] },
    getBoundingClientRect: { configurable: true, value: () => new DOMRect() }
  })

  try {
    const mounted = render(
      <LocalFilePreview
        reloadKey={0}
        target={{
          kind: 'file',
          label: 'report.md',
          path: '/workspace/report.md',
          previewKind: 'text',
          language: 'markdown',
          source: '/workspace/report.md',
          url: 'file:///workspace/report.md',
          fileOwner: owner
        }}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))
    const editor = EditorView.findFromDOM(mounted.container.querySelector('.cm-content')!)!
    act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: '# Edited B' } }))
    backendText = 'External edit on B'
    fireEvent.click(screen.getByRole('button', { name: /Save/ }))
    const overwrite = await screen.findByRole('button', { name: /Overwrite/ })
    expect(api.mock.calls.some(([request]) => request.path === '/api/fs/write-text')).toBe(false)
    await act(async () => {
      $connection.set({ connectionId: 'backend-c', mode: 'remote', profile: 'other' } as never)
      setApiRequestConnection('backend-c')
    })
    fireEvent.click(overwrite)
    await waitFor(() =>
      expect(api).toHaveBeenCalledWith(
        expect.objectContaining({
          ...owner,
          path: '/api/fs/write-text',
          body: { path: '/workspace/report.md', content: '# Edited B' }
        })
      )
    )
    expect(
      api.mock.calls.filter(([request]) => request.path.startsWith('/api/fs/read-text?')).length
    ).toBeGreaterThanOrEqual(2)
  } finally {
    for (const key of ['getClientRects', 'getBoundingClientRect']) {
      if (descriptors[key]) {
        Object.defineProperty(Range.prototype, key, descriptors[key])
      } else {
        Reflect.deleteProperty(Range.prototype, key)
      }
    }
  }
})
