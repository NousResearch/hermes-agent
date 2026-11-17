import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { StrictMode, useEffect, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setSessionOwnerResolver } from '@/api/client'
import { createPluginContext } from '@/contrib/plugin'
import type { HermesApiRequest, HermesConnection } from '@/global'
import {
  FILE_PREVIEW_RENDERER_AREA,
  type FilePreviewFile,
  type FilePreviewReadResult,
  type FilePreviewRendererProps
} from '@/sdk/areas'
import type { PreviewTarget } from '@/store/preview'
import { $connection } from '@/store/session'

import { PreviewPane } from './preview-pane'

vi.mock('./real-profile-consent-dialog', () => ({ RealProfileConsentDialog: () => null }))

const XLSX_BYTES = [0x50, 0x4b, 0x03, 0x04, 0x00, 0xff]
const XLSX_DATA_URL = `data:application/octet-stream;base64,${btoa(String.fromCharCode(...XLSX_BYTES))}`
const BINARY_TITLE = 'This looks like a binary file'

const disposers: (() => void)[] = []

function fileTarget(path: string, extra: Partial<PreviewTarget> = {}): PreviewTarget {
  return {
    binary: true,
    byteSize: XLSX_BYTES.length,
    kind: 'file',
    label: path.split('/').pop() ?? path,
    mimeType: 'application/octet-stream',
    path,
    previewKind: 'binary',
    source: path,
    url: `file://${path}`,
    ...extra
  }
}

function connection(fields: Pick<HermesConnection, 'connectionId' | 'mode' | 'profile'>): HermesConnection {
  return {
    baseUrl: '',
    isFullscreen: false,
    logs: [],
    nativeOverlayWidth: 0,
    token: '',
    windowButtonPosition: null,
    wsUrl: '',
    ...fields
  }
}

function stubBridge(bridge: Record<string, unknown>) {
  vi.stubGlobal('hermesDesktop', { onPreviewFileChanged: () => () => {}, ...bridge })
}

function registerRenderer(
  pluginId: string,
  data: { extensions?: string[]; mimeTypes?: string[]; render: (props: FilePreviewRendererProps) => unknown },
  order?: number
) {
  createPluginContext(pluginId, dispose => disposers.push(dispose)).register({
    area: FILE_PREVIEW_RENDERER_AREA,
    data,
    id: 'preview',
    order
  })
}

function deactivatePlugins() {
  act(() => disposers.splice(0).forEach(dispose => dispose()))
}

function Sheet({ file, onFile }: { file: FilePreviewFile; onFile?: (file: FilePreviewFile) => void }) {
  const [result, setResult] = useState<FilePreviewReadResult | null>(null)

  useEffect(() => {
    let live = true
    onFile?.(file)
    void file.read().then(next => live && setResult(next))

    return () => {
      live = false
    }
  }, [file, onFile])

  return (
    <div data-testid="sheet">
      {file.name}|{file.extension}|{result ? (result.ok ? `bytes:${[...result.bytes].join(',')}` : result.error) : '…'}
    </div>
  )
}

function Counter({ onMount }: { onMount: () => void }) {
  const [count, setCount] = useState(0)

  useEffect(() => onMount(), [onMount])

  return (
    <button data-testid="counter" onClick={() => setCount(value => value + 1)} type="button">
      {`count: ${count}`}
    </button>
  )
}

function ReadOnMount({ file, results }: { file: FilePreviewFile; results: FilePreviewReadResult[] }) {
  useEffect(() => {
    void file.read().then(result => results.push(result))
  }, [file, results])

  return <span>reader</span>
}

const textTarget = (path: string) => fileTarget(path, { binary: false, mimeType: 'text/plain', previewKind: 'text' })
const htmlTarget = (path: string) => fileTarget(path, { binary: false, mimeType: 'text/html', previewKind: 'html' })

async function renderPane(target: PreviewTarget, sessionId?: string) {
  let view!: ReturnType<typeof render>

  await act(async () => {
    view = render(<PreviewPane sessionId={sessionId} target={target} />)
  })

  return view
}

afterEach(() => {
  deactivatePlugins()
  cleanup()
  $connection.set(null)
  setSessionOwnerResolver(null)
  vi.unstubAllGlobals()
})

describe('file preview renderers in the preview pane', () => {
  it('mounts the claiming plugin for a binary xlsx and reads exactly that file', async () => {
    // Arrange
    const expected = `book.xlsx|xlsx|bytes:${XLSX_BYTES.join(',')}`
    const readFileDataUrl = vi.fn(async () => XLSX_DATA_URL)
    stubBridge({ readFileDataUrl })
    registerRenderer('grid', { extensions: ['xlsx'], render: props => <Sheet file={props.file} /> })

    // Act
    await renderPane(fileTarget('/work/book.xlsx'))

    // Assert
    await waitFor(() => expect(screen.getByTestId('sheet').textContent).toBe(expected))
    expect(readFileDataUrl).toHaveBeenCalledWith('/work/book.xlsx')
    expect(screen.queryByText(BINARY_TITLE)).toBeNull()
  })

  it('keeps the binary refusal for an unclaimed file and reads nothing', async () => {
    // Arrange
    const readFileDataUrl = vi.fn(async () => XLSX_DATA_URL)
    const readFileText = vi.fn()
    stubBridge({ readFileDataUrl, readFileText })
    registerRenderer('grid', { extensions: ['xlsx'], render: props => <Sheet file={props.file} /> })

    // Act
    await renderPane(fileTarget('/work/archive.bin'))

    // Assert
    expect(screen.getByText(BINARY_TITLE)).toBeTruthy()
    expect(screen.queryByTestId('sheet')).toBeNull()
    expect(readFileDataUrl).not.toHaveBeenCalled()
    expect(readFileText).not.toHaveBeenCalled()
  })

  it.each([
    ['/work/REPORT.XLSX', 'xlsx'],
    ['/work/report #2.xlsx', 'xlsx'],
    ['/work/q?a/book.xlsx', 'xlsx'],
    ['/work/book.xlsx#Sheet1', 'xlsx']
  ])('claims %s by its file-name extension', async (path, extension) => {
    // Arrange
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('grid', { extensions: ['.XLSX'], render: props => <span>{`ext:${props.file.extension}`}</span> })

    // Act
    await renderPane(fileTarget(path))

    // Assert
    expect(screen.getByText(`ext:${extension}`)).toBeTruthy()
  })

  it('does not claim by a dotted folder or a fragment-only extension', async () => {
    // Arrange
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('grid', { extensions: ['xlsx'], render: () => <span>claimed</span> })

    // Act
    await renderPane(fileTarget('/work/v1.xlsx/README'))

    // Assert
    expect(screen.queryByText('claimed')).toBeNull()
    expect(screen.getByText(BINARY_TITLE)).toBeTruthy()
  })

  it('matches a reported MIME type but never application/octet-stream', async () => {
    // Arrange
    const sheetMime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('mime', {
      mimeTypes: [sheetMime, 'application/octet-stream'],
      render: props => <span>{`mime:${props.file.mimeType}`}</span>
    })

    // Act
    const view = await renderPane(fileTarget('/work/data', { mimeType: `${sheetMime.toUpperCase()}; charset=binary` }))
    const claimed = screen.queryByText(`mime:${sheetMime}`)
    view.unmount()
    await renderPane(fileTarget('/work/blob'))

    // Assert
    expect(claimed).toBeTruthy()
    expect(screen.queryByText(/^mime:/)).toBeNull()
    expect(screen.getByText(BINARY_TITLE)).toBeTruthy()
  })

  it('picks the lowest order, then the contribution id, whatever the load order', async () => {
    // Arrange
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('zeta', { extensions: ['xlsx'], render: () => <span>zeta</span> })
    registerRenderer('alpha', { extensions: ['xlsx'], render: () => <span>alpha</span> })
    registerRenderer('late', { extensions: ['xlsx'], render: () => <span>late</span> }, 5)

    // Act
    await renderPane(fileTarget('/work/book.xlsx'))

    // Assert
    expect(screen.getByText('alpha')).toBeTruthy()
    expect(screen.queryByText('zeta')).toBeNull()
  })

  it('ignores a malformed claim and keeps the built-in preview', async () => {
    // Arrange
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    createPluginContext('broken', dispose => disposers.push(dispose)).register({
      area: FILE_PREVIEW_RENDERER_AREA,
      data: { extensions: 'xlsx', render: () => <span>claimed</span> },
      id: 'preview'
    })

    // Act
    await renderPane(fileTarget('/work/book.xlsx'))

    // Assert
    expect(screen.queryByText('claimed')).toBeNull()
    expect(screen.getByText(BINARY_TITLE)).toBeTruthy()
  })

  it('keeps an open unclaimed file mounted when a renderer is registered or removed', async () => {
    // Arrange
    const readFileText = vi.fn(async () => ({ binary: false, byteSize: 5, path: '/work/notes.txt', text: 'hello' }))
    stubBridge({ readFileText })
    await renderPane(fileTarget('/work/notes.txt', { binary: false, mimeType: 'text/plain', previewKind: 'text' }))
    await waitFor(() => expect(readFileText).toHaveBeenCalledTimes(1))

    // Act
    await act(async () => registerRenderer('grid', { extensions: ['xlsx'], render: () => <span>claimed</span> }))
    deactivatePlugins()
    await act(async () => {})

    // Assert
    expect(readFileText).toHaveBeenCalledTimes(1)
  })

  it('returns to the built-in preview when the plugin is disabled', async () => {
    // Arrange
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('grid', { extensions: ['xlsx'], render: () => <span>claimed</span> })
    await renderPane(fileTarget('/work/book.xlsx'))
    const before = screen.queryByText('claimed')

    // Act
    deactivatePlugins()

    // Assert
    expect(before).toBeTruthy()
    expect(screen.queryByText('claimed')).toBeNull()
    expect(screen.getByText(BINARY_TITLE)).toBeTruthy()
  })

  it.each([
    [
      'throws',
      () => {
        throw new Error('renderer failed')
      }
    ],
    ['returns null', () => null]
  ])('shows the built-in preview when the renderer %s', async (_label, renderFn) => {
    // Arrange
    vi.spyOn(console, 'error').mockImplementation(() => {})
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('grid', { extensions: ['xlsx'], render: renderFn })

    // Act
    await renderPane(fileTarget('/work/book.xlsx'))

    // Assert
    expect(screen.getByText(BINARY_TITLE)).toBeTruthy()
  })

  it('reads through the backend that owns the tab session on a remote connection', async () => {
    // Arrange
    const requests: HermesApiRequest[] = []
    $connection.set(connection({ connectionId: 'gw', mode: 'remote', profile: 'default' }))
    setSessionOwnerResolver(id => (id === 'sess-1' ? { connectionId: 'gw-2', profile: 'work' } : null))
    stubBridge({
      api: async (request: HermesApiRequest) => {
        requests.push(request)

        return { dataUrl: XLSX_DATA_URL }
      }
    })
    registerRenderer('grid', { extensions: ['xlsx'], render: props => <Sheet file={props.file} /> })

    // Act
    await renderPane(fileTarget('/workspace/out/book.xlsx'), 'sess-1')

    // Assert
    await waitFor(() => expect(screen.getByTestId('sheet').textContent).toContain('bytes:'))
    expect(requests).toEqual([
      expect.objectContaining({
        connectionId: 'gw-2',
        path: `/api/fs/read-data-url?path=${encodeURIComponent('/workspace/out/book.xlsx')}&session_id=sess-1`,
        profile: 'work'
      })
    ])
  })

  it('answers scope-changed when the connection changes while a read is in flight', async () => {
    // Arrange
    let release!: (value: string) => void

    const pending = new Promise<string>(resolve => {
      release = resolve
    })

    const seen: FilePreviewFile[] = []
    stubBridge({ readFileDataUrl: () => pending })
    registerRenderer('grid', {
      extensions: ['xlsx'],
      render: props => {
        seen.push(props.file)

        return <span>claimed</span>
      }
    })
    await renderPane(fileTarget('/work/book.xlsx'))
    const inFlight = seen[0]?.read()

    // Act
    act(() => $connection.set(connection({ connectionId: 'other', mode: 'local', profile: 'second' })))
    await act(async () => release(XLSX_DATA_URL))

    // Assert
    expect(await inFlight).toEqual(expect.objectContaining({ error: 'scope-changed', ok: false }))
    expect(seen.at(-1)).not.toBe(seen[0])
  })

  it('gives the renderer a new handle for a new file and retires the old one', async () => {
    // Arrange
    const readFileDataUrl = vi.fn(async (_path: string) => XLSX_DATA_URL)
    const seen: FilePreviewFile[] = []
    const onFile = (file: FilePreviewFile) => seen.push(file)
    stubBridge({ readFileDataUrl })
    registerRenderer('grid', { extensions: ['xlsx'], render: props => <Sheet file={props.file} onFile={onFile} /> })
    const view = await renderPane(fileTarget('/work/a.xlsx'))

    // Act
    await act(async () => {
      view.rerender(<PreviewPane target={fileTarget('/work/b.xlsx')} />)
    })
    const stale = await seen[0]?.read()

    // Assert
    expect(seen.map(file => file.path)).toEqual(['/work/a.xlsx', '/work/b.xlsx'])
    expect(stale).toEqual(expect.objectContaining({ error: 'scope-changed', ok: false }))
    expect(readFileDataUrl.mock.calls.map(call => call[0])).toEqual(['/work/a.xlsx', '/work/b.xlsx'])
  })

  it.each([
    [
      'too-large',
      new Error(
        "Error invoking remote method 'hermes:readFileDataUrl': Error: File preview failed: file is too large (99 bytes; limit 9 bytes)."
      )
    ],
    ['too-large', new Error("Error invoking remote method 'hermes:api': Error: 413: File too large")],
    ['missing', { error: 'ENOENT', message: 'gone', ok: false }],
    ['missing', new Error("Error invoking remote method 'hermes:api': Error: 404: File not found")],
    ['io-error', new Error('File preview failed: /work/404/413 too large/book.xlsx is not readable.')]
  ])('reports %s as a result instead of throwing', async (code, outcome) => {
    // Arrange
    stubBridge({
      readFileDataUrl: async () => {
        if (outcome instanceof Error) {
          throw outcome
        }

        return outcome
      }
    })
    registerRenderer('grid', { extensions: ['xlsx'], render: props => <Sheet file={props.file} /> })

    // Act
    await renderPane(fileTarget('/work/book.xlsx'))

    // Assert
    await waitFor(() => expect(screen.getByTestId('sheet').textContent).toContain(code))
  })

  it('offers a copy through the OS save dialog without touching the source path', async () => {
    // Arrange
    const clicks: { download: string; href: string }[] = []
    const writeTextFile = vi.fn()
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicks.push({ download: this.download, href: this.href })
    })
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:copy', revokeObjectURL: () => {} }))
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL, writeTextFile })
    let file!: FilePreviewFile
    registerRenderer('grid', {
      extensions: ['xlsx'],
      render: props => {
        file = props.file

        return <span>claimed</span>
      }
    })
    await renderPane(fileTarget('/work/book.xlsx'))

    // Act
    file.saveCopy(new Uint8Array(XLSX_BYTES), '../../etc/edited.xlsx')

    // Assert
    expect(clicks).toEqual([{ download: 'edited.xlsx', href: 'blob:copy' }])
    expect(writeTextFile).not.toHaveBeenCalled()
  })

  it('gives a replaced render callback fresh hook state', async () => {
    // Arrange
    function First() {
      const [state] = useState('old state')

      return <span data-testid="host">{`first: ${state}`}</span>
    }

    function Second() {
      const [state] = useState('new state')

      return <span data-testid="host">{`second: ${state}`}</span>
    }

    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('grid', { extensions: ['xlsx'], render: First })
    await renderPane(fileTarget('/work/book.xlsx'))

    // Act
    await act(async () => registerRenderer('grid', { extensions: ['xlsx'], render: Second }))

    // Assert
    expect(screen.getByTestId('host').textContent).toBe('second: new state')
  })

  it('re-renders a render callback that calls hooks when its own state changes', async () => {
    // Arrange
    let bump!: () => void

    function Counting() {
      const [count, setCount] = useState(0)
      bump = () => setCount(value => value + 1)

      return <span data-testid="host">{`count: ${count}`}</span>
    }

    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('grid', { extensions: ['xlsx'], render: Counting })
    await renderPane(fileTarget('/work/book.xlsx'))

    // Act
    act(() => bump())

    // Assert
    expect(screen.getByTestId('host').textContent).toBe('count: 1')
  })

  it('keeps an unchanged renderer mounted while other contributions change', async () => {
    // Arrange
    const mounts = vi.fn()
    const readFileText = vi.fn(async () => ({ binary: false, byteSize: 5, path: '/work/notes.txt', text: 'hello' }))
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL, readFileText })
    registerRenderer('grid', { extensions: ['xlsx'], render: () => <Counter onMount={mounts} /> })
    registerRenderer('sheets', { extensions: ['ods'], render: () => <span>first</span> })
    await renderPane(fileTarget('/work/book.xlsx'))
    await renderPane(textTarget('/work/notes.txt'))
    await waitFor(() => expect(readFileText).toHaveBeenCalledTimes(1))
    act(() => screen.getByTestId('counter').click())

    // Act
    await act(async () => registerRenderer('csv', { extensions: ['csv'], render: () => <span>csv</span> }))
    await act(async () => registerRenderer('sheets', { extensions: ['ods'], render: () => <span>second</span> }))

    // Assert
    expect(screen.getByTestId('counter').textContent).toBe('count: 1')
    expect(mounts).toHaveBeenCalledTimes(1)
    expect(readFileText).toHaveBeenCalledTimes(1)
  })

  it('keeps the file and the renderer through a file-watch refresh', async () => {
    // Arrange
    let changed!: (payload: { id: string; path: string; url: string }) => void
    const mounts = vi.fn()
    const seen: FilePreviewFile[] = []
    const watchPreviewFile = vi.fn(async () => ({ id: 'watch-1', path: '/work/book.xlsx' }))
    stubBridge({
      onPreviewFileChanged: (callback: typeof changed) => {
        changed = callback

        return () => {}
      },
      readFileDataUrl: async () => XLSX_DATA_URL,
      watchPreviewFile
    })
    registerRenderer('grid', {
      extensions: ['xlsx'],
      render: props => {
        seen.push(props.file)

        return <Counter onMount={mounts} />
      }
    })
    await renderPane(fileTarget('/work/book.xlsx'))
    await waitFor(() => expect(watchPreviewFile).toHaveBeenCalled())
    await act(async () => {})

    // Act
    act(() => changed({ id: 'watch-1', path: '/work/book.xlsx', url: 'file:///work/book.xlsx' }))
    await act(async () => new Promise(resolve => setTimeout(resolve, 300)))
    const result = await seen[0]?.read()

    // Assert
    expect(new Set(seen).size).toBe(1)
    expect(mounts).toHaveBeenCalledTimes(1)
    expect(result).toEqual(expect.objectContaining({ ok: true }))
  })

  it('retires the old handle when the tab switches to a rendered HTML file', async () => {
    // Arrange
    const readFileDataUrl = vi.fn(async (_path: string) => XLSX_DATA_URL)
    const seen: FilePreviewFile[] = []
    stubBridge({ readFileDataUrl })
    registerRenderer('grid', {
      extensions: ['xlsx'],
      render: props => {
        seen.push(props.file)

        return <span>claimed</span>
      }
    })
    const view = await renderPane(fileTarget('/work/book.xlsx'))

    // Act
    await act(async () => {
      view.rerender(<PreviewPane target={htmlTarget('/work/page.html')} />)
    })
    const stale = await seen[0]?.read()

    // Assert
    expect(screen.queryByText('claimed')).toBeNull()
    expect(stale).toEqual(expect.objectContaining({ error: 'scope-changed', ok: false }))
    expect(readFileDataUrl).not.toHaveBeenCalled()
  })

  it('answers scope-changed for a read in flight when the tab switches to a rendered HTML file', async () => {
    // Arrange
    let release!: (value: string) => void

    const pending = new Promise<string>(resolve => {
      release = resolve
    })

    const seen: FilePreviewFile[] = []
    stubBridge({ readFileDataUrl: () => pending })
    registerRenderer('grid', {
      extensions: ['xlsx'],
      render: props => {
        seen.push(props.file)

        return <span>claimed</span>
      }
    })
    const view = await renderPane(fileTarget('/work/book.xlsx'))
    const inFlight = seen[0]?.read()

    // Act
    await act(async () => {
      view.rerender(<PreviewPane target={htmlTarget('/work/page.html')} />)
    })
    await act(async () => release(XLSX_DATA_URL))

    // Assert
    expect(await inFlight).toEqual(expect.objectContaining({ error: 'scope-changed', ok: false }))
  })

  it('lets a renderer read on mount under StrictMode', async () => {
    // Arrange
    const results: FilePreviewReadResult[] = []
    stubBridge({ readFileDataUrl: async () => XLSX_DATA_URL })
    registerRenderer('grid', {
      extensions: ['xlsx'],
      render: props => <ReadOnMount file={props.file} results={results} />
    })

    // Act
    await act(async () => {
      render(
        <StrictMode>
          <PreviewPane target={fileTarget('/work/book.xlsx')} />
        </StrictMode>
      )
    })
    await waitFor(() => expect(results.length).toBeGreaterThanOrEqual(2))

    // Assert
    expect(results.map(result => result.ok)).toEqual(results.map(() => true))
  })
})
