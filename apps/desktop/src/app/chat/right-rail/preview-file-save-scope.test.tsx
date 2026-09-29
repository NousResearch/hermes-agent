import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { HermesApiRequest, HermesConnection } from '@/global'
import { $connection } from '@/store/session'

import { LocalFilePreview } from './preview-file'

vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre>
}))

it.each(['connection', 'profile', 'local', 'api-route'] as const)(
  'never retargets a pending preview save after a %s switch and permits retry on its owner',
  async destination => {
    const directory = await mkdtemp(path.join(tmpdir(), 'hermes-save-scope-'))
    const originFile = path.join(directory, 'origin.txt')
    const otherFile = path.join(directory, 'other.txt')
    const logicalPath = '/workspace/note.txt'
    const origin = { connectionId: 'gateway-a', mode: 'remote', profile: 'default' } as HermesConnection

    const other = {
      connectionId:
        destination === 'connection' || destination === 'api-route'
          ? 'gateway-b'
          : destination === 'local'
            ? 'local'
            : 'gateway-a',
      mode: destination === 'local' ? 'local' : 'remote',
      profile: destination === 'profile' ? 'other' : 'default'
    } as HermesConnection

    const rangeDescriptors = Object.getOwnPropertyDescriptors(Range.prototype)
    let releaseRead!: () => void

    const pendingRead = new Promise<void>(resolve => {
      releaseRead = resolve
    })

    let originReads = 0
    let validationStarted!: () => void

    const started = new Promise<void>(resolve => {
      validationStarted = resolve
    })

    const writes: string[] = []

    const read = async (file: string) => ({
      path: logicalPath,
      text: await readFile(file, 'utf8'),
      binary: false,
      truncated: false
    })

    const write = async (file: string, content: string) => {
      writes.push(file)
      await writeFile(file, content)

      return { path: logicalPath }
    }

    const activate = (connection: HermesConnection) => {
      $connection.set(connection)
      setApiRequestConnection(connection.connectionId ?? null)
      setApiRequestProfile(connection.profile ?? null)
    }

    try {
      await writeFile(originFile, 'origin baseline')
      await writeFile(otherFile, 'unrelated valuable contents')
      vi.stubGlobal('hermesDesktop', {
        api: async (request: HermesApiRequest) => {
          const file =
            request.connectionId === origin.connectionId && request.profile === origin.profile ? originFile : otherFile

          if (request.path.startsWith('/api/fs/read-text?')) {
            const snapshot = await read(file)

            if (file === originFile && ++originReads === 2) {
              validationStarted()
              await pendingRead
            }

            return snapshot
          }

          if (request.path === '/api/fs/write-text') {
            return write(file, (request.body as { content: string }).content)
          }

          if (request.path.startsWith('/api/fs/git-root?')) {
            return { root: null }
          }

          throw new Error(`Unexpected request: ${request.path}`)
        },
        readFileText: () => read(otherFile),
        writeTextFile: (_file: string, content: string) => write(otherFile, content)
      })
      Object.defineProperties(Range.prototype, {
        getClientRects: { configurable: true, value: () => [] },
        getBoundingClientRect: { configurable: true, value: () => new DOMRect() }
      })
      activate(origin)

      const rendered = render(
        <LocalFilePreview
          reloadKey={0}
          target={{
            kind: 'file',
            label: 'note.txt',
            path: logicalPath,
            previewKind: 'text',
            source: logicalPath,
            url: `file://${logicalPath}`
          }}
        />
      )

      fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))
      const editor = EditorView.findFromDOM(rendered.container.querySelector('.cm-content')!)!
      act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: 'draft for origin' } }))
      fireEvent.click(screen.getByRole('button', { name: /Save/ }))
      await started

      await act(async () => {
        if (destination === 'api-route') {
          setApiRequestConnection(other.connectionId ?? null)
        } else {
          activate(other)
        }

        releaseRead()
      })
      await waitFor(() =>
        expect(
          screen.queryByRole('button', { name: 'Edit' }) || screen.queryByText(/original connection/)
        ).not.toBeNull()
      )
      expect(await readFile(otherFile, 'utf8')).toBe('unrelated valuable contents')
      expect(writes).toEqual([])
      expect(editor.dom.isConnected).toBe(true)
      expect(editor.state.doc.toString()).toBe('draft for origin')

      // Save is still refused when invoked after the switch.
      fireEvent.click(screen.getByRole('button', { name: /Save/ }))
      await screen.findByText(/original connection/)
      expect(writes).toEqual([])

      await act(async () => activate(origin))

      if (destination === 'connection') {
        // An Overwrite decision belongs to the same owner as the conflict.
        await writeFile(originFile, 'external edit at origin')
        fireEvent.click(screen.getByRole('button', { name: /Save/ }))
        await screen.findByRole('button', { name: /Overwrite/ })
        await act(async () => activate(other))
        fireEvent.click(screen.getByRole('button', { name: /Overwrite/ }))
        await screen.findByText(/original connection/)
        expect(writes).toEqual([])
        expect(await readFile(otherFile, 'utf8')).toBe('unrelated valuable contents')
        await act(async () => activate(origin))
        fireEvent.click(screen.getByRole('button', { name: /Overwrite/ }))
      } else {
        fireEvent.click(screen.getByRole('button', { name: /Save/ }))
      }

      await screen.findByRole('button', { name: 'Edit' })
      expect(writes).toEqual([originFile])
      expect(await readFile(originFile, 'utf8')).toBe('draft for origin')
      expect(await readFile(otherFile, 'utf8')).toBe('unrelated valuable contents')
    } finally {
      releaseRead()
      cleanup()
      vi.unstubAllGlobals()
      $connection.set(null)
      setApiRequestConnection(null)
      setApiRequestProfile(null)

      for (const key of ['getClientRects', 'getBoundingClientRect']) {
        if (rangeDescriptors[key]) {
          Object.defineProperty(Range.prototype, key, rangeDescriptors[key])
        } else {
          Reflect.deleteProperty(Range.prototype, key)
        }
      }

      await rm(directory, { recursive: true, force: true })
    }
  }
)
