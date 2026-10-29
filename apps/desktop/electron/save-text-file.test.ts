import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { registerSaveDialogIpc, SAVE_TEXT_FILE_MAX_BYTES, saveTextFileAs } from './save-text-file'

const scratch: string[] = []

function tempDir(): string {
  const dir = fs.mkdtempSync(path.join(process.env.TMPDIR || os.tmpdir(), 'save-text-'))
  scratch.push(dir)

  return dir
}

afterEach(() => {
  for (const dir of scratch.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

describe('saveTextFileAs (Electron-side /save write)', () => {
  it('writes the content where the user chose, defaulting to a bare name in the default dir', async () => {
    const dir = tempDir()
    const target = path.join(dir, 'picked.md')
    let seen: { defaultPath?: string; filters?: unknown; title: string } | null = null

    const result = await saveTextFileAs(
      {
        content: '# hello',
        defaultPath: '../../etc/hermes_session_x.md',
        filters: [{ name: 'Markdown', extensions: ['md'] }]
      },
      {
        defaultDir: () => dir,
        showSaveDialog: async options => {
          seen = options

          return { canceled: false, filePath: target }
        }
      }
    )

    expect(result).toEqual({ saved: true, path: target })
    expect(fs.readFileSync(target, 'utf8')).toBe('# hello')
    // The renderer's suggestion never picks the directory.
    expect(seen!.defaultPath).toBe(path.join(dir, 'hermes_session_x.md'))
    expect(seen!.filters).toEqual([{ name: 'Markdown', extensions: ['md'] }])
  })

  it('asks before overwriting an existing file and can create a directory (Linux needs both flags)', async () => {
    let seen: { properties?: string[] } | null = null

    await saveTextFileAs(
      { content: 'x', defaultPath: 'a.md' },
      {
        defaultDir: () => '/home/u',
        showSaveDialog: async options => {
          seen = options

          return { canceled: true }
        }
      }
    )

    expect(seen!.properties).toEqual(['createDirectory', 'showOverwriteConfirmation'])
  })

  it('writes nothing when the dialog is cancelled', async () => {
    const writes: string[] = []

    const result = await saveTextFileAs(
      { content: 'x', defaultPath: 'a.json' },
      {
        defaultDir: () => '/nowhere',
        showSaveDialog: async () => ({ canceled: true }),
        writeFile: async filePath => void writes.push(filePath)
      }
    )

    expect(result).toEqual({ saved: false, canceled: true })
    expect(writes).toEqual([])
  })

  it('refuses non-string or oversized content before opening a dialog', async () => {
    let opened = false

    const deps = {
      defaultDir: () => '/nowhere',
      showSaveDialog: async () => {
        opened = true

        return { canceled: true }
      }
    }

    await expect(saveTextFileAs({ content: 42 }, deps)).rejects.toThrow('Invalid content')
    await expect(saveTextFileAs({ content: 'x'.repeat(SAVE_TEXT_FILE_MAX_BYTES + 1) }, deps)).rejects.toThrow(
      'Content too large'
    )
    expect(opened).toBe(false)
  })
})

describe('registerSaveDialogIpc', () => {
  it('registers the path picker and the main-process write', async () => {
    const handlers = new Map<string, (event: unknown, ...args: any[]) => unknown>()
    const dir = tempDir()
    const target = path.join(dir, 'out.json')
    registerSaveDialogIpc(
      { handle: (channel, listener) => void handlers.set(channel, listener) },
      { defaultDir: () => dir, showSaveDialog: async () => ({ canceled: false, filePath: target }) }
    )

    expect([...handlers.keys()].sort()).toEqual(['hermes:saveTextFileAs', 'hermes:selectSavePath'])
    expect(await handlers.get('hermes:selectSavePath')!(null, { title: 'Export' })).toBe(target)
    expect(await handlers.get('hermes:saveTextFileAs')!(null, { content: '{}' })).toEqual({ saved: true, path: target })
    expect(fs.readFileSync(target, 'utf8')).toBe('{}')
  })
})
