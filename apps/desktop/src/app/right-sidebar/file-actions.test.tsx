import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  onComposerFocusRequest,
  onComposerInsertRequest
} from '@/app/chat/composer/focus'
import { $currentCwd } from '@/store/session'

import {
  FileEntryContextMenu,
  insertFileTargetIntoChat,
  isInsertIntoChatShortcut,
  isRenameShortcut
} from './file-actions'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key,
  useI18n: () => ({
    t: {
      fileMenu: {
        insertIntoChat: 'Insert into Chat',
        revealFinder: 'Reveal in Finder',
        revealExplorer: 'Reveal in File Explorer',
        revealFileManager: 'Open Containing Folder',
        copyPath: 'Copy Path',
        copyRelativePath: 'Copy Relative Path',
        rename: 'Rename…',
        delete: 'Delete',
        deleteTitle: (name: string) => `Delete ${name}?`,
        deleteBody: 'It will be moved to the Trash.'
      }
    }
  })
}))

vi.mock('@/lib/desktop-fs', () => ({
  copyTextToClipboard: vi.fn(),
  isDesktopFsRemoteMode: () => false,
  readDesktopFileDataUrl: vi.fn(),
  renameDesktopPath: vi.fn(),
  revealDesktopPath: vi.fn(),
  trashDesktopPath: vi.fn()
}))

// Radix menus measure themselves; jsdom has no ResizeObserver.
beforeAll(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
})

/** The insert/focus bus defers dispatch by a macrotask — flush it. */
const flushBus = () => new Promise(resolve => setTimeout(resolve, 1))

interface CapturedInsert {
  mode: string
  target: string
  text: string
}

function captureBus() {
  const inserts: CapturedInsert[] = []
  const focuses: string[] = []
  const offInsert = onComposerInsertRequest(detail => inserts.push(detail))
  const offFocus = onComposerFocusRequest(detail => focuses.push(detail.target))

  return { focuses, inserts, stop: () => (offInsert(), offFocus()) }
}

beforeEach(() => {
  $currentCwd.set('/repo')
})

describe('insertFileTargetIntoChat', () => {
  it('inserts a workspace-relative @file: chip inline', async () => {
    const bus = captureBus()

    expect(insertFileTargetIntoChat({ isDirectory: false, path: '/repo/src/main.ts' })).toBe(true)
    await flushBus()
    bus.stop()

    expect(bus.inserts).toEqual([{ mode: 'inline', target: 'main', text: '@file:src/main.ts' }])
    expect(bus.focuses).toEqual([])
  })

  it('inserts folders as @folder: chips', async () => {
    const bus = captureBus()

    expect(insertFileTargetIntoChat({ isDirectory: true, path: '/repo/src' })).toBe(true)
    await flushBus()
    bus.stop()

    expect(bus.inserts).toEqual([{ mode: 'inline', target: 'main', text: '@folder:src' }])
  })

  it('keeps paths outside the workspace absolute', async () => {
    const bus = captureBus()

    insertFileTargetIntoChat({ isDirectory: false, path: '/elsewhere/notes.md' })
    await flushBus()
    bus.stop()

    expect(bus.inserts.map(entry => entry.text)).toEqual(['@file:/elsewhere/notes.md'])
  })

  it('focuses the composer only when asked (context-menu flow)', async () => {
    const bus = captureBus()

    insertFileTargetIntoChat({ isDirectory: false, path: '/repo/a.ts' }, { focusComposer: true })
    await flushBus()
    bus.stop()

    expect(bus.focuses).toEqual(['main'])
  })

  it('is a no-op for empty paths', async () => {
    const bus = captureBus()

    expect(insertFileTargetIntoChat({ isDirectory: false, path: '' })).toBe(false)
    await flushBus()
    bus.stop()

    expect(bus.inserts).toEqual([])
  })
})

describe('row shortcuts', () => {
  it('Enter (unmodified) is the insert-into-chat shortcut', () => {
    expect(isInsertIntoChatShortcut(new KeyboardEvent('keydown', { key: 'Enter' }))).toBe(true)

    for (const modifier of ['altKey', 'ctrlKey', 'metaKey', 'shiftKey'] as const) {
      const event = new KeyboardEvent('keydown', { key: 'Enter', [modifier]: true })

      expect(isInsertIntoChatShortcut(event)).toBe(false)
    }
  })

  it('rename is F2-only — Enter no longer renames', () => {
    expect(isRenameShortcut(new KeyboardEvent('keydown', { key: 'F2' }))).toBe(true)
    expect(isRenameShortcut(new KeyboardEvent('keydown', { key: 'Enter' }))).toBe(false)
  })
})

describe('FileEntryContextMenu', () => {
  it('offers "Insert into Chat" and routes it through the composer bus', async () => {
    const bus = captureBus()

    render(
      <FileEntryContextMenu isDirectory={false} name="main.ts" path="/repo/src/main.ts" relativeTo="/repo">
        <div data-testid="row">main.ts</div>
      </FileEntryContextMenu>
    )

    fireEvent.contextMenu(screen.getByTestId('row'))
    const item = await screen.findByText('Insert into Chat')
    fireEvent.click(item)
    await flushBus()
    bus.stop()

    expect(bus.inserts).toEqual([{ mode: 'inline', target: 'main', text: '@file:src/main.ts' }])
    // Mouse flow: hand focus to the composer so the user can type the prompt.
    expect(bus.focuses).toEqual(['main'])
  })

  it('inserts folder entries as @folder: chips', async () => {
    const bus = captureBus()

    render(
      <FileEntryContextMenu isDirectory name="src" path="/repo/src" relativeTo="/repo">
        <div data-testid="row">src</div>
      </FileEntryContextMenu>
    )

    fireEvent.contextMenu(screen.getByTestId('row'))
    fireEvent.click(await screen.findByText('Insert into Chat'))
    await flushBus()
    bus.stop()

    expect(bus.inserts.map(entry => entry.text)).toEqual(['@folder:src'])
  })
})
