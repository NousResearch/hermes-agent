import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { onComposerInsertRefsRequest } from '@/app/chat/composer/focus'

import { FileEntryContextMenu } from './file-actions'

afterEach(cleanup)

interface Seen {
  refs: unknown[]
  target: string
}

async function openMenu(children: React.ReactNode) {
  const utils = render(<>{children}</>)
  const trigger = utils.container.firstElementChild as HTMLElement

  await act(async () => {
    fireEvent.contextMenu(trigger)
  })

  return utils
}

async function clickSend(seen: Seen[]) {
  await act(async () => {
    fireEvent.click(screen.getByText('Send to Chat'))
  })

  // The bus defers dispatch to a macrotask.
  await waitFor(() => expect(seen.length).toBe(1))
}

describe('FileEntryContextMenu — Send to Chat', () => {
  it('inserts an @file: ref that shows the name but carries the workspace-relative path', async () => {
    const seen: Seen[] = []
    const unsubscribe = onComposerInsertRefsRequest(({ refs, target }) => seen.push({ refs, target }))

    try {
      await openMenu(
        <FileEntryContextMenu isDirectory={false} name="notes.md" path="/repo/src/notes.md" relativeTo="/repo">
          <div>notes.md</div>
        </FileEntryContextMenu>
      )

      await clickSend(seen)
      expect(seen[0].target).toBe('main')
      // Display label = basename; the value Hermes resolves stays the full
      // workspace-relative path.
      expect(seen[0].refs).toEqual([{ kind: 'file', label: 'notes.md', value: 'src/notes.md' }])
    } finally {
      unsubscribe()
    }
  })

  it('inserts an @folder: ref for directories', async () => {
    const seen: Seen[] = []
    const unsubscribe = onComposerInsertRefsRequest(({ refs, target }) => seen.push({ refs, target }))

    try {
      await openMenu(
        <FileEntryContextMenu isDirectory name="src" path="/repo/src" relativeTo="/repo">
          <div>src</div>
        </FileEntryContextMenu>
      )

      await clickSend(seen)
      expect(seen[0].target).toBe('main')
      expect(seen[0].refs).toEqual([{ kind: 'folder', label: 'src', value: 'src' }])
    } finally {
      unsubscribe()
    }
  })

  it('falls back to the absolute path when outside the workspace', async () => {
    const seen: Seen[] = []
    const unsubscribe = onComposerInsertRefsRequest(({ refs }) => seen.push({ refs, target: '' }))

    try {
      await openMenu(
        <FileEntryContextMenu isDirectory={false} name="photo.png" path="/elsewhere/photo.png" relativeTo="/repo">
          <div>photo.png</div>
        </FileEntryContextMenu>
      )

      await clickSend(seen)
      expect(seen[0].refs).toEqual([{ kind: 'file', label: 'photo.png', value: '/elsewhere/photo.png' }])
    } finally {
      unsubscribe()
    }
  })

  it('sends the absolute path unchanged when there is no workspace root', async () => {
    const seen: Seen[] = []
    const unsubscribe = onComposerInsertRefsRequest(({ refs }) => seen.push({ refs, target: '' }))

    try {
      await openMenu(
        <FileEntryContextMenu isDirectory={false} name="data.csv" path="/elsewhere/data.csv" relativeTo={null}>
          <div>data.csv</div>
        </FileEntryContextMenu>
      )

      await clickSend(seen)
      expect(seen[0].refs).toEqual([{ kind: 'file', label: 'data.csv', value: '/elsewhere/data.csv' }])
    } finally {
      unsubscribe()
    }
  })
})
