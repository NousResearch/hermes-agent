import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DroppedFile } from './use-composer-actions'
import { useFileDropZone } from './use-file-drop-zone'

/**
 * jsdom's bundled `DataTransfer` constructor sets `types: ['Files']`
 * unconditionally, so we cannot use it to model a Windows Explorer sparse
 * drag (both `types` and `items` empty at dragenter, payload only at drop).
 * The shape below covers everything `useFileDropZone` and the
 * `extractDroppedFiles` helper read in the test path, including the
 * `item(i)` method `extractDroppedFiles` calls on `FileList`.
 */
function makeTransfer(options: {
  types?: string[]
  files?: File[]
  items?: { kind: string; getAsFile: () => File | null }[]
} = {}): DataTransfer {
  const types = options.types ?? []
  const items = options.items ?? options.files?.map(file => ({ kind: 'file', getAsFile: () => file })) ?? []
  const files = options.files ?? []

  const fileListLike = Object.assign([...files], {
    item: (i: number) => files[i] ?? null,
    length: files.length
  })

  return {
    types: Object.freeze([...types]) as unknown as DOMStringList,
    items: Object.freeze([...items]) as unknown as DataTransferItemList,
    files: fileListLike as unknown as FileList,
    getData: () => '',
    setData: () => undefined,
    clearData: () => undefined,
    setDragImage: () => undefined,
    dropEffect: 'none',
    effectAllowed: 'all'
  } as unknown as DataTransfer
}

function Harness({
  onDropFiles,
  enabled
}: {
  onDropFiles: ReturnType<typeof vi.fn<(files: DroppedFile[]) => void>>
  enabled?: boolean
}) {
  const { dragKind, dropHandlers } = useFileDropZone({
    enabled: enabled ?? true,
    onDropFiles
  })

  return (
    <div data-drag-kind={dragKind ?? 'none'} data-testid="zone" {...dropHandlers}>
      <input data-testid="inner" />
    </div>
  )
}

afterEach(cleanup)

/**
 * Regression: Windows Explorer / Chromium exposes a *fully empty* native
 * DataTransfer during `dragenter`/`dragover` and only populates `files`
 * at `drop`. The pre-fix code rejected that dragenter (returning early
 * without `preventDefault`), which let the app-lifetime react-dnd
 * HTML5Backend's capture-time `dropEffect='none'` suppress the eventual
 * drop. The new stateful gate must (a) accept the sparse dragenter, (b)
 * keep `dragover` alive for the same drag, and (c) still hand the real
 * `File` payload to `onDropFiles` at drop.
 */
describe('useFileDropZone (#97702 Windows sparse file drag)', () => {
  it('accepts a sparse dragenter, keeps dragover alive, and ingests the file at drop', () => {
    const onDropFiles = vi.fn<(files: DroppedFile[]) => void>()
    const file = new File(['hello'], 'greeting.txt', { type: 'text/plain' })
    // Dragenter / dragover: empty (sparse). Drop: payload materialises.
    const sparse = makeTransfer()
    const populated = makeTransfer({ types: ['Files'], files: [file] })

    render(<Harness onDropFiles={onDropFiles} />)
    const zone = screen.getByTestId('zone')

    // 1. Dragenter with a fully empty transfer — the only known production
    //    signal of a Windows Explorer drag. The gate must accept (and the
    //    handler must preventDefault) so the eventual drop survives.
    act(() => {
      fireEvent.dragEnter(zone, { dataTransfer: sparse })
    })
    expect(zone.getAttribute('data-drag-kind')).toBe('files')
    // Browsers fire several dragovers before drop — each must keep the
    // gate armed (preventDefault'd, dropEffect='copy').
    act(() => {
      fireEvent.dragOver(zone, { dataTransfer: sparse })
    })
    act(() => {
      fireEvent.dragOver(zone, { dataTransfer: sparse })
    })
    expect(zone.getAttribute('data-drag-kind')).toBe('files')

    // 2. Drop with the real payload: a real File must be ingested, not
    //    silently swallowed, and the overlay must clear.
    act(() => {
      fireEvent.drop(zone, { dataTransfer: populated })
    })

    expect(onDropFiles).toHaveBeenCalledTimes(1)
    expect(onDropFiles.mock.calls[0]?.[0]?.[0]?.file?.name).toBe('greeting.txt')
    expect(zone.getAttribute('data-drag-kind')).toBe('none')
  })

  it('REJECTS a sparse dragover that has no prior dragenter (no blanket accept)', () => {
    const onDropFiles = vi.fn<(files: DroppedFile[]) => void>()
    const sparse = makeTransfer()

    render(<Harness onDropFiles={onDropFiles} />)
    const zone = screen.getByTestId('zone')

    // No dragenter — a stray dragover alone must NOT set dragKind or
    // preventDefault. The fix is intentionally NOT "classify every empty
    // DataTransfer as a file drag"; it is "trust a sparse dragenter as
    // a credible signal, then keep the gate armed for that drag's
    // dragovers".
    act(() => {
      fireEvent.dragOver(zone, { dataTransfer: sparse })
    })
    act(() => {
      fireEvent.dragOver(zone, { dataTransfer: sparse })
    })
    expect(zone.getAttribute('data-drag-kind')).toBe('none')

    // And a drop that follows a never-armed drag must NOT call
    // onDropFiles — the zone never accepted the drag.
    act(() => {
      fireEvent.drop(zone, { dataTransfer: sparse })
    })
    expect(onDropFiles).not.toHaveBeenCalled()
  })

  it('still accepts a typed "Files" dragenter (existing typed-drag path is unchanged)', () => {
    const onDropFiles = vi.fn<(files: DroppedFile[]) => void>()
    const typed = makeTransfer({ types: ['Files'] })

    render(<Harness onDropFiles={onDropFiles} />)
    const zone = screen.getByTestId('zone')

    act(() => {
      fireEvent.dragEnter(zone, { dataTransfer: typed })
    })
    expect(zone.getAttribute('data-drag-kind')).toBe('files')
  })

  it('rejects a text drag (text/plain) at the enter and over steps', () => {
    const onDropFiles = vi.fn<(files: DroppedFile[]) => void>()
    const text = makeTransfer({ types: ['text/plain'] })

    render(<Harness onDropFiles={onDropFiles} />)
    const zone = screen.getByTestId('zone')

    act(() => {
      fireEvent.dragEnter(zone, { dataTransfer: text })
    })
    act(() => {
      fireEvent.dragOver(zone, { dataTransfer: text })
    })
    expect(zone.getAttribute('data-drag-kind')).toBe('none')
  })

  it('disarms after a root-depth dragleave, so a subsequent sparse drag re-evaluates', () => {
    const onDropFiles = vi.fn<(files: DroppedFile[]) => void>()
    const sparse = makeTransfer()
    const populated = makeTransfer({ types: ['Files'], files: [new File(['x'], 'a.txt')] })

    render(<Harness onDropFiles={onDropFiles} />)
    const zone = screen.getByTestId('zone')

    act(() => {
      fireEvent.dragEnter(zone, { dataTransfer: sparse })
    })
    act(() => {
      fireEvent.dragLeave(zone, { dataTransfer: sparse })
    })
    expect(zone.getAttribute('data-drag-kind')).toBe('none')

    // A second, independent drag must work as cleanly as the first.
    act(() => {
      fireEvent.dragEnter(zone, { dataTransfer: sparse })
    })
    expect(zone.getAttribute('data-drag-kind')).toBe('files')
    act(() => {
      fireEvent.drop(zone, { dataTransfer: populated })
    })
    expect(onDropFiles).toHaveBeenCalledTimes(1)
  })

  it('does nothing when enabled: false (sparse or typed)', () => {
    const onDropFiles = vi.fn<(files: DroppedFile[]) => void>()
    const sparse = makeTransfer()

    render(<Harness enabled={false} onDropFiles={onDropFiles} />)
    const zone = screen.getByTestId('zone')

    act(() => {
      fireEvent.dragEnter(zone, { dataTransfer: sparse })
    })
    act(() => {
      fireEvent.dragOver(zone, { dataTransfer: sparse })
    })
    expect(zone.getAttribute('data-drag-kind')).toBe('none')

    act(() => {
      fireEvent.drop(zone, { dataTransfer: makeTransfer({ types: ['Files'], files: [new File(['x'], 'a.txt')] }) })
    })
    expect(onDropFiles).not.toHaveBeenCalled()
  })
})
