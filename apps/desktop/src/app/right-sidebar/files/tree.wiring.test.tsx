import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// ── Module seams ─────────────────────────────────────────────────────────────
// The point of this suite is the WIRING: which user gesture ends up calling
// which handler prop on <ProjectTree>. Everything arborist/dnd/context-menu
// brings that isn't part of that contract is stubbed.

type RowNodeOverrides = Partial<{ id: string; isDirectory: boolean; placeholder: boolean | 'error' }>

interface CapturedNode {
  data: { id: string; isDirectory: boolean; name: string; placeholder?: boolean | 'error' }
  handleClick: ReturnType<typeof vi.fn>
  select: ReturnType<typeof vi.fn>
  toggle: ReturnType<typeof vi.fn>
}

let capturedTreeProps: Record<string, unknown> = {}
let currentNode: CapturedNode

vi.mock('react-arborist', () => ({
  Tree: (props: Record<string, unknown>) => {
    capturedTreeProps = props

    // Arborist hands its ref a TreeApi; the component reads selectedNodes off
    // it for the rename shortcut. Provide the minimum shape.
    const treeRef = props.ref as { current: unknown }
    treeRef.current = { selectedNodes: [currentNode] }

    // Mimic the real structure: renderRow wraps the row renderer's output,
    // and the CONTAINER carries arborist's own onClick (node.handleClick).
    const children = (props.children as (rowProps: unknown) => React.ReactNode)({
      attrs: {},
      dragHandle: null,
      node: currentNode as never,
      style: {}
    })

    return (props.renderRow as (rowProps: unknown) => React.ReactNode)({
      attrs: {},
      children,
      innerRef: null,
      node: currentNode as never
    })
  }
}))

vi.mock('./dnd-manager', () => ({ getFileTreeDndManager: () => ({}) }))

vi.mock('../file-actions', () => ({
  // Same contract as the real helper: F2 anywhere, Enter on a focused row.
  isRenameShortcut: (event: { key: string }) => event.key === 'F2' || event.key === 'Enter',
  FileEntryContextMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  InlineRenameInput: () => <input aria-label="inline-rename" />
}))

// jsdom has no ResizeObserver; feed the component one synchronous size so the
// size gate (size.height > 0) opens and <Tree> mounts.
vi.mock('@/hooks/use-resize-observer', () => ({
  useResizeObserver: (callback: (entries: unknown[], el: Element) => void, ref: { current: Element | null }) => {
    setTimeout(() => {
      if (ref.current) {
        callback([{ contentRect: { height: 600, width: 240 }, target: ref.current }], ref.current)
      }
    }, 0)
  }
}))

import { $renamingPath } from '@/store/file-actions'

import { ProjectTree } from './tree'

const FILE_PATH = '/w/README.md'
const FOLDER_PATH = '/w/src'

function makeNode(overrides: RowNodeOverrides = {}): CapturedNode {
  const isDirectory = overrides.isDirectory ?? false

  return {
    data: {
      id: overrides.id ?? (isDirectory ? FOLDER_PATH : FILE_PATH),
      isDirectory,
      name: (overrides.id ?? (isDirectory ? FOLDER_PATH : FILE_PATH)).split('/').pop() ?? '',
      ...(overrides.placeholder !== undefined ? { placeholder: overrides.placeholder } : {})
    },
    handleClick: vi.fn(),
    select: vi.fn(),
    toggle: vi.fn()
  }
}

function renderTree(handlers: {
  onActivateFile?: ReturnType<typeof vi.fn>
  onActivateFolder?: ReturnType<typeof vi.fn>
  onPreviewFile?: ReturnType<typeof vi.fn>
}) {
  const props = {
    collapseNonce: 0,
    cwd: '/w',
    data: [{ id: currentNode.data.id, isDirectory: currentNode.data.isDirectory, name: currentNode.data.name }],
    onActivateFile: handlers.onActivateFile,
    onActivateFolder: handlers.onActivateFolder,
    onLoadChildren: () => {},
    onNodeOpenChange: () => {},
    onPreviewFile: handlers.onPreviewFile,
    openState: {}
  }

  // Test doubles are intentionally looser than the real prop signatures.
  return render(<ProjectTree {...(props as unknown as React.ComponentProps<typeof ProjectTree>)} />)
}

async function mountWith(overrides: RowNodeOverrides, handlers: Parameters<typeof renderTree>[0]) {
  currentNode = makeNode(overrides)
  renderTree(handlers)

  // Wait for the resize-driven size state so the tree mounts.
  await screen.findByText(currentNode.data.name)

  return screen.getByText(currentNode.data.name)
}

beforeEach(() => {
  $renamingPath.set(null)
})

afterEach(() => {
  cleanup()
  $renamingPath.set(null)
})

describe('ProjectTree gesture wiring', () => {
  it('a single click on a file opens the preview and selects the row, without reaching arborist’s click handler', async () => {
    const onPreviewFile = vi.fn()
    const label = await mountWith({}, { onPreviewFile })

    fireEvent.click(label)

    expect(onPreviewFile).toHaveBeenCalledTimes(1)
    expect(onPreviewFile).toHaveBeenCalledWith(FILE_PATH)
    expect(currentNode.select).toHaveBeenCalledTimes(1)
    // stopPropagation in the row handler keeps arborist's own click handling
    // (select+activate) out of the way — opening happens exactly once.
    expect(currentNode.handleClick).not.toHaveBeenCalled()
  })

  it('clicking a folder toggles it and never previews', async () => {
    const onPreviewFile = vi.fn()
    const label = await mountWith({ isDirectory: true }, { onPreviewFile })

    fireEvent.click(label)

    expect(currentNode.toggle).toHaveBeenCalledTimes(1)
    expect(onPreviewFile).not.toHaveBeenCalled()
  })

  it('shift-click attaches the file instead of opening it', async () => {
    const onActivateFile = vi.fn()
    const onPreviewFile = vi.fn()
    const label = await mountWith({}, { onActivateFile, onPreviewFile })

    fireEvent.click(label, { shiftKey: true })

    expect(onActivateFile).toHaveBeenCalledWith(FILE_PATH)
    expect(onPreviewFile).not.toHaveBeenCalled()
  })

  it('a click on the row being renamed is ignored entirely', async () => {
    const onPreviewFile = vi.fn()
    await mountWith({}, { onPreviewFile })

    $renamingPath.set(FILE_PATH)
    fireEvent.click(screen.getByText('README.md'))

    expect(onPreviewFile).not.toHaveBeenCalled()
    expect(currentNode.select).not.toHaveBeenCalled()
  })

  it('a click on a placeholder row is ignored', async () => {
    const onPreviewFile = vi.fn()
    const label = await mountWith({ id: '/w/loading', placeholder: true }, { onPreviewFile })

    fireEvent.click(label)

    expect(onPreviewFile).not.toHaveBeenCalled()
    expect(currentNode.select).not.toHaveBeenCalled()
  })

  it.each(['Enter', 'F2'])('%s on the selected file starts an inline rename instead of opening', async key => {
    const onPreviewFile = vi.fn()
    await mountWith({}, { onPreviewFile })

    fireEvent.keyDown(screen.getByText('README.md'), { key })

    expect($renamingPath.get()).toBe(FILE_PATH)
    expect(onPreviewFile).not.toHaveBeenCalled()
  })
})
