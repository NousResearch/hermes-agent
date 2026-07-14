import { useStore } from '@nanostores/react'
import { type KeyboardEvent as ReactKeyboardEvent, useCallback, useEffect, useRef, useState } from 'react'
import { type NodeApi, type NodeRendererProps, type RowRendererProps, Tree, type TreeApi } from 'react-arborist'

import { TreeSkeleton } from '@/components/chat/skeletons'
import { Codicon } from '@/components/ui/codicon'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { cn } from '@/lib/utils'
import { $repoChangeByPath, type RepoChangeKind } from '@/store/coding-status'
import { $renamingPath, beginInlineRename } from '@/store/file-actions'
import { $revealInTreeRequest } from '@/store/layout'

import { FileEntryContextMenu, InlineRenameInput, isRenameShortcut } from '../file-actions'

import { getFileTreeDndManager } from './dnd-manager'
import type { TreeNode } from './use-project-tree'

const ROW_HEIGHT = 22
const INDENT = 10
/** Fixed base inset (`px-6.5`) layered on top of arborist's depth indent. */
const TREE_ROW_INSET = '17px'

function withTreeInset(paddingLeft: number | string | undefined): string {
  if (typeof paddingLeft === 'number') {
    return `calc(${paddingLeft}px + ${TREE_ROW_INSET})`
  }

  if (!paddingLeft) {
    return TREE_ROW_INSET
  }

  return `calc(${paddingLeft} + ${TREE_ROW_INSET})`
}

interface ProjectTreeProps {
  collapseNonce: number
  cwd: string
  data: TreeNode[]
  onActivateFile: (path: string) => void
  onActivateFolder: (path: string) => void
  onLoadChildren: (id: string) => void | Promise<void>
  onNodeOpenChange: (id: string, open: boolean) => void
  onPreviewFile?: (path: string) => void
  openState: Record<string, boolean>
}

export function ProjectTree({
  collapseNonce,
  cwd,
  data,
  onActivateFile,
  onActivateFolder,
  onLoadChildren,
  onNodeOpenChange,
  onPreviewFile,
  openState
}: ProjectTreeProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const treeRef = useRef<TreeApi<TreeNode> | null>(null)
  const [size, setSize] = useState({ height: 0, width: 0 })
  const changeByPath = useStore($repoChangeByPath)

  const syncTreeSize = useCallback(() => {
    const el = containerRef.current

    if (!el) {
      return
    }

    const { height, width } = el.getBoundingClientRect()

    setSize(prev => {
      if (prev.height === height && prev.width === width) {
        return prev
      }

      return { height, width }
    })
  }, [])

  useResizeObserver(syncTreeSize, containerRef)

  const handleToggle = useCallback(
    (id: string) => {
      const node = treeRef.current?.get(id)

      if (!node) {
        return
      }

      onNodeOpenChange(id, node.isOpen)

      if (node.isOpen && node.data?.isDirectory && node.data.children === undefined) {
        void onLoadChildren(id)
      }
    },
    [onLoadChildren, onNodeOpenChange]
  )

  // "Reveal in side bar": expand each ancestor folder top-down (lazy-loading its
  // children first so the node exists), then select + scroll to the target. The
  // pane is opened by the caller; this drives the tree to the file.
  const revealNode = useCallback(
    async (absPath: string) => {
      const root = cwd.replace(/[\\/]+$/, '')
      const target = absPath.replace(/[\\/]+$/, '')
      const rel = target.startsWith(root) ? target.slice(root.length).replace(/^[\\/]+/, '') : ''
      const segments = rel.split(/[\\/]/).filter(Boolean)

      let acc = root

      for (let i = 0; i < segments.length - 1; i += 1) {
        acc = `${acc}/${segments[i]}`
        const node = treeRef.current?.get(acc)

        if (node?.data?.isDirectory && node.data.children === undefined) {
          await onLoadChildren(acc)
        }

        onNodeOpenChange(acc, true)
        treeRef.current?.open(acc)
        await new Promise(resolve => requestAnimationFrame(() => resolve(undefined)))
      }

      treeRef.current?.select(target)
      // 'start' lands the file at/near the top (instant — arborist sets scrollTop
      // directly, no smooth scroll).
      treeRef.current?.scrollTo(target, 'start')
    },
    [cwd, onLoadChildren, onNodeOpenChange]
  )

  useEffect(
    () =>
      $revealInTreeRequest.subscribe(path => {
        if (!path) {
          return
        }

        $revealInTreeRequest.set(null)
        void revealNode(path)
      }),
    [revealNode]
  )

  const handleActivate = useCallback(
    (node: NodeApi<TreeNode>) => {
      // arborist fires onActivate on click/dblclick/Enter — independent of the
      // row's own handlers. Suppress it for the row being renamed so the
      // context-menu "Rename" (and its fall-through) can't open the preview.
      if (node.data && !node.data.isDirectory && $renamingPath.get() !== node.data.id) {
        onPreviewFile?.(node.data.id)
      }
    },
    [onPreviewFile]
  )

  // F2 / Enter on the selected row begins an inline rename. Capture-phase so it
  // beats arborist's own Enter-to-activate; skipped while an edit is in progress
  // (the editor input owns Enter/Esc then) and for placeholder rows.
  const handleRenameShortcut = useCallback((event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!isRenameShortcut(event) || $renamingPath.get()) {
      return
    }

    const node = treeRef.current?.selectedNodes?.[0]

    if (!node?.data || node.data.placeholder) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    beginInlineRename(node.data.id)
  }, [])

  return (
    <div className="min-h-0 flex-1 overflow-hidden" onKeyDownCapture={handleRenameShortcut} ref={containerRef}>
      {size.height > 0 && size.width > 0 ? (
        <Tree<TreeNode>
          childrenAccessor={node => (node?.isDirectory ? (node.children ?? []) : null)}
          data={data}
          disableDrag
          disableDrop
          disableEdit
          dndManager={getFileTreeDndManager()}
          height={size.height}
          indent={INDENT}
          initialOpenState={openState}
          key={`${cwd}:${collapseNonce}`}
          onActivate={handleActivate}
          onToggle={handleToggle}
          openByDefault={false}
          padding={0}
          ref={treeRef}
          renderRow={ProjectTreeRowContainer}
          rowHeight={ROW_HEIGHT}
          width={size.width}
        >
          {props => (
            <ProjectTreeRow
              {...props}
              changeKind={props.node.data ? changeByPath.get(props.node.data.id) : undefined}
              onAttachFile={onActivateFile}
              onAttachFolder={onActivateFolder}
              onPreviewFile={onPreviewFile}
              relativeTo={cwd}
            />
          )}
        </Tree>
      ) : (
        <TreeSizingState />
      )}
    </div>
  )
}

function TreeSizingState() {
  return <TreeSkeleton />
}

// arborist's default row hardcodes `min-width: max-content` (so a highlight can
// span horizontally-scrolled content), which grows the row to its full name
// width and defeats the inner `truncate`. We don't scroll sideways — pin the row
// to the viewport so long names ellipsize instead of clipping at the pane edge.
function ProjectTreeRowContainer({ attrs, children, innerRef, node }: RowRendererProps<TreeNode>) {
  return (
    <div
      {...attrs}
      onClick={node.handleClick}
      onFocus={e => e.stopPropagation()}
      ref={innerRef}
      style={{ ...attrs.style, minWidth: 0, width: '100%' }}
    >
      {children}
    </div>
  )
}

const CHANGE_TINT: Record<RepoChangeKind, string> = {
  added: 'text-(--ui-green)',
  conflicted: 'text-(--ui-red)',
  modified: 'text-(--ui-yellow)'
}

function ProjectTreeRow({
  changeKind,
  dragHandle,
  node,
  onAttachFile,
  onAttachFolder,
  onPreviewFile,
  relativeTo,
  style
}: NodeRendererProps<TreeNode> & {
  changeKind?: RepoChangeKind
  onAttachFile: (path: string) => void
  onAttachFolder: (path: string) => void
  onPreviewFile?: (path: string) => void
  relativeTo?: null | string
}) {
  const renamingPath = useStore($renamingPath)

  if (!node.data) {
    return <div style={style} />
  }

  const isFolder = node.data.isDirectory
  const isPlaceholder = Boolean(node.data.placeholder)
  const isErrorPlaceholder = node.data.placeholder === 'error'
  const editing = !isPlaceholder && renamingPath === node.data.id

  const row = (
    <div
      aria-expanded={isFolder ? node.isOpen : undefined}
      aria-selected={node.isSelected}
      className={cn(
        'group/row flex h-full cursor-pointer select-none items-center gap-1 border border-transparent px-3 text-xs font-normal leading-(--file-tree-row-height) text-(--ui-text-secondary) transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:text-foreground hover:transition-none',
        node.isSelected && 'bg-(--ui-row-active-background) text-foreground',
        isPlaceholder && 'pointer-events-none italic text-muted-foreground/70'
      )}
      draggable={!isPlaceholder && !editing}
      onClick={event => {
        event.stopPropagation()

        // Read the rename atom LIVE (not the render closure): the fall-through
        // click from a context-menu close can fire before the editing re-render
        // commits, so a stale closure would still select/activate and yank focus.
        if (isPlaceholder || $renamingPath.get() === node.data.id) {
          return
        }

        if (event.shiftKey) {
          ;(isFolder ? onAttachFolder : onAttachFile)(node.data.id)

          return
        }

        if (isFolder) {
          node.toggle()
        } else {
          node.select()
        }
      }}
      onDoubleClick={event => {
        event.stopPropagation()

        if (!isFolder && !isPlaceholder && $renamingPath.get() !== node.data.id) {
          onPreviewFile?.(node.data.id)
        }
      }}
      onDragStart={event => {
        if (isPlaceholder || $renamingPath.get() === node.data.id) {
          event.preventDefault()

          return
        }

        const payload = JSON.stringify([{ isDirectory: isFolder, path: node.data.id }])

        event.dataTransfer.effectAllowed = 'copy'
        event.dataTransfer.setData('application/x-hermes-paths', payload)
        event.dataTransfer.setData('text/plain', node.data.id)
      }}
      ref={dragHandle}
      style={{
        ...style,
        paddingLeft: withTreeInset(style.paddingLeft)
      }}
      title={node.data.id}
    >
      {/* No chevron column — the folder icon (open/closed) already carries the
          expand state, so the extra glyph was pure noise. */}
      <span aria-hidden className="flex w-3.5 items-center justify-center text-(--ui-text-tertiary)">
        {isPlaceholder && !isErrorPlaceholder ? (
          <Codicon name="loading" size="0.75rem" spinning />
        ) : isErrorPlaceholder ? (
          <Codicon name="warning" size="0.75rem" />
        ) : isFolder ? (
          <Codicon name={node.isOpen ? 'folder-opened' : 'folder'} size="0.875rem" />
        ) : (
          <Codicon name="file" size="0.875rem" />
        )}
      </span>
      {editing ? (
        <InlineRenameInput name={node.data.name} path={node.data.id} />
      ) : (
        // Git decoration (VS Code-style): tint changed files; the explicit color
        // wins over the row's hover/selected text color, so it persists.
        <span className={cn('min-w-0 flex-1 truncate', changeKind && CHANGE_TINT[changeKind])}>{node.data.name}</span>
      )}
    </div>
  )

  if (isPlaceholder) {
    return row
  }

  return (
    <FileEntryContextMenu isDirectory={isFolder} name={node.data.name} path={node.data.id} relativeTo={relativeTo}>
      {row}
    </FileEntryContextMenu>
  )
}
