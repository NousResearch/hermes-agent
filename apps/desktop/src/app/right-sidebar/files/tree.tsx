import { useCallback, useRef, useState } from 'react'
import { type NodeApi, type NodeRendererProps, Tree, type TreeApi } from 'react-arborist'

import { Codicon } from '@/components/ui/codicon'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { cn } from '@/lib/utils'

import type { TreeNode } from './use-project-tree'

const ROW_HEIGHT = 22
const INDENT = 10

interface ProjectTreeProps {
  data: TreeNode[]
  onActivateFile: (path: string) => void
  onActivateFolder: (path: string) => void
  onLoadChildren: (id: string) => void | Promise<void>
  onNodeOpenChange: (id: string, open: boolean) => void
  onPreviewFile?: (path: string) => void
  openState: Record<string, boolean>
}

export function ProjectTree({
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

      if (node.isOpen && node.data.children === undefined) {
        void onLoadChildren(id)
      }
    },
    [onLoadChildren, onNodeOpenChange]
  )

  const handleActivate = useCallback(
    (node: NodeApi<TreeNode>) => {
      if (!node.data.isDirectory) {
        onPreviewFile?.(node.data.id)
      }
    },
    [onPreviewFile]
  )

  return (
    <div className="min-h-0 flex-1 overflow-hidden" ref={containerRef}>
      {size.height > 0 && size.width > 0 ? (
        <Tree<TreeNode>
          childrenAccessor={node => (node.isDirectory ? (node.children ?? []) : null)}
          data={data}
          disableDrag
          disableDrop
          disableEdit
          height={size.height}
          indent={INDENT}
          initialOpenState={openState}
          onActivate={handleActivate}
          onToggle={handleToggle}
          openByDefault={false}
          padding={0}
          ref={treeRef}
          rowHeight={ROW_HEIGHT}
          width={size.width}
        >
          {props => (
            <ProjectTreeRow
              {...props}
              onAttachFile={onActivateFile}
              onAttachFolder={onActivateFolder}
              onPreviewFile={onPreviewFile}
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
  return (
    <div className="flex h-full min-h-24 items-center justify-center px-3 text-[0.68rem] text-(--ui-text-tertiary)">
      Loading files...
    </div>
  )
}

function ProjectTreeRow({
  dragHandle,
  node,
  onAttachFile,
  onAttachFolder,
  onPreviewFile,
  style
}: NodeRendererProps<TreeNode> & {
  onAttachFile: (path: string) => void
  onAttachFolder: (path: string) => void
  onPreviewFile?: (path: string) => void
}) {
  const isFolder = node.data.isDirectory
  const isPlaceholder = node.data.id.endsWith('::__loading__')

  return (
    <div
      aria-expanded={isFolder ? node.isOpen : undefined}
      aria-selected={node.isSelected}
      className={cn(
        'group/row flex h-full cursor-pointer select-none items-center gap-1 border border-transparent px-3 text-xs font-normal leading-(--file-tree-row-height) text-(--ui-text-secondary) transition-colors hover:bg-(--ui-row-hover-background) hover:text-foreground',
        node.isSelected && 'bg-(--ui-row-active-background) text-foreground',
        isPlaceholder && 'pointer-events-none italic text-muted-foreground/70'
      )}
      draggable={!isPlaceholder}
      onClick={event => {
        event.stopPropagation()

        if (isPlaceholder) {
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

        if (!isFolder && !isPlaceholder) {
          onPreviewFile?.(node.data.id)
        }
      }}
      onDragStart={event => {
        if (isPlaceholder) {
          event.preventDefault()

          return
        }

        const payload = JSON.stringify([{ isDirectory: isFolder, path: node.data.id }])

        event.dataTransfer.effectAllowed = 'copy'
        event.dataTransfer.setData('application/x-hermes-paths', payload)
        event.dataTransfer.setData('text/plain', node.data.id)
      }}
      ref={dragHandle}
      style={style}
    >
      {isFolder && !isPlaceholder && (
        <span aria-hidden className="flex w-3 items-center justify-center">
          <Codicon
            className="text-(--ui-text-tertiary)"
            name={node.isOpen ? 'chevron-down' : 'chevron-right'}
            size="0.75rem"
          />
        </span>
      )}
      {!isFolder && <span aria-hidden className="w-3 shrink-0" />}
      <span aria-hidden className="flex w-3.5 items-center justify-center text-(--ui-text-tertiary)">
        {isPlaceholder ? (
          <Codicon name="loading" size="0.75rem" spinning />
        ) : isFolder ? (
          <Codicon name={node.isOpen ? 'folder-opened' : 'folder'} size="0.875rem" />
        ) : (
          <Codicon name="file" size="0.875rem" />
        )}
      </span>
      <span className="min-w-0 flex-1 truncate">{node.data.name}</span>
    </div>
  )
}
