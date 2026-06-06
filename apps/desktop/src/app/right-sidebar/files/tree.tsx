import { useCallback, useRef, useState } from 'react'
import { type NodeApi, type NodeRendererProps, Tree, type TreeApi } from 'react-arborist'

import { PageLoader } from '@/components/page-loader'
import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { trashFile } from './ipc'
import type { TreeNode } from './use-project-tree'

const ROW_HEIGHT = 22
const INDENT = 10

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
  const [deleteTarget, setDeleteTarget] = useState<TreeNode | null>(null)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)

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

  const handleActivate = useCallback(
    (node: NodeApi<TreeNode>) => {
      if (node.data && !node.data.isDirectory) {
        onPreviewFile?.(node.data.id)
      }
    },
    [onPreviewFile]
  )

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) {
      return
    }

    const path = deleteTarget.id
    const result = await trashFile(path)

    if (!result.ok) {
      const name = deleteTarget.isDirectory ? 'folder' : 'file'
      // eslint-disable-next-line no-alert
      window.alert(`Could not move ${name} to trash: ${result.error || 'unknown error'}`)
    }

    setDeleteDialogOpen(false)
    setDeleteTarget(null)
  }, [deleteTarget])

  const handleDeleteRequest = useCallback(
    (node: TreeNode) => {
      setDeleteTarget(node)
      setDeleteDialogOpen(true)
    },
    []
  )

  return (
    <div className="min-h-0 flex-1 overflow-hidden" ref={containerRef}>
      {size.height > 0 && size.width > 0 ? (
        <>
          <Tree<TreeNode>
            childrenAccessor={node => (node?.isDirectory ? (node.children ?? []) : null)}
            data={data}
            disableDrag
            disableDrop
            disableEdit
            height={size.height}
            indent={INDENT}
            initialOpenState={openState}
            key={`${cwd}:${collapseNonce}`}
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
                onDeleteFile={handleDeleteRequest}
                onPreviewFile={onPreviewFile}
              />
            )}
          </Tree>
          <DeleteConfirmDialog
            deleteTarget={deleteTarget}
            open={deleteDialogOpen}
            onCancel={() => {
              setDeleteDialogOpen(false)
              setDeleteTarget(null)
            }}
            onConfirm={confirmDelete}
          />
        </>
      ) : (
        <TreeSizingState />
      )}
    </div>
  )
}

function TreeSizingState() {
  const { t } = useI18n()

  return <PageLoader aria-label={t.rightSidebar.loadingFiles} className="min-h-24 px-3" />
}

function DeleteConfirmDialog({
  deleteTarget,
  open,
  onCancel,
  onConfirm
}: {
  deleteTarget: TreeNode | null
  open: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  const name = deleteTarget?.name ?? ''
  const kind = deleteTarget?.isDirectory ? 'folder' : 'file'

  return (
    <Dialog onOpenChange={open => !open && onCancel()} open={open}>
      <DialogContent className="sm:max-w-md" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Move to trash</DialogTitle>
          <DialogDescription>
            Are you sure you want to move the {kind} <strong>{name}</strong> to the trash?
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button onClick={onCancel} variant="ghost">
            Cancel
          </Button>
          <Button onClick={onConfirm} variant="destructive">
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ProjectTreeRow({
  dragHandle,
  node,
  onAttachFile,
  onAttachFolder,
  onDeleteFile,
  onPreviewFile,
  style
}: NodeRendererProps<TreeNode> & {
  onAttachFile: (path: string) => void
  onAttachFolder: (path: string) => void
  onDeleteFile?: (node: TreeNode) => void
  onPreviewFile?: (path: string) => void
}) {
  if (!node.data) {
    return <div style={style} />
  }

  const isFolder = node.data.isDirectory
  const isPlaceholder = node.data.id.endsWith('::__loading__')

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
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
          onKeyDown={event => {
            if (isPlaceholder) {
              return
            }

            if ((event.key === 'Delete' || event.key === 'Backspace') && onDeleteFile && node.data) {
              event.preventDefault()
              event.stopPropagation()
              onDeleteFile(node.data)
            }
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
      </ContextMenuTrigger>
      {!isPlaceholder && onDeleteFile && node.data && (
        <ContextMenuContent>
          <ContextMenuItem
            onSelect={() => {
              if (node.data) {
                onDeleteFile(node.data)
              }
            }}
            variant="destructive"
          >
            <Codicon name="trash" size="0.75rem" />
            Delete
          </ContextMenuItem>
        </ContextMenuContent>
      )}
    </ContextMenu>
  )
}
