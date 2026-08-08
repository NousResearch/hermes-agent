import type * as React from 'react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import type { SessionInfo } from '@/hermes'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import {
  createConversationGroup,
  deleteConversationGroup,
  renameConversationGroup,
  reorderConversationGroups
} from '@/store/projects'

import { SidebarRowStack } from '../chrome'

import { type SidebarConversationGroup, type SidebarProjectTree } from './workspace-groups'
import { WorkspaceHeader } from './workspace-header'

const copy = {
  add: '대화 그룹 추가',
  cancel: '취소',
  create: '생성',
  createDescription: '이 프로젝트 안에서 대화 세션을 정리할 그룹 이름을 입력하세요.',
  createTitle: '새 대화 그룹',
  delete: '삭제',
  deleteDescription: '그룹만 삭제됩니다. 포함된 대화는 삭제되지 않고 미분류로 이동합니다.',
  deleteTitle: '대화 그룹 삭제',
  edit: '이름 변경',
  editDescription: '대화 그룹의 새 이름을 입력하세요.',
  editTitle: '대화 그룹 이름 변경',
  groupName: '그룹 이름',
  menu: '대화 그룹 작업',
  moveDown: '아래로 이동',
  moveUp: '위로 이동',
  save: '저장',
  section: '대화 그룹'
}

type EditorState = { group?: SidebarConversationGroup; mode: 'create' | 'rename' }

export function ConversationGroups({
  project,
  renderRows
}: {
  project: SidebarProjectTree
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
}) {
  const groups = project.conversationGroups ?? []
  const [editor, setEditor] = useState<EditorState | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<SidebarConversationGroup | null>(null)
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const openCreate = () => {
    setName('')
    setEditor({ mode: 'create' })
  }

  const openRename = (group: SidebarConversationGroup) => {
    setName(group.label)
    setEditor({ group, mode: 'rename' })
  }

  const submit = async () => {
    const trimmed = name.trim()

    if (!editor || !trimmed) {
      return
    }

    setSubmitting(true)

    try {
      if (editor.mode === 'create') {
        await createConversationGroup(project.id, trimmed)
      } else if (editor.group) {
        await renameConversationGroup(editor.group.id, trimmed)
      }

      setEditor(null)
    } catch (error) {
      notifyError(error, editor.mode === 'create' ? copy.createTitle : copy.editTitle)
    } finally {
      setSubmitting(false)
    }
  }

  const move = async (index: number, delta: -1 | 1) => {
    const target = index + delta

    if (target < 0 || target >= groups.length) {
      return
    }

    const ids = groups.map(group => group.id)

    ;[ids[index], ids[target]] = [ids[target], ids[index]]

    try {
      await reorderConversationGroups(project.id, ids)
    } catch (error) {
      notifyError(error, copy.menu)
    }
  }

  const confirmDelete = async () => {
    const target = deleteTarget
    setDeleteTarget(null)

    if (!target) {
      return
    }

    try {
      await deleteConversationGroup(target.id)
    } catch (error) {
      notifyError(error, copy.deleteTitle)
    }
  }

  return (
    <>
      <div className="flex min-h-7 items-center gap-1 px-2 pt-1 text-[0.6875rem] font-semibold text-(--ui-text-secondary)">
        <Codicon className="text-(--ui-text-tertiary)" name="list-tree" size="0.75rem" />
        <span className="min-w-0 flex-1 truncate">{copy.section}</span>
        <button
          aria-label={copy.add}
          className="grid size-5 shrink-0 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={openCreate}
          type="button"
        >
          <Codicon name="add" size="0.75rem" />
        </button>
      </div>

      {groups.map((group, index) => (
        <ConversationGroupRow
          group={group}
          index={index}
          key={group.id}
          onDelete={() => setDeleteTarget(group)}
          onMove={delta => void move(index, delta)}
          onRename={() => openRename(group)}
          projectId={project.id}
          renderRows={renderRows}
          total={groups.length}
        />
      ))}

      <Dialog onOpenChange={open => !open && setEditor(null)} open={Boolean(editor)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editor?.mode === 'rename' ? copy.editTitle : copy.createTitle}</DialogTitle>
            <DialogDescription>
              {editor?.mode === 'rename' ? copy.editDescription : copy.createDescription}
            </DialogDescription>
          </DialogHeader>
          <Input
            aria-label={copy.groupName}
            autoFocus
            disabled={submitting}
            onChange={event => setName(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                void submit()
              }
            }}
            placeholder={copy.groupName}
            value={name}
          />
          <DialogFooter>
            <Button disabled={submitting} onClick={() => setEditor(null)} variant="ghost">
              {copy.cancel}
            </Button>
            <Button disabled={submitting || !name.trim()} onClick={() => void submit()}>
              {editor?.mode === 'rename' ? copy.save : copy.create}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog onOpenChange={open => !open && setDeleteTarget(null)} open={Boolean(deleteTarget)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{`${copy.deleteTitle}: ${deleteTarget?.label ?? ''}`}</DialogTitle>
            <DialogDescription>{copy.deleteDescription}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button onClick={() => setDeleteTarget(null)} variant="ghost">
              {copy.cancel}
            </Button>
            <Button onClick={() => void confirmDelete()} variant="destructive">
              {copy.delete}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

function ConversationGroupRow({
  group,
  index,
  onDelete,
  onMove,
  onRename,
  projectId,
  renderRows,
  total
}: {
  group: SidebarConversationGroup
  index: number
  onDelete: () => void
  onMove: (delta: -1 | 1) => void
  onRename: () => void
  projectId: string
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  total: number
}) {
  const [open, setOpen] = useState(true)

  return (
    <SidebarRowStack
      className={cn(
        'rounded-md transition-colors',
        'data-[session-drag-over=true]:bg-(--ui-control-active-background) data-[session-drag-over=true]:ring-1 data-[session-drag-over=true]:ring-(--ui-stroke-secondary)'
      )}
      data-conversation-group-drop
      data-group-id={group.id}
      data-project-id={projectId}
    >
      <WorkspaceHeader
        action={
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                aria-label={copy.menu}
                className="grid size-4 shrink-0 place-items-center rounded-sm text-(--ui-text-quaternary) opacity-0 hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100 data-[state=open]:opacity-100"
                onClick={event => event.stopPropagation()}
                type="button"
              >
                <Codicon name="kebab-vertical" size="0.75rem" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-44">
              <DropdownMenuItem onSelect={onRename}>
                <Codicon name="edit" size="0.75rem" />
                {copy.edit}
              </DropdownMenuItem>
              <DropdownMenuItem disabled={index === 0} onSelect={() => onMove(-1)}>
                <Codicon name="arrow-up" size="0.75rem" />
                {copy.moveUp}
              </DropdownMenuItem>
              <DropdownMenuItem disabled={index === total - 1} onSelect={() => onMove(1)}>
                <Codicon name="arrow-down" size="0.75rem" />
                {copy.moveDown}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-destructive" onSelect={onDelete}>
                <Codicon name="trash" size="0.75rem" />
                {copy.delete}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        }
        icon={<Codicon className="text-(--ui-text-tertiary)" name="folder" size="0.75rem" />}
        label={`${group.label} (${group.sessions.length})`}
        onToggle={() => setOpen(value => !value)}
        open={open}
      />
      {open && <SidebarRowStack className="pl-2.5">{renderRows(group.sessions)}</SidebarRowStack>}
    </SidebarRowStack>
  )
}
