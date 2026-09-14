import {
  closestCenter,
  DndContext,
  type DragEndEvent,
  type DragMoveEvent,
  type DragOverEvent,
  type DragStartEvent,
  PointerSensor,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import { SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { composerPanelCard } from '@/components/chat/composer-dock'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Kbd } from '@/components/ui/kbd'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import {
  ChevronDown,
  ChevronRight,
  Clipboard,
  FileText,
  FolderOpen,
  type IconComponent,
  ImageIcon,
  Link,
  MessageSquareText,
  Pencil,
  Plus,
  Trash2
} from '@/lib/icons'
import { REORDER_DRAG_TRANSITION_CSS, REORDER_RAIL_TRANSITION_CSS, reorderCommitHaptic } from '@/lib/reorder'
import { cn } from '@/lib/utils'
import {
  $promptTemplates,
  addFolder,
  addTemplate,
  deleteTemplate,
  type DropPlacement,
  ensureSeeded,
  placeNode,
  type PromptTemplate,
  resetToBuiltins,
  toggleFolderCollapsed,
  updateTemplate,
  visibleTreeRows
} from '@/store/prompt-templates'

import { useComposerAttachmentProviders } from './contrib'
import { GHOST_ICON_BTN } from './controls'
import type { ChatBarState } from './types'

/** Vertical-only auto-scroll when the pointer nears the list edge (same feel as sidebar). */
const TEMPLATE_LIST_AUTO_SCROLL = { threshold: { x: 0, y: 0.18 } } as const

export function ContextMenu({
  state,
  onInsertText,
  onOpenUrlDialog,
  onPasteClipboardImage,
  onPickFiles,
  onPickFolders,
  onPickImages
}: ContextMenuProps) {
  const { t } = useI18n()
  const c = t.composer
  // Prompt templates used to be a Radix submenu. That submenu didn't open
  // reliably when the parent menu was positioned at the bottom of the
  // window (composer "+" anchor), so we promoted it to a real Dialog —
  // easier to grow with search / descriptions, and no positioning math.
  const [templatesOpen, setTemplatesOpen] = useState(false)
  // `composer.attachments` contributions — plugin/core-registered rows that
  // extend this menu through the same registry as every other surface.
  const attachmentProviders = useComposerAttachmentProviders()

  return (
    <>
      <DropdownMenu>
        <Tip label={state.tools.label} side="top">
          <DropdownMenuTrigger asChild>
            <Button
              aria-label={state.tools.label}
              className={cn(
                GHOST_ICON_BTN,
                'data-[state=open]:bg-(--chrome-action-hover) data-[state=open]:text-foreground'
              )}
              disabled={!state.tools.enabled}
              size="icon"
              type="button"
              variant="ghost"
            >
              <Codicon name="add" size="0.875rem" />
            </Button>
          </DropdownMenuTrigger>
        </Tip>
        <DropdownMenuContent align="start" className={cn('w-60', composerPanelCard)} side="top" sideOffset={6}>
          <DropdownMenuLabel className="px-2 pb-0.5 pt-0.5 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
            {c.attachLabel}
          </DropdownMenuLabel>
          <ContextMenuItem disabled={!onPickFiles} icon={FileText} onSelect={onPickFiles}>
            {c.files}
          </ContextMenuItem>
          <ContextMenuItem disabled={!onPickFolders} icon={FolderOpen} onSelect={onPickFolders}>
            {c.folder}
          </ContextMenuItem>
          <ContextMenuItem disabled={!onPickImages} icon={ImageIcon} onSelect={onPickImages}>
            {c.images}
          </ContextMenuItem>
          <ContextMenuItem
            disabled={!onPasteClipboardImage}
            icon={Clipboard}
            onSelect={onPasteClipboardImage ? () => void onPasteClipboardImage() : undefined}
          >
            {c.pasteImage}
          </ContextMenuItem>
          <ContextMenuItem icon={Link} onSelect={onOpenUrlDialog}>
            {c.url}
          </ContextMenuItem>

          <DropdownMenuSeparator />

          <ContextMenuItem icon={MessageSquareText} onSelect={() => setTemplatesOpen(true)}>
            {c.promptTemplates}
          </ContextMenuItem>

          {attachmentProviders.length > 0 && <DropdownMenuSeparator />}
          {attachmentProviders.map(provider => (
            <DropdownMenuItem
              className="text-[length:var(--conversation-tool-font-size)] focus:bg-(--ui-bg-tertiary)"
              key={provider.key}
              onSelect={() => void provider.run({ insertText: onInsertText })}
            >
              <Codicon name={provider.icon ?? 'plug'} size="0.875rem" />
              <span>{provider.label}</span>
            </DropdownMenuItem>
          ))}

          <DropdownMenuSeparator />

          <div className="px-2 py-1 text-[0.7rem] text-muted-foreground/80">
            {c.tipPre}
            <Kbd size="sm">@</Kbd>
            {c.tipPost}
          </div>
        </DropdownMenuContent>
      </DropdownMenu>

      <PromptTemplatesDialog onInsertText={onInsertText} onOpenChange={setTemplatesOpen} open={templatesOpen} />
    </>
  )
}

function PromptTemplatesDialog({ onInsertText, onOpenChange, open }: PromptTemplatesDialogProps) {
  const { t } = useI18n()
  const c = t.composer
  const templates = useStore($promptTemplates)
  const rows = visibleTreeRows(templates)
  const rowIds = useMemo(() => rows.map(r => r.node.id), [rows])
  const [editingId, setEditingId] = useState<string | null>(null)
  const [dropHint, setDropHint] = useState<{ overId: string; placement: DropPlacement } | null>(null)
  const activeIdRef = useRef<string | null>(null)
  /** Live pointer Y — updated from dnd-kit delta (ul onPointerMove misses capture during drag). */
  const pointerYRef = useRef(0)
  const listRef = useRef<HTMLUListElement | null>(null)

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }))

  useEffect(() => {
    if (open) {
      ensureSeeded()
    }
  }, [open])

  useEffect(() => {
    if (editingId && !templates.some(s => s.id === editingId)) {
      setEditingId(null)
    }
  }, [editingId, templates])

  function handleAdd() {
    const created = addTemplate()
    setEditingId(created.id)
  }

  function handleAddFolder() {
    const created = addFolder()
    setEditingId(created.id)
  }

  function handleReset() {
    if (window.confirm(c.templateResetConfirm)) {
      resetToBuiltins()
      setEditingId(null)
    }
  }

  function handleInsert(template: PromptTemplate) {
    if (template.kind === 'folder') {
      return
    }

    onInsertText(template.text)
    onOpenChange(false)
  }

  function handleDelete(node: PromptTemplate) {
    const message = node.kind === 'folder' ? c.templateConfirmDeleteFolder : c.templateConfirmDelete

    if (window.confirm(message)) {
      deleteTemplate(node.id)
    }
  }

  /**
   * Pointer Y while dragging. The list's onPointerMove often stops once
   * PointerSensor captures the pointer — derive from activator + delta instead.
   */
  function pointerYFromDrag(event: { activatorEvent: Event; delta: { y: number } }): number {
    const act = event.activatorEvent

    if (act instanceof PointerEvent || act instanceof MouseEvent || act instanceof TouchEvent) {
      const startY =
        'clientY' in act
          ? act.clientY
          : act.touches[0]?.clientY ?? act.changedTouches[0]?.clientY ?? pointerYRef.current

      return startY + event.delta.y
    }

    return pointerYRef.current
  }

  function resolvePlacement(overId: string, clientY: number): DropPlacement {
    const overNode = templates.find(s => s.id === overId)
    const el = listRef.current?.querySelector<HTMLElement>(`[data-template-id="${CSS.escape(overId)}"]`)

    if (!el || !overNode) {
      return 'after'
    }

    const rect = el.getBoundingClientRect()
    const ratio = (clientY - rect.top) / Math.max(rect.height, 1)

    if (overNode.kind === 'folder') {
      // Collapsed folder: the whole row is a nest target — reorder-before must not
      // win over "put into this pack" (children are hidden, so nest is the intent).
      if (overNode.collapsed) {
        return 'inside'
      }

      // Expanded: thin top edge = sibling before; rest of row = nest inside.
      if (ratio < 0.15) {
        return 'before'
      }

      return 'inside'
    }

    return ratio < 0.5 ? 'before' : 'after'
  }

  function updateDropHint(overId: string | null, clientY: number) {
    if (!overId || overId === activeIdRef.current) {
      setDropHint(null)

      return
    }

    pointerYRef.current = clientY
    setDropHint({ overId, placement: resolvePlacement(overId, clientY) })
  }

  function handleDragStart({ active, activatorEvent }: DragStartEvent) {
    activeIdRef.current = String(active.id)
    setEditingId(null)

    if (activatorEvent instanceof PointerEvent || activatorEvent instanceof MouseEvent) {
      pointerYRef.current = activatorEvent.clientY
    }
  }

  function handleDragMove(event: DragMoveEvent) {
    const y = pointerYFromDrag(event)
    updateDropHint(event.over ? String(event.over.id) : null, y)
  }

  function handleDragOver(event: DragOverEvent) {
    const y = pointerYFromDrag(event)
    updateDropHint(event.over ? String(event.over.id) : null, y)
  }

  function handleDragCancel() {
    activeIdRef.current = null
    setDropHint(null)
  }

  function handleDragEnd(event: DragEndEvent) {
    const { activatorEvent, active, over } = event

    // Match sidebar: drop grabber focus so hover affordance doesn't stick "on".
    if (!(activatorEvent instanceof KeyboardEvent)) {
      ;(document.activeElement as HTMLElement | null)?.blur()
    }

    const fromId = String(active.id)
    const toId = over ? String(over.id) : null
    const y = pointerYFromDrag(event)
    const placement = toId ? resolvePlacement(toId, y) : null

    activeIdRef.current = null
    setDropHint(null)

    if (!toId || !placement || fromId === toId) {
      // Illegal / cancelled drop → no store write; dnd-kit snaps the row back.
      return
    }

    if (placeNode(fromId, toId, placement)) {
      reorderCommitHaptic()
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="flex max-h-[calc(85vh-2rem)] flex-col gap-3 overflow-hidden"
        className="max-w-lg"
      >
        <DialogHeader className="shrink-0">
          <DialogTitle>{c.templatesTitle}</DialogTitle>
          <DialogDescription>{c.templatesDesc}</DialogDescription>
        </DialogHeader>

        {templates.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted-foreground">{c.templateEmpty}</p>
        ) : (
          <DndContext
            autoScroll={TEMPLATE_LIST_AUTO_SCROLL}
            collisionDetection={closestCenter}
            onDragCancel={handleDragCancel}
            onDragEnd={handleDragEnd}
            onDragMove={handleDragMove}
            onDragOver={handleDragOver}
            onDragStart={handleDragStart}
            sensors={sensors}
          >
            <SortableContext items={rowIds} strategy={verticalListSortingStrategy}>
              <ul
                className="grid min-h-0 flex-1 gap-1 overflow-y-auto"
                onPointerMove={event => {
                  pointerYRef.current = event.clientY
                }}
                ref={listRef}
              >
                {rows.map(({ depth, node }) => (
                  <li key={node.id}>
                    {editingId === node.id ? (
                      <TemplateEditor
                        depth={depth}
                        onCancel={() => setEditingId(null)}
                        onSave={() => setEditingId(null)}
                        template={node}
                      />
                    ) : (
                      <SortableTemplateRow
                        depth={depth}
                        dropHint={dropHint?.overId === node.id ? dropHint.placement : null}
                        onDelete={() => handleDelete(node)}
                        onEdit={() => setEditingId(node.id)}
                        onInsert={() => handleInsert(node)}
                        onToggleFolder={() => toggleFolderCollapsed(node.id)}
                        template={node}
                      />
                    )}
                  </li>
                ))}
              </ul>
            </SortableContext>
          </DndContext>
        )}

        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-t pt-3">
          <Button onClick={handleReset} size="sm" variant="ghost">
            {c.templateReset}
          </Button>
          <div className="flex flex-wrap gap-2">
            <Button onClick={handleAddFolder} size="sm" variant="outline">
              <FolderOpen className="size-4" />
              {c.templateAddFolder}
            </Button>
            <Button onClick={handleAdd} size="sm" variant="outline">
              <Plus className="size-4" />
              {c.templateAdd}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function SortableTemplateRow({
  depth,
  dropHint,
  onDelete,
  onEdit,
  onInsert,
  onToggleFolder,
  template
}: SortableTemplateRowProps) {
  const { t } = useI18n()
  const c = t.composer
  const isFolder = template.kind === 'folder'
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id: template.id })

  return (
    <div
      className={cn(
        'group/template relative flex w-full items-start gap-1 rounded-md border border-transparent py-2 pr-2.5 text-left transition-colors hover:border-(--ui-stroke-tertiary) hover:bg-(--ui-control-hover-background)',
        isDragging && 'z-10 border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) opacity-90 shadow-sm',
        dropHint === 'inside' && 'border-(--ui-stroke-primary) bg-(--ui-control-hover-background)',
        dropHint === 'before' && 'border-t-2 border-t-(--ui-stroke-primary)',
        dropHint === 'after' && 'border-b-2 border-b-(--ui-stroke-primary)'
      )}
      data-template-id={template.id}
      ref={setNodeRef}
      style={{
        paddingLeft: `${0.625 + depth * 0.75}rem`,
        transform: transform ? `translate3d(0px, ${transform.y}px, 0)` : undefined,
        transition: isDragging ? REORDER_DRAG_TRANSITION_CSS : transition || REORDER_RAIL_TRANSITION_CSS,
        willChange: isDragging ? 'transform' : undefined
      }}
    >
      <button
        aria-label={c.templateReorder}
        className={cn(
          'mt-0.5 flex size-5 shrink-0 cursor-grab touch-none items-center justify-center rounded text-(--ui-text-quaternary) opacity-0 transition-opacity hover:text-foreground group-hover/template:opacity-100 active:cursor-grabbing',
          isDragging && 'opacity-100'
        )}
        type="button"
        {...attributes}
        {...listeners}
      >
        <Codicon name="grabber" size="0.75rem" />
      </button>

      {isFolder ? (
        <button
          aria-expanded={!template.collapsed}
          className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded text-(--ui-text-tertiary) hover:text-foreground"
          onClick={onToggleFolder}
          type="button"
        >
          {template.collapsed ? <ChevronRight className="size-3.5" /> : <ChevronDown className="size-3.5" />}
        </button>
      ) : (
        <span className="mt-0.5 size-5 shrink-0" />
      )}

      <button
        className="grid min-w-0 flex-1 cursor-pointer items-start gap-0.5 text-left"
        onClick={isFolder ? onToggleFolder : onInsert}
        type="button"
      >
        <span className="flex items-center gap-1.5 text-sm font-medium text-foreground">
          {isFolder ? <FolderOpen className="size-3.5 shrink-0 text-(--ui-text-tertiary)" /> : null}
          {template.label || (
            <span className="text-muted-foreground italic">
              {isFolder ? c.templateFolderPlaceholder : c.templateLabelPlaceholder}
            </span>
          )}
        </span>
        {!isFolder && template.description ? (
          <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {template.description}
          </span>
        ) : null}
        {!isFolder ? (
          <span className="truncate text-[length:var(--conversation-caption-font-size)] text-muted-foreground/70">
            {template.text}
          </span>
        ) : null}
      </button>

      <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover/template:opacity-100">
        <Tip label={c.templateEdit} side="top">
          <Button onClick={onEdit} size="icon" type="button" variant="ghost">
            <Pencil className="size-3.5" />
          </Button>
        </Tip>
        <Tip label={c.templateDelete} side="top">
          <Button onClick={onDelete} size="icon" type="button" variant="ghost">
            <Trash2 className="size-3.5" />
          </Button>
        </Tip>
      </div>
    </div>
  )
}

function TemplateEditor({ depth, onCancel, onSave, template }: TemplateEditorProps) {
  const { t } = useI18n()
  const c = t.composer
  const isFolder = template.kind === 'folder'
  const [label, setLabel] = useState(template.label)
  const [description, setDescription] = useState(template.description)
  const [text, setText] = useState(template.text)

  const labelRef = (el: HTMLInputElement | null) => {
    el?.focus()
  }

  function handleSave() {
    if (isFolder) {
      updateTemplate(template.id, { label: label.trim() })
    } else {
      updateTemplate(template.id, {
        label: label.trim(),
        description: description.trim(),
        text: text.trim()
      })
    }

    onSave()
  }

  return (
    <div
      className="grid gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary) p-3"
      style={{ marginLeft: `${depth * 0.75}rem` }}
    >
      <Input
        onChange={e => setLabel(e.target.value)}
        placeholder={isFolder ? c.templateFolderPlaceholder : c.templateLabelPlaceholder}
        ref={labelRef}
        value={label}
      />
      {isFolder ? null : (
        <>
          <Input
            onChange={e => setDescription(e.target.value)}
            placeholder={c.templateDescPlaceholder}
            value={description}
          />
          <Textarea
            onChange={e => setText(e.target.value)}
            placeholder={c.templateTextPlaceholder}
            rows={3}
            value={text}
          />
        </>
      )}
      <div className="flex justify-end gap-2">
        <Button onClick={onCancel} size="sm" type="button" variant="ghost">
          {c.templateCancel}
        </Button>
        <Button onClick={handleSave} size="sm" type="button">
          {c.templateSave}
        </Button>
      </div>
    </div>
  )
}

export function ContextMenuItem({ children, disabled, icon: Icon, onSelect }: ContextMenuItemProps) {
  return (
    // Override font size + highlight to match the / · @ completion rows exactly.
    <DropdownMenuItem
      className="text-[length:var(--conversation-tool-font-size)] focus:bg-(--ui-bg-tertiary)"
      disabled={disabled}
      onSelect={onSelect}
    >
      <Icon />
      <span>{children}</span>
    </DropdownMenuItem>
  )
}

interface ContextMenuItemProps {
  children: string
  disabled?: boolean
  icon: IconComponent
  onSelect?: () => void
}

interface ContextMenuProps {
  onInsertText: (text: string) => void
  onOpenUrlDialog: () => void
  onPasteClipboardImage?: (opts?: { silent?: boolean }) => Promise<boolean> | void
  onPickFiles?: () => void
  onPickFolders?: () => void
  onPickImages?: () => void
  state: ChatBarState
}

interface PromptTemplatesDialogProps {
  onInsertText: (text: string) => void
  onOpenChange: (open: boolean) => void
  open: boolean
}

interface SortableTemplateRowProps {
  depth: number
  dropHint: DropPlacement | null
  onDelete: () => void
  onEdit: () => void
  onInsert: () => void
  onToggleFolder: () => void
  template: PromptTemplate
}

interface TemplateEditorProps {
  depth: number
  onCancel: () => void
  onSave: () => void
  template: PromptTemplate
}
