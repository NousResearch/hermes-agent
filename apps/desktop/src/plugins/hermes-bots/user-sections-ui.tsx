/**
 * The chrome for user sections: the foldable heading (the roster's own
 * `RosterSectionHeader`, with a ⋯ menu and a right-click menu that drive the
 * same actions), the name dialog used for both New section and Rename (the
 * same shape the app's session rename uses), and the drop zone a section
 * block sits in. The model is in `user-sections.ts`; nothing here holds state
 * that outlives a dialog.
 */

import {
  Button,
  cn,
  Codicon,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger,
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
  Input,
  isSubmitEnter,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
import { type DragEvent, type ReactNode, useEffect, useId, useRef, useState } from 'react'

import { useBots } from './i18n'
import { RosterSectionHeader } from './roster-sections'
import { $draggingBot, BOT_DRAG_MIME, SECTION_DRAG_PREFIX } from './user-sections'

// ── name dialog ──────────────────────────────────────────────────────────────

interface SectionNameDialogProps {
  /** Blank for New section, the current name for Rename. */
  initialName: string
  mode: 'create' | 'rename'
  onOpenChange: (open: boolean) => void
  onSubmit: (name: string) => void
  open: boolean
}

/** One small dialog for both creating and renaming a section — the app renames
 *  sessions through the same Dialog + Input + Cancel/Save shape, so a section
 *  rename feels like every other rename. */
export function SectionNameDialog({ initialName, mode, onOpenChange, onSubmit, open }: SectionNameDialogProps) {
  const { t } = useI18n()
  const b = useBots()
  const [value, setValue] = useState(initialName)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(initialName)
      window.setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [initialName, open])

  const submit = () => {
    const next = value.trim()

    if (!next) {
      return
    }

    onOpenChange(false)

    if (mode === 'create' || next !== initialName.trim()) {
      onSubmit(next)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>{mode === 'create' ? b.sections.newTitle : b.sections.renameTitle}</DialogTitle>
        </DialogHeader>
        <Input
          aria-label={b.sections.nameLabel}
          autoFocus
          maxLength={40}
          onChange={event => setValue(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event)) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder={b.sections.namePlaceholder}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={!value.trim()} onClick={submit} type="button">
            {mode === 'create' ? b.sections.create : t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ── heading ──────────────────────────────────────────────────────────────────

/** A folder this section can move under, labelled with its full path. */
export interface SectionParentOption {
  id: null | string
  label: string
}

interface UserSectionHeaderProps {
  canMoveDown: boolean
  canMoveUp: boolean
  collapsed: boolean
  count: number
  /** null for Unassigned, which has no record and therefore no menu. */
  id: null | string
  name: string
  onDelete: () => void
  onMove: (delta: number) => void
  onMoveInto: (parentId: null | string) => void
  onNewSubsection: () => void
  onRename: () => void
  onToggle: () => void
  /** Where "Move into" may put this section: never itself or a descendant.
   *  The current parent is listed but disabled. */
  parentId: null | string
  parentOptions: SectionParentOption[]
}

export function UserSectionHeader({
  canMoveDown,
  canMoveUp,
  collapsed,
  count,
  id,
  name,
  onDelete,
  onMove,
  onMoveInto,
  onNewSubsection,
  onRename,
  onToggle,
  parentId,
  parentOptions
}: UserSectionHeaderProps) {
  const b = useBots()
  const { t } = useI18n()
  const dragHintId = useId()

  // Unassigned has no record to rename, reorder or delete — it is whatever is
  // left over — so it gets the plain heading rather than a menu of disabled
  // items.
  if (!id) {
    return (
      <RosterSectionHeader
        collapsed={collapsed}
        count={count}
        icon="inbox"
        label={b.sections.unassigned}
        onToggle={onToggle}
      />
    )
  }

  // RIGHT-CLICK IS THE SAME MENU. The ⋯ button only appears on hover and is a
  // small target; right-clicking the heading is what people actually try
  // first. Both drive the identical actions, so neither can drift.
  const items = [
    { icon: 'edit', label: b.sections.rename, onSelect: onRename },
    { disabled: !canMoveUp, icon: 'arrow-up', label: b.sections.moveUp, onSelect: () => onMove(-1) },
    { disabled: !canMoveDown, icon: 'arrow-down', label: b.sections.moveDown, onSelect: () => onMove(1) },
    { icon: 'new-folder', label: b.sections.newSubsection, onSelect: onNewSubsection }
  ]

  // Nesting goes through a menu, not only drag, so it works from the keyboard.
  const destinations = parentOptions.map(option => ({
    ...option,
    disabled: option.id === parentId,
    onSelect: () => onMoveInto(option.id)
  }))

  const action = (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          aria-label={b.sections.options(name)}
          className="shrink-0 rounded-md p-0.5 text-(--ui-text-quaternary) opacity-0 transition hover:text-foreground group-hover/section:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100"
          type="button"
        >
          <Codicon name="ellipsis" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {items.map(item => (
          <DropdownMenuItem disabled={item.disabled} key={item.label} onSelect={item.onSelect}>
            <Codicon className="mr-1.5" name={item.icon} />
            {item.label}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>
            <Codicon className="mr-1.5" name="folder" />
            {b.sections.moveInto}
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            {destinations.map(item => (
              <DropdownMenuItem disabled={item.disabled} key={item.id ?? ''} onSelect={item.onSelect}>
                {item.label}
              </DropdownMenuItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={onDelete} variant="destructive">
          <Codicon className="mr-1.5" name="trash" />
          {t.common.delete}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        {/* The heading is the drag handle for sorting folders; Move up/down
            and Move section into are the keyboard path to the same result. */}
        <div
          data-section-id={id}
          data-slot="bots-section-heading"
          draggable
          onDragEnd={() => $draggingBot.set(null)}
          onDragStart={event => {
            const key = `${SECTION_DRAG_PREFIX}${id}`

            event.dataTransfer.setData(BOT_DRAG_MIME, key)
            event.dataTransfer.effectAllowed = 'move'
            $draggingBot.set(key)
          }}
        >
          <RosterSectionHeader
            action={action}
            collapsed={collapsed}
            count={count}
            describedBy={dragHintId}
            icon="folder"
            label={name}
            onDoubleClick={onRename}
            onToggle={onToggle}
            tip={b.sections.headingTip}
          />
          <span className="sr-only" id={dragHintId}>
            {b.sections.dragHint}
          </span>
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        {items.map(item => (
          <ContextMenuItem disabled={item.disabled} key={item.label} onSelect={item.onSelect}>
            {item.label}
          </ContextMenuItem>
        ))}
        <ContextMenuSub>
          <ContextMenuSubTrigger>{b.sections.moveInto}</ContextMenuSubTrigger>
          <ContextMenuSubContent>
            {destinations.map(item => (
              <ContextMenuItem disabled={item.disabled} key={item.id ?? ''} onSelect={item.onSelect}>
                {item.label}
              </ContextMenuItem>
            ))}
          </ContextMenuSubContent>
        </ContextMenuSub>
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={onDelete} variant="destructive">
          {t.common.delete}
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}

// ── drop zone ────────────────────────────────────────────────────────────────

/** While a bot is in flight, Escape cancels the gesture. Mount once in the
 *  roster pane. */
export function useEscapeCancelsBotDrag(): void {
  const dragging = useValue($draggingBot)

  useEffect(() => {
    if (!dragging) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        $draggingBot.set(null)
      }
    }

    window.addEventListener('keydown', onKeyDown, true)

    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [dragging])
}

interface SectionDropZoneProps {
  children: ReactNode
  /** Whether the dragged bot is already filed here — then the zone is not a
   *  target, and the OS shows the no-drop cursor instead of a highlight that
   *  promises a move that would change nothing. */
  isSource: boolean
  /** Indent levels: one for a block drawn inside a gateway bucket, plus one
   *  per parent folder. Drawn under a hairline rail so the heading levels
   *  read as parent and child. */
  indent?: number
  onDropBot: (rosterKey: string) => void
}

/** A section block as a drop target: the whole block (heading + rows, or the
 *  empty placeholder) lights up while a bot is over it. */
export function SectionDropZone({ children, indent = 0, isSource, onDropBot }: SectionDropZoneProps) {
  const dragging = useValue($draggingBot)
  const [over, setOver] = useState(false)
  const armed = Boolean(dragging) && !isSource
  const lit = armed && over
  // A folder in flight lands BEFORE the block it is dropped on, so mark the
  // insertion edge rather than lighting the block as a container.
  const sorting = Boolean(dragging?.startsWith(SECTION_DRAG_PREFIX))

  // Escape cancels the gesture: the in-flight key is cleared (see the
  // keydown hook in the roster pane), so every zone disarms at once and a
  // drop that still lands is refused below. Reset the hover so the next drag
  // starts clean.
  useEffect(() => {
    if (!dragging) {
      setOver(false)
    }
  }, [dragging])

  const accepts = (event: DragEvent) => armed && event.dataTransfer.types.includes(BOT_DRAG_MIME)

  return (
    <div
      className={cn(
        'relative min-w-0 rounded-md transition-[background-color,box-shadow] duration-100',
        indent > 0 && 'border-l border-(--ui-stroke-tertiary) pl-1',
        // While a drag is live, every valid target gets a faint outline so the
        // user can see where a drop is allowed before hovering one.
        armed && 'ring-1 ring-inset ring-(--ui-stroke-secondary)',
        lit && !sorting && 'bg-(--ui-accent)/10 ring-(--ui-accent)',
        lit && sorting && 'shadow-[inset_0_2px_0_var(--ui-accent)]'
      )}
      data-drop-over={lit ? 'true' : undefined}
      data-slot="bots-section"
      onDragEnter={event => {
        if (accepts(event)) {
          event.preventDefault()
          setOver(true)
        }
      }}
      onDragLeave={event => {
        // Only clear when the pointer leaves the BLOCK, not when it crosses
        // between the rows inside it — dragleave fires on every child
        // boundary, which otherwise strobes the highlight.
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setOver(false)
        }
      }}
      onDragOver={event => {
        if (!accepts(event)) {
          return
        }

        // preventDefault is what MAKES this a drop target — without it the
        // browser refuses the drop and the cursor stays "no entry".
        event.preventDefault()
        event.dataTransfer.dropEffect = 'move'

        if (!over) {
          setOver(true)
        }
      }}
      onDrop={event => {
        setOver(false)
        // The dropped row remounts under its new section, so its own dragend
        // never reaches the new node — clear the in-flight state here or the
        // row stays faded after a successful drop.
        $draggingBot.set(null)

        const key = event.dataTransfer.getData(BOT_DRAG_MIME)

        // No in-flight key means the user pressed Escape mid-drag: refuse.
        if (!key || !dragging || isSource) {
          return
        }

        event.preventDefault()
        onDropBot(key)
      }}
      style={indent > 0 ? { marginLeft: `${0.625 * indent}rem` } : undefined}
    >
      {children}
    </div>
  )
}
