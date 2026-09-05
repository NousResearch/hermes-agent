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
  DropdownMenuTrigger,
  Input,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
import { type DragEvent, type ReactNode, useEffect, useRef, useState } from 'react'

import { useBots } from './i18n'
import { RosterSectionHeader } from './roster-sections'
import { $draggingBot, BOT_DRAG_MIME } from './user-sections'

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
            if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
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

interface UserSectionHeaderProps {
  canMoveDown: boolean
  canMoveUp: boolean
  collapsed: boolean
  count: number
  hasCustomOrder?: boolean
  /** null for Unassigned, which has no record and therefore no menu. */
  id: null | string
  name: string
  onDelete: () => void
  onMove: (delta: number) => void
  onRename: () => void
  onResetOrder?: () => void
  onToggle: () => void
}

export function UserSectionHeader({
  canMoveDown,
  canMoveUp,
  collapsed,
  count,
  hasCustomOrder,
  id,
  name,
  onDelete,
  onMove,
  onRename,
  onResetOrder,
  onToggle
}: UserSectionHeaderProps) {
  const b = useBots()
  const { t } = useI18n()

  // Unassigned has no record to rename, reorder or delete - it is whatever is
  // left over - but when it has a custom order, offer Reset custom order.
  if (!id) {
    if (!hasCustomOrder || !onResetOrder) {
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

    const unassignedItems = [{ icon: 'clear-all', label: b.sections.resetOrder, onSelect: onResetOrder }]

    const unassignedAction = (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            aria-label={b.sections.options(b.sections.unassigned)}
            className="shrink-0 rounded-md p-0.5 text-(--ui-text-quaternary) opacity-0 transition hover:text-foreground group-hover/section:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100"
            type="button"
          >
            <Codicon name="ellipsis" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {unassignedItems.map(item => (
            <DropdownMenuItem key={item.label} onSelect={item.onSelect}>
              <Codicon className="mr-1.5" name={item.icon} />
              {item.label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    )

    return (
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <div data-section-id="unassigned" data-slot="bots-section-heading">
            <RosterSectionHeader
              action={unassignedAction}
              collapsed={collapsed}
              count={count}
              icon="inbox"
              label={b.sections.unassigned}
              onToggle={onToggle}
            />
          </div>
        </ContextMenuTrigger>
        <ContextMenuContent>
          {unassignedItems.map(item => (
            <ContextMenuItem key={item.label} onSelect={item.onSelect}>
              {item.label}
            </ContextMenuItem>
          ))}
        </ContextMenuContent>
      </ContextMenu>
    )
  }

  // RIGHT-CLICK IS THE SAME MENU. The ⋯ button only appears on hover and is a
  // small target; right-clicking the heading is what people actually try
  // first. Both drive the identical actions, so neither can drift.
  const items = [
    { icon: 'edit', label: b.sections.rename, onSelect: onRename },
    { disabled: !canMoveUp, icon: 'arrow-up', label: b.sections.moveUp, onSelect: () => onMove(-1) },
    { disabled: !canMoveDown, icon: 'arrow-down', label: b.sections.moveDown, onSelect: () => onMove(1) },
    ...(hasCustomOrder && onResetOrder
      ? [{ icon: 'clear-all', label: b.sections.resetOrder, onSelect: onResetOrder }]
      : [])
  ]

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
        <div data-section-id={id} data-slot="bots-section-heading">
          <RosterSectionHeader
            action={action}
            collapsed={collapsed}
            count={count}
            icon="folder"
            label={name}
            onDoubleClick={onRename}
            onToggle={onToggle}
            tip={b.sections.headingTip}
          />
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        {items.map(item => (
          <ContextMenuItem disabled={item.disabled} key={item.label} onSelect={item.onSelect}>
            {item.label}
          </ContextMenuItem>
        ))}
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
  /** Drawn inside a gateway bucket: indented under a hairline rail so the
   *  two heading levels read as parent and child. */
  nested?: boolean
  onDropBot: (rosterKey: string) => void
}

/** A section block as a drop target: the whole block (heading + rows, or the
 *  empty placeholder) lights up while a bot is over it. */
export function SectionDropZone({ children, isSource, nested, onDropBot }: SectionDropZoneProps) {
  const dragging = useValue($draggingBot)
  const [over, setOver] = useState(false)
  const armed = Boolean(dragging) && !isSource
  const lit = armed && over

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
        nested && 'ml-2.5 border-l border-(--ui-stroke-tertiary) pl-1',
        // While a drag is live, every valid target gets a faint outline so the
        // user can see where a drop is allowed before hovering one.
        armed && 'ring-1 ring-inset ring-(--ui-stroke-secondary)',
        lit && 'bg-(--ui-accent)/10 ring-(--ui-accent)'
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
    >
      {children}
    </div>
  )
}
