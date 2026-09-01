/**
 * The chrome for user sections: one foldable heading with an inline rename and
 * a small menu. The model is in `user-sections.ts`; nothing here holds state
 * that outlives a caret.
 */

import {
  cn,
  Codicon,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
  DisclosureCaret,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  RowButton,
  useValue
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { $botSections, $renamingSection, renameBotSection, setBotSectionIcon } from './user-sections'

interface UserSectionHeaderProps {
  collapsed: boolean
  count: number
  /** null for Unassigned, which has no record and therefore no menu. */
  id: null | string
  name: string
  onDelete: () => void
  onMove: (delta: number) => void
  onToggle: () => void
}

export function UserSectionHeader({
  collapsed,
  count,
  id,
  name,
  onDelete,
  onMove,
  onToggle
}: UserSectionHeaderProps) {
  const renamingId = useValue($renamingSection)
  // Absent means show it, so only an explicit false hides the glyph.
  const showIcon = useValue($botSections).find(section => section.id === id)?.icon !== false
  const renaming = Boolean(id) && renamingId === id
  const [draft, setDraft] = useState(name)

  const commit = () => {
    $renamingSection.set(null)

    if (id && draft.trim() && draft.trim() !== name) {
      renameBotSection(id, draft)
    }
  }

  // RIGHT-CLICK IS THE SAME MENU. The ⋯ button only appears on hover and is a
  // small target; right-clicking the heading is what people actually try
  // first. Both drive the identical actions, so neither can drift.
  const sectionMenu = id ? (
    <ContextMenuContent>
      <ContextMenuItem
        onSelect={() => {
          setDraft(name)
          $renamingSection.set(id)
        }}
      >
        Rename
      </ContextMenuItem>
      <ContextMenuItem onSelect={() => setBotSectionIcon(id, !showIcon)}>
        {showIcon ? 'Hide icon' : 'Show icon'}
      </ContextMenuItem>
      <ContextMenuItem onSelect={() => onMove(-1)}>Move up</ContextMenuItem>
      <ContextMenuItem onSelect={() => onMove(1)}>Move down</ContextMenuItem>
      <ContextMenuItem onSelect={onDelete} variant="destructive">
        Delete section (keeps its bots)
      </ContextMenuItem>
    </ContextMenuContent>
  ) : null

  const header = (
    <div className="group/section mt-1 flex w-full min-w-0 items-center gap-1 pr-1">
      {renaming ? (
        <input
          autoFocus
          className="ml-2 min-w-0 flex-1 rounded-[3px] border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-1 py-0.5 text-[0.6875rem] font-semibold uppercase tracking-wider outline-none"
          onBlur={commit}
          onChange={event => setDraft(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              commit()
            }

            if (event.key === 'Escape') {
              $renamingSection.set(null)
            }
          }}
          value={draft}
        />
      ) : (
        <RowButton
          aria-expanded={!collapsed}
          className={cn(
            'flex min-w-0 flex-1 items-center gap-1.5 rounded-md px-2 py-1.5 text-left',
            'text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)',
            'transition-colors hover:bg-(--chrome-action-hover) hover:text-(--ui-text-secondary)'
          )}
          onClick={onToggle}
          onDoubleClick={() => {
            if (id) {
              setDraft(name)
              $renamingSection.set(id)
            }
          }}
        >
          <DisclosureCaret open={!collapsed} />
          {showIcon ? <Codicon className="shrink-0" name={id ? 'folder' : 'inbox'} /> : null}
          <span className="min-w-0 truncate">{name}</span>
          <span aria-hidden className="min-w-0 flex-1" />
          <span className="shrink-0 font-normal tabular-nums">{count}</span>
        </RowButton>
      )}
      {/* Unassigned has no record to rename, reorder or delete — it is
          whatever is left over — so it gets no menu rather than a menu of
          disabled items. */}
      {id && !renaming ? (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              aria-label={`${name} section options`}
              className="shrink-0 rounded-md p-0.5 text-(--ui-text-quaternary) opacity-0 transition hover:text-foreground group-hover/section:opacity-100 focus-visible:opacity-100"
              type="button"
            >
              <Codicon name="ellipsis" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onSelect={() => {
                setDraft(name)
                $renamingSection.set(id)
              }}
            >
              Rename
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => setBotSectionIcon(id, !showIcon)}>
              {showIcon ? 'Hide icon' : 'Show icon'}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onMove(-1)}>Move up</DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onMove(1)}>Move down</DropdownMenuItem>
            {/* Deleting a section keeps every bot in it — they fall back to
                Unassigned. Said plainly here so nobody has to find out. */}
            <DropdownMenuItem onSelect={onDelete} variant="destructive">
              Delete section (keeps its bots)
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : null}
    </div>
  )

  return sectionMenu ? (
    <ContextMenu>
      <ContextMenuTrigger asChild>{header}</ContextMenuTrigger>
      {sectionMenu}
    </ContextMenu>
  ) : (
    header
  )
}
