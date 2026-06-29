import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import { $bindings } from '@/store/keybinds'

import { setTerminalTakeover } from '../store'

import {
  $activeTerminalId,
  $terminals,
  closeOtherTerminals,
  closeTerminal,
  createTerminal,
  selectTerminal,
  type TerminalEntry
} from './terminals'

const RAIL_ACTION =
  'grid size-6 place-items-center rounded text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring [-webkit-app-region:no-drag]'

/** Tooltip label with a trailing hotkey hint (the user's live binding). */
function hintLabel(text: string, combo?: string) {
  return combo ? (
    <span className="flex items-center gap-2">
      <span>{text}</span>
      <span className="opacity-55">{formatCombo(combo)}</span>
    </span>
  ) : (
    text
  )
}

/** Thin icon "bookmark" strip blended into the terminal surface, shown whenever a
 *  terminal exists. Each square is a tab (name + hotkey on hover); close via the
 *  shell's `exit`, middle-click, or the context menu. */
export function TerminalRail() {
  const { t } = useI18n()
  const terminals = useStore($terminals)
  const activeId = useStore($activeTerminalId)
  const bindings = useStore($bindings)
  const toggleHint = bindings['view.showTerminal']?.[0]
  const newHint = bindings['view.newTerminal']?.[0]

  return (
    <div
      className="group/rail relative z-40 flex h-full w-9 shrink-0 flex-col items-center border-l border-(--ui-stroke-quaternary) bg-(--ui-editor-surface-background)"
      // The rail sits at the pane's outer edge, under the collapsed sidebars'
      // hover-reveal triggers; mark it so those triggers go pointer-transparent
      // while it's hovered (see the suppression rules in styles.css) and a reach
      // for a tab can't drag in the file-browser/review panel.
      data-suppress-pane-reveal=""
    >
      <ul
        aria-label={t.rightSidebar.terminalsAria}
        className="flex min-h-0 flex-1 flex-col items-center gap-0.5 self-stretch overflow-y-auto overflow-x-hidden overscroll-contain py-1 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        role="tablist"
      >
        {terminals.map((term, index) => (
          <TerminalRailItem
            active={term.id === activeId}
            canCloseOthers={terminals.length > 1}
            index={index}
            key={term.id}
            term={term}
            toggleHint={toggleHint}
          />
        ))}
        <li className="flex w-full justify-center">
          <Tip label={hintLabel(t.rightSidebar.terminalNew, newHint)} side="left">
            <button
              aria-label={t.rightSidebar.terminalNew}
              className={cn(RAIL_ACTION, 'size-7 text-(--ui-text-quaternary)')}
              onClick={() => createTerminal()}
              type="button"
            >
              <Codicon name="add" size="0.8125rem" />
            </button>
          </Tip>
        </li>
      </ul>

      <div className="flex shrink-0 flex-col items-center pb-1.5">
        <Tip label={t.rightSidebar.terminalHide} side="left">
          <button
            aria-label={t.rightSidebar.terminalHide}
            className={cn(RAIL_ACTION, 'opacity-0 transition-opacity group-hover/rail:opacity-100')}
            onClick={() => setTerminalTakeover(false)}
            type="button"
          >
            <Codicon name="chevron-down" size="0.8125rem" />
          </button>
        </Tip>
      </div>
    </div>
  )
}

interface TerminalRailItemProps {
  active: boolean
  canCloseOthers: boolean
  index: number
  term: TerminalEntry
  toggleHint?: string
}

function TerminalRailItem({ active, canCloseOthers, index, term, toggleHint }: TerminalRailItemProps) {
  const { t } = useI18n()
  const label = `${index + 1}. ${term.title}`

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <li className="relative flex w-full justify-center [-webkit-app-region:no-drag]">
          {active && (
            <span
              aria-hidden="true"
              className="absolute inset-y-0.5 right-0 w-0.5 rounded-l-sm bg-(--ui-stroke-primary)"
            />
          )}
          <Tip label={hintLabel(label, toggleHint)} side="left">
            <button
              aria-label={label}
              aria-selected={active}
              className={cn(
                'grid size-7 place-items-center rounded-md transition-colors',
                active
                  ? 'bg-(--chrome-action-hover) text-foreground'
                  : 'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
              )}
              onAuxClick={event => {
                if (event.button === 1) {
                  event.preventDefault()
                  closeTerminal(term.id)
                }
              }}
              onClick={() => selectTerminal(term.id)}
              onMouseDown={event => {
                if (event.button === 1) {
                  event.preventDefault()
                }
              }}
              role="tab"
              type="button"
            >
              <Codicon
                className={cn(term.kind === 'agent' && !active && 'text-primary')}
                name={term.kind === 'agent' ? 'agent' : 'terminal'}
                size="0.875rem"
              />
            </button>
          </Tip>
        </li>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => closeTerminal(term.id)}>{t.common.close}</ContextMenuItem>
        <ContextMenuItem disabled={!canCloseOthers} onSelect={() => closeOtherTerminals(term.id)}>
          {t.rightSidebar.terminalCloseOthers}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => setTerminalTakeover(false)}>{t.rightSidebar.terminalHide}</ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}
