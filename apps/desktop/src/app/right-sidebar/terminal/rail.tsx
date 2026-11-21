import { useStore } from '@nanostores/react'

import { ConnectionGlyph } from '@/app/chat/sidebar/connection-glyph'
import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { Tip, TipHintLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { isMetaClose, middleClickHandlers } from '@/lib/middle-click'
import { cn } from '@/lib/utils'
import { $connectionsRegistry } from '@/store/connections'
import { $bindings } from '@/store/keybinds'
import { usePortalShell } from '@/store/shell-variant'

import { setTerminalTakeover } from '../store'

import { NewTerminalButton } from './rail-new-terminal'
import { toggleTerminalPane } from './reveal-focus'
import { terminalHost, type TerminalHost } from './terminal-gateway'
import {
  $activeTerminalId,
  $terminalHostsMixed,
  $terminals,
  closeAllTerminals,
  closeOtherTerminals,
  closeTerminal,
  selectTerminal,
  type TerminalEntry
} from './terminals'

const RAIL_ACTION =
  'grid size-6 place-items-center rounded text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:bg-(--chrome-action-hover) focus-visible:text-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-sidebar-ring [-webkit-app-region:no-drag]'

/** Thin icon "bookmark" strip blended into the terminal surface, shown whenever a
 *  terminal exists. Each square is a tab (name + hotkey on hover); close via the
 *  shell's `exit`, middle-click, or the context menu. */
export function TerminalRail() {
  const redesigned = usePortalShell()
  const { t } = useI18n()
  const terminals = useStore($terminals)
  const activeId = useStore($activeTerminalId)
  const bindings = useStore($bindings)
  const toggleHint = bindings['view.showTerminal']?.[0]

  return (
    <div
      className={cn(
        'group/rail relative z-40 flex shrink-0 items-center border-(--ui-stroke-quaternary) bg-(--ui-terminal-surface-background)',
        redesigned ? 'h-10 w-full gap-1 border-b px-2.5' : 'h-full w-9 flex-col border-l'
      )}
      // The rail sits at the pane's outer edge, under the collapsed sidebars'
      // hover-reveal triggers; mark it so those triggers go pointer-transparent
      // while it's hovered (see the suppression rules in styles.css) and a reach
      // for a tab can't drag in the file-browser/review panel.
      data-suppress-pane-reveal=""
    >
      <ul
        aria-label={t.rightSidebar.terminalsAria}
        aria-orientation={redesigned ? 'horizontal' : 'vertical'}
        className={cn(
          'flex min-h-0 min-w-0 flex-1 items-center gap-0.5 self-stretch overscroll-contain py-1 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden',
          redesigned ? 'overflow-x-auto overflow-y-hidden' : 'flex-col overflow-y-auto overflow-x-hidden'
        )}
        role="tablist"
      >
        {terminals.map((term, index) => (
          <TerminalRailItem
            active={term.id === activeId}
            canCloseOthers={terminals.length > 1}
            horizontal={redesigned}
            index={index}
            key={term.id}
            term={term}
            toggleHint={toggleHint}
          />
        ))}
        <NewTerminalButton className={RAIL_ACTION} horizontal={redesigned} />
      </ul>

      <div className={cn('flex shrink-0 items-center', !redesigned && 'flex-col pb-1.5')}>
        <Tip label={t.rightSidebar.terminalHide} placement={redesigned ? 'toolbar' : 'right-rail'}>
          <button
            aria-label={t.rightSidebar.terminalHide}
            className={cn(
              RAIL_ACTION,
              !redesigned && 'opacity-0 transition-opacity focus-visible:opacity-100 group-hover/rail:opacity-100'
            )}
            onClick={() => (redesigned ? toggleTerminalPane() : setTerminalTakeover(false))}
            type="button"
          >
            <Codicon name={redesigned ? 'close' : 'chevron-down'} size="0.8125rem" />
          </button>
        </Tip>
      </div>
    </div>
  )
}

interface TerminalRailItemProps {
  active: boolean
  canCloseOthers: boolean
  horizontal: boolean
  index: number
  term: TerminalEntry
  toggleHint?: string
}

function TerminalRailItem({ active, canCloseOthers, horizontal, index, term, toggleHint }: TerminalRailItemProps) {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const mixed = useStore($terminalHostsMixed)
  const host = terminalHost(term, registry)
  // Where the shell runs. This device stays unlabeled until a tab runs elsewhere.
  const shown = host === 'removed' || (host && (mixed || host.connection.kind !== 'local')) ? host : null
  const hostLabel = shown === 'removed' ? t.rightSidebar.terminalGatewayRemoved : shown?.connection.label
  const label = `${index + 1}. ${term.title}${hostLabel ? ` · ${hostLabel}` : ''}`

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <li
          className={cn(
            'relative flex justify-center [-webkit-app-region:no-drag]',
            horizontal ? 'shrink-0' : 'w-full'
          )}
        >
          {/* The vertical rail marks its active tab with an edge bar; the redesign's
              horizontal pills read from their fill alone. */}
          {active && !horizontal && (
            <span
              aria-hidden="true"
              className="absolute inset-y-0.5 right-0 w-0.5 rounded-l-sm bg-(--ui-stroke-primary)"
            />
          )}
          <Tip
            label={<TipHintLabel hint={toggleHint && formatCombo(toggleHint)} text={label} />}
            placement={horizontal ? 'toolbar' : 'right-rail'}
          >
            <button
              aria-label={label}
              aria-selected={active}
              className={cn(
                'relative flex h-7 items-center justify-center transition-colors',
                horizontal ? 'max-w-40 gap-1.5 rounded-[3px] px-2.5 text-[0.8125rem]' : 'w-7 rounded-md',
                active
                  ? 'bg-(--chrome-action-hover) text-foreground'
                  : 'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
              )}
              {...middleClickHandlers(() => closeTerminal(term.id))}
              // ⌘-click closes (the pane-tab gesture); a plain click selects.
              onClick={event => (isMetaClose(event) ? closeTerminal(term.id) : selectTerminal(term.id))}
              role="tab"
              type="button"
            >
              <Codicon
                className={cn(term.kind === 'agent' && !active && 'text-primary')}
                name={term.kind === 'agent' ? 'agent' : 'terminal'}
                size="0.875rem"
              />
              {horizontal && <span className="truncate">{term.title}</span>}
              <TerminalHostBadge host={shown} />
            </button>
          </Tip>
        </li>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => closeTerminal(term.id)}>{t.common.close}</ContextMenuItem>
        <ContextMenuItem disabled={!canCloseOthers} onSelect={() => closeOtherTerminals(term.id)}>
          {t.rightSidebar.terminalCloseOthers}
        </ContextMenuItem>
        <ContextMenuItem onSelect={closeAllTerminals}>{t.rightSidebar.terminalCloseAll}</ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => (horizontal ? toggleTerminalPane() : setTerminalTakeover(false))}>
          {t.rightSidebar.terminalHide}
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}

function TerminalHostBadge({ host }: { host: TerminalHost }) {
  if (!host) {
    return null
  }

  return (
    <span
      className="pointer-events-none absolute bottom-0 right-0.5 grid place-items-center rounded-sm bg-(--ui-terminal-surface-background)"
      data-terminal-host={host === 'removed' ? 'removed' : host.connection.id}
    >
      {host === 'removed' ? (
        <Codicon className="text-(--ui-text-quaternary)" name="debug-disconnect" size="0.625rem" />
      ) : (
        <ConnectionGlyph className="size-2.5 [&_svg]:size-2.5" connection={host.connection} />
      )}
    </span>
  )
}
