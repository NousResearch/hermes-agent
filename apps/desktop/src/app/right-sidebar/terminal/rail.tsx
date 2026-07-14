import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Tip, TipHintLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import { $bindings } from '@/store/keybinds'

import { setTerminalTakeover } from '../store'

import {
  $activeTerminalId,
  $terminals,
  closeAllTerminals,
  closeOtherTerminals,
  closeTerminal,
  createTerminal,
  selectTerminal,
  type TerminalEntry
} from './terminals'

type TerminalMode = 'smart' | 'wsl2' | 'windows-native'

interface TerminalModeInfo {
  configuredMode: TerminalMode
  distribution: null | string
  resolvedMode: 'windows-native' | 'wsl2'
  supported: boolean
  wslDistributions: string[]
}

const RAIL_ACTION =
  'grid size-6 place-items-center rounded text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring [-webkit-app-region:no-drag]'

const TERMINAL_MODE_LABELS: Record<TerminalMode, string> = {
  smart: 'Smart',
  wsl2: 'WSL2',
  'windows-native': 'Windows Native'
}

const TERMINAL_MODE_ICONS: Record<TerminalMode, string> = {
  smart: 'server-environment',
  wsl2: 'terminal-linux',
  'windows-native': 'terminal-powershell'
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

  const [modeInfo, setModeInfo] = useState<TerminalModeInfo | null>(null)

  useEffect(() => {
    const api = window.hermesDesktop?.terminal?.mode

    if (!api) {
      return
    }

    void api
      .get()
      .then(setModeInfo)
      .catch(() => setModeInfo(null))
  }, [])

  const selectMode = (mode: TerminalMode) => {
    const api = window.hermesDesktop?.terminal?.mode

    if (!api) {
      return
    }

    void api
      .set(mode)
      .then(setModeInfo)
      .catch(() => undefined)
  }

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
          <Tip
            label={<TipHintLabel hint={newHint && formatCombo(newHint)} text={t.rightSidebar.terminalNew} />}
            side="left"
          >
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

      <div className="flex shrink-0 flex-col items-center gap-0.5 pb-1.5">
        {modeInfo?.supported && (
          <DropdownMenu>
            <Tip
              label={`Terminal mode: ${TERMINAL_MODE_LABELS[modeInfo.configuredMode]} (new terminals; agent commands update immediately)`}
              side="left"
            >
              <DropdownMenuTrigger asChild>
                <button
                  aria-label={`Terminal mode: ${TERMINAL_MODE_LABELS[modeInfo.configuredMode]}`}
                  className={RAIL_ACTION}
                  type="button"
                >
                  <Codicon name={TERMINAL_MODE_ICONS[modeInfo.configuredMode]} size="0.8125rem" />
                </button>
              </DropdownMenuTrigger>
            </Tip>
            <DropdownMenuContent align="end" side="left">
              {(Object.keys(TERMINAL_MODE_LABELS) as TerminalMode[]).map(mode => (
                <DropdownMenuItem
                  disabled={mode === 'wsl2' && modeInfo.wslDistributions.length === 0}
                  key={mode}
                  onSelect={() => selectMode(mode)}
                >
                  <Codicon name={TERMINAL_MODE_ICONS[mode]} />
                  <span className="flex-1">{TERMINAL_MODE_LABELS[mode]}</span>
                  {modeInfo.configuredMode === mode && <Codicon name="check" />}
                </DropdownMenuItem>
              ))}
              {modeInfo.configuredMode === 'smart' && (
                <div className="max-w-56 px-2 py-1 text-[0.65rem] text-(--ui-text-tertiary)">
                  Smart uses WSL2 for projects opened through a WSL UNC path; other projects stay Windows native.
                </div>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
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
          <Tip label={<TipHintLabel hint={toggleHint && formatCombo(toggleHint)} text={label} />} side="left">
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
        <ContextMenuItem onSelect={closeAllTerminals}>{t.rightSidebar.terminalCloseAll}</ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onSelect={() => setTerminalTakeover(false)}>{t.rightSidebar.terminalHide}</ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}
