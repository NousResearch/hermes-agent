import { useStore } from '@nanostores/react'
import * as React from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
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
import { Input } from '@/components/ui/input'
import { Tip, TipHintLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { isMetaClose, middleClickHandlers } from '@/lib/middle-click'
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
  renameTerminal,
  selectTerminal,
  TERMINAL_COLORS,
  TERMINAL_ICONS,
  type TerminalColor,
  type TerminalEntry,
  type TerminalIcon,
  updateTerminalAppearance
} from './terminals'

const RAIL_ACTION =
  'grid size-6 place-items-center rounded text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:bg-(--chrome-action-hover) focus-visible:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring [-webkit-app-region:no-drag]'

/** Rail tints, mapped onto the skin's own palette variables so every theme and
 *  light/dark mode keeps them legible. Exported so a test can assert the mapping
 *  without restating it. */
export const TERMINAL_COLOR_VARS: Record<TerminalColor, string> = {
  blue: 'var(--ui-blue)',
  cyan: 'var(--ui-cyan)',
  green: 'var(--ui-green)',
  orange: 'var(--ui-orange)',
  purple: 'var(--ui-purple)',
  red: 'var(--ui-red)',
  yellow: 'var(--ui-yellow)'
}

export const TERMINAL_ICON_LABELS: Record<TerminalIcon, string> = {
  beaker: 'Beaker',
  code: 'Code',
  database: 'Database',
  github: 'GitHub',
  server: 'Server',
  terminal: 'Terminal'
}

export const TERMINAL_COLOR_LABELS: Record<TerminalColor, string> = {
  blue: 'Blue',
  cyan: 'Cyan',
  green: 'Green',
  orange: 'Orange',
  purple: 'Purple',
  red: 'Red',
  yellow: 'Yellow'
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
      className="group/rail relative z-40 flex h-full w-9 shrink-0 flex-col items-center border-l border-(--ui-stroke-quaternary) bg-(--ui-terminal-surface-background)"
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
  const [renaming, setRenaming] = React.useState(false)
  const [draft, setDraft] = React.useState(term.title)
  const label = `${index + 1}. ${term.title}`

  // An agent mirror is a fixed, app-owned view of a background process: its name
  // and glyph come from the process, so only user tabs are restyleable.
  const stylable = term.kind === 'user'
  const icon = term.icon ?? (term.kind === 'agent' ? 'agent' : 'terminal')
  const tint = term.color ? TERMINAL_COLOR_VARS[term.color] : undefined

  return (
    <>
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
                {...middleClickHandlers(() => closeTerminal(term.id))}
                // ⌘-click closes (the pane-tab gesture); a plain click selects.
                onClick={event => (isMetaClose(event) ? closeTerminal(term.id) : selectTerminal(term.id))}
                role="tab"
                type="button"
              >
                <Codicon
                  className={cn(term.kind === 'agent' && !active && 'text-primary')}
                  data-terminal-color={term.color}
                  name={icon}
                  size="0.875rem"
                  style={tint ? { color: tint } : undefined}
                />
              </button>
            </Tip>
          </li>
        </ContextMenuTrigger>
        <ContextMenuContent>
          <ContextMenuItem
            disabled={!stylable}
            onSelect={() => {
              setDraft(term.title)
              setRenaming(true)
            }}
          >
            {t.rightSidebar.terminalRename}
          </ContextMenuItem>
          <ContextMenuSeparator />
          <ContextMenuSub>
            <ContextMenuSubTrigger disabled={!stylable}>{t.rightSidebar.terminalIcon}</ContextMenuSubTrigger>
            <ContextMenuSubContent>
              {TERMINAL_ICONS.map(name => (
                <ContextMenuItem key={name} onSelect={() => updateTerminalAppearance(term.id, { icon: name })}>
                  <Codicon name={name} size="0.875rem" />
                  {TERMINAL_ICON_LABELS[name]}
                  {term.icon === name && <Codicon className="ml-auto text-(--ui-accent)" name="check" size="0.75rem" />}
                </ContextMenuItem>
              ))}
            </ContextMenuSubContent>
          </ContextMenuSub>
          <ContextMenuSub>
            <ContextMenuSubTrigger disabled={!stylable}>{t.rightSidebar.terminalColor}</ContextMenuSubTrigger>
            <ContextMenuSubContent>
              {TERMINAL_COLORS.map(name => (
                <ContextMenuItem key={name} onSelect={() => updateTerminalAppearance(term.id, { color: name })}>
                  <Codicon name="circle-filled" size="0.75rem" style={{ color: TERMINAL_COLOR_VARS[name] }} />
                  {TERMINAL_COLOR_LABELS[name]}
                  {term.color === name && (
                    <Codicon className="ml-auto text-(--ui-accent)" name="check" size="0.75rem" />
                  )}
                </ContextMenuItem>
              ))}
            </ContextMenuSubContent>
          </ContextMenuSub>
          <ContextMenuItem
            disabled={!stylable || (!term.color && !term.icon && term.auto)}
            onSelect={() => {
              updateTerminalAppearance(term.id, {})

              if (!term.auto) {
                renameTerminal(term.id, '')
              }
            }}
          >
            {t.rightSidebar.terminalAppearanceReset}
          </ContextMenuItem>
          <ContextMenuSeparator />
          <ContextMenuItem onSelect={() => closeTerminal(term.id)}>{t.common.close}</ContextMenuItem>
          <ContextMenuItem disabled={!canCloseOthers} onSelect={() => closeOtherTerminals(term.id)}>
            {t.rightSidebar.terminalCloseOthers}
          </ContextMenuItem>
          <ContextMenuItem onSelect={closeAllTerminals}>{t.rightSidebar.terminalCloseAll}</ContextMenuItem>
          <ContextMenuSeparator />
          <ContextMenuItem onSelect={() => setTerminalTakeover(false)}>{t.rightSidebar.terminalHide}</ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>

      <TerminalRenameDialog
        onChange={setDraft}
        onClose={() => setRenaming(false)}
        onRename={title => {
          renameTerminal(term.id, title)
          setRenaming(false)
        }}
        open={renaming}
        value={draft}
      />
    </>
  )
}

/** Rename prompt for a terminal tab. A dialog rather than an inline field: the
 *  rail is a 36px icon strip with no room for one. */
function TerminalRenameDialog({
  onChange,
  onClose,
  onRename,
  open,
  value
}: {
  onChange: (value: string) => void
  onClose: () => void
  onRename: (title: string) => void
  open: boolean
  value: string
}) {
  const { t } = useI18n()
  const trimmed = value.trim()

  return (
    <Dialog onOpenChange={open => (open ? undefined : onClose())} open={open}>
      <DialogContent bodyClassName="gap-5" className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t.rightSidebar.terminalRenameTitle}</DialogTitle>
          <DialogDescription>{t.rightSidebar.terminalRenameDesc}</DialogDescription>
        </DialogHeader>
        <form
          className="grid gap-4"
          onSubmit={event => {
            event.preventDefault()
            onRename(trimmed)
          }}
        >
          <Input
            autoComplete="off"
            autoCorrect="off"
            onChange={event => onChange(event.target.value)}
            spellCheck={false}
            value={value}
          />
          <DialogFooter>
            <Button onClick={onClose} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={!trimmed} type="submit">
              {t.common.save}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
