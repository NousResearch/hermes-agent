import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { ConnectionGlyph } from '@/app/chat/sidebar/connection-glyph'
import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuLabel,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip, TipHintLabel } from '@/components/ui/tooltip'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import { $connectionsRegistry } from '@/store/connections'
import { $bindings } from '@/store/keybinds'
import { $activeGatewayProfile } from '@/store/profile'

import { $localGateway, $terminalShellGateways, defaultTerminalGateway, shellGatewayFor } from './terminal-gateway'
import { createTerminal } from './terminals'

type GatewayChoice = Pick<DesktopRegistryConnection, 'id' | 'kind' | 'label'>

// Same {connectionId, profile} record as +, so the SSH tunnel is the active
// profile's. The session's cwd is a path on its own host: a tab on another
// gateway starts in its home instead.
function openOn(choice: GatewayChoice) {
  const gateway = shellGatewayFor(choice.id, $activeGatewayProfile.get(), $connectionsRegistry.get())
  const session = defaultTerminalGateway()
  const sameHost = !session || gateway?.connectionId === session.connectionId

  createTerminal(sameHost ? undefined : '', gateway)
}

/** The rail's new-terminal control. Click opens on the gateway picked in
 *  Sessions; with an SSH gateway registered, the chevron (or a right-click on
 *  +) opens one on any shell-capable gateway, so hosts sit side by side. */
export function NewTerminalButton({ className }: { className: string }) {
  const { t } = useI18n()
  const bindings = useStore($bindings)
  const local = useStore($localGateway)
  const sshGateways = useStore($terminalShellGateways)
  const newHint = bindings['view.newTerminal']?.[0]
  const choices: GatewayChoice[] = sshGateways.length ? [...(local ? [local] : []), ...sshGateways] : []

  const button = (
    <button
      aria-label={t.rightSidebar.terminalNew}
      className={cn(className, 'size-7 text-(--ui-text-quaternary)')}
      onClick={() => createTerminal()}
      type="button"
    >
      <Codicon name="add" size="0.8125rem" />
    </button>
  )

  // The context-menu trigger (when there is one) sits inside the Tip, right on
  // the button: Tip hands its extra props to the tooltip content, so a trigger
  // wrapped around it never reaches the button.
  const tipped = (trigger: ReactNode) => (
    <Tip
      label={<TipHintLabel hint={newHint && formatCombo(newHint)} text={t.rightSidebar.terminalNew} />}
      placement="right-rail"
    >
      {trigger}
    </Tip>
  )

  if (!choices.length) {
    return <li className="flex w-full justify-center">{tipped(button)}</li>
  }

  const items = (Item: typeof ContextMenuItem | typeof DropdownMenuItem): ReactNode =>
    choices.map(choice => (
      <Item data-terminal-gateway={choice.id} key={choice.id} onSelect={() => openOn(choice)}>
        <ConnectionGlyph connection={choice} />
        {choice.label}
      </Item>
    ))

  return (
    <li className="flex w-full flex-col items-center">
      <ContextMenu>
        {tipped(<ContextMenuTrigger asChild>{button}</ContextMenuTrigger>)}
        <ContextMenuContent>
          <ContextMenuLabel>{t.rightSidebar.terminalNewOn}</ContextMenuLabel>
          {items(ContextMenuItem)}
        </ContextMenuContent>
      </ContextMenu>
      <DropdownMenu>
        <Tip label={t.rightSidebar.terminalNewOn} placement="right-rail">
          <DropdownMenuTrigger asChild>
            <button
              aria-label={t.rightSidebar.terminalNewOn}
              className={cn(className, 'h-3.5 w-7 text-(--ui-text-quaternary)')}
              data-slot="terminal-new-on"
              type="button"
            >
              <Codicon name="chevron-down" size="0.625rem" />
            </button>
          </DropdownMenuTrigger>
        </Tip>
        <DropdownMenuContent align="start" side="left">
          <DropdownMenuLabel>{t.rightSidebar.terminalNewOn}</DropdownMenuLabel>
          {items(DropdownMenuItem)}
        </DropdownMenuContent>
      </DropdownMenu>
    </li>
  )
}
