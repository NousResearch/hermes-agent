import './portal-status-bar.css'

import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router'

import {
  STATUSBAR_WAITING,
  StatusbarControls,
  type StatusbarItem,
  StatusbarItemView
} from '@/app/shell/statusbar-controls'
import { Tip } from '@/components/ui/tooltip'
import { $portalStatusBarOpen, setPortalStatusBarOpen } from '@/store/portal-status-bar'
import { $portalShell } from '@/store/shell-variant'

import { usePortalStatusBarCopy } from './portal-status-bar-copy'

/** The classic statusbar's items that describe the chat being worked in, in the bar's order: where
 *  it runs first, then what the turn is doing, then what it may do unasked. */
const CHAT_STATUS_IDS: readonly string[] = [
  'workspace-cwd',
  'running-timer',
  'context-usage',
  'cache-hit-rate',
  'tokens-per-second',
  'session-timer',
  'approval-mode'
]

/** Items the redesign's own chrome already carries: the popover's header opens the Command Center,
 *  the tool rail the terminal, and the popover's first card is the gateway's health. */
const CARRIED_BY_CHROME = new Set(['command-center', 'gateway-health', 'terminal'])

/** The frame makes room for the open bar off this attribute (portal-status-bar.css). */
const OPEN_ATTRIBUTE = 'data-portal-status-bar'

const onBar = (item: StatusbarItem, contributed: ReadonlySet<string>) =>
  CHAT_STATUS_IDS.includes(item.id) || contributed.has(item.id)

/** The bar's chips: the chat's readouts, then what plugins contributed (Figma's "plugin" chip). A
 *  readout with nothing to report yet stays off: an unlabelled dash says nothing, where the classic
 *  bar keeps it as proof the item is switched on. */
export function statusBarItems(items: readonly StatusbarItem[], contributed: ReadonlySet<string>): StatusbarItem[] {
  const shown = items.filter(item => !item.hidden && item.label !== STATUSBAR_WAITING)

  return [
    ...CHAT_STATUS_IDS.flatMap(id => shown.filter(item => item.id === id)),
    ...shown.filter(item => contributed.has(item.id) && !CHAT_STATUS_IDS.includes(item.id))
  ]
}

/** Everything else the classic bar held; the bar's own items too while it is collapsed, so
 *  collapsing it never strands a readout. */
export function systemStatusItems(
  items: readonly StatusbarItem[],
  contributed: ReadonlySet<string>,
  barOpen: boolean
): StatusbarItem[] {
  return items.filter(item => !CARRIED_BY_CHROME.has(item.id) && !(barOpen && onBar(item, contributed)))
}

/** The bar's frame, mounted once by the rail: the chips it is handed while open, and its toggle.
 *  Portalled to the body so it stands in the canvas under the workspace frame on every page. */
export function PortalStatusBar({ children }: { children?: ReactNode }) {
  const portal = useStore($portalShell)
  const open = useStore($portalStatusBarOpen)
  const copy = usePortalStatusBarCopy()
  const label = open ? copy.collapse : copy.open

  useEffect(() => {
    const root = document.documentElement

    root.toggleAttribute(OPEN_ATTRIBUTE, open)

    return () => root.removeAttribute(OPEN_ATTRIBUTE)
  }, [open])

  if (!portal) {
    return null
  }

  return createPortal(
    <div data-open={open || undefined} data-slot="portal-status-bar">
      {children}
      <Tip label={label}>
        <button
          aria-expanded={open}
          aria-label={label}
          data-slot="portal-status-bar-toggle"
          onClick={() => setPortalStatusBarOpen(!open)}
          type="button"
        >
          <span aria-hidden />
        </button>
      </Tip>
    </div>,
    document.body
  )
}

/** The open bar's chips, rendered where the status data lives. */
export function PortalStatusBarItems(props: { contributed: ReadonlySet<string>; items: readonly StatusbarItem[] }) {
  const navigate = useNavigate()

  return (
    <div data-slot="portal-status-bar-items">
      {statusBarItems(props.items, props.contributed).map(item => (
        <StatusbarItemView item={item} key={item.id} navigate={navigate} />
      ))}
    </div>
  )
}

/** The System popover's rows under the machine card: what the classic bar held that the open status
 *  bar does not show. */
export function PortalSystemStatus(props: { contributed: ReadonlySet<string>; items: readonly StatusbarItem[] }) {
  const open = useStore($portalStatusBarOpen)

  return (
    <div data-slot="portal-system-status">
      <StatusbarControls items={systemStatusItems(props.items, props.contributed, open)} sidebar />
    </div>
  )
}
