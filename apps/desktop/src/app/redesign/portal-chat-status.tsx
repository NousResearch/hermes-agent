import './portal-chat-status.css'

import { useStore } from '@nanostores/react'
import { useCallback, useId } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router'

import {
  STATUSBAR_WAITING,
  StatusbarControls,
  type StatusbarItem,
  StatusbarItemView
} from '@/app/shell/statusbar-controls'
import { Switch } from '@/components/ui/switch'
import {
  $portalChatStatusEnabled,
  $portalChatStatusShown,
  $portalChatStatusSlot,
  $portalChatStatusTarget,
  setPortalChatStatusEnabled
} from '@/store/portal-chat-status'

import { usePortalChatStatusCopy } from './portal-chat-status-copy'

/** The classic statusbar's items that describe the chat being worked in, in the line's order: where
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

/** The line leads with the workspace; the readouts sit at its end. */
const CHAT_STATUS_LEAD = 'workspace-cwd'

/** Items the redesign's own chrome already carries: the popover's header opens the Command Center,
 *  the tool rail the terminal, and the popover's first card is the gateway's health. */
const CARRIED_BY_CHROME = new Set(['command-center', 'gateway-health', 'terminal'])

/** A readout with nothing to report yet stays off the line: an unlabelled dash under the composer
 *  says nothing, where the classic bar keeps it as proof the item is switched on. */
export function chatStatusItems(items: readonly StatusbarItem[]): StatusbarItem[] {
  return CHAT_STATUS_IDS.flatMap(id =>
    items.filter(item => item.id === id && !item.hidden && item.label !== STATUSBAR_WAITING)
  )
}

/** Everything else the bar held; the chat's own items too while its line is off, so turning the
 *  line off never strands a readout. */
export function systemStatusItems(items: readonly StatusbarItem[], chatLineShown: boolean): StatusbarItem[] {
  return items.filter(
    item => !CARRIED_BY_CHROME.has(item.id) && !(chatLineShown && CHAT_STATUS_IDS.includes(item.id))
  )
}

/** The focused chat's composer mounts this under itself; the line is portalled into it. */
export function PortalChatStatusSlot({ target }: { target: string }) {
  const shown = useStore($portalChatStatusShown)
  const focused = useStore($portalChatStatusTarget) === target

  const mount = useCallback((element: HTMLDivElement | null) => {
    if (!element) {
      return
    }

    $portalChatStatusSlot.set(element)

    return () => {
      if ($portalChatStatusSlot.get() === element) {
        $portalChatStatusSlot.set(null)
      }
    }
  }, [])

  return shown && focused ? <div data-slot="portal-chat-status" ref={mount} /> : null
}

/** The line itself: rendered where the status data lives, shown under the focused composer. */
export function PortalChatStatus({ items }: { items: readonly StatusbarItem[] }) {
  const slot = useStore($portalChatStatusSlot)
  const navigate = useNavigate()

  if (!slot) {
    return null
  }

  const line = chatStatusItems(items)
  const view = (item: StatusbarItem) => <StatusbarItemView item={item} key={item.id} navigate={navigate} />

  return createPortal(
    <>
      <div data-slot="portal-chat-status-lead">{line.filter(item => item.id === CHAT_STATUS_LEAD).map(view)}</div>
      <div data-slot="portal-chat-status-readouts">{line.filter(item => item.id !== CHAT_STATUS_LEAD).map(view)}</div>
    </>,
    slot
  )
}

/** The System popover's rows under the machine card: what the bar held that is not the chat's, and
 *  (with developer chrome) the switch for the chat's line. */
export function PortalSystemStatus({ advanced, items }: { advanced: boolean; items: readonly StatusbarItem[] }) {
  const copy = usePortalChatStatusCopy()
  const shown = useStore($portalChatStatusShown)
  const enabled = useStore($portalChatStatusEnabled)
  const switchId = useId()

  return (
    <div data-slot="portal-system-status">
      <StatusbarControls items={systemStatusItems(items, shown)} sidebar />
      {advanced && (
        <label data-slot="portal-chat-status-switch" htmlFor={switchId}>
          <span>{copy.line}</span>
          <Switch checked={enabled} id={switchId} onCheckedChange={setPortalChatStatusEnabled} />
        </label>
      )}
    </div>
  )
}
