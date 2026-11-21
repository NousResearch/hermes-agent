import { atom, computed, type WritableAtom } from 'nanostores'

import { allPaneIds } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { $routeCollapsedPanes, setPaneRouteCollapsed } from '@/components/pane-shell/tree/store-route-collapse'
import { persistentAtom } from '@/lib/persisted'
import { $chatItemsOpen } from '@/store/chat-items'
import { $fileBrowserOpen } from '@/store/layout'
import { currentSessionId } from '@/store/preview-ownership'
import { $reviewOpen } from '@/store/review'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $portalShell } from '@/store/shell-variant'

import { $terminalTakeover } from '../right-sidebar/store'

import { isTool } from './developer-tool-panel'

/**
 * The redesign's right-hand panel belongs to the chat on screen (design review 2026-11-17): no page
 * shows it, and every chat comes back to the panel it had.
 *
 * - Off a chat the panel's panes are ROUTE-collapsed: off screen but mounted, so a running terminal or
 *   a loaded Browser page survives the visit, and nothing the user chose is rewritten.
 * - Which tools a chat had open is remembered per stored session. The tools' own open stores (one
 *   each, written by the rail, the keybinds, the agent) stay the authority for what is on screen now;
 *   this module records them for the focused chat and puts them back when focus returns to it.
 *   Browser tabs need no entry: each is already owned by its session (store/preview-ownership.ts).
 */

const TOOL_STORES = {
  files: $fileBrowserOpen,
  review: $reviewOpen,
  terminal: $terminalTakeover,
  items: $chatItemsOpen
} satisfies Record<string, WritableAtom<boolean>>

export type ChatTool = keyof typeof TOOL_STORES

const CHAT_TOOLS = Object.keys(TOOL_STORES) as ChatTool[]

/** Most chats are visited a few times and forgotten; the oldest entries make room. */
const MAX_REMEMBERED_CHATS = 256

const isChatTool = (value: unknown): value is ChatTool => CHAT_TOOLS.includes(value as ChatTool)

function decodePanels(raw: string): Record<string, ChatTool[]> {
  const value: unknown = JSON.parse(raw)

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const entries = Object.entries(value)
    .map(([chat, tools]): [string, ChatTool[]] => [chat, Array.isArray(tools) ? tools.filter(isChatTool) : []])
    .filter(([chat, tools]) => chat.length > 0 && tools.length > 0)

  return Object.fromEntries(entries.slice(-MAX_REMEMBERED_CHATS))
}

/** Stored session id → the tools it had open, oldest first. A chat with nothing open has no entry,
 *  which is also what an unvisited chat reads: closed. Keyed by the stored id alone (unique across
 *  profiles and machines: a timestamp and a random suffix), so a profile rename keeps every entry and
 *  a deleted profile's entries age out under the bound. */
export const $chatToolPanels = persistentAtom<Record<string, ChatTool[]>>('hermes.desktop.chatToolPanels.v1', {}, {
  decode: decodePanels,
  encode: value => (Object.keys(value).length ? JSON.stringify(value) : null)
})

function remember(chat: string, tools: readonly ChatTool[]): void {
  const { [chat]: previous, ...rest } = $chatToolPanels.get()

  if ((previous ?? []).join() === tools.join()) {
    return
  }

  // Re-inserting moves the chat to the newest end, so the bound drops the least recently used.
  const next = tools.length ? { ...rest, [chat]: [...tools] } : rest
  const keys = Object.keys(next)

  $chatToolPanels.set(
    keys.length > MAX_REMEMBERED_CHATS
      ? Object.fromEntries(keys.slice(-MAX_REMEMBERED_CHATS).map(key => [key, next[key]]))
      : next
  )
}

const openTools = (): ChatTool[] => CHAT_TOOLS.filter(tool => TOOL_STORES[tool].get())

// A new chat has no stored id until its first send; what it opens before then is this window's alone.
let draftTools: ChatTool[] = []

/** The draft on screen just became `storedSessionId` (its first send): the panel it opened is that
 *  conversation's, and the next new chat starts closed. Called beside the draft's other hand-overs. */
export function adoptDraftToolPanel(storedSessionId: string): void {
  if (!$portalShell.get()) {
    return
  }

  remember(storedSessionId, openTools())
  draftTools = []
}

/** Follow the focused chat: record its tools while it is focused, put them back when it returns.
 *  Redesign only; classic keeps one panel for the whole window. */
export function bindChatToolPanels(): () => void {
  // The chat the tool stores currently describe: undefined until the redesign is bound, null a draft.
  let shown: null | string | undefined
  let applying = false

  const record = () => {
    if (applying || shown === undefined) {
      return
    }

    if (shown === null) {
      draftTools = openTools()
    } else {
      remember(shown, openTools())
    }
  }

  const apply = (tools: readonly ChatTool[]) => {
    applying = true

    try {
      for (const tool of CHAT_TOOLS) {
        if (TOOL_STORES[tool].get() !== tools.includes(tool)) {
          TOOL_STORES[tool].set(tools.includes(tool))
        }
      }
    } finally {
      applying = false
    }
  }

  const follow = () => {
    const chat = $focusedStoredSessionId.get()

    if (!$portalShell.get()) {
      shown = undefined

      return
    }

    if (shown === undefined) {
      // Binding (boot, or the switch from classic): the persisted stores already describe this chat.
      shown = chat
      record()

      return
    }

    if (chat === shown) {
      return
    }

    // Compression rotated the conversation onto a new stored id: same chat, its entry moves along.
    if (shown !== null && chat !== null && currentSessionId(shown) === currentSessionId(chat)) {
      remember(chat, $chatToolPanels.get()[shown] ?? [])
      remember(shown, [])
      shown = chat

      return
    }

    shown = chat
    apply(chat === null ? draftTools : ($chatToolPanels.get()[chat] ?? []))
  }

  const stops = [
    $focusedStoredSessionId.subscribe(follow),
    $portalShell.listen(follow),
    ...CHAT_TOOLS.map(tool => TOOL_STORES[tool].listen(record))
  ]

  return () => stops.forEach(stop => stop())
}

/** True while the route shows something other than a chat (a page, or an overlay over one). */
const $chatToolsOffRoute = atom(false)

export function setChatToolsOffRoute(off: boolean): void {
  $chatToolsOffRoute.set(off)
}

const $toolPanesInTree = computed($layoutTree, tree => (tree ? allPaneIds(tree).filter(isTool) : []))

/** Route-collapse every tool pane while off a chat, including one that joins the tree meanwhile (an
 *  agent opening a Browser tab in the background); release them all on return. */
export function bindChatToolRouteCollapse(): () => void {
  const sync = () => {
    const off = $chatToolsOffRoute.get()
    const panes = new Set($toolPanesInTree.get())

    for (const id of $routeCollapsedPanes.get()) {
      if (isTool(id) && !(off && panes.has(id))) {
        setPaneRouteCollapsed(id, false)
      }
    }

    if (off) {
      panes.forEach(id => setPaneRouteCollapsed(id, true))
    }
  }

  const stops = [$chatToolsOffRoute.subscribe(sync), $toolPanesInTree.listen(sync)]

  return () => stops.forEach(stop => stop())
}
