/**
 * Which surface a tool call renders as.
 *
 * Two consumers have to agree on this and they sit on opposite sides of the
 * app: the transcript decides what to draw, and the DOM render budget decides
 * how much of the transcript to mount. Pricing a turn correctly means pricing
 * what the grouping actually renders, so the classification lives on its own
 * rather than inside either one.
 */

const FILE_EDIT_TOOL_NAMES = new Set(['edit_file', 'patch', 'write_file'])

/** Renders a diff — the deliverable of the turn, and the one card whose cost scales. */
export function isFileEditTool(toolName: string): boolean {
  return FILE_EDIT_TOOL_NAMES.has(toolName)
}

// Tools that draw their own surface and must never be folded into a run's
// summary. Two kinds, for the same reason — the thing on screen IS the point:
//
//   - File edits are the deliverable, not scaffolding. The diff is what the
//     user reviews, so it stays visible at its place in the turn, live and
//     settled, the way a PR shows its changes.
//   - `clarify`, `image_generate` and `delegate_task` bypass ToolEntry to
//     render their own markup: a question the user has to answer, an image
//     they asked for, the several agents a fan-out is running.
//   - `manage_connections` and `manage_catalog` are consent cards; their controls must stay visible.
//   - `set_theme` shows a picker card while it waits for the user, then a one-line receipt.
//   - `setup_beat` is the onboarding chat's card for one beat, then a one-line receipt.
//
// Everything else is ephemeral activity — reads, searches, commands — which is
// what a run summarizes and what the live ticker cycles through.
const CARD_TOOL_NAMES = [
  'clarify',
  'delegate_task',
  'image_generate',
  'manage_catalog',
  'manage_connections',
  'set_theme',
  'setup_beat'
] as const

export type CardToolName = (typeof CARD_TOOL_NAMES)[number]

// Card kinds a tool may declare (`ui_card=` on registration) → the built-in card
// that draws them. The gateway stamps `card` on tool.start / tool.complete and on
// rehydrated history rows.
const TOOL_CARD_KINDS = { delegate: 'delegate_task' } as const satisfies Record<string, CardToolName>

export type ToolCardKind = keyof typeof TOOL_CARD_KINDS

// Plugin tools (e.g. a fan-out `agents_run`) declare their card at registration
// time in Python, so a static name list can never know them. Learn the mapping
// from the first event or history row that carries it and consult it wherever the
// built-in names are consulted. Module-level on purpose: a reloaded transcript
// and the live stream must classify the same tool identically.
const learnedCards = new Map<string, CardToolName>()

function isToolCardKind(kind: string): kind is ToolCardKind {
  return Object.hasOwn(TOOL_CARD_KINDS, kind)
}

export function learnToolCard(toolName: string, kind: unknown): void {
  if (!toolName || typeof kind !== 'string' || !isToolCardKind(kind)) {
    return
  }

  learnedCards.set(toolName, TOOL_CARD_KINDS[kind])
}

/** The built-in card that draws this tool: its own name if built in, else the card it declared. */
export function cardToolName(toolName: string): CardToolName | undefined {
  if ((CARD_TOOL_NAMES as readonly string[]).includes(toolName)) {
    return toolName as CardToolName
  }

  return learnedCards.get(toolName)
}

export function isCardToolName(toolName: string): toolName is CardToolName {
  return cardToolName(toolName) !== undefined
}

export function _resetToolCardsForTests(): void {
  learnedCards.clear()
}

// Name the run splitter uses for a manage_connections part it has classified as a card.
export const CONNECTION_CARD_KEY = 'manage_connections:card'

export function isCardTool(toolName: string): boolean {
  return isCardToolName(toolName) || toolName === CONNECTION_CARD_KEY || isFileEditTool(toolName)
}

// Activity tools that render nothing at all: `todo` parts are hoisted to a
// dedicated panel above the message content, and a reaction's UI is the emoji
// landing on the bubble. Both still render when they FAIL, which is a bounded
// error row either way.
const SILENT_TOOL_NAMES = new Set(['react_to_message', 'todo', 'todo_list'])

export function isSilentTool(toolName: string): boolean {
  return SILENT_TOOL_NAMES.has(toolName)
}
