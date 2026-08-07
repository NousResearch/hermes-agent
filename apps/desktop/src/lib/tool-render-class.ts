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
//
// Everything else is ephemeral activity — reads, searches, commands — which is
// what a run summarizes and what the live ticker cycles through.
const CARD_TOOL_NAMES = new Set(['clarify', 'delegate_task', 'image_generate'])

export function isCardTool(toolName: string): boolean {
  return CARD_TOOL_NAMES.has(toolName) || isFileEditTool(toolName)
}

// Activity tools that render nothing at all: `todo` parts are hoisted to a
// dedicated panel above the message content, and a reaction's UI is the emoji
// landing on the bubble. Both still render when they FAIL, which is a bounded
// error row either way.
const SILENT_TOOL_NAMES = new Set(['react_to_message', 'todo'])

export function isSilentTool(toolName: string): boolean {
  return SILENT_TOOL_NAMES.has(toolName)
}
