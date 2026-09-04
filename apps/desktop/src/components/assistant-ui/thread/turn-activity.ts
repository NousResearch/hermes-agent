import { isSilentTool } from '@/lib/tool-render-class'

/**
 * Seconds of silence before an unnamed wait earns a row of its own.
 *
 * Long enough that the pause between one tool call finishing and the next
 * arriving stays silent — a row that appeared for 300ms between every call
 * would strobe down a long run — short enough that a real gap is timed almost
 * as soon as it starts.
 */
export const TURN_QUIET_S = 2

export interface ActivityPart {
  result?: unknown
  text?: unknown
  toolName?: string
  type: string
}

/**
 * What the tail message has produced so far, as a value that changes exactly
 * when the turn makes visible progress.
 *
 * Part count and text length cover streamed prose and each new call. Settled
 * calls are counted separately because a result lands by MUTATING the part
 * that was already there: neither the count nor the text changes, so a
 * signature without it reads a finished tool call as more of the same silence
 * and dates the gap after it from whenever the call started.
 */
export function activitySignature(content: readonly ActivityPart[]): string {
  let textLength = 0
  let settledTools = 0

  for (const part of content) {
    if (typeof part.text === 'string') {
      textLength += part.text.length
    }

    if (part.type === 'tool-call' && part.result !== undefined) {
      settledTools += 1
    }
  }

  return `${content.length}:${textLength}:${settledTools}`
}

/**
 * Whether a tool call is already narrating this wait.
 *
 * A call in flight renders its own row, with its own timer, so a second
 * spinner under it would count the same seconds twice. Silent tools don't:
 * `todo` is hoisted to its own panel and a reaction's UI is the emoji landing
 * on the bubble, so neither leaves anything on screen to time — a wait on one
 * of those is as unnarrated as a wait on nothing at all.
 */
export function toolNarratesWait(content: readonly ActivityPart[]): boolean {
  return content.some(
    part => part.type === 'tool-call' && part.result === undefined && !isSilentTool(part.toolName ?? '')
  )
}
