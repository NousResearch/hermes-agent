// Renderer client for the screen-analysis scanline overlay. The active
// renderer is the source of truth for which `computer_use` capture tool calls
// are in flight: it registers a `tool_id` when a capture `tool.start` arrives
// and retires it when the matching `tool.complete` lands. The overlay is
// visible while at least one capture is in flight.
//
// `tool.complete` carries no `args` (only `tool_id`), so a completion is only
// treated as a capture ending if that `tool_id` was previously registered —
// non-capture tool completions are ignored here.

export type ScanlineState = 'active' | 'hidden'

const inFlightCaptures = new Set<string>()

let lastState: ScanlineState = 'hidden'

function pushState(state: ScanlineState): void {
  if (state === lastState) {
    return
  }

  lastState = state
  window.hermesDesktop?.scanline?.setState(state)
}

function keyFor(toolId: string | undefined, fallback: string): string {
  return toolId && toolId.length > 0 ? toolId : fallback
}

/** A `computer_use` capture just started — light the overlay. */
export function scanlineStart(toolId?: string, fallback = 'unknown'): void {
  inFlightCaptures.add(keyFor(toolId, fallback))
  pushState('active')
}

/** A tool call completed — retire it if it was a registered capture. */
export function scanlineComplete(toolId?: string): void {
  if (!toolId) {
    return
  }

  if (inFlightCaptures.delete(toolId)) {
    pushState(inFlightCaptures.size === 0 ? 'hidden' : 'active')
  }
}

/** Drop all bookkeeping (e.g. when the active session is torn down). */
export function resetScanline(): void {
  inFlightCaptures.clear()
  pushState('hidden')
}
