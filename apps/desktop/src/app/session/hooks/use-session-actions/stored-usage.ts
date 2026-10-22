import { setCurrentUsage } from '@/store/session'

// Reflect a stored row's persisted token counts and spend into the live usage
// atom (total is derived, so callers can't drift it out of sync with
// input/output). Spend needs the row because the gateway builds a FRESH agent
// on resume with every counter at 0 and only reports cost again once a call
// lands — the row is the only figure available before the next turn.
// Split out of `index.ts`, which is over the code-health line cap.
export function applyStoredUsage(stored: {
  actual_cost_usd?: null | number
  estimated_cost_usd?: null | number
  input_tokens?: null | number
  output_tokens?: null | number
}) {
  const input = stored.input_tokens || 0
  const output = stored.output_tokens || 0
  // Same precedence as the sidebar's `sessionCostUsd`: the provider-reported
  // figure when the row carries one, our estimate otherwise. A row with
  // neither reports no status, so the chip reads "not measured" rather than
  // claiming a fresh $0.
  const cost = stored.actual_cost_usd || stored.estimated_cost_usd || 0
  const costStatus = stored.actual_cost_usd ? 'actual' : cost > 0 ? 'estimated' : undefined

  setCurrentUsage(current => ({
    ...current,
    cost_status: costStatus,
    cost_usd: cost,
    input,
    output,
    total: input + output
  }))
}
