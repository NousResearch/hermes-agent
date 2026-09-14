import { describe, expect, it } from 'vitest'

import { BACKEND_EVENT_NAMES } from './gateway-events'
import contract from './gateway-events.json'

/**
 * Two-sided contract with `tests/tui_gateway/test_gateway_event_contract.py`: the
 * Python side pins the emitter call sites to `gateway-events.json`; this side pins
 * `BACKEND_EVENT_NAMES` (which `BackendGatewayEventMap` is checked against via
 * `satisfies`) to the same JSON. A name added on either side alone goes red here.
 */
describe('gateway-events.json ⇄ BackendGatewayEventMap', () => {
  it('lists exactly the backend-emitted notification names', () => {
    expect([...BACKEND_EVENT_NAMES]).toEqual(contract)
  })

  it('keeps both lists sorted and duplicate-free (stable diffs)', () => {
    const sorted = [...contract].sort()
    expect(contract).toEqual(sorted)
    expect(new Set(contract).size).toBe(contract.length)
  })
})
