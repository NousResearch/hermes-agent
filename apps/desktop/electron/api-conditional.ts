// The `hermes:api` conditional-GET seam, shared by the main-process handler
// and the renderer so both sides agree on one literal (mirror of
// api-expected-404.ts).
//
// A renderer list read opts in with `request.conditional` and main answers
// with a BRANDED envelope `{ etag, status, data }` instead of the bare parsed
// body: only a GET /api/sessions read (the route that stamps content ETags)
// ever travels as one, so the general `hermes:api` surface stays unchanged
// for every other call. An older main process ignores the request fields and
// resolves the plain body — the renderer detects the missing brand and falls
// back to today's unconditional behavior (the capability gate).

import { normalizeConditionalResult } from './session-conditional-reads'

const HERMES_API_CONDITIONAL = '__hermesConditional__'

export interface HermesApiConditionalPayload {
  etag: string | null
  status: number
  data: unknown
}

export type HermesApiConditionalEnvelope = HermesApiConditionalPayload & {
  [HERMES_API_CONDITIONAL]: true
}

// True when `value` is the handler-resolved conditional envelope, not real
// response data. The brand is the only discriminator: a session list payload
// is `{ sessions, total, ... }` and can never carry it.
export function isConditionalEnvelope(value: unknown): value is HermesApiConditionalEnvelope {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const obj = value as Record<string, unknown>

  return (
    obj[HERMES_API_CONDITIONAL] === true &&
    typeof obj.status === 'number' &&
    (obj.etag === null || typeof obj.etag === 'string') &&
    'data' in obj
  )
}

export function buildConditionalEnvelope(payload: HermesApiConditionalPayload): HermesApiConditionalEnvelope {
  return { [HERMES_API_CONDITIONAL]: true, ...payload }
}

// The main-side post-processing both `hermes:api` handler paths run when a
// conditional list read resolves (main.ts is too heavy to load in a unit
// test, so the seam is tested through this helper): the transport
// (api-json-fetch) resolves the PLAIN `{ data, status, etag }` — the brand is
// applied here, exactly once, on the main side, and the renderer
// capability-gates on it. The OAuth cookie leg resolves the plain parsed body
// instead (it cannot capture status/ETag); normalization reads that as a
// 200 with no validator, so the renderer simply never revalidates it.
// `transformData` is the per-route post-processing (registry row tagging)
// that must run on the payload's data before it is wrapped.
export function conditionalResultForRenderer(
  value: unknown,
  transformData?: (data: unknown) => unknown
): HermesApiConditionalEnvelope {
  const result = normalizeConditionalResult(value)

  return buildConditionalEnvelope({
    etag: result.etag,
    status: result.status,
    data: transformData ? transformData(result.data) : result.data
  })
}

export { HERMES_API_CONDITIONAL }
