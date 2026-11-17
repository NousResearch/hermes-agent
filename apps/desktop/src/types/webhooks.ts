// -- Webhooks (subscription CRUD) --------------------------------------------
// Incoming HTTP event routes served by the webhook gateway platform. Backed by
// the same JSON store the CLI/dashboard use; per-route HMAC secrets are
// redacted on read and surfaced exactly once on create.

export interface WebhookRoute {
  created_at: null | string
  deliver: string
  deliver_only: boolean
  description: string
  enabled: boolean
  events: string[]
  name: string
  prompt: string
  secret_set: boolean
  signature: null | WebhookSignature
  skills: string[]
  url: string
}

export type WebhookSignature = 'generic' | 'github' | 'gitlab' | 'linear'

export interface WebhooksResponse {
  base_url: string
  enabled: boolean
  subscriptions: WebhookRoute[]
}

export interface WebhookCreatePayload {
  deliver?: string
  deliver_chat_id?: string
  deliver_only?: boolean
  description?: string
  events?: string[]
  name: string
  prompt?: string
  signature?: WebhookSignature
  skills?: string[]
}

// Create echoes the route summary plus the one-time secret.
export interface WebhookCreateResponse extends WebhookRoute {
  secret: string
}

export interface WebhookEnableResponse {
  enabled: true
  needs_restart: boolean
  ok: boolean
  platform: 'webhook'
  restart_action?: string
  restart_error?: string
  restart_pid?: null | number
  restart_started?: boolean
}

// POST /api/webhooks/{name}/test: the backend signs and sends one test event; `ok` is a 2xx answer.
export interface WebhookTestResponse {
  delivery_id: string
  ok: boolean
  response: string
  status: number
  url: string
}
