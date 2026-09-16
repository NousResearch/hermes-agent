import type {
  MessagingPlatformsResponse,
  MessagingPlatformTestResponse,
  MessagingPlatformUpdate,
  PairingResponse,
  PairingUser,
  TelegramOnboardingApplyResponse,
  TelegramOnboardingStartResponse,
  TelegramOnboardingStatusResponse,
  WebhookCreatePayload,
  WebhookCreateResponse,
  WebhookEnableResponse,
  WebhooksResponse
} from '@/types/hermes'

import { hermesApi, profileScoped } from './client'

export function getMessagingPlatforms(profile?: null | string): Promise<MessagingPlatformsResponse> {
  return hermesApi<MessagingPlatformsResponse>({
    ...profileScoped(profile),
    path: '/api/messaging/platforms'
  })
}

/** `hot_served`: a live multiplexer serving this named profile rebuilt its adapters from the new
 *  credentials right away — no gateway restart is needed for the change to take effect. */
export interface MessagingPlatformUpdateResponse {
  hot_served?: boolean
  ok: boolean
  platform: string
}

export function updateMessagingPlatform(
  platformId: string,
  body: MessagingPlatformUpdate,
  profile?: null | string
): Promise<MessagingPlatformUpdateResponse> {
  return hermesApi<MessagingPlatformUpdateResponse>({
    ...profileScoped(profile),
    path: `/api/messaging/platforms/${encodeURIComponent(platformId)}`,
    method: 'PUT',
    body
  })
}

export function testMessagingPlatform(
  platformId: string,
  profile?: null | string
): Promise<MessagingPlatformTestResponse> {
  return hermesApi<MessagingPlatformTestResponse>({
    ...profileScoped(profile),
    path: `/api/messaging/platforms/${encodeURIComponent(platformId)}/test`,
    method: 'POST'
  })
}

// -- Telegram QR onboarding ---------------------------------------------------
// Pairing state lives in the memory of the backend process that started it, so
// every call in one flow carries the SAME profile scope — the Electron router
// picks the backend from it, and a mismatched apply would 404 the pairing.

export function startTelegramOnboarding(
  botName?: string,
  profile?: null | string
): Promise<TelegramOnboardingStartResponse> {
  return hermesApi<TelegramOnboardingStartResponse>({
    ...profileScoped(profile),
    path: '/api/messaging/telegram/onboarding/start',
    method: 'POST',
    body: botName ? { bot_name: botName } : {}
  })
}

export function getTelegramOnboardingStatus(
  pairingId: string,
  profile?: null | string
): Promise<TelegramOnboardingStatusResponse> {
  return hermesApi<TelegramOnboardingStatusResponse>({
    ...profileScoped(profile),
    path: `/api/messaging/telegram/onboarding/${encodeURIComponent(pairingId)}`
  })
}

export function applyTelegramOnboarding(
  pairingId: string,
  allowedUserIds: string[],
  profile?: null | string
): Promise<TelegramOnboardingApplyResponse> {
  return hermesApi<TelegramOnboardingApplyResponse>({
    ...profileScoped(profile),
    path: `/api/messaging/telegram/onboarding/${encodeURIComponent(pairingId)}/apply`,
    method: 'POST',
    body: { allowed_user_ids: allowedUserIds, ...profileScoped(profile) }
  })
}

export function cancelTelegramOnboarding(pairingId: string, profile?: null | string): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(profile),
    path: `/api/messaging/telegram/onboarding/${encodeURIComponent(pairingId)}`,
    method: 'DELETE'
  })
}

// -- Pairing (who may DM the bot) --------------------------------------------
// Unknown DMers get a one-time code and land in `pending` until an admin
// approves them. Approval grants on the row's `request_id`, never on the code:
// the code is the requester's proof that the channel is theirs and is never
// returned by the API, while an authenticated admin is only ever identifying
// a row they can already see.

export function getPairing(profile?: null | string): Promise<PairingResponse> {
  return hermesApi<PairingResponse>({
    ...profileScoped(profile),
    path: '/api/pairing'
  })
}

export function approvePairing(
  platform: string,
  requestId: string,
  profile?: null | string
): Promise<{ ok: boolean; user: PairingUser }> {
  return hermesApi<{ ok: boolean; user: PairingUser }>({
    ...profileScoped(profile),
    path: '/api/pairing/approve',
    method: 'POST',
    // These endpoints read the profile off the body, not the query string —
    // `profileScoped()` alone would approve into the wrong profile's store.
    body: { platform, request_id: requestId, ...profileScoped(profile) }
  })
}

export function revokePairing(platform: string, userId: string, profile?: null | string): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(profile),
    path: '/api/pairing/revoke',
    method: 'POST',
    body: { platform, user_id: userId, ...profileScoped(profile) }
  })
}

// -- Webhooks (subscription CRUD) --------------------------------------------
// The webhook receiver is its own gateway platform; subscriptions live in a
// shared JSON store the CLI/dashboard also drive. Enable mutates config and
// best-effort restarts the gateway; subscription changes hot-reload.

export function getWebhooks(): Promise<WebhooksResponse> {
  return hermesApi<WebhooksResponse>({
    ...profileScoped(),
    path: '/api/webhooks'
  })
}

export function enableWebhooks(): Promise<WebhookEnableResponse> {
  return hermesApi<WebhookEnableResponse>({
    ...profileScoped(),
    path: '/api/webhooks/enable',
    method: 'POST'
  })
}

export function createWebhook(body: WebhookCreatePayload): Promise<WebhookCreateResponse> {
  return hermesApi<WebhookCreateResponse>({
    ...profileScoped(),
    path: '/api/webhooks',
    method: 'POST',
    body
  })
}

export function deleteWebhook(name: string): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(),
    path: `/api/webhooks/${encodeURIComponent(name)}`,
    method: 'DELETE'
  })
}

export function setWebhookEnabled(
  name: string,
  enabled: boolean
): Promise<{ enabled: boolean; name: string; ok: boolean }> {
  return hermesApi<{ enabled: boolean; name: string; ok: boolean }>({
    ...profileScoped(),
    path: `/api/webhooks/${encodeURIComponent(name)}/enabled`,
    method: 'PUT',
    body: { enabled }
  })
}
