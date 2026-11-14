import type {
  MessagingPlatformsResponse,
  MessagingPlatformTestResponse,
  MessagingPlatformUpdate,
  PairingResponse,
  PairingUser,
  RelayLinkCode,
  RelayLinkCodeState,
  RelayLinksResponse,
  TelegramOnboardingApplyResponse,
  TelegramOnboardingStartResponse,
  TelegramOnboardingStatusResponse
} from '@/types/hermes'
import type {
  WebhookCreatePayload,
  WebhookCreateResponse,
  WebhookEnableResponse,
  WebhooksResponse
} from '@/types/webhooks'

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

/** `POST /api/messaging/discord/check`: what Discord reports for the saved bot token. */
export interface DiscordBotCheckResponse {
  app_id: string
  bot_name: string
  bot_settings_url: string
  invite_url: string
  message_content: boolean
  owners: { id: string; username: string }[]
  server_count: null | number
  server_members: boolean
}

export function checkDiscordBot(profile?: null | string): Promise<DiscordBotCheckResponse> {
  return hermesApi<DiscordBotCheckResponse>({
    ...profileScoped(profile),
    path: '/api/messaging/discord/check',
    method: 'POST'
  })
}

// -- Team Gateway account links -----------------------------------------------
// A relay-fronted instance reaches Telegram/Discord/Slack only through the shared bot; these mint
// the one-time `/link <code>` that binds a chat account to this agent. The profile tag picks the
// backend (a profile may point at another host); the relay itself serves that gateway's own profile.

export function getRelayLinks(profile?: null | string): Promise<RelayLinksResponse> {
  return hermesApi<RelayLinksResponse>({ ...profileScoped(profile), path: '/api/messaging/relay' })
}

export function requestRelayLink(platform: string, profile?: null | string): Promise<RelayLinkCode> {
  return hermesApi<RelayLinkCode>({
    ...profileScoped(profile),
    path: `/api/messaging/relay/${encodeURIComponent(platform)}/link`,
    method: 'POST'
  })
}

export function getRelayLinkState(code: string, profile?: null | string): Promise<RelayLinkCodeState> {
  return hermesApi<RelayLinkCodeState>({
    ...profileScoped(profile),
    path: `/api/messaging/relay/link/${encodeURIComponent(code)}`
  })
}

export function unlinkRelayAccount(
  platform: string,
  platformUserId: string,
  profile?: null | string
): Promise<{ unlinked: boolean }> {
  return hermesApi<{ unlinked: boolean }>({
    ...profileScoped(profile),
    path: `/api/messaging/relay/${encodeURIComponent(platform)}/accounts/${encodeURIComponent(platformUserId)}`,
    method: 'DELETE'
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
  const scope = profileScoped(profile)

  return hermesApi<TelegramOnboardingApplyResponse>({
    ...scope,
    path: `/api/messaging/telegram/onboarding/${encodeURIComponent(pairingId)}/apply`,
    method: 'POST',
    body: { allowed_user_ids: allowedUserIds, profile: scope.profile }
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
  const scope = profileScoped(profile)

  return hermesApi<{ ok: boolean; user: PairingUser }>({
    ...scope,
    path: '/api/pairing/approve',
    method: 'POST',
    // These endpoints read the profile off the body, not the query string —
    // the request scope alone would approve into the wrong profile's store.
    body: { platform, request_id: requestId, profile: scope.profile }
  })
}

export function revokePairing(platform: string, userId: string, profile?: null | string): Promise<{ ok: boolean }> {
  const scope = profileScoped(profile)

  return hermesApi<{ ok: boolean }>({
    ...scope,
    path: '/api/pairing/revoke',
    method: 'POST',
    body: { platform, user_id: userId, profile: scope.profile }
  })
}

// -- Webhooks (subscription CRUD) --------------------------------------------
// The webhook receiver is its own gateway platform; subscriptions live in a
// shared JSON store the CLI/dashboard also drive. Enable mutates config and
// best-effort restarts the gateway; subscription changes hot-reload.

/** `profile` = the backend owner this view captured at mount; omitted → the
 *  ambient request scope. Keyed explicitly so a view that survives an active
 *  profile switch keeps talking to the profile it belongs to (#71352). */
export function getWebhooks(profile?: null | string): Promise<WebhooksResponse> {
  return hermesApi<WebhooksResponse>({
    ...profileScoped(profile),
    path: '/api/webhooks'
  })
}

export function enableWebhooks(profile?: null | string): Promise<WebhookEnableResponse> {
  return hermesApi<WebhookEnableResponse>({
    ...profileScoped(profile),
    path: '/api/webhooks/enable',
    method: 'POST'
  })
}

export function createWebhook(body: WebhookCreatePayload, profile?: null | string): Promise<WebhookCreateResponse> {
  return hermesApi<WebhookCreateResponse>({
    ...profileScoped(profile),
    path: '/api/webhooks',
    method: 'POST',
    body
  })
}

export function deleteWebhook(name: string, profile?: null | string): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(profile),
    path: `/api/webhooks/${encodeURIComponent(name)}`,
    method: 'DELETE'
  })
}

export function setWebhookEnabled(
  name: string,
  enabled: boolean,
  profile?: null | string
): Promise<{ enabled: boolean; name: string; ok: boolean }> {
  return hermesApi<{ enabled: boolean; name: string; ok: boolean }>({
    ...profileScoped(profile),
    path: `/api/webhooks/${encodeURIComponent(name)}/enabled`,
    method: 'PUT',
    body: { enabled }
  })
}
