import type { ProfileScope } from '@/hermes'
import type { MessagingPlatformsResponse, MessagingPlatformUpdate } from '@/types/hermes'

import { capabilityScoped, hermesApi } from './client'

export interface DiscordDeskStatus {
  channel_id: null | string
  channel_name: null | string
  configured: boolean
  ok: boolean
  token_set: boolean
  user_id: null | string
}

export interface DiscordTextChannel {
  id: string
  name: string
}

export interface DiscordGuildChannels {
  channels: DiscordTextChannel[]
  guild: string
}

export interface DiscordSetupResult {
  channel_id: string
  channel_name: string
  ok: boolean
  restart_error?: string
  restart_started?: boolean
}

export function getDiscordDesk(profile?: ProfileScope): Promise<DiscordDeskStatus> {
  return hermesApi<DiscordDeskStatus>({
    ...capabilityScoped(profile),
    path: '/api/messaging/discord'
  })
}

export function listDiscordChannels(token: string, profile?: ProfileScope): Promise<{ guilds: DiscordGuildChannels[] }> {
  return hermesApi<{ guilds: DiscordGuildChannels[] }>({
    ...capabilityScoped(profile),
    path: '/api/messaging/discord/channels',
    method: 'POST',
    body: { token }
  })
}

export function saveDiscordDesk(
  body: { channel_id: string; channel_name: string; token: string; user_id: string },
  profile?: ProfileScope
): Promise<DiscordSetupResult> {
  return hermesApi<DiscordSetupResult>({
    ...capabilityScoped(profile),
    path: '/api/messaging/discord',
    method: 'PUT',
    body
  })
}

export function listDeskPlatforms(profile?: ProfileScope): Promise<MessagingPlatformsResponse> {
  return hermesApi<MessagingPlatformsResponse>({
    ...capabilityScoped(profile),
    path: '/api/messaging/platforms'
  })
}

export function saveDeskPlatform(
  platformId: string,
  body: MessagingPlatformUpdate,
  profile?: ProfileScope
): Promise<{ hot_served?: boolean; ok: boolean }> {
  return hermesApi<{ hot_served?: boolean; ok: boolean }>({
    ...capabilityScoped(profile),
    path: `/api/messaging/platforms/${encodeURIComponent(platformId)}`,
    method: 'PUT',
    body
  })
}

export function deleteDiscordDesk(profile?: ProfileScope): Promise<DiscordSetupResult> {
  return hermesApi<DiscordSetupResult>({
    ...capabilityScoped(profile),
    path: '/api/messaging/discord',
    method: 'DELETE'
  })
}

export function postDiscordMessage(text: string, profile?: ProfileScope): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/messaging/discord/send',
    method: 'POST',
    body: { text }
  })
}
