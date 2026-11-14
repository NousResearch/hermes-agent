import { capabilityScoped, hermesApi } from '@/api/client'
import type {
  CronJob,
  CronJobCreatePayload,
  CronJobUpdates,
  MessagingPlatformsResponse,
  PaginatedSessions
} from '@/hermes'

export interface AgentSettingsOwner {
  connectionId?: string
  profile: string
}

export const agentSettingsKey = (owner: AgentSettingsOwner) =>
  ['agent-settings', owner.connectionId ?? '', owner.profile] as const

export function getAgentRoutines(owner: AgentSettingsOwner): Promise<CronJob[]> {
  return hermesApi({ ...capabilityScoped(owner), path: `/api/cron/jobs?profile=${encodeURIComponent(owner.profile)}` })
}

export function getAgentMessaging(owner: AgentSettingsOwner): Promise<MessagingPlatformsResponse> {
  return hermesApi({
    ...capabilityScoped(owner),
    path: `/api/messaging/platforms?profile=${encodeURIComponent(owner.profile)}`
  })
}

export function getAgentRecentSessions(owner: AgentSettingsOwner, limit = 5, offset = 0): Promise<PaginatedSessions> {
  return hermesApi({
    ...capabilityScoped(owner),
    path: `/api/profiles/sessions?profile=${encodeURIComponent(owner.profile)}&limit=${limit}&offset=${offset}&archived=exclude&order=recent`
  })
}

export function saveAgentRoutine(
  owner: AgentSettingsOwner,
  values: CronJobCreatePayload | CronJobUpdates,
  id?: string
): Promise<CronJob> {
  const path = id ? `/api/cron/jobs/${encodeURIComponent(id)}` : '/api/cron/jobs'

  return hermesApi({
    ...capabilityScoped(owner),
    path: `${path}?profile=${encodeURIComponent(owner.profile)}`,
    method: id ? 'PUT' : 'POST',
    body: id ? { updates: values } : values
  })
}
