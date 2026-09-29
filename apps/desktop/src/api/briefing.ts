import { hermesApi, profileScoped } from './client'

export interface BriefingHeadline {
  link: string
  published: null | number
  source: string
  summary: string
  title: string
}

/** `GET /api/briefing` — the real state the daily briefing is told from. */
export interface BriefingResponse {
  ai: BriefingHeadline[]
  feeds_failed: string[]
  generated_at: number
  /** Today's calendar and newest unread mail; null when Google is not connected. Mail text is external data. */
  google?: null | { events: { start: string; summary: string }[]; unread: { from: string; subject: string }[] }
  window: { since: number; until: number }
  world: BriefingHeadline[]
  workspace: {
    jobs: null | { active: number; failing: string[]; next: { at: string; name: string }[] }
    model: null | string
    provider: null | string
    sessions: null | { titles: string[]; today: number; yesterday: number }
  }
}

export function getBriefing(): Promise<BriefingResponse> {
  return hermesApi<BriefingResponse>({ ...profileScoped(), path: '/api/briefing', timeoutMs: 20_000 })
}
