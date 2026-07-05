// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { AnalyticsResponse } from '@/types/hermes'

const getUsageAnalytics = vi.fn()

vi.mock('@/hermes', () => ({
  getUsageAnalytics: (days: number) => getUsageAnalytics(days)
}))

import { $authState } from '@/store/auth'

import { ProfileStatsView } from './index'

const DAY_MS = 86_400_000

function isoDaysAgo(days: number): string {
  return new Date(Date.now() - days * DAY_MS).toISOString().slice(0, 10)
}

function dailyEntry(day: string, input: number, output: number) {
  return {
    actual_cost: 0,
    api_calls: 3,
    cache_read_tokens: 0,
    day,
    estimated_cost: 0.02,
    input_tokens: input,
    output_tokens: output,
    reasoning_tokens: 0,
    sessions: 2
  }
}

// Real-shaped payload: everything present → every block renders.
const FULL_USAGE: AnalyticsResponse = {
  by_model: [
    {
      api_calls: 80,
      estimated_cost: 0.4,
      input_tokens: 1_100_000,
      model: 'deepseek-v4-pro',
      output_tokens: 250_000,
      sessions: 30
    }
  ],
  daily: [dailyEntry(isoDaysAgo(2), 400_000, 100_000), dailyEntry(isoDaysAgo(1), 800_000, 200_000)],
  period_days: 370,
  skills: {
    summary: { distinct_skills_used: 3, total_skill_actions: 20, total_skill_edits: 2, total_skill_loads: 18 },
    top_skills: [
      { last_used_at: null, manage_count: 0, percentage: 60, skill: 'douyin-data', total_count: 12, view_count: 12 },
      { last_used_at: null, manage_count: 0, percentage: 40, skill: 'image-gen', total_count: 8, view_count: 8 }
    ]
  },
  totals: {
    total_actual_cost: 0,
    total_api_calls: 91,
    total_cache_read: 0,
    total_estimated_cost: 0.42,
    total_input: 1_200_000,
    total_output: 300_000,
    total_reasoning: 0,
    total_sessions: 42
  }
}

// Sessions exist but nothing else was recorded: per-day rows, skill counters,
// model rows and api-call counts are all absent → those blocks must not render.
const SPARSE_USAGE: AnalyticsResponse = {
  by_model: [],
  daily: [],
  period_days: 370,
  skills: {
    summary: { distinct_skills_used: 0, total_skill_actions: 0, total_skill_edits: 0, total_skill_loads: 0 },
    top_skills: []
  },
  totals: {
    total_actual_cost: 0,
    total_api_calls: null,
    total_cache_read: null,
    total_estimated_cost: 0,
    total_input: null,
    total_output: null,
    total_reasoning: null,
    total_sessions: 5
  }
}

function signIn() {
  $authState.set({
    account: { email: 'kael@apex-nodes.com', name: 'Kael', plan: 'pro' },
    enabled: true,
    gateReason: null,
    status: 'signed-in'
  })
}

function signOut() {
  $authState.set({
    account: { email: '', name: '', plan: '' },
    enabled: true,
    gateReason: null,
    status: 'signed-out'
  })
}

beforeEach(() => {
  getUsageAnalytics.mockReset()
})

afterEach(() => {
  cleanup()
  signOut()
})

describe('ProfileStatsView', () => {
  it('renders account header, stat cards, heatmap and both columns off real analytics data', async () => {
    signIn()
    getUsageAnalytics.mockResolvedValue(FULL_USAGE)

    render(<ProfileStatsView onClose={vi.fn()} />)

    // Header — managed-account identity (name / @handle / plan badge).
    expect(await screen.findByText('Kael')).toBeTruthy()
    expect(screen.getByText('@kael')).toBeTruthy()
    expect(screen.getByText('pro')).toBeTruthy()

    // Stat cards — real totals (42 sessions, 1.2M + 300K = 1.5M tokens).
    expect(await screen.findByText('42')).toBeTruthy()
    expect(screen.getByText('Sessions')).toBeTruthy()
    expect(screen.getByText('1.5M')).toBeTruthy()
    expect(screen.getByText('API calls')).toBeTruthy()

    // Token-activity heatmap with its 每日/每周/累计 toggle.
    expect(screen.getByText('Token activity')).toBeTruthy()
    expect(screen.getByRole('tab', { name: 'Daily' })).toBeTruthy()
    expect(screen.getByRole('tab', { name: 'Weekly' })).toBeTruthy()
    expect(screen.getByRole('tab', { name: 'Cumulative' })).toBeTruthy()

    // Insights + top plugins columns.
    expect(screen.getByText('Activity insights')).toBeTruthy()
    expect(screen.getByText('deepseek-v4-pro')).toBeTruthy()
    expect(screen.getByText('Top plugins')).toBeTruthy()
    expect(screen.getByText('douyin-data')).toBeTruthy()
    expect(screen.getByText('12 uses')).toBeTruthy()

    expect(getUsageAnalytics).toHaveBeenCalledTimes(1)
  })

  it('omits heatmap, insights, plugins and null-metric stat cards when the data is absent', async () => {
    signIn()
    getUsageAnalytics.mockResolvedValue(SPARSE_USAGE)

    render(<ProfileStatsView onClose={vi.fn()} />)

    // Sessions card still renders (the one metric that exists)…
    expect(await screen.findByText('5')).toBeTruthy()
    expect(screen.getByText('Sessions')).toBeTruthy()

    // …while unavailable blocks disappear wholesale instead of faking zeros.
    expect(screen.queryByText('Token activity')).toBeNull()
    expect(screen.queryByText('Activity insights')).toBeNull()
    expect(screen.queryByText('Top plugins')).toBeNull()
    expect(screen.queryByText('API calls')).toBeNull()
    expect(screen.queryByText('Total tokens')).toBeNull()
  })

  it('shows the 未登录 placeholder when no account is signed in', async () => {
    signOut()
    getUsageAnalytics.mockResolvedValue(SPARSE_USAGE)

    render(<ProfileStatsView onClose={vi.fn()} />)

    expect(await screen.findByText('Not signed in')).toBeTruthy()
    expect(screen.queryByText(/@/)).toBeNull()
  })
})
