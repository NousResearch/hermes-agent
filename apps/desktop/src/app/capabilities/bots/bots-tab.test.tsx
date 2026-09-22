import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { closeBotMarketplaceRequest, openBotMarketplaceRequest } from '@/store/bot-marketplace'

import { BotMarketplaceTab, isTrustedBotPickerMessage } from './bots-tab'
import { $foregroundBotProfile } from './marketplace-actions'

const catalogEntry = {
  name: 'research-assistant',
  version: '1.0.0',
  maintainer: 'Nous Research',
  tier: 'official' as const,
  category: 'Research',
  tags: ['evidence'],
  title: 'Research Assistant',
  summary: 'Finds and synthesizes evidence.',
  profile: {
    suggested_name: 'research-assistant',
    description: 'A careful research partner.',
    soul: 'Verify claims. Cite primary sources. State uncertainty.',
    starter_prompt: 'What are we researching?'
  },
  capabilities: { skills: ['research/arxiv'], toolsets: ['web'] },
  setup: { requirements: [] },
  routines: [{ id: 'digest', name: 'Research digest', prompt: 'Summarize the latest evidence.', schedule: '0 9 * * 1' }],
  presentation: { emoji: '🔎', color: 'blue' }
}

const pendingStatus = {
  profile: 'research-assistant',
  catalog_name: catalogEntry.name,
  setup_state: 'needs_setup' as const,
  requirements: [{
    kind: 'connector' as const,
    id: 'gmail',
    required: true,
    purpose: 'Deliver digests.',
    status: 'needs_setup' as const,
    action: 'connect',
    detail: 'Connect an account.'
  }],
  runtime: {
    ok: true,
    provider: 'nous',
    model: 'Hermes-4',
    source: 'shared-auth',
    reused_sign_in: true
  },
  first_task: { status: 'pending' as const },
  starter_prompt: catalogEntry.profile.starter_prompt,
  can_start_first_task: false,
  can_activate_routines: false
}

const readyStatus = {
  ...pendingStatus,
  setup_state: 'ready' as const,
  requirements: pendingStatus.requirements.map(requirement => ({ ...requirement, status: 'ready' as const, action: null })),
  first_task: { status: 'complete' as const, session_id: 'bot-chat' },
  can_activate_routines: true
}

const routine = {
  id: 'digest',
  name: 'Research digest',
  prompt: 'Summarize the latest evidence.',
  schedule: '0 9 * * 1',
  state: 'paused' as const,
  timezone: null,
  destination: null
}

const request = vi.fn(async (method: string, _params?: Record<string, unknown>): Promise<any> => {
  if (method === 'bots.catalog') {return { entries: [catalogEntry], removed: [] }}

  if (method === 'bots.installed') {return { bots: [{
    profile: 'research-assistant',
    path: '/profiles/research-assistant',
    catalog_name: catalogEntry.name,
    title: catalogEntry.title,
    summary: catalogEntry.summary,
    setup_state: pendingStatus.setup_state,
    presentation: catalogEntry.presentation
  }] }}

  if (method === 'bots.status' || method === 'bots.setup') {return pendingStatus}

  if (method === 'bots.routines.list') {return { routines: [routine] }}

  if (method === 'bots.install') {return {
    committed: true,
    name: 'research-assistant-2',
    path: '/profiles/research-assistant-2',
    catalog_name: catalogEntry.name,
    catalog_version: catalogEntry.version,
    source_profile: 'default',
    setup_state: 'needs_setup'
  }}

  return {}
})

const kickoff = vi.fn(async () => 'session')
const open = vi.fn(async () => 'session')
const setupAction = vi.fn()
const invalidateRoster = vi.fn(async () => undefined)
const notify = vi.fn()

function renderTab(scope: string | { connectionId: string; profile: string } = 'default') {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <BotMarketplaceTab
        invalidateRoster={invalidateRoster}
        notify={notify}
        onKickoff={kickoff}
        onOpen={open}
        onSetupAction={setupAction}
        request={request}
        scope={scope}
      />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  closeBotMarketplaceRequest()
  $foregroundBotProfile.set('')
  vi.clearAllMocks()
})

describe('Bot Marketplace picker boundary', () => {
  it('requires both the public origin and exact iframe window', () => {
    const iframeWindow = {} as Window
    const otherWindow = {} as Window
    const data = { type: 'hermes-bot-pick', catalog: catalogEntry.name }

    expect(isTrustedBotPickerMessage({ origin: 'https://evil.invalid', source: iframeWindow, data }, iframeWindow)).toBeNull()
    expect(isTrustedBotPickerMessage({ origin: 'https://hermes-agent.nousresearch.com', source: otherWindow, data }, iframeWindow)).toBeNull()
    expect(isTrustedBotPickerMessage({ origin: 'https://hermes-agent.nousresearch.com', source: iframeWindow, data }, iframeWindow)).toBe(catalogEntry.name)
  })
})

describe('native Bot Marketplace behavior (mocked backend)', () => {
  it('renders a searchable native fallback and persistent installed setup inventory', async () => {
    renderTab()

    expect(await screen.findByRole('button', { name: 'Add Bot Research Assistant' })).toBeTruthy()
    expect(await screen.findByRole('button', { name: 'Finish setup research-assistant' })).toBeTruthy()
    expect(request.mock.calls.map(([method]) => method)).toEqual(['bots.catalog', 'bots.installed'])

    fireEvent.click(screen.getByRole('button', { name: 'Finish setup research-assistant' }))
    expect(await screen.findByText('Reusing the existing sign-in for this profile.')).toBeTruthy()
    expect(screen.getByText('What are we researching?')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Set up gmail' }))
    expect(setupAction).toHaveBeenCalledWith('connect', 'gmail', expect.objectContaining({
      scope: 'research-assistant'
    }))
  })

  it('reviews the backend blueprint and keeps a committed install in setup instead of starting chat', async () => {
    renderTab({ connectionId: 'homelab', profile: 'source' })
    fireEvent.click(await screen.findByRole('button', { name: 'Add Bot Research Assistant' }))

    expect((await screen.findByRole('dialog')).textContent).toContain('Verify claims. Cite primary sources. State uncertainty.')
    fireEvent.click(screen.getByRole('button', { name: 'Add Bot' }))

    await waitFor(() => expect(request).toHaveBeenCalledWith(
      'bots.status',
      { profile: 'research-assistant-2' },
      { connectionId: 'homelab', profile: 'research-assistant-2' }
    ))
    expect(kickoff).not.toHaveBeenCalled()
    expect((await screen.findAllByRole('button', { name: 'Finish setup research-assistant-2' })).length).toBeGreaterThan(0)
  })

  it('starts the stored starter only when authoritative status allows it, then refreshes to completion', async () => {
    request.mockImplementation(async (method: string) => {
      if (method === 'bots.catalog') {return { entries: [catalogEntry], removed: [] }}

      if (method === 'bots.installed') {return { bots: [{
    profile: 'research-assistant',
    path: '/profiles/research-assistant',
    catalog_name: catalogEntry.name,
    title: catalogEntry.title,
    summary: catalogEntry.summary,
    setup_state: pendingStatus.setup_state,
    presentation: catalogEntry.presentation
  }] }}

      if (method === 'bots.routines.list') {return { routines: [routine] }}

      if (method === 'bots.status') {return kickoff.mock.calls.length ? readyStatus : { ...pendingStatus, requirements: [], can_start_first_task: true }}

      return readyStatus
    })
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Finish setup research-assistant' }))
    const runSample = await screen.findByRole('button', { name: 'Run sample task research-assistant' })
    fireEvent.click(runSample)
    fireEvent.click(runSample)

    await waitFor(() => expect(kickoff).toHaveBeenCalledTimes(1))
    expect(kickoff).toHaveBeenCalledWith('research-assistant', 'research-assistant', 'What are we researching?')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Open research-assistant' })).toBeTruthy())
  })

  it('reconciles only the selected foreground bot when stale metadata still says setup is pending', async () => {
    const otherInstalled = {
      profile: 'other-bot',
      path: '/profiles/other-bot',
      catalog_name: catalogEntry.name,
      title: catalogEntry.title,
      summary: catalogEntry.summary,
      setup_state: pendingStatus.setup_state,
      presentation: catalogEntry.presentation
    }

    request.mockImplementation(async (method: string, params?: { profile?: string }) => {
      if (method === 'bots.catalog') {return { entries: [catalogEntry], removed: [] }}

      if (method === 'bots.installed') {return { bots: [{
        profile: 'research-assistant',
        path: '/profiles/research-assistant',
        catalog_name: catalogEntry.name,
        title: catalogEntry.title,
        summary: catalogEntry.summary,
        setup_state: pendingStatus.setup_state,
        presentation: catalogEntry.presentation
      }, otherInstalled] }}

      if (method === 'bots.status' && params?.profile === 'research-assistant') {return readyStatus}

      if (method === 'bots.routines.list') {return { routines: [routine] }}

      throw new Error(`Unexpected ${method} for ${params?.profile ?? 'unknown profile'}`)
    })
    $foregroundBotProfile.set('research-assistant')

    renderTab()

    expect(await screen.findByRole('button', { name: 'Open research-assistant' })).toBeTruthy()
    expect(request.mock.calls.filter(([method]) => method === 'bots.status').map(([, params]) => params)).toEqual([
      { profile: 'research-assistant' }
    ])
    expect(request).not.toHaveBeenCalledWith('bots.status', { profile: 'other-bot' }, expect.anything())
  })

  it('keeps removed installed bots manageable from their saved metadata', async () => {
    request.mockImplementation(async (method: string) => {
      if (method === 'bots.catalog') {return { entries: [], removed: [catalogEntry.name] }}

      if (method === 'bots.installed') {return { bots: [{
        profile: 'legacy-researcher',
        path: '/profiles/legacy-researcher',
        catalog_name: catalogEntry.name,
        title: catalogEntry.title,
        summary: catalogEntry.summary,
        setup_state: 'ready',
        presentation: catalogEntry.presentation
      }] }}

      return readyStatus
    })

    renderTab()

    expect(await screen.findByText('Research Assistant')).toBeTruthy()
    expect(screen.getByText('No longer available in the live catalog')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Open legacy-researcher' }))
    expect(open).toHaveBeenCalledWith('legacy-researcher', 'legacy-researcher')
  })

  it('shows an unknown deep-link error inside the dialog with recovery actions', async () => {
    request.mockImplementation(async (method: string) => {
      if (method === 'bots.catalog') {return { entries: [catalogEntry], removed: [] }}

      if (method === 'bots.installed') {return { bots: [] }}

      return {}
    })
    openBotMarketplaceRequest('removed-catalog-entry')
    renderTab()

    expect((await screen.findByRole('alert')).textContent).toContain('“removed-catalog-entry” is not available from this backend.')
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(screen.getAllByRole('button', { name: 'Close' }).length).toBeGreaterThan(0)
  })

  it('requires routine schedule, timezone, and destination before explicit activation', async () => {
    request.mockImplementation(async (method: string) => {
      if (method === 'bots.catalog') {return { entries: [catalogEntry], removed: [] }}

      if (method === 'bots.installed') {return { bots: [{
    profile: 'research-assistant',
    path: '/profiles/research-assistant',
    catalog_name: catalogEntry.name,
    title: catalogEntry.title,
    summary: catalogEntry.summary,
    setup_state: readyStatus.setup_state,
    presentation: catalogEntry.presentation
  }] }}

      if (method === 'bots.status') {return readyStatus}

      if (method === 'bots.routines.list') {return { routines: [routine] }}

      return {}
    })
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Open research-assistant' }).then(() => screen.getByText('research-assistant')))

    const activate = await screen.findByRole('button', { name: 'Activate Research digest' })
    expect((activate as HTMLButtonElement).disabled).toBe(true)
    fireEvent.change(screen.getByLabelText('Research digest Destination'), { target: { value: 'discord:research' } })
    expect((activate as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(activate)

    await waitFor(() => expect(request).toHaveBeenCalledWith(
      'bots.routines.activate',
      expect.objectContaining({
        profile: 'research-assistant',
        routine_id: 'digest',
        schedule: '0 9 * * 1',
        timezone: expect.any(String),
        destination: 'discord:research'
      }),
      'research-assistant'
    ))
  })
})
