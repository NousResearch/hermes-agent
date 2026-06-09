import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.fn()
const openExternal = vi.fn()
const writeText = vi.fn()

const cockpitResponse = {
  status: 'ok',
  updated_at: '2026-06-07T10:00:00+02:00',
  dispatch_boundary: {
    dispatch_embedded: false,
    dispatch_url: 'http://127.0.0.1:3001',
    rule: 'Dispatch stays in real Dispatch Dashboard'
  },
  jarvis_todo: [{ id: 'cockpit-local-report', title: 'Regenerate local Cockpit report', status: 'todo' }],
  gates: [
    {
      id: 'gate-cockpit-health',
      title: 'Aktivera Hermes-status v1',
      status: 'waiting',
      decision: 'Ska Jarvis bygga enkel trafikljusstatus i Cockpit?',
      recommendation: 'Kör lokal v1 utan externa writes.',
      scope: 'Endast lokal Cockpit-UI och read-only kontrakt.',
      default_action: 'Kör',
      risk: 'Låg',
      owner: 'Jarvis',
      source: 'jarvis-cockpit-session'
    }
  ],
  status_cards: [{ id: 'hermes-jarvis-health', title: 'Hermes / Jarvis hälsa', state: 'attention', summary: 'Lokal trafikljusstatus.' }],
  artifacts: [],
  graphify: {
    status: 'ok',
    state: 'ok',
    scope: 'Hermes Desktop / Jarvis Cockpit',
    nodes: 344,
    edges: 366,
    communities: 14,
    token_reduction: '11.3x',
    updated_at: '2026-06-09T01:03:21+02:00',
    report_path: '/tmp/graphify-jarvis-cockpit/GRAPH_REPORT.md',
    html_path: '/tmp/graphify-jarvis-cockpit/graph.html',
    notes_dir: '/tmp/graphify-jarvis-cockpit/notes',
    latest_note_path: '/tmp/graphify-jarvis-cockpit/notes/20260609-graphify.md',
    latest_note_title: 'Affected: healthTone',
    latest_note_updated_at: '2026-06-09T01:04:21+02:00',
    safety: {
      local_only: true,
      structural_only: true,
      external_writes: false,
      semantic_llm: false,
      hooks: false,
      mcp: false,
      watch: false,
      secrets_found: false
    }
  },
  sources: [],
  missing: [],
  safety: {
    read_only: true,
    microsoft_writes: false,
    blikk_writes: false,
    mail_mutation: false,
    secrets_read: false,
    dispatch_embedded: false
  },
  improvements: {
    suggestions: [
      {
        id: 'cockpit-improvements-history',
        title: 'Förbättringar med Historik',
        classification: 'Bygg nu',
        status: 'active',
        why: 'Förslag ska kunna köras, parkeras eller avfärdas.',
        benefit: 'Mindre brus.',
        risk: 'Låg',
        next_step: 'Bygg flikar och knappar.'
      },
      {
        id: 'cron-improvement-engine',
        title: 'Styr om nattligt cronjobb till förbättringsmotor',
        classification: 'Förbered',
        status: 'parked',
        why: 'Cron ska läsa historik.'
      }
    ],
    history: []
  }
}

beforeEach(() => {
  writeText.mockResolvedValue(undefined)
  api.mockImplementation(({ body, method, path }: { body?: Record<string, unknown>; method?: string; path: string }) => {
    if (path === '/api/jarvis/cockpit/local-report' && method === 'POST') {
      return Promise.resolve({
        status: 'ok',
        report_path: '/tmp/jarvis-cockpit-local-report.md',
        safety: { local_only: true, external_writes: false }
      })
    }
    if (path === '/api/jarvis/cockpit/improvements/action' && method === 'POST') {
      return Promise.resolve({
        status: 'ok',
        action: 'park',
        suggestion_id: 'cockpit-improvements-history',
        improvements: {
          suggestions: [
            {
              ...cockpitResponse.improvements.suggestions[0],
              status: 'parked'
            }
          ],
          history: [
            {
              id: 'hist-1',
              suggestion_id: 'cockpit-improvements-history',
              title: 'Förbättringar med Historik',
              action: 'park',
              classification: 'Bygg nu',
              actor: 'Tobias',
              at: '2026-06-07T10:01:00+02:00'
            }
          ]
        },
        safety: { local_only: true, external_writes: false }
      })
    }
    if (path === '/api/jarvis/cockpit/graphify/query' && method === 'POST') {
      const mode = String(body?.mode || 'query')
      return Promise.resolve({
        status: 'ok',
        mode,
        output: mode === 'affected' ? 'Affected nodes for healthTone()\n- JarvisCockpitView() [calls]' : '2 nodes found\nJarvisCockpitView -> healthTone',
        safety: { local_only: true, structural_only: true, external_writes: false, semantic_llm: false, hooks: false, mcp: false, watch: false, query_log: false }
      })
    }
    if (path === '/api/jarvis/cockpit/graphify/note' && method === 'POST') {
      return Promise.resolve({
        status: 'ok',
        action: 'jarvis-cockpit-graphify-note',
        note_path: '/tmp/jarvis-cockpit-graphify/notes/20260609-graphify.md',
        note_title: String(body?.title || 'Graphify note'),
        notes_dir: '/tmp/jarvis-cockpit-graphify/notes',
        updated_at: '2026-06-09T01:05:21+02:00',
        safety: { local_only: true, external_writes: false }
      })
    }
    if (path === '/api/jarvis/cockpit') {
      return Promise.resolve(cockpitResponse)
    }
    return Promise.reject(new Error(`unexpected api call: ${method || 'GET'} ${path}`))
  })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api, openExternal }
  })
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('JarvisCockpitView', () => {
  it('generates a local-only cockpit report via the safe POST endpoint', async () => {
    const { JarvisCockpitView } = await import('./index')

    render(<JarvisCockpitView />)

    const button = await screen.findByRole('button', { name: /generate local report/i })
    fireEvent.click(button)

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({ path: '/api/jarvis/cockpit/local-report', method: 'POST' })
    )
    expect(await screen.findByText(/Local report created/i)).toBeTruthy()
  })

  it('renders Hermes health traffic light and Gates decision cards', async () => {
    const { JarvisCockpitView } = await import('./index')

    render(<JarvisCockpitView />)

    expect(await screen.findAllByText('Hermes / Jarvis hälsa')).toHaveLength(2)
    expect(await screen.findByText('Gul')).toBeTruthy()
    expect(await screen.findByText(/Cockpit fungerar, men något behöver koll eller beslut/i)).toBeTruthy()
    expect(await screen.findByText('Aktivera Hermes-status v1')).toBeTruthy()
    expect(await screen.findByText('Ska Jarvis bygga enkel trafikljusstatus i Cockpit?')).toBeTruthy()
    expect(await screen.findByText('Kör lokal v1 utan externa writes.')).toBeTruthy()
    expect(await screen.findByText('Endast lokal Cockpit-UI och read-only kontrakt.')).toBeTruthy()
  })

  it('renders Graphify local project graph lane', async () => {
    const { JarvisCockpitView } = await import('./index')

    render(<JarvisCockpitView />)

    expect(await screen.findByText('Graphify project graph')).toBeTruthy()
    expect(await screen.findByText(/344 nodes \/ 366 edges/i)).toBeTruthy()
    expect(await screen.findByText(/11.3x/i)).toBeTruthy()
    expect(await screen.findByText(/semantic LLM, hooks, MCP and watch remain gated/i)).toBeTruthy()
    expect(await screen.findByText('Local notes')).toBeTruthy()
    expect((await screen.findAllByText('Affected: healthTone')).length).toBeGreaterThanOrEqual(1)
    expect(await screen.findByRole('button', { name: /Open latest note/i })).toBeTruthy()
    expect(await screen.findByRole('button', { name: /Open notes folder/i })).toBeTruthy()
  })

  it('runs Graphify local query helper from the Cockpit lane', async () => {
    const { JarvisCockpitView } = await import('./index')

    render(<JarvisCockpitView />)

    fireEvent.change(await screen.findByLabelText('Graphify query'), { target: { value: 'JarvisCockpitView' } })
    fireEvent.click(screen.getByRole('button', { name: 'Query' }))

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        path: '/api/jarvis/cockpit/graphify/query',
        method: 'POST',
        body: { mode: 'query', question: 'JarvisCockpitView', budget: 1200 }
      })
    )
    expect(await screen.findAllByText(/JarvisCockpitView -> healthTone/i)).toHaveLength(2)
  })

  it('runs Graphify preset helpers, keeps insight cards, saves notes and copies the result', async () => {
    const { JarvisCockpitView } = await import('./index')

    render(<JarvisCockpitView />)

    fireEvent.click(await screen.findByRole('button', { name: /What affects Cockpit health\?/i }))

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        path: '/api/jarvis/cockpit/graphify/query',
        method: 'POST',
        body: { mode: 'affected', node: 'healthTone', depth: 2 }
      })
    )
    expect(await screen.findAllByText(/Affected nodes for healthTone/i)).toHaveLength(2)
    expect(await screen.findByText(/query log off/i)).toBeTruthy()
    expect(await screen.findByText('Latest insights')).toBeTruthy()
    expect((await screen.findAllByText('Affected: healthTone')).length).toBeGreaterThanOrEqual(1)

    fireEvent.click(screen.getByRole('button', { name: /Save as local Cockpit note/i }))
    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        path: '/api/jarvis/cockpit/graphify/note',
        method: 'POST',
        body: {
          title: 'Affected: healthTone',
          mode: 'affected',
          question: 'healthTone',
          output: expect.stringContaining('Affected nodes for healthTone')
        }
      })
    )
    expect(await screen.findByText(/Saved local Graphify note/i)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /Copy result/i }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(expect.stringContaining('Affected nodes for healthTone')))
    expect(await screen.findByRole('button', { name: /Copied/i })).toBeTruthy()
  })

  it('renders improvement tabs and records local-only park actions', async () => {
    const { JarvisCockpitView } = await import('./index')

    render(<JarvisCockpitView />)

    expect(await screen.findByText('Förbättringar med Historik')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Parkera' }))

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        path: '/api/jarvis/cockpit/improvements/action',
        method: 'POST',
        body: { suggestion_id: 'cockpit-improvements-history', action: 'park', actor: 'Tobias' }
      })
    )
    expect(await screen.findByText(/Lokal förbättringshistorik uppdaterad/i)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Historik \(1\)/i }))
    expect(await screen.findByText(/Tobias · Bygg nu/i)).toBeTruthy()
  })

})
