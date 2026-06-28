import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getSkills = vi.fn()
const getToolsets = vi.fn()
const toggleSkill = vi.fn()
const toggleToolset = vi.fn()
const getToolsetConfig = vi.fn()
const selectToolsetProvider = vi.fn()

vi.mock('@/hermes', () => ({
  getSkills: () => getSkills(),
  getToolsets: () => getToolsets(),
  toggleSkill: (name: string, enabled: boolean) => toggleSkill(name, enabled),
  toggleToolset: (name: string, enabled: boolean) => toggleToolset(name, enabled),
  getToolsetConfig: (name: string) => getToolsetConfig(name),
  selectToolsetProvider: (toolset: string, provider: string) => selectToolsetProvider(toolset, provider),
  deleteEnvVar: vi.fn(),
  revealEnvVar: vi.fn(),
  setEnvVar: vi.fn()
}))

// Notifications hit nanostores/timers we don't care about here.
vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function toolset(overrides: Record<string, unknown> = {}) {
  return {
    name: 'web',
    label: 'Web Search',
    description: 'web_search, web_extract',
    enabled: true,
    available: true,
    configured: true,
    tools: ['web_search', 'web_extract'],
    ...overrides
  }
}

function renderSkills() {
  return import('./index').then(({ SkillsView }) =>
    render(
      <MemoryRouter initialEntries={['/skills?tab=toolsets']}>
        <SkillsView />
      </MemoryRouter>
    )
  )
}

beforeEach(() => {
  getSkills.mockResolvedValue([])
  getToolsets.mockResolvedValue([toolset()])
  toggleToolset.mockResolvedValue({ ok: true, name: 'web', enabled: false })
  getToolsetConfig.mockResolvedValue({ has_category: false, active_provider: null, providers: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('SkillsView toolset management', () => {
  it('renders a switch for each toolset and toggles it off', async () => {
    await renderSkills()

    const sw = await screen.findByRole('switch', { name: 'Toggle Web Search toolset' })
    expect(sw.getAttribute('aria-checked')).toBe('true')

    fireEvent.click(sw)

    await waitFor(() => expect(toggleToolset).toHaveBeenCalledWith('web', false))
  }, 10_000)

  it('renders toolset titles without leading emoji', async () => {
    getToolsets.mockResolvedValue([toolset({ name: 'cronjob', label: '⏰ Cron Jobs', description: 'cron tools' })])

    await renderSkills()

    expect(await screen.findByText('Cron Jobs')).toBeTruthy()
    expect(screen.queryByText(/⏰/)).toBeNull()
  })

  it('keeps the configured pill alongside the switch', async () => {
    await renderSkills()

    await screen.findByRole('switch', { name: 'Toggle Web Search toolset' })
    expect(screen.getByText('Configured')).toBeTruthy()
  })

  it('expands the provider config panel when the configured pill is clicked', async () => {
    await renderSkills()

    const configureBtn = await screen.findByRole('button', { name: 'Configure Web Search' })
    fireEvent.click(configureBtn)

    await waitFor(() => expect(getToolsetConfig).toHaveBeenCalledWith('web'))
  })

  it('renders plugin-provided SurfSense video planning in the desktop toolset list', async () => {
    getToolsets.mockResolvedValue([
      toolset({
        name: 'surfsense',
        label: 'SurfSense',
        description:
          'Connect Hermes to a self-hosted SurfSense NotebookLM-style knowledge base, including source-grounded video planning for Manim, HeyGen, HyperFrames, AITuber OnAir, irodoriTTS, MP4 audio muxing, LLM-wiki, codegraph, sleep digests, and memory.',
        tools: ['surfsense_search', 'surfsense_ask', 'surfsense_video_plan', 'surfsense_video_mux']
      })
    ])

    await renderSkills()

    expect(await screen.findByText('SurfSense')).toBeTruthy()
    expect(screen.getByText(/Manim, HeyGen, HyperFrames, AITuber OnAir, irodoriTTS, MP4 audio muxing/)).toBeTruthy()
    expect(screen.getByText('surfsense_video_plan')).toBeTruthy()
    expect(screen.getByText('surfsense_video_mux')).toBeTruthy()
  })
})
