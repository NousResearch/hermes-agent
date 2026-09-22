import { expect, it } from 'vitest'

import { orderConnectorPicks } from '@/components/onboarding-chat/options'
import { onboardingRecommendations } from '@/lib/onboarding-recommendations'
import type { McpCatalogEntry } from '@/types/hermes'

it('keeps new enabled managed apps searchable without disturbing the curated leaders', () => {
  const next = { connector: 'new-reviewed-app', enabled: true }
  const lead = { connector: 'gmail', enabled: true }
  const disabled = { connector: 'disabled-app', enabled: false }
  const channel = { connector: 'discord', enabled: true }
  expect(orderConnectorPicks([next, disabled, channel, lead])).toEqual([lead, next])
})

function entry(name: string, patch: Partial<McpCatalogEntry> = {}): McpCatalogEntry {
  return {
    name,
    description: `${name} integration`,
    source: 'https://example.test',
    transport: 'stdio',
    auth_type: 'none',
    required_env: [],
    command: 'example',
    args: [],
    url: null,
    install_url: null,
    install_ref: null,
    bootstrap: [],
    default_enabled: null,
    post_install: 'Enable the app integration.',
    needs_install: false,
    installed: false,
    enabled: false,
    suggest: {
      keywords: [name],
      hosts: [],
      applications: [name],
      examples: [`Make something in ${name}`],
      requires_app: true
    },
    ...patch
  }
}

it('recommends catalog entries from evidence and intent, never from installation as proof of access', () => {
  const future = entry('future-paint', { suggest: null, detected_apps: ['Future Paint'] })
  expect(onboardingRecommendations([future])).toMatchObject([
    {
      name: future.name,
      description: future.description,
      examples: [],
      setupAction: 'install'
    }
  ])
  expect(onboardingRecommendations([])).toEqual([])
  const local = entry('modeler', { detected_apps: ['Modeler'] })
  const newEntry = entry('future-studio', { detected_apps: ['Future Studio'] })
  const longerExample = 'Describe the lighting and materials in my scene before suggesting a different render treatment'
  newEntry.suggest!.examples!.push(longerExample)
  const absent = entry('missing-studio')
  const disabled = entry('paused-studio', { installed: true, detected_apps: ['Paused Studio'] })
  const configured = entry('notes', { installed: true, enabled: true })
  const rows = [absent, disabled, configured, local, newEntry]

  const suggestions = onboardingRecommendations(rows)
  expect(new Set(suggestions.map(row => row.name))).toEqual(new Set([newEntry.name, local.name, configured.name]))
  expect(suggestions[0].name).toBe(configured.name)
  expect(suggestions.every(row => row.examples.length === 1)).toBe(true)
  expect(onboardingRecommendations(rows, { context: 'Use future studio' })[0].examples).toContain(longerExample)
  expect(suggestions.map(row => row.readiness)).not.toContain('connected')
  expect(suggestions.find(row => row.name === local.name)).toMatchObject({
    readiness: 'setup_required',
    detectedApps: ['Modeler'],
    setupAction: 'install'
  })
  expect(suggestions.find(row => row.name === configured.name)?.setupAction).toBeNull()
  expect(onboardingRecommendations(rows, { context: 'Use paused studio' })[0].setupAction).toBe('enable')
  expect(suggestions.find(row => row.name === configured.name)?.readiness).toBe('configured_unverified')
  expect(onboardingRecommendations(rows, { context: 'Help with my notes' })[0].name).toBe(configured.name)
  expect(onboardingRecommendations([entry('legacy', { suggest: { keywords: ['legacy'], hosts: [] } })])).toEqual([])
})
