import type { ConnectorsListResult } from '@hermes/shared'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { type FactSources, readConnectorList } from '../facts'
import { closeQuestionnaire, openQuestionnaire, setFacts } from '../store'

import { ConnectorsStep } from './looks'

// What connectors.list answers for a signed-in account: hosted apps, MCP variants among them.
const LIVE_SLUGS = [
  'airtable',
  'asana',
  'better_stack_mcp',
  'canva_mcp',
  'github',
  'gmail',
  'googlecalendar',
  'googledrive',
  'hubspot',
  'linear',
  'notion',
  'slack',
  'stripe',
  'stripe_mcp'
]

function listResult(slugs: string[]): ConnectorsListResult {
  return {
    available: true,
    connectors: slugs.map(connector => ({
      connected: false,
      connection_status: null,
      connector,
      enabled: true,
      gateway_disabled_tools: [],
      status_reason: null
    }))
  }
}

const sources = (slugs: string[]): FactSources => ({
  listConnectors: async () => listResult(slugs),
  localFit: async () => null,
  request: () => Promise.reject(new Error('not read by the connector list'))
})

async function showConnectors(slugs: string[]) {
  openQuestionnaire()
  setFacts({ connectors: await readConnectorList(sources(slugs)) })
  render(<ConnectorsStep />)
}

afterEach(() => {
  cleanup()
  closeQuestionnaire('skipped')
})

describe('ConnectorsStep', () => {
  it('gives every live connector its own label', async () => {
    await showConnectors(LIVE_SLUGS)

    // The chip's label line; the logo beside it may carry the name too.
    const labels = screen
      .getAllByRole('button', { pressed: false })
      .map(button => button.querySelector('span.truncate')?.textContent)

    expect(labels).toContain('Stripe')
    expect(labels).toContain('Stripe MCP')
    expect(new Set(labels).size).toBe(labels.length)
  })

  it('keeps a long connector list inside a height-capped scroller', async () => {
    await showConnectors(Array.from({ length: 45 }, (_, index) => `app_${index}`))

    const scroller = screen.getByText('App 44').closest<HTMLElement>('.overflow-y-auto')

    expect(scroller?.style.maxHeight).toMatch(/rem$/)
    expect(scroller?.contains(screen.getByText('App 0'))).toBe(true)
  })
})
