/**
 * A gateway section heading must stay addressable in the DOM.
 *
 * `e2e/cloud-gateway-roster.spec.ts` addresses ONE section's health through
 * `[data-roster-section="gateway:<id>"]` (its `sectionOrder` assertion, and the
 * stopped-gateway test's per-section "Unavailable" text). The heading is always
 * tipped (`GatewaySectionHeading` builds `${label} · ${status}`), so the
 * attribute has to survive the `Tip` → `TooltipTrigger asChild` wrapper: an
 * `asChild` trigger merges its own props into the child, and anything the merge
 * drops makes every section assertion silently match nothing.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    // The plugin bundle normally lands via `ctx.i18n.register` at load, so
    // without this every localized label renders empty.
    usePluginI18n: () => translateBots
  }
})

const { GatewaySectionHeading, RosterSectionHeader } = await import('./roster-sections')

afterEach(() => {
  cleanup()
})

describe('roster section headings keep data-roster-section', () => {
  it('stamps the section key on an untipped heading', () => {
    const { container } = render(
      <RosterSectionHeader collapsed={false} count={9} label="This device" onToggle={() => undefined} sectionKey="gateway:local" />
    )

    expect(container.querySelectorAll('[data-roster-section]')).toHaveLength(1)
    expect(container.querySelector('[data-roster-section]')?.getAttribute('data-roster-section')).toBe('gateway:local')
  })

  it('keeps the section key when a tip wraps the heading', () => {
    const { container } = render(
      <RosterSectionHeader
        collapsed={false}
        count={1}
        label="Cloud 123"
        onToggle={() => undefined}
        sectionKey="gateway:cloud-123"
        tip="Cloud 123 · Available"
      />
    )

    expect(container.querySelector('[data-roster-section="gateway:cloud-123"]')).not.toBeNull()
  })

  it('keeps the section key on a gateway heading (which always passes a tip)', () => {
    const { container } = render(
      <GatewaySectionHeading
        collapsed={false}
        count={1}
        onToggle={() => undefined}
        option={{ connectionId: 'cloud-123', kind: 'remote', label: 'Cloud 123', name: 'Cloud 123', reachable: true }}
      />
    )

    expect(container.querySelector('[data-roster-section="gateway:cloud-123"]')).not.toBeNull()
  })

  it('keeps the section key on a local gateway heading', () => {
    const { container } = render(
      <GatewaySectionHeading
        collapsed={false}
        count={9}
        onToggle={() => undefined}
        option={{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }}
      />
    )

    expect(container.querySelector('[data-roster-section="gateway:local"]')).not.toBeNull()
  })
})