import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useState } from 'react'

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PageSearchShell } from '../page-search-shell'

import { BillingTab } from './billing-tab'
import { ClientsTab } from './clients-tab'
import { ConnectTab } from './connect-tab'
import { CopilotTab } from './copilot-tab'
import { EngagementsTab } from './engagements-tab'
import { SkillsTab } from './skills-tab'
import { IxStatusStrip } from './status-strip'
import { $agencyBook } from './store'
import { ToolsTab } from './tools-tab'

// IX Agency workspace — fully native, no webview: the native copilot
// (LiteLLM + full admin-mcp tool estate, gated behind the native OTP
// sign-in), the CRM trio (clients / engagements / billing, local-first),
// the org's copilot skill + MCP tool catalogs, and Connect (company
// WireGuard VPN, admin-mcp gateway, LiteLLM, Cognito S2S + Hermes init).
const IX_MODES = ['copilot', 'clients', 'engagements', 'billing', 'skills', 'tools', 'connect'] as const

type IxMode = (typeof IX_MODES)[number]

const TAB_LABEL: Record<IxMode, string> = {
  billing: 'Billing',
  clients: 'Clients',
  connect: 'Connect',
  copilot: 'Copilot',
  engagements: 'Engagements',
  skills: 'Org skills',
  tools: 'Org tools'
}

const SEARCH_PLACEHOLDER: Record<IxMode, string> = {
  billing: 'Search invoices…',
  clients: 'Search clients…',
  connect: '',
  copilot: '',
  engagements: 'Search engagements…',
  skills: 'Search org skills…',
  tools: 'Search org tools…'
}

const SEARCHLESS_MODES: ReadonlySet<IxMode> = new Set<IxMode>(['connect', 'copilot'])

export function IxAgencyView(props: React.ComponentProps<'section'>) {
  const [mode, setMode] = useRouteEnumParam('tab', IX_MODES, 'copilot')
  const [query, setQuery] = useState('')
  const book = useStore($agencyBook)

  const counts: Partial<Record<IxMode, number>> = {
    billing: book.invoices.length,
    clients: book.clients.length,
    engagements: book.engagements.length
  }

  return (
    <PageSearchShell
      {...props}
      activeTab={mode}
      filters={<IxStatusStrip />}
      onSearchChange={setQuery}
      onTabChange={id => {
        setMode(id as IxMode)
        setQuery('')
      }}
      searchHidden={SEARCHLESS_MODES.has(mode)}
      searchPlaceholder={SEARCH_PLACEHOLDER[mode]}
      searchValue={query}
      tabs={IX_MODES.map(id => ({
        id,
        label: TAB_LABEL[id],
        meta: counts[id] || undefined
      }))}
    >
      <aside
        aria-label="Intelliverse native engine availability"
        className="mb-3 rounded-lg border border-border bg-muted/20 p-3 text-xs text-muted-foreground"
      >
        <p className="font-medium text-foreground">Three native engines, one shared Memory</p>
        <p className="mt-1">
          Map each client workspace to one Brand or App ID so Knowledge, customer Memory,
          credentials, and spend stay isolated. Subject identity and consent determine what may
          continue across approved channels.
        </p>
        <ul className="mt-2 grid gap-1 sm:grid-cols-3">
          <li><strong className="text-foreground">QuestX:</strong> connected per-App setup; kiosk completion approved pilots.</li>
          <li><strong className="text-foreground">ContentX + Postiz:</strong> publishing live; Memory automation rolling out.</li>
          <li><strong className="text-foreground">KioskX:</strong> connected deployment; kiosk Worlds approved pilots.</li>
        </ul>
        <p className="mt-2">
          Discount-code and payout review workflows are roadmap, not live Desktop actions.
        </p>
      </aside>
      {mode === 'copilot' && <CopilotTab />}
      {mode === 'clients' && <ClientsTab query={query} />}
      {mode === 'engagements' && <EngagementsTab query={query} />}
      {mode === 'billing' && <BillingTab query={query} />}
      {mode === 'skills' && <SkillsTab onRunNatively={() => setMode('copilot')} query={query} />}
      {mode === 'tools' && <ToolsTab query={query} />}
      {mode === 'connect' && <ConnectTab />}
    </PageSearchShell>
  )
}
