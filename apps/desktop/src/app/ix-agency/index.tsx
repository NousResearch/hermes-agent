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
