// IX Agency domain model — clients, engagements, invoices. Local-first: the
// whole book persists as one JSON blob in localStorage (see store.ts), the
// same posture as pinned sessions / layout prefs. No server round-trips.

export type ClientStatus = 'active' | 'churned' | 'lead' | 'paused'

export interface AgencyClient {
  id: string
  name: string
  company: string
  email: string
  status: ClientStatus
  notes: string
  createdAt: string
  updatedAt: string
}

export type EngagementStatus = 'active' | 'done' | 'on-hold' | 'proposal'

export type EngagementBilling = 'fixed' | 'hourly' | 'monthly'

export interface AgencyEngagement {
  id: string
  clientId: string
  title: string
  status: EngagementStatus
  billing: EngagementBilling
  /** Contract value in whole USD (fixed: total; monthly: per month; hourly: rate). */
  amount: number
  startDate: string
  endDate: string
  notes: string
  createdAt: string
  updatedAt: string
}

export type InvoiceStatus = 'draft' | 'overdue' | 'paid' | 'sent'

export interface AgencyInvoice {
  id: string
  clientId: string
  engagementId: string
  /** Human invoice number, e.g. "IX-2026-014". */
  number: string
  /** Whole USD. */
  amount: number
  status: InvoiceStatus
  issuedDate: string
  dueDate: string
  paidDate: string
  notes: string
  createdAt: string
  updatedAt: string
}

export interface AgencyBook {
  clients: AgencyClient[]
  engagements: AgencyEngagement[]
  invoices: AgencyInvoice[]
}

export const CLIENT_STATUSES: readonly ClientStatus[] = ['lead', 'active', 'paused', 'churned']
export const ENGAGEMENT_STATUSES: readonly EngagementStatus[] = ['proposal', 'active', 'on-hold', 'done']
export const ENGAGEMENT_BILLINGS: readonly EngagementBilling[] = ['fixed', 'monthly', 'hourly']
export const INVOICE_STATUSES: readonly InvoiceStatus[] = ['draft', 'sent', 'paid', 'overdue']

// Bundled fallback data (scripts/generate-ix-agency-data.mjs regenerates).

export interface IxSkillItem {
  id: string
  title: string
  description: string
  persona: string
  rank: null | number
  superAdminOnly: boolean
  /** Full playbook prompt — injected into the native chat as an ACTIVE SKILL. */
  content: string
  starterPrompts: string[]
  // Raw scoping metadata (kept for future scope filtering).
  tiers: string[]
  bundles: string[]
  appIds: string[]
}

export interface IxMcpTileItem {
  id: string
  label: string
  blurb?: string
  group?: string
  mcpUrl: string
  domain: string
  mcpAuthHint?: string
  hasDefaultToken?: boolean
}
