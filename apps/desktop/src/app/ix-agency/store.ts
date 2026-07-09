// IX Agency book store. One persisted atom holds the entire book (clients +
// engagements + invoices) so every mutation is a pure book → book map and the
// whole state survives restarts via the shared storage choke point.
import { Codecs, persistentAtom } from '@/lib/persisted'

import {
  type AgencyBook,
  type AgencyClient,
  type AgencyEngagement,
  type AgencyInvoice,
  CLIENT_STATUSES,
  ENGAGEMENT_BILLINGS,
  ENGAGEMENT_STATUSES,
  INVOICE_STATUSES
} from './types'

const BOOK_KEY = 'hermes.desktop.ixAgency.book'

const EMPTY_BOOK: AgencyBook = { clients: [], engagements: [], invoices: [] }

const str = (value: unknown): string => (typeof value === 'string' ? value : '')

const num = (value: unknown): number => {
  const parsed = Number(value)

  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return allowed.includes(value as T) ? (value as T) : fallback
}

// The persisted blob is untrusted (hand-edited storage, older schema versions):
// rebuild every record field-by-field and drop anything without an id.
function sanitizeBook(raw: unknown): AgencyBook {
  if (!raw || typeof raw !== 'object') {
    return { ...EMPTY_BOOK }
  }

  const source = raw as Record<string, unknown>

  const rows = (value: unknown): Record<string, unknown>[] =>
    Array.isArray(value) ? value.filter((row): row is Record<string, unknown> => !!row && typeof row === 'object') : []

  const clients: AgencyClient[] = rows(source.clients)
    .filter(row => str(row.id))
    .map(row => ({
      id: str(row.id),
      name: str(row.name),
      company: str(row.company),
      email: str(row.email),
      status: oneOf(row.status, CLIENT_STATUSES, 'lead'),
      notes: str(row.notes),
      createdAt: str(row.createdAt),
      updatedAt: str(row.updatedAt)
    }))

  const engagements: AgencyEngagement[] = rows(source.engagements)
    .filter(row => str(row.id))
    .map(row => ({
      id: str(row.id),
      clientId: str(row.clientId),
      title: str(row.title),
      status: oneOf(row.status, ENGAGEMENT_STATUSES, 'proposal'),
      billing: oneOf(row.billing, ENGAGEMENT_BILLINGS, 'fixed'),
      amount: num(row.amount),
      startDate: str(row.startDate),
      endDate: str(row.endDate),
      notes: str(row.notes),
      createdAt: str(row.createdAt),
      updatedAt: str(row.updatedAt)
    }))

  const invoices: AgencyInvoice[] = rows(source.invoices)
    .filter(row => str(row.id))
    .map(row => ({
      id: str(row.id),
      clientId: str(row.clientId),
      engagementId: str(row.engagementId),
      number: str(row.number),
      amount: num(row.amount),
      status: oneOf(row.status, INVOICE_STATUSES, 'draft'),
      issuedDate: str(row.issuedDate),
      dueDate: str(row.dueDate),
      paidDate: str(row.paidDate),
      notes: str(row.notes),
      createdAt: str(row.createdAt),
      updatedAt: str(row.updatedAt)
    }))

  return { clients, engagements, invoices }
}

export const $agencyBook = persistentAtom<AgencyBook>(BOOK_KEY, EMPTY_BOOK, Codecs.json(sanitizeBook))

const newId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`

const nowIso = () => new Date().toISOString()

const todayIso = () => nowIso().slice(0, 10)

function mutate(update: (book: AgencyBook) => AgencyBook) {
  $agencyBook.set(update($agencyBook.get()))
}

// ── Clients ─────────────────────────────────────────────────────────────────

export function addClient(input: Partial<AgencyClient> = {}): AgencyClient {
  const stamp = nowIso()

  const client: AgencyClient = {
    id: newId(),
    name: input.name ?? '',
    company: input.company ?? '',
    email: input.email ?? '',
    status: input.status ?? 'lead',
    notes: input.notes ?? '',
    createdAt: stamp,
    updatedAt: stamp
  }

  mutate(book => ({ ...book, clients: [client, ...book.clients] }))

  return client
}

export function updateClient(id: string, patch: Partial<Omit<AgencyClient, 'createdAt' | 'id'>>) {
  mutate(book => ({
    ...book,
    clients: book.clients.map(client => (client.id === id ? { ...client, ...patch, updatedAt: nowIso() } : client))
  }))
}

/** Removing a client also removes its engagements and invoices — no orphans. */
export function removeClient(id: string) {
  mutate(book => ({
    clients: book.clients.filter(client => client.id !== id),
    engagements: book.engagements.filter(engagement => engagement.clientId !== id),
    invoices: book.invoices.filter(invoice => invoice.clientId !== id)
  }))
}

// ── Engagements ─────────────────────────────────────────────────────────────

export function addEngagement(input: Partial<AgencyEngagement> = {}): AgencyEngagement {
  const stamp = nowIso()

  const engagement: AgencyEngagement = {
    id: newId(),
    clientId: input.clientId ?? '',
    title: input.title ?? '',
    status: input.status ?? 'proposal',
    billing: input.billing ?? 'fixed',
    amount: input.amount ?? 0,
    startDate: input.startDate ?? todayIso(),
    endDate: input.endDate ?? '',
    notes: input.notes ?? '',
    createdAt: stamp,
    updatedAt: stamp
  }

  mutate(book => ({ ...book, engagements: [engagement, ...book.engagements] }))

  return engagement
}

export function updateEngagement(id: string, patch: Partial<Omit<AgencyEngagement, 'createdAt' | 'id'>>) {
  mutate(book => ({
    ...book,
    engagements: book.engagements.map(engagement =>
      engagement.id === id ? { ...engagement, ...patch, updatedAt: nowIso() } : engagement
    )
  }))
}

/** Invoices keep their client but drop the engagement link. */
export function removeEngagement(id: string) {
  mutate(book => ({
    ...book,
    engagements: book.engagements.filter(engagement => engagement.id !== id),
    invoices: book.invoices.map(invoice =>
      invoice.engagementId === id ? { ...invoice, engagementId: '' } : invoice
    )
  }))
}

// ── Invoices ────────────────────────────────────────────────────────────────

/** Next sequential invoice number: IX-<year>-NNN over the current book. */
export function nextInvoiceNumber(book: AgencyBook): string {
  const year = new Date().getFullYear()
  const prefix = `IX-${year}-`

  const max = book.invoices.reduce((acc, invoice) => {
    if (!invoice.number.startsWith(prefix)) {
      return acc
    }

    const seq = Number(invoice.number.slice(prefix.length))

    return Number.isFinite(seq) ? Math.max(acc, seq) : acc
  }, 0)

  return `${prefix}${String(max + 1).padStart(3, '0')}`
}

export function addInvoice(input: Partial<AgencyInvoice> = {}): AgencyInvoice {
  const stamp = nowIso()

  const invoice: AgencyInvoice = {
    id: newId(),
    clientId: input.clientId ?? '',
    engagementId: input.engagementId ?? '',
    number: input.number ?? nextInvoiceNumber($agencyBook.get()),
    amount: input.amount ?? 0,
    status: input.status ?? 'draft',
    issuedDate: input.issuedDate ?? todayIso(),
    dueDate: input.dueDate ?? '',
    paidDate: input.paidDate ?? '',
    notes: input.notes ?? '',
    createdAt: stamp,
    updatedAt: stamp
  }

  mutate(book => ({ ...book, invoices: [invoice, ...book.invoices] }))

  return invoice
}

export function updateInvoice(id: string, patch: Partial<Omit<AgencyInvoice, 'createdAt' | 'id'>>) {
  mutate(book => ({
    ...book,
    invoices: book.invoices.map(invoice =>
      invoice.id === id ? { ...invoice, ...patch, updatedAt: nowIso() } : invoice
    )
  }))
}

export function removeInvoice(id: string) {
  mutate(book => ({ ...book, invoices: book.invoices.filter(invoice => invoice.id !== id) }))
}

// ── Derived ─────────────────────────────────────────────────────────────────

export interface BillingTotals {
  outstanding: number
  overdue: number
  paidThisYear: number
}

export function billingTotals(book: AgencyBook): BillingTotals {
  const year = String(new Date().getFullYear())

  return book.invoices.reduce<BillingTotals>(
    (acc, invoice) => {
      if (invoice.status === 'sent') {
        acc.outstanding += invoice.amount
      }

      if (invoice.status === 'overdue') {
        acc.outstanding += invoice.amount
        acc.overdue += invoice.amount
      }

      if (invoice.status === 'paid' && (invoice.paidDate || invoice.issuedDate).startsWith(year)) {
        acc.paidThisYear += invoice.amount
      }

      return acc
    },
    { outstanding: 0, overdue: 0, paidThisYear: 0 }
  )
}

export const clientById = (book: AgencyBook, id: string): AgencyClient | undefined =>
  book.clients.find(client => client.id === id)

export const engagementById = (book: AgencyBook, id: string): AgencyEngagement | undefined =>
  book.engagements.find(engagement => engagement.id === id)

/** Whole-USD display: $12,500. Money never goes through compactNumber. */
export function formatUsd(amount: number): string {
  return `$${Math.round(amount).toLocaleString('en-US')}`
}
