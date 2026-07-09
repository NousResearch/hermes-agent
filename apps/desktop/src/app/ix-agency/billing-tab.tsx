import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { normalize } from '@/lib/text'

import { DetailColumn, ListColumn, MasterDetail } from '../master-detail'
import { PanelAddButton, PanelEmpty, PanelListRow, PanelMeta, PanelPill, PanelRowMenu } from '../overlays/panel'

import { Field, FieldRow, INVOICE_DOT, StatCard } from './bits'
import { $agencyBook, addInvoice, billingTotals, clientById, formatUsd, removeInvoice, updateInvoice } from './store'
import { type AgencyInvoice, INVOICE_STATUSES, type InvoiceStatus } from './types'

const STATUS_PILL_TONE = { draft: 'muted', overdue: 'bad', paid: 'good', sent: 'muted' } as const

function InvoiceDetail({ invoice }: { invoice: AgencyInvoice }) {
  const book = useStore($agencyBook)

  const engagements = book.engagements.filter(
    engagement => !invoice.clientId || engagement.clientId === invoice.clientId
  )

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
          {invoice.number || 'New invoice'}
        </h3>
        <PanelPill tone={STATUS_PILL_TONE[invoice.status]}>{invoice.status}</PanelPill>
      </div>
      <FieldRow>
        <Field label="Invoice number">
          <Input onChange={e => updateInvoice(invoice.id, { number: e.target.value })} value={invoice.number} />
        </Field>
        <Field label="Amount (USD)">
          <Input
            min={0}
            onChange={e => updateInvoice(invoice.id, { amount: Math.max(0, Number(e.target.value) || 0) })}
            type="number"
            value={invoice.amount || ''}
          />
        </Field>
      </FieldRow>
      <FieldRow>
        <Field label="Client">
          <Select
            // Re-picking the client resets the engagement link: engagements are
            // client-scoped, so a stale link would point across accounts.
            onValueChange={clientId => updateInvoice(invoice.id, { clientId, engagementId: '' })}
            value={invoice.clientId || undefined}
          >
            <SelectTrigger>
              <SelectValue placeholder="Pick a client" />
            </SelectTrigger>
            <SelectContent>
              {book.clients.map(client => (
                <SelectItem key={client.id} value={client.id}>
                  {client.name || 'Untitled client'}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="Engagement">
          <Select
            onValueChange={engagementId => updateInvoice(invoice.id, { engagementId })}
            value={invoice.engagementId || undefined}
          >
            <SelectTrigger>
              <SelectValue placeholder="Optional" />
            </SelectTrigger>
            <SelectContent>
              {engagements.map(engagement => (
                <SelectItem key={engagement.id} value={engagement.id}>
                  {engagement.title || 'Untitled engagement'}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      </FieldRow>
      <FieldRow>
        <Field label="Status">
          <Select
            onValueChange={status => updateInvoice(invoice.id, { status: status as InvoiceStatus })}
            value={invoice.status}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {INVOICE_STATUSES.map(status => (
                <SelectItem key={status} value={status}>
                  {status}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="Issued">
          <Input
            onChange={e => updateInvoice(invoice.id, { issuedDate: e.target.value })}
            type="date"
            value={invoice.issuedDate}
          />
        </Field>
      </FieldRow>
      <FieldRow>
        <Field label="Due">
          <Input
            onChange={e => updateInvoice(invoice.id, { dueDate: e.target.value })}
            type="date"
            value={invoice.dueDate}
          />
        </Field>
        <Field label="Paid">
          <Input
            onChange={e => updateInvoice(invoice.id, { paidDate: e.target.value })}
            type="date"
            value={invoice.paidDate}
          />
        </Field>
      </FieldRow>
      <Field label="Notes">
        <Textarea
          onChange={e => updateInvoice(invoice.id, { notes: e.target.value })}
          placeholder="Line items, payment terms…"
          value={invoice.notes}
        />
      </Field>
      <PanelMeta rows={[{ label: 'Added', value: invoice.createdAt.slice(0, 10) || '—' }]} />
    </div>
  )
}

export function BillingTab({ query }: { query: string }) {
  const book = useStore($agencyBook)
  const [selectedId, setSelectedId] = useState<null | string>(null)

  const totals = billingTotals(book)

  const q = normalize(query)

  const invoices = book.invoices.filter(invoice => {
    if (!q) {
      return true
    }

    const client = clientById(book, invoice.clientId)

    return normalize(`${invoice.number} ${client?.name ?? ''} ${client?.company ?? ''}`).includes(q)
  })

  const selected = book.invoices.find(invoice => invoice.id === selectedId) ?? null

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="grid shrink-0 grid-cols-3 gap-2 px-3 pb-1 pt-2">
        <StatCard label="Outstanding" value={formatUsd(totals.outstanding)} />
        <StatCard label="Overdue" tone={totals.overdue > 0 ? 'bad' : 'default'} value={formatUsd(totals.overdue)} />
        <StatCard label="Paid this year" tone="good" value={formatUsd(totals.paidThisYear)} />
      </div>
      <div className="min-h-0 flex-1">
        <MasterDetail>
          <ListColumn>
            {invoices.map(invoice => {
              const client = clientById(book, invoice.clientId)

              return (
                <PanelListRow
                  active={invoice.id === selectedId}
                  dotClassName={INVOICE_DOT[invoice.status]}
                  key={invoice.id}
                  menu={
                    <PanelRowMenu
                      items={[
                        ...(invoice.status !== 'paid'
                          ? [
                              {
                                icon: 'check',
                                label: 'Mark paid',
                                onSelect: () =>
                                  updateInvoice(invoice.id, {
                                    paidDate: new Date().toISOString().slice(0, 10),
                                    status: 'paid' as const
                                  })
                              }
                            ]
                          : []),
                        {
                          icon: 'trash',
                          label: 'Delete invoice',
                          onSelect: () => {
                            removeInvoice(invoice.id)

                            if (selectedId === invoice.id) {
                              setSelectedId(null)
                            }
                          },
                          tone: 'danger' as const
                        }
                      ]}
                    />
                  }
                  meta={formatUsd(invoice.amount)}
                  onSelect={() => setSelectedId(invoice.id)}
                  title={`${invoice.number || 'Draft'}${client?.name ? ` · ${client.name}` : ''}`}
                />
              )
            })}
            <PanelAddButton label="Add invoice" onClick={() => setSelectedId(addInvoice().id)} />
          </ListColumn>
          <DetailColumn footer={selected ? 'Changes save automatically.' : undefined}>
            {selected ? (
              <InvoiceDetail invoice={selected} />
            ) : (
              <PanelEmpty
                description={
                  book.invoices.length === 0
                    ? 'Invoices across all clients and engagements. Add your first invoice to get started.'
                    : 'Select an invoice on the left, or add a new one.'
                }
                icon="credit-card"
                title="Billing"
              />
            )}
          </DetailColumn>
        </MasterDetail>
      </div>
    </div>
  )
}
