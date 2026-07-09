import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { normalize } from '@/lib/text'

import { DetailColumn, ListColumn, MasterDetail } from '../master-detail'
import { PanelAddButton, PanelEmpty, PanelListRow, PanelMeta, PanelPill, PanelRowMenu } from '../overlays/panel'

import { ENGAGEMENT_DOT, Field, FieldRow } from './bits'
import { $agencyBook, addEngagement, clientById, formatUsd, removeEngagement, updateEngagement } from './store'
import {
  type AgencyEngagement,
  ENGAGEMENT_BILLINGS,
  ENGAGEMENT_STATUSES,
  type EngagementBilling,
  type EngagementStatus
} from './types'

const STATUS_PILL_TONE = { active: 'good', done: 'muted', 'on-hold': 'warn', proposal: 'muted' } as const

const BILLING_LABEL: Record<EngagementBilling, string> = {
  fixed: 'Fixed price',
  hourly: 'Hourly',
  monthly: 'Monthly retainer'
}

const amountLabel = (billing: EngagementBilling) =>
  billing === 'monthly' ? 'Amount (USD / month)' : billing === 'hourly' ? 'Rate (USD / hour)' : 'Amount (USD)'

function EngagementDetail({ engagement }: { engagement: AgencyEngagement }) {
  const book = useStore($agencyBook)

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
          {engagement.title || 'New engagement'}
        </h3>
        <PanelPill tone={STATUS_PILL_TONE[engagement.status]}>{engagement.status}</PanelPill>
      </div>
      <Field label="Title">
        <Input
          onChange={e => updateEngagement(engagement.id, { title: e.target.value })}
          placeholder="Website revamp, SEO retainer…"
          value={engagement.title}
        />
      </Field>
      <FieldRow>
        <Field label="Client">
          <Select
            onValueChange={clientId => updateEngagement(engagement.id, { clientId })}
            value={engagement.clientId || undefined}
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
        <Field label="Status">
          <Select
            onValueChange={status => updateEngagement(engagement.id, { status: status as EngagementStatus })}
            value={engagement.status}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ENGAGEMENT_STATUSES.map(status => (
                <SelectItem key={status} value={status}>
                  {status}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      </FieldRow>
      <FieldRow>
        <Field label="Billing">
          <Select
            onValueChange={billing => updateEngagement(engagement.id, { billing: billing as EngagementBilling })}
            value={engagement.billing}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ENGAGEMENT_BILLINGS.map(billing => (
                <SelectItem key={billing} value={billing}>
                  {BILLING_LABEL[billing]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label={amountLabel(engagement.billing)}>
          <Input
            min={0}
            onChange={e => updateEngagement(engagement.id, { amount: Math.max(0, Number(e.target.value) || 0) })}
            type="number"
            value={engagement.amount || ''}
          />
        </Field>
      </FieldRow>
      <FieldRow>
        <Field label="Start date">
          <Input
            onChange={e => updateEngagement(engagement.id, { startDate: e.target.value })}
            type="date"
            value={engagement.startDate}
          />
        </Field>
        <Field label="End date">
          <Input
            onChange={e => updateEngagement(engagement.id, { endDate: e.target.value })}
            type="date"
            value={engagement.endDate}
          />
        </Field>
      </FieldRow>
      <Field label="Notes">
        <Textarea
          onChange={e => updateEngagement(engagement.id, { notes: e.target.value })}
          placeholder="Deliverables, milestones, links…"
          value={engagement.notes}
        />
      </Field>
      <PanelMeta rows={[{ label: 'Added', value: engagement.createdAt.slice(0, 10) || '—' }]} />
    </div>
  )
}

export function EngagementsTab({ query }: { query: string }) {
  const book = useStore($agencyBook)
  const [selectedId, setSelectedId] = useState<null | string>(null)

  const q = normalize(query)

  const engagements = book.engagements.filter(engagement => {
    if (!q) {
      return true
    }

    const client = clientById(book, engagement.clientId)

    return normalize(`${engagement.title} ${client?.name ?? ''} ${client?.company ?? ''}`).includes(q)
  })

  const selected = book.engagements.find(engagement => engagement.id === selectedId) ?? null

  return (
    <MasterDetail>
      <ListColumn>
        {engagements.map(engagement => (
          <PanelListRow
            active={engagement.id === selectedId}
            dotClassName={ENGAGEMENT_DOT[engagement.status]}
            key={engagement.id}
            menu={
              <PanelRowMenu
                items={[
                  {
                    icon: 'trash',
                    label: 'Delete engagement',
                    onSelect: () => {
                      removeEngagement(engagement.id)

                      if (selectedId === engagement.id) {
                        setSelectedId(null)
                      }
                    },
                    tone: 'danger'
                  }
                ]}
              />
            }
            meta={engagement.amount > 0 ? formatUsd(engagement.amount) : undefined}
            onSelect={() => setSelectedId(engagement.id)}
            title={engagement.title || 'Untitled engagement'}
          />
        ))}
        <PanelAddButton label="Add engagement" onClick={() => setSelectedId(addEngagement().id)} />
      </ListColumn>
      <DetailColumn footer={selected ? 'Changes save automatically.' : undefined}>
        {selected ? (
          <EngagementDetail engagement={selected} />
        ) : (
          <PanelEmpty
            description={
              book.engagements.length === 0
                ? 'Projects and retainers you run for clients. Add your first engagement to get started.'
                : 'Select an engagement on the left, or add a new one.'
            }
            icon="briefcase"
            title="Engagements"
          />
        )}
      </DetailColumn>
    </MasterDetail>
  )
}
