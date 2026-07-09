import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { normalize } from '@/lib/text'

import { DetailColumn, ListColumn, MasterDetail } from '../master-detail'
import { PanelAddButton, PanelEmpty, PanelListRow, PanelMeta, PanelPill, PanelRowMenu } from '../overlays/panel'

import { CLIENT_DOT, Field, FieldRow } from './bits'
import { $agencyBook, addClient, removeClient, updateClient } from './store'
import { type AgencyClient, CLIENT_STATUSES, type ClientStatus } from './types'

const STATUS_PILL_TONE = { active: 'good', churned: 'muted', lead: 'muted', paused: 'warn' } as const

function ClientDetail({ client, engagementCount }: { client: AgencyClient; engagementCount: number }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
          {client.name || 'New client'}
        </h3>
        <PanelPill tone={STATUS_PILL_TONE[client.status]}>{client.status}</PanelPill>
      </div>
      <FieldRow>
        <Field label="Name">
          <Input onChange={e => updateClient(client.id, { name: e.target.value })} value={client.name} />
        </Field>
        <Field label="Company">
          <Input onChange={e => updateClient(client.id, { company: e.target.value })} value={client.company} />
        </Field>
      </FieldRow>
      <FieldRow>
        <Field label="Email">
          <Input
            onChange={e => updateClient(client.id, { email: e.target.value })}
            type="email"
            value={client.email}
          />
        </Field>
        <Field label="Status">
          <Select
            onValueChange={status => updateClient(client.id, { status: status as ClientStatus })}
            value={client.status}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CLIENT_STATUSES.map(status => (
                <SelectItem key={status} value={status}>
                  {status}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      </FieldRow>
      <Field label="Notes">
        <Textarea
          onChange={e => updateClient(client.id, { notes: e.target.value })}
          placeholder="Context, scope, contacts…"
          value={client.notes}
        />
      </Field>
      <PanelMeta
        rows={[
          { label: 'Engagements', value: String(engagementCount) },
          { label: 'Added', value: client.createdAt.slice(0, 10) || '—' }
        ]}
      />
    </div>
  )
}

export function ClientsTab({ query }: { query: string }) {
  const book = useStore($agencyBook)
  const [selectedId, setSelectedId] = useState<null | string>(null)

  const q = normalize(query)

  const clients = book.clients.filter(
    client => !q || normalize(`${client.name} ${client.company} ${client.email}`).includes(q)
  )

  const selected = book.clients.find(client => client.id === selectedId) ?? null

  const engagementCount = (clientId: string) =>
    book.engagements.filter(engagement => engagement.clientId === clientId).length

  return (
    <MasterDetail>
      <ListColumn>
        {clients.map(client => (
          <PanelListRow
            active={client.id === selectedId}
            dotClassName={CLIENT_DOT[client.status]}
            key={client.id}
            menu={
              <PanelRowMenu
                items={[
                  {
                    icon: 'trash',
                    label: 'Delete client',
                    onSelect: () => {
                      removeClient(client.id)

                      if (selectedId === client.id) {
                        setSelectedId(null)
                      }
                    },
                    tone: 'danger'
                  }
                ]}
              />
            }
            meta={client.company || undefined}
            onSelect={() => setSelectedId(client.id)}
            title={client.name || 'Untitled client'}
          />
        ))}
        <PanelAddButton label="Add client" onClick={() => setSelectedId(addClient().id)} />
      </ListColumn>
      <DetailColumn footer={selected ? 'Changes save automatically.' : undefined}>
        {selected ? (
          <ClientDetail client={selected} engagementCount={engagementCount(selected.id)} />
        ) : (
          <PanelEmpty
            description={
              book.clients.length === 0
                ? 'Track the accounts your agency serves. Add your first client to get started.'
                : 'Select a client on the left, or add a new one.'
            }
            icon="organization"
            title="Clients"
          />
        )}
      </DetailColumn>
    </MasterDetail>
  )
}
