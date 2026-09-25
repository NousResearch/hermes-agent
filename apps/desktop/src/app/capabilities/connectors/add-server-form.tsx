import { type ReactNode, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { parseMcpImport } from '@/lib/mcp-import'

import {
  type AddServerAuth,
  type AddServerDraft,
  type AddServerTransport,
  draftFromEntry,
  type DraftPair,
  type DraftValue,
  emptyPair,
  emptyValue
} from './add-server-draft'

type AddCopy = Translations['connectorsPage']['add']

type SetDraft = (patch: Partial<AddServerDraft>) => void

export interface AddServerFormProps {
  draft: AddServerDraft
  nameTaken: boolean
  onChange: (next: AddServerDraft) => void
}

export function AddServerForm({ draft, nameTaken, onChange }: AddServerFormProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.add
  const set: SetDraft = patch => onChange({ ...draft, ...patch })

  return (
    <div className="grid gap-4" data-slot="add-server-form">
      <PasteBox copy={copy} onFill={onChange} previous={draft} />

      <Field label={copy.name}>
        <Input onChange={event => set({ name: event.currentTarget.value })} size="sm" value={draft.name} />
        {nameTaken ? <p className="text-[0.65rem] text-(--ui-red)">{copy.nameTaken}</p> : null}
      </Field>

      <Fieldset label={copy.type}>
        <SegmentedControl
          onChange={(next: AddServerTransport) => set({ transport: next })}
          options={[
            { id: 'stdio', label: copy.typeStdio },
            { id: 'http', label: copy.typeHttp }
          ]}
          value={draft.transport}
        />
      </Fieldset>

      {draft.transport === 'stdio' ? (
        <StdioFields copy={copy} draft={draft} set={set} />
      ) : (
        <HttpFields copy={copy} draft={draft} set={set} />
      )}
    </div>
  )
}

function StdioFields({ copy, draft, set }: { copy: AddCopy; draft: AddServerDraft; set: SetDraft }) {
  return (
    <>
      <Field label={copy.command}>
        <Input onChange={event => set({ command: event.currentTarget.value })} size="sm" value={draft.command} />
      </Field>

      <ValueRows
        addLabel={copy.addArg}
        label={copy.args}
        onChange={args => set({ args })}
        removeLabel={copy.removeRow}
        rows={draft.args}
      />

      <PairRows
        addLabel={copy.addEnvVar}
        copy={copy}
        label={copy.envVars}
        onChange={env => set({ env })}
        rows={draft.env}
      />

      <ValueRows
        addLabel={copy.addPassthrough}
        label={copy.passthrough}
        onChange={passthrough => set({ passthrough })}
        placeholder={copy.keyPlaceholder}
        removeLabel={copy.removeRow}
        rows={draft.passthrough}
      />

      <Field label={copy.cwd}>
        <Input onChange={event => set({ cwd: event.currentTarget.value })} size="sm" value={draft.cwd} />
      </Field>
    </>
  )
}

const AUTH_OPTIONS: readonly AddServerAuth[] = ['none', 'oauth', 'bearer']

function HttpFields({ copy, draft, set }: { copy: AddCopy; draft: AddServerDraft; set: SetDraft }) {
  const authLabel = { bearer: copy.authBearer, none: copy.authNone, oauth: copy.authOauth }

  return (
    <>
      <Field label={copy.url}>
        <Input onChange={event => set({ url: event.currentTarget.value })} size="sm" value={draft.url} />
      </Field>

      <PairRows
        addLabel={copy.addHeader}
        copy={copy}
        label={copy.headers}
        onChange={headers => set({ headers })}
        rows={draft.headers}
      />

      <Fieldset label={copy.auth}>
        <SegmentedControl
          onChange={(next: AddServerAuth) => set({ auth: next })}
          options={AUTH_OPTIONS.map(id => ({ id, label: authLabel[id] }))}
          value={draft.auth}
        />
      </Fieldset>

      {draft.auth === 'bearer' ? (
        <Field label={copy.authBearer}>
          <Input
            onChange={event => set({ bearer: event.currentTarget.value })}
            size="sm"
            type="password"
            value={draft.bearer}
          />
        </Field>
      ) : null}
    </>
  )
}

function PasteBox({
  copy,
  onFill,
  previous
}: {
  copy: AddCopy
  onFill: (next: AddServerDraft) => void
  previous: AddServerDraft
}) {
  const [text, setText] = useState('')
  const [failed, setFailed] = useState(false)

  const read = (value: string) => {
    setText(value)

    if (value.trim() === '') {
      setFailed(false)

      return
    }

    const entries = parseMcpImport(value)

    setFailed(entries === null)

    if (entries !== null) {
      onFill(draftFromEntry(entries[0].name, entries[0].config, previous))
    }
  }

  return (
    <Fieldset label={copy.pasteLabel}>
      <Textarea
        aria-label={copy.pasteLabel}
        className="max-h-32 min-h-16 font-mono text-[0.68rem]"
        onChange={event => read(event.currentTarget.value)}
        placeholder={copy.pastePlaceholder}
        value={text}
      />
      {failed ? <p className="text-[0.65rem] text-(--ui-text-tertiary)">{copy.pasteNoMatch}</p> : null}
    </Fieldset>
  )
}

function FieldLabel({ children }: { children: string }) {
  return <span className="text-xs font-medium text-(--ui-text-primary)">{children}</span>
}

function Field({ children, label }: { children: ReactNode; label: string }) {
  return (
    <label className="grid gap-1.5">
      <FieldLabel>{label}</FieldLabel>
      {children}
    </label>
  )
}

function Fieldset({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div aria-label={label} className="grid gap-1.5" role="group">
      <FieldLabel>{label}</FieldLabel>
      {children}
    </div>
  )
}

function TrashButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Button
      aria-label={label}
      className="shrink-0 text-(--ui-text-tertiary)"
      onClick={onClick}
      size="icon-xs"
      variant="ghost"
    >
      <Codicon name="trash" size="0.8125rem" />
    </Button>
  )
}

function RowsFrame({
  addLabel,
  children,
  label,
  onAdd
}: {
  addLabel: string
  children: ReactNode
  label: string
  onAdd: () => void
}) {
  return (
    <Fieldset label={label}>
      {children}
      <Button className="justify-self-start" onClick={onAdd} size="xs" variant="text">
        {addLabel}
      </Button>
    </Fieldset>
  )
}

function ValueRows({
  addLabel,
  label,
  onChange,
  placeholder,
  removeLabel,
  rows
}: {
  addLabel: string
  label: string
  onChange: (rows: DraftValue[]) => void
  placeholder?: string
  removeLabel: string
  rows: DraftValue[]
}) {
  const replace = (row: DraftValue, value: string) =>
    onChange(rows.map(candidate => (candidate.id === row.id ? { ...row, value } : candidate)))

  return (
    <RowsFrame addLabel={addLabel} label={label} onAdd={() => onChange([...rows, emptyValue()])}>
      {rows.map(row => (
        <div className="flex items-center gap-1.5" key={row.id}>
          <Input
            aria-label={label}
            className="min-w-0 flex-1"
            onChange={event => replace(row, event.currentTarget.value)}
            placeholder={placeholder}
            size="sm"
            value={row.value}
          />
          <TrashButton label={removeLabel} onClick={() => onChange(rows.filter(other => other.id !== row.id))} />
        </div>
      ))}
    </RowsFrame>
  )
}

function PairRows({
  addLabel,
  copy,
  label,
  onChange,
  rows
}: {
  addLabel: string
  copy: AddCopy
  label: string
  onChange: (rows: DraftPair[]) => void
  rows: DraftPair[]
}) {
  const patch = (row: DraftPair, next: Partial<DraftPair>) =>
    onChange(rows.map(candidate => (candidate.id === row.id ? { ...row, ...next } : candidate)))

  return (
    <RowsFrame addLabel={addLabel} label={label} onAdd={() => onChange([...rows, emptyPair()])}>
      {rows.map(row => (
        <div className="flex items-center gap-1.5" key={row.id}>
          <Input
            aria-label={copy.keyPlaceholder}
            className="min-w-0 flex-1"
            onChange={event => patch(row, { key: event.currentTarget.value })}
            placeholder={copy.keyPlaceholder}
            size="sm"
            value={row.key}
          />
          <Input
            aria-label={copy.valuePlaceholder}
            className="min-w-0 flex-1"
            onChange={event => patch(row, { value: event.currentTarget.value })}
            placeholder={copy.valuePlaceholder}
            size="sm"
            value={row.value}
          />
          <TrashButton label={copy.removeRow} onClick={() => onChange(rows.filter(other => other.id !== row.id))} />
        </div>
      ))}
    </RowsFrame>
  )
}
