import type * as React from 'react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tip } from '@/components/ui/tooltip'
import type { MessagingEnvVarInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { Plus, X } from '@/lib/icons'

import { CREDENTIAL_CONTROL_CLASS } from '../settings/credential-key-ui'

interface AllowlistRow {
  id: number
  value: string
}

function toRows(raw: null | string | undefined): AllowlistRow[] {
  const entries = splitAllowlist(raw)

  return (entries.length ? entries : ['']).map(value => ({ id: rowSeq++, value }))
}

/** Allowlists are stored as one comma-separated env value; a pasted list may also use newlines. */
export function splitAllowlist(raw: null | string | undefined): string[] {
  return (raw || '')
    .split(/[,\n]/)
    .map(entry => entry.trim())
    .filter(Boolean)
}

// Row identity for React keys and focus; module-wide so render-time re-seeding needs no ref.
let rowSeq = 0

export function joinAllowlist(entries: string[]): string {
  return [...new Set(entries.map(entry => entry.trim()).filter(Boolean))].join(',')
}

export interface AllowlistFieldProps {
  field: MessagingEnvVarInfo
  fieldId: string
  /** Visible field label; each entry is announced as "<label> <n>". */
  label: string
  /** The field's unsaved value, or undefined when untouched. */
  pending: string | undefined
  onEdit: (key: string, value: string) => void
  /** Docs link, shared with the single-input fields. */
  tools: React.ReactNode
}

/** One input per allowed user/ID, with + to add and × to remove; saved as one comma-separated value. */
export function AllowlistField({ field, fieldId, label, pending, onEdit, tools }: AllowlistFieldProps) {
  const { t } = useI18n()
  const m = t.messaging

  const [rows, setRows] = useState<AllowlistRow[]>(() => toRows(pending ?? field.value))
  const [syncedValue, setSyncedValue] = useState(field.value)
  const [focusId, setFocusId] = useState<null | number>(null)

  // A save or clear changes the server value: re-seed from it (render-time sync, no effect).
  if (field.value !== syncedValue) {
    setSyncedValue(field.value)
    setRows(toRows(field.value))
  }

  function commit(next: AllowlistRow[]) {
    setRows(next)
    onEdit(field.key, joinAllowlist(next.map(row => row.value)))
  }

  function change(id: number, value: string) {
    // A pasted "123, 456" becomes one box per entry.
    const parts = /[,\n]/.test(value) ? splitAllowlist(value) : [value]
    const index = rows.findIndex(row => row.id === id)
    const replacement = parts.map((part, i) => (i === 0 ? { id, value: part } : { id: rowSeq++, value: part }))
    commit([
      ...rows.slice(0, index),
      ...(replacement.length ? replacement : [{ id, value: '' }]),
      ...rows.slice(index + 1)
    ])
  }

  function add() {
    const id = rowSeq++
    setFocusId(id)
    setRows([...rows, { id, value: '' }])
  }

  function remove(id: number) {
    const next = rows.filter(row => row.id !== id)
    commit(next.length ? next : [{ id: rowSeq++, value: '' }])
  }

  const lone = rows.length === 1 && !rows[0].value

  return (
    <div className="grid w-full gap-1.5 @2xl:w-88" data-slot="allowlist-field">
      {rows.map((row, index) => (
        <div className="flex items-center gap-2" key={row.id}>
          <Input
            aria-label={`${label} ${index + 1}`}
            autoFocus={row.id === focusId}
            className={CREDENTIAL_CONTROL_CLASS}
            id={index === 0 ? fieldId : undefined}
            onChange={event => change(row.id, event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter' && row.value.trim()) {
                event.preventDefault()
                add()
              }
            }}
            placeholder={m.listEntryPlaceholder}
            type="text"
            value={row.value}
          />
          <Tip label={m.removeListEntry}>
            <Button
              aria-label={m.removeListEntry}
              className="size-8 shrink-0"
              disabled={lone}
              onClick={() => remove(row.id)}
              variant="ghost"
            >
              <X className="size-3.5" />
            </Button>
          </Tip>
        </div>
      ))}
      <div className="flex items-center gap-2">
        <Button className="h-7 gap-1 px-2 text-xs" onClick={add} size="sm" variant="ghost">
          <Plus className="size-3.5" />
          {m.addListEntry}
        </Button>
        <span className="flex-1" />
        {tools}
      </div>
    </div>
  )
}
