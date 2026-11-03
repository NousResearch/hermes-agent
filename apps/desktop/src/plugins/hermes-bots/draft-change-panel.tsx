/**
 * Edit Profile's "Describe a change": the backend proposes a revised title,
 * description and SOUL sections; each changed part is accepted or rejected on
 * its own. Accepted parts are handed to the dialog, which writes them on Save.
 * The bot's running chat keeps its prompt — the new SOUL is read the next time
 * that chat builds its prompt, so the cached conversation is never rewritten.
 */

import { Button, GlyphSpinner, Textarea } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { labeled } from './dialog-parts'
import { useBots } from './i18n'
import { type DraftRequest, mergeSoulSections, type ProfileDraftResult } from './profile-draft'

export interface AcceptedChange {
  description?: string
  soul?: string
  title?: string
}

interface ChangeRow {
  after: string
  before: string
  key: string
  label: string
}

interface DraftChangePanelProps {
  name: string
  onChange: (accepted: AcceptedChange) => void
  request: DraftRequest
}

/** The parts the proposal changes, removals (empty new text) included. */
export function changedRows(res: ProfileDraftResult, labels: { description: string; title: string }): ChangeRow[] {
  const prev = res.previous

  if (!prev) {
    return []
  }

  const before = new Map(prev.sections.map(s => [s.key, s.text]))

  const rows: ChangeRow[] = [
    { key: 'title', label: labels.title, before: prev.title, after: res.title },
    { key: 'description', label: labels.description, before: prev.description, after: res.description },
    ...res.sections.map(s => ({ key: s.key, label: s.title, before: before.get(s.key) || '', after: s.text }))
  ]

  return rows.filter(r => r.after.trim() !== r.before.trim())
}

function acceptedChange(res: ProfileDraftResult, accepted: Record<string, boolean>): AcceptedChange {
  const sections = res.sections.filter(s => accepted[s.key])

  return {
    ...(accepted.title ? { title: res.title } : {}),
    ...(accepted.description ? { description: res.description } : {}),
    ...(sections.length && res.previous ? { soul: mergeSoulSections(res.previous.soul, sections) } : {})
  }
}

export function DraftChangePanel({ name, onChange, request }: DraftChangePanelProps) {
  const b = useBots()
  const d = b.draft
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const [result, setResult] = useState<null | ProfileDraftResult>(null)
  const [accepted, setAccepted] = useState<Record<string, boolean>>({})

  const propose = async () => {
    setBusy(true)
    setError(null)

    try {
      const res = await request<ProfileDraftResult>('profiles.draft', { name, description: text })

      if (res.ok) {
        setResult(res)
        setAccepted({})
        onChange({})
      } else {
        setError(d.failed(res.reason))
      }
    } catch (err) {
      setError(d.failed(err instanceof Error ? err.message : String(err)))
    } finally {
      setBusy(false)
    }
  }

  const decide = (key: string, value: boolean) => {
    const next = { ...accepted, [key]: value }
    setAccepted(next)

    if (result) {
      onChange(acceptedChange(result, next))
    }
  }

  const rows = result ? changedRows(result, { title: b.editor.title, description: b.editor.description }) : []

  return (
    <div className="grid gap-2 rounded-md border border-(--ui-stroke-secondary) p-3" data-testid="bot-draft-change">
      {labeled(
        d.changeLabel,
        <Textarea
          className="min-h-12"
          onChange={event => setText(event.target.value)}
          placeholder={d.changePlaceholder}
          value={text}
        />
      )}
      <div className="flex items-center gap-2">
        <Button disabled={!text.trim() || busy} onClick={() => void propose()} size="sm">
          {busy ? d.drafting : d.propose}
        </Button>
        {busy ? <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" /> : null}
      </div>
      {error ? <div className="text-xs text-(--ui-accent)">{error}</div> : null}
      {result && !rows.length ? <div className="text-xs text-(--ui-text-tertiary)">{d.noChanges}</div> : null}
      {rows.map(row => (
        <div className="grid gap-1 rounded border border-(--ui-stroke-tertiary) p-2" data-change={row.key} key={row.key}>
          <div className="flex items-center gap-1">
            <span className="flex-1 text-xs font-medium text-(--ui-text-primary)">{row.label}</span>
            <Button onClick={() => decide(row.key, true)} size="sm" variant={accepted[row.key] ? 'default' : 'ghost'}>
              {d.accept}
            </Button>
            <Button
              onClick={() => decide(row.key, false)}
              size="sm"
              variant={accepted[row.key] === false ? 'default' : 'ghost'}
            >
              {d.reject}
            </Button>
          </div>
          <div className="text-[0.7rem] text-(--ui-text-tertiary)">{d.before}</div>
          <div className="whitespace-pre-wrap text-xs leading-5 text-(--ui-text-tertiary) line-through decoration-(--ui-stroke-secondary)">
            {row.before || d.empty}
          </div>
          <div className="text-[0.7rem] text-(--ui-text-tertiary)">{d.after}</div>
          <div className="whitespace-pre-wrap text-xs leading-5 text-(--ui-text-primary)">{row.after || d.empty}</div>
        </div>
      ))}
      {rows.length ? <div className="text-[0.7rem] leading-5 text-(--ui-text-tertiary)">{d.appliesNext}</div> : null}
    </div>
  )
}
