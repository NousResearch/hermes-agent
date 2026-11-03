/**
 * New Agent's "Describe the bot you want" box: one Draft fills the name,
 * title and description fields and proposes a sectioned SOUL.md that the user
 * reviews section by section (Keep / Edit / Regenerate). Nothing is written
 * here — the dialog's Create saves the reviewed draft.
 */

import { Button, GlyphSpinner, Input, Textarea } from '@hermes/plugin-sdk'
import { type Dispatch, type SetStateAction, useEffect, useRef, useState } from 'react'

import { labeled } from './dialog-parts'
import { useBots } from './i18n'
import {
  draftAsCurrent,
  type DraftRequest,
  type DraftSection,
  mergeDraftPart,
  type ProfileDraftResult
} from './profile-draft'

/** The dialog's identity fields a fresh Draft fills (a typed name is kept). */
interface DraftIdentity {
  setDescription: (value: string) => void
  setName: (update: (current: string) => string) => void
  setTitle: (value: string) => void
}

interface DraftPanelProps {
  draft: null | ProfileDraftResult
  identity: DraftIdentity
  /** Takes an updater so a Regenerate reply lands on the latest draft, not the one it was sent from. */
  onDraft: Dispatch<SetStateAction<null | ProfileDraftResult>>
  request: DraftRequest
  /** Profile on the create target whose installed skills / MCP servers are the catalog. */
  source: string
}

export function DraftPanel({ draft, identity, onDraft, request, source }: DraftPanelProps) {
  const b = useBots()
  const d = b.draft
  const [prompt, setPrompt] = useState('')
  const [note, setNote] = useState('')
  // In-flight part keys ('*' = a whole Draft); several sections can regenerate at once.
  const [busy, setBusy] = useState<string[]>([])
  const [error, setError] = useState<null | string>(null)
  const [editing, setEditing] = useState<null | string>(null)
  const [kept, setKept] = useState<Record<string, boolean>>({})
  // Bumped by a fresh Draft, a new catalog source and the dialog closing: a reply sent before
  // any of those is stale and must not touch the dialog (Cancel has already reset it).
  const generation = useRef(0)

  useEffect(
    () => () => {
      generation.current += 1
    },
    [source]
  )

  const run = async (only: string) => {
    const part = only || '*'

    if (!only) {
      generation.current += 1
    }

    const sent = generation.current
    setBusy(prev => [...prev, part])
    setError(null)

    try {
      const res = await request<ProfileDraftResult>('profiles.draft', {
        description: prompt,
        source,
        ...(only && draft ? { current: draftAsCurrent(draft), note, only } : {})
      })

      if (sent !== generation.current) {
        return
      }

      if (!res.ok) {
        setError(d.failed(res.reason))
      } else if (only) {
        onDraft(current => (current ? mergeDraftPart(current, res, only) : current))
      } else {
        onDraft(res)
        setKept({})
        identity.setName(current => current || res.name)
        identity.setTitle(res.title)
        identity.setDescription(res.description)
      }
    } catch (err) {
      if (sent === generation.current) {
        setError(d.failed(err instanceof Error ? err.message : String(err)))
      }
    } finally {
      setBusy(prev => prev.filter(key => key !== part))
    }
  }

  const editSection = (key: string, text: string) =>
    onDraft(current =>
      current ? { ...current, sections: current.sections.map(s => (s.key === key ? { ...s, text } : s)) } : current
    )

  const picks = draft ? [...draft.skills, ...draft.toolsets, ...draft.mcp_servers] : []

  return (
    <div className="grid gap-2 rounded-md border border-(--ui-stroke-secondary) p-3" data-testid="bot-draft-panel">
      {labeled(
        d.describeLabel,
        <Textarea
          className="min-h-14"
          onChange={event => setPrompt(event.target.value)}
          placeholder={d.describePlaceholder}
          value={prompt}
        />
      )}
      <div className="flex items-center gap-2">
        <Button disabled={!prompt.trim() || busy.length > 0} onClick={() => void run('')} size="sm">
          {busy.includes('*') ? d.drafting : d.draft}
        </Button>
        {busy.length ? <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" /> : null}
        <span className="text-[0.7rem] text-(--ui-text-tertiary)">{d.nothingSaved}</span>
      </div>
      {error ? <div className="text-xs text-(--ui-accent)">{error}</div> : null}
      {draft ? (
        <div className="grid gap-2">
          {draft.sections.map(section => (
            <SectionCard
              busy={busy.includes(section.key)}
              editing={editing === section.key}
              kept={Boolean(kept[section.key])}
              key={section.key}
              onEdit={() => setEditing(editing === section.key ? null : section.key)}
              onKeep={() => setKept(prev => ({ ...prev, [section.key]: true }))}
              onRegenerate={() => void run(section.key)}
              onText={text => editSection(section.key, text)}
              section={section}
            />
          ))}
          <Input onChange={event => setNote(event.target.value)} placeholder={d.notePlaceholder} value={note} />
          <div className="text-[0.7rem] leading-5 text-(--ui-text-secondary)">
            {picks.length ? `${d.suggested} ${picks.join(', ')}` : d.noSuggestions}
          </div>
          {draft.dropped.length ? (
            <div className="text-[0.7rem] leading-5 text-(--ui-text-tertiary)" data-testid="bot-draft-dropped">
              {d.dropped(draft.dropped.map(x => x.name).join(', '))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

interface SectionCardProps {
  busy: boolean
  editing: boolean
  kept: boolean
  onEdit: () => void
  onKeep: () => void
  onRegenerate: () => void
  onText: (text: string) => void
  section: DraftSection
}

function SectionCard({ busy, editing, kept, onEdit, onKeep, onRegenerate, onText, section }: SectionCardProps) {
  const d = useBots().draft

  return (
    <div className="grid gap-1 rounded border border-(--ui-stroke-tertiary) p-2" data-section={section.key}>
      <div className="flex items-center gap-1">
        <span className="flex-1 text-xs font-medium text-(--ui-text-primary)">
          {kept ? '✓ ' : ''}
          {section.title}
        </span>
        <Button disabled={busy} onClick={onKeep} size="sm" variant="ghost">
          {d.keep}
        </Button>
        <Button disabled={busy} onClick={onEdit} size="sm" variant="ghost">
          {editing ? d.done : d.edit}
        </Button>
        <Button disabled={busy} onClick={onRegenerate} size="sm" variant="ghost">
          {busy ? d.drafting : d.regenerate}
        </Button>
      </div>
      {editing ? (
        <Textarea
          className="min-h-20 font-mono text-xs leading-5"
          onChange={event => onText(event.target.value)}
          value={section.text}
        />
      ) : (
        <div className="whitespace-pre-wrap text-xs leading-5 text-(--ui-text-secondary)">
          {section.text || d.empty}
        </div>
      )}
    </div>
  )
}
