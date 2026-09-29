import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router'

import { createVaultNote, getVaultGraph, type VaultNoteNode } from '@/api/vault'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { ArrowUpRight, Brain, FileText, Loader2, Plus } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'

import { STARMAP_ROUTE } from '../routes'

import { RailCard } from './rail-cards'

const RECENT_SHOWN = 4
const REFRESH_MS = 2 * 60_000
const INBOX_FOLDER = 'Inbox'

const COPY = {
  en: {
    capture: 'Remember something…',
    captureLabel: 'Save a note to memory',
    empty: 'Memory is empty. Jot down the first thing Czesiek should remember.',
    open: 'Open map',
    recent: 'Recently changed',
    saved: 'Remembered',
    stats: (n: number, e: number) => `${n} notes · ${e} links`,
    title: 'Memory',
    unavailable: 'The vault is not available yet.'
  },
  pl: {
    capture: 'Zapamiętaj coś…',
    captureLabel: 'Zapisz notatkę w pamięci',
    empty: 'Pamięć jest pusta. Zapisz pierwszą rzecz, którą Czesiek ma pamiętać.',
    open: 'Otwórz mapę',
    recent: 'Ostatnio zmienione',
    saved: 'Zapamiętane',
    stats: (n: number, e: number) => `${n} notatek · ${e} połączeń`,
    title: 'Pamięć',
    unavailable: 'Vault nie jest jeszcze dostępny.'
  }
} as const

/** Inbox note title from free text: the first line, trimmed to a readable length. */
export function captureTitle(text: string, now = new Date()): string {
  const first = text.trim().split('\n')[0]?.trim() ?? ''
  const stamp = now.toISOString().slice(0, 16).replace('T', ' ')

  return first.length > 0 ? `${stamp} ${first.slice(0, 48)}` : stamp
}

export function recentNotes(nodes: readonly VaultNoteNode[], limit = RECENT_SHOWN): VaultNoteNode[] {
  return [...nodes].sort((a, b) => b.timestamp - a.timestamp).slice(0, limit)
}

/** The vault at a glance, plus one-line capture: the rail's door into Mapa wiedzy. */
export function JarvisMemoryCard({ connected }: { connected: boolean }) {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const navigate = useNavigate()
  const profile = useStore($activeGatewayProfile)
  const queryClient = useQueryClient()
  const [text, setText] = useState('')
  const [saving, setSaving] = useState(false)
  const key = ['jarvis-vault-rail', profile]

  const vault = useQuery({
    enabled: connected,
    queryFn: () => getVaultGraph(),
    queryKey: key,
    refetchInterval: REFRESH_MS,
    staleTime: REFRESH_MS
  })

  const graph = vault.data
  const recent = graph ? recentNotes(graph.nodes) : []
  const open = (query: string) => navigate(`${STARMAP_ROUTE}?view=vault${query}`)

  const capture = async () => {
    const value = text.trim()

    if (!value || saving) {
      return
    }

    setSaving(true)

    try {
      await createVaultNote(captureTitle(value), INBOX_FOLDER, `${value}\n`)
      setText('')
      notify({ kind: 'success', message: copy.saved, durationMs: 1800 })
      await queryClient.invalidateQueries({ queryKey: key })
    } catch (error) {
      notifyError(error, copy.captureLabel)
    } finally {
      setSaving(false)
    }
  }

  return (
    <RailCard
      action={
        <Button onClick={() => open('')} size="xs" type="button" variant="textStrong">
          {copy.open}
          <ArrowUpRight />
        </Button>
      }
      icon={Brain}
      testId="memory"
      title={copy.title}
    >
      <form
        className="jarvis-well mb-3 flex items-center gap-1 px-3"
        onSubmit={event => {
          event.preventDefault()
          void capture()
        }}
      >
        <input
          aria-label={copy.captureLabel}
          className="min-h-10 min-w-0 flex-1 bg-transparent text-sm text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-tertiary)"
          disabled={!connected || saving}
          onChange={event => setText(event.target.value)}
          placeholder={copy.capture}
          value={text}
        />
        <Button aria-label={copy.captureLabel} disabled={!text.trim() || saving} size="icon-sm" type="submit" variant="ghost">
          {saving ? <Loader2 className="animate-spin" /> : <Plus />}
        </Button>
      </form>

      {vault.isError ? (
        <p className="text-xs text-(--ui-text-secondary)">{copy.unavailable}</p>
      ) : graph && graph.nodes.length === 0 ? (
        <p className="text-xs text-(--ui-text-secondary)">{copy.empty}</p>
      ) : graph ? (
        <>
          <p className="mb-1 text-xs font-medium text-(--ui-text-secondary)">
            {copy.recent} · <span className="text-(--ui-text-tertiary)">{copy.stats(graph.nodes.length, graph.edges.length)}</span>
          </p>
          <ul className="grid gap-0.5">
            {recent.map(note => (
              <li key={note.id}>
                <button
                  className="flex min-h-10 w-full items-center gap-2 rounded-lg px-2 text-left text-sm text-(--ui-text-primary) outline-none hover:bg-(--chrome-action-hover) focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)"
                  onClick={() => open(`&note=${encodeURIComponent(note.id)}`)}
                  type="button"
                >
                  <FileText className="size-3.5 shrink-0 text-(--ui-text-tertiary)" />
                  <span className="min-w-0 flex-1 truncate">{note.label}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </RailCard>
  )
}
