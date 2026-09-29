import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocation } from 'react-router'

import {
  createVaultNote,
  deleteVaultNote,
  getVaultGraph,
  getVaultNote,
  saveVaultNote,
  type VaultGraph
} from '@/api/vault'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { ExternalLink, Plus, RefreshCw, Save, Trash2, X } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

import { folderHues, VaultGraphCanvas } from './vault-graph'

const COPY = {
  en: {
    close: 'Close note',
    delete: 'Delete note',
    deleteBody: 'The note moves to .trash in the vault, so you can restore it from Obsidian.',
    deleteTitle: 'Delete this note?',
    emptyBody: 'Your memory lives as plain markdown notes. Create the first one and link notes with [[double brackets]].',
    emptyTitle: 'The vault is empty',
    links: 'Linked notes',
    newNote: 'New note',
    newPrompt: 'Note title',
    noMatches: 'No notes match',
    obsidian: 'Open in Obsidian',
    refresh: 'Refresh',
    save: 'Save',
    saved: 'Saved',
    search: 'Search notes',
    stats: (n: number, e: number) => `${n} notes · ${e} links`,
    unsaved: 'Unsaved changes'
  },
  pl: {
    close: 'Zamknij notatkę',
    delete: 'Usuń notatkę',
    deleteBody: 'Notatka trafi do .trash w vaulcie, więc odzyskasz ją z Obsidiana.',
    deleteTitle: 'Usunąć tę notatkę?',
    emptyBody: 'Twoja pamięć to zwykłe notatki markdown. Utwórz pierwszą i łącz notatki za pomocą [[podwójnych nawiasów]].',
    emptyTitle: 'Vault jest pusty',
    links: 'Powiązane notatki',
    newNote: 'Nowa notatka',
    newPrompt: 'Tytuł notatki',
    noMatches: 'Brak pasujących notatek',
    obsidian: 'Otwórz w Obsidianie',
    refresh: 'Odśwież',
    save: 'Zapisz',
    saved: 'Zapisano',
    search: 'Szukaj notatek',
    stats: (n: number, e: number) => `${n} notatek · ${e} połączeń`,
    unsaved: 'Niezapisane zmiany'
  }
} as const

export function VaultView() {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const [graph, setGraph] = useState<null | VaultGraph>(null)
  const [error, setError] = useState<null | string>(null)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<null | string>(null)
  const [draft, setDraft] = useState<null | { base: string; id: string; text: string }>(null)
  const [naming, setNaming] = useState(false)
  const [title, setTitle] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const params = new URLSearchParams(useLocation().search)
  const linkedNote = params.get('note')
  const wantsNew = params.get('new') === '1'

  // Deep links from the rail: `?note=<id>` opens that note, `?new=1` starts one.
  useEffect(() => {
    if (linkedNote) {
      setSelected(linkedNote)
    }
  }, [linkedNote])

  useEffect(() => {
    if (wantsNew) {
      setNaming(true)
    }
  }, [wantsNew])

  const load = useCallback(async () => {
    try {
      setGraph(await getVaultGraph())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  // Opening a note reads it fresh from disk; a slow read for a note the user
  // already left must not replace the one on screen.
  useEffect(() => {
    if (!selected) {
      setDraft(null)

      return undefined
    }

    let stale = false

    getVaultNote(selected)
      .then(note => !stale && setDraft({ base: note.content, id: selected, text: note.content }))
      .catch(err => !stale && notifyError(err, copy.save))

    return () => {
      stale = true
    }
  }, [copy.save, selected])

  const matches = useMemo(() => {
    const q = query.trim().toLocaleLowerCase()

    if (!q || !graph) {
      return null
    }

    return new Set(
      graph.nodes
        .filter(n => `${n.label} ${n.excerpt} ${n.tags.join(' ')} ${n.id}`.toLocaleLowerCase().includes(q))
        .map(n => n.id)
    )
  }, [graph, query])

  const hues = useMemo(() => folderHues(graph?.nodes ?? []), [graph])
  const byId = useMemo(() => new Map((graph?.nodes ?? []).map(n => [n.id, n])), [graph])

  const related = useMemo(() => {
    if (!graph || !selected) {
      return []
    }

    return graph.edges
      .flatMap(e => (e.source === selected ? [e.target] : e.target === selected ? [e.source] : []))
      .flatMap(id => byId.get(id) ?? [])
  }, [byId, graph, selected])

  const dirty = draft !== null && draft.text !== draft.base

  const save = useCallback(async () => {
    if (!draft || draft.text === draft.base) {
      return
    }

    try {
      await saveVaultNote(draft.id, draft.text)
      setDraft(d => (d && d.id === draft.id ? { ...d, base: draft.text } : d))
      notify({ kind: 'success', message: copy.saved, durationMs: 1800 })
      void load()
    } catch (err) {
      notifyError(err, copy.save)
    }
  }, [copy.save, copy.saved, draft, load])

  const create = async () => {
    const value = title.trim()

    if (!value) {
      return
    }

    try {
      const made = await createVaultNote(value)
      setNaming(false)
      setTitle('')
      await load()
      setSelected(made.id)
    } catch (err) {
      notifyError(err, copy.newNote)
    }
  }

  const remove = async () => {
    if (!selected) {
      return
    }

    await deleteVaultNote(selected)
    setSelected(null)
    void load()
  }

  const openInObsidian = () => {
    if (graph) {
      const abs = selected ? `${graph.vault.path}/${selected}` : graph.vault.path
      void window.hermesDesktop.openExternal(`obsidian://open?path=${encodeURIComponent(abs)}`)
    }
  }

  if (error) {
    return <p className="m-auto max-w-sm text-center text-sm text-(--ui-text-secondary)">{error}</p>
  }

  if (!graph) {
    return <PageLoader className="min-h-0 flex-1" />
  }

  const empty = graph.nodes.length === 0

  return (
    <div className="relative flex min-h-0 flex-1 gap-3 overflow-hidden" data-testid="vault-view">
      <div className="relative min-w-0 flex-1">
        {empty ? (
          <div className="absolute inset-0 grid place-items-center px-6 text-center">
            <div className="flex max-w-sm flex-col items-center gap-3">
              <h2 className="text-lg font-semibold text-(--ui-text-primary)">{copy.emptyTitle}</h2>
              <p className="text-sm text-(--ui-text-secondary)">{copy.emptyBody}</p>
              <Button onClick={() => setNaming(true)} type="button">
                <Plus />
                {copy.newNote}
              </Button>
            </div>
          </div>
        ) : (
          <VaultGraphCanvas graph={graph} matches={matches} onSelect={setSelected} selected={selected} />
        )}

        <div className="pointer-events-none absolute inset-x-3 top-3 flex flex-wrap items-center gap-2">
          <div className="pointer-events-auto flex items-center gap-1 rounded-xl border border-(--stroke-nous) bg-(--ui-widget-surface-background) px-2 py-1 shadow-nous backdrop-blur-xl">
            <SearchField
              aria-label={copy.search}
              containerClassName="w-44"
              onChange={setQuery}
              placeholder={copy.search}
              value={query}
            />
            <Button aria-label={copy.newNote} onClick={() => setNaming(true)} size="icon-sm" type="button" variant="ghost">
              <Plus />
            </Button>
            <Button aria-label={copy.refresh} onClick={() => void load()} size="icon-sm" type="button" variant="ghost">
              <RefreshCw />
            </Button>
            <Button aria-label={copy.obsidian} onClick={openInObsidian} size="icon-sm" type="button" variant="ghost">
              <ExternalLink />
            </Button>
          </div>
          <span className="pointer-events-auto text-xs text-(--ui-text-tertiary)">
            {matches && matches.size === 0 ? copy.noMatches : copy.stats(graph.nodes.length, graph.edges.length)}
          </span>
        </div>

        {naming ? (
          <form
            className="absolute inset-x-3 top-16 flex max-w-sm items-center gap-2 rounded-xl border border-(--stroke-nous) bg-(--ui-widget-surface-background) p-2 shadow-nous backdrop-blur-xl"
            onSubmit={event => {
              event.preventDefault()
              void create()
            }}
          >
            <input
              aria-label={copy.newPrompt}
              autoFocus
              className="min-w-0 flex-1 bg-transparent px-2 text-sm text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-tertiary)"
              onChange={event => setTitle(event.target.value)}
              onKeyDown={event => event.key === 'Escape' && setNaming(false)}
              placeholder={copy.newPrompt}
              value={title}
            />
            <Button disabled={!title.trim()} size="sm" type="submit">
              {copy.newNote}
            </Button>
          </form>
        ) : null}

        {!empty && hues.size > 1 ? (
          <ul className="pointer-events-none absolute bottom-3 left-3 flex max-w-[60%] flex-wrap gap-x-3 gap-y-1 text-xs text-(--ui-text-tertiary)">
            {[...hues].map(([folder, hue]) => (
              <li className="flex items-center gap-1.5" key={folder}>
                <span className="size-2 rounded-full" style={{ background: `hsl(${hue}, 72%, 58%)` }} />
                {folder || 'vault'}
              </li>
            ))}
          </ul>
        ) : null}
      </div>

      {selected && draft?.id === selected ? (
        <aside className="flex w-[min(26rem,45%)] shrink-0 flex-col gap-3 rounded-xl bg-(--ui-widget-surface-background) p-4">
          <header className="flex items-center gap-2">
            <h2 className="min-w-0 flex-1 truncate text-sm font-semibold text-(--ui-text-primary)">
              {byId.get(selected)?.label ?? selected}
            </h2>
            {dirty ? <span className="text-xs text-(--ui-text-tertiary)">{copy.unsaved}</span> : null}
            <Button aria-label={copy.close} onClick={() => setSelected(null)} size="icon-sm" type="button" variant="ghost">
              <X />
            </Button>
          </header>
          <p className="truncate text-xs text-(--ui-text-tertiary)">{selected}</p>
          <textarea
            aria-label={byId.get(selected)?.label ?? selected}
            className="min-h-0 flex-1 resize-none rounded-lg bg-(--ui-bg-quaternary) p-3 font-mono text-xs leading-relaxed text-(--ui-text-primary) outline-none"
            onChange={event => setDraft({ ...draft, text: event.target.value })}
            onKeyDown={event => {
              if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
                event.preventDefault()
                void save()
              }
            }}
            spellCheck={false}
            value={draft.text}
          />
          {related.length > 0 ? (
            <div className="flex flex-col gap-1">
              <p className="text-xs font-medium text-(--ui-text-secondary)">{copy.links}</p>
              <div className="flex max-h-20 flex-wrap gap-1 overflow-auto">
                {related.map(note => (
                  <Button key={note.id} onClick={() => setSelected(note.id)} size="xs" type="button" variant="secondary">
                    {note.label}
                  </Button>
                ))}
              </div>
            </div>
          ) : null}
          <footer className="flex items-center justify-between gap-2">
            <Button onClick={() => setConfirmDelete(true)} size="sm" type="button" variant="text">
              <Trash2 />
              {copy.delete}
            </Button>
            <Button disabled={!dirty} onClick={() => void save()} size="sm" type="button">
              <Save />
              {copy.save}
            </Button>
          </footer>
        </aside>
      ) : null}

      <ConfirmDialog
        description={copy.deleteBody}
        destructive
        onClose={() => setConfirmDelete(false)}
        onConfirm={remove}
        open={confirmDelete}
        title={copy.deleteTitle}
      />
    </div>
  )
}
