import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  getLearningProviderSession,
  getSession,
  materializeLearningProviderSession,
  type ProviderSessionMessage,
  searchSessions
} from '@/hermes'
import { useI18n } from '@/i18n'
import { fmtDateTime } from '@/lib/time'
import { notifyError } from '@/store/notifications'
import type { StarmapNode } from '@/types/hermes'

import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

import { isProviderSource } from './sources'

// One resolved conversation behind a node. `direct` = provenance-recorded
// (the node carries its originating session id); false = found by content
// search, which can surface several candidate sessions.
interface SessionHit {
  direct: boolean
  id: string
  started: null | number
  title: string
}

function fmtTs(ts: null | number): string {
  if (!ts) {
    return ''
  }

  try {
    return fmtDateTime.format(new Date(ts * 1000))
  } catch {
    return ''
  }
}

// Search FTS with the node's title. Memory labels arrive truncated with a
// trailing ellipsis — strip it so the last (cut) word can't sink the query.
function searchQuery(label: string): string {
  const clean = label.replace(/…$/, '').trim()
  const words = clean.split(/\s+/).filter(Boolean)

  return words.length > 8 ? words.slice(0, 8).join(' ') : clean
}

/** Double-click drill-down for a star-map node: the conversation(s) the
 *  knowledge came from, and (provider nodes) the raw source corpus. */
export function NodeSessionsDialog({
  onClose,
  onOpenSession,
  target
}: {
  onClose: () => void
  onOpenSession: (storedSessionId: string) => void
  target: StarmapNode | null
}) {
  const { t } = useI18n()
  const [hits, setHits] = useState<SessionHit[]>([])
  const [loading, setLoading] = useState(false)
  const [corpus, setCorpus] = useState<null | ProviderSessionMessage[]>(null)
  const [corpusLoading, setCorpusLoading] = useState(false)
  const [showCorpus, setShowCorpus] = useState(false)
  const [recreating, setRecreating] = useState(false)

  // Bumped when the dialog's context dies (close / profile switch) so an
  // in-flight openCorpus/recreate from the old context can't apply late.
  // Deliberately NOT written inside useEffect (repo lint convention) — the
  // fetch effect below uses a local staleness flag instead.
  const epochRef = useRef(0)

  const handleClose = () => {
    epochRef.current += 1
    onClose()
  }

  useOnProfileSwitch(() => {
    epochRef.current += 1
    onClose()
  })

  // Provider-contributed node (Honcho etc.): the source corpus lives in the
  // provider backend and is readable via the provider-session endpoint.
  const isProviderNode = Boolean(target?.sessionId) && isProviderSource(target?.memorySource)

  useEffect(() => {
    setHits([])
    setCorpus(null)
    setShowCorpus(false)

    if (!target) {
      return
    }

    // Local staleness flag (repo convention): covers target change, close,
    // and unmount. epochRef guards only the imperative callbacks below.
    let stale = false

    setLoading(true)

    const resolve = async () => {
      const out: SessionHit[] = []

      // Provenance-recorded id first: for Hermes-born provider sessions the
      // provider session id IS the Hermes session id, so a direct lookup
      // beats any content search. 404 (imported/provider-only history) just
      // falls through to search.
      if (target.sessionId) {
        try {
          const s = await getSession(target.sessionId)

          out.push({
            direct: true,
            id: s.id,
            started: s.started_at ?? null,
            title: s.title || s.preview || s.id
          })
        } catch {
          // Not a Hermes session — provider-only (e.g. imported history).
        }
      }

      if (out.length === 0) {
        try {
          const res = await searchSessions(searchQuery(target.label))

          for (const r of res.results.slice(0, 12)) {
            out.push({
              direct: false,
              id: r.session_id,
              started: r.session_started,
              title: r.snippet ? r.snippet.replace(/<\/?b>/g, '') : r.session_id
            })
          }
        } catch {
          // Search unavailable — the empty state explains it.
        }
      }

      if (!stale) {
        setHits(out)
        setLoading(false)
      }
    }

    void resolve()

    return () => {
      stale = true
    }
  }, [target])

  const openCorpus = async () => {
    if (!target?.sessionId) {
      return
    }

    const epoch = epochRef.current
    setShowCorpus(true)

    if (corpus !== null) {
      return
    }

    setCorpusLoading(true)

    try {
      const res = await getLearningProviderSession(target.sessionId)

      if (epochRef.current === epoch) {
        setCorpus(res.messages)
      }
    } catch {
      if (epochRef.current === epoch) {
        setCorpus([])
      }
    } finally {
      if (epochRef.current === epoch) {
        setCorpusLoading(false)
      }
    }
  }

  // Materialize the provider-side conversation as a real Hermes session and
  // open it. Backend import skips existing ids, so re-running is safe — an
  // already-recreated conversation just reopens.
  const recreate = async () => {
    if (!target?.sessionId || recreating) {
      return
    }

    const epoch = epochRef.current
    setRecreating(true)

    try {
      const res = await materializeLearningProviderSession(target.sessionId)

      if (epochRef.current !== epoch) {
        return
      }

      onOpenSession(res.session_id)
    } catch (e) {
      if (epochRef.current === epoch) {
        notifyError(e instanceof Error ? e : new Error(String(e)), t.starmap.recreateFailed)
      }
    } finally {
      if (epochRef.current === epoch) {
        setRecreating(false)
      }
    }
  }

  // Only offer "recreate" when the conversation doesn't already exist as a
  // Hermes session — if a direct hit resolved, opening it IS the action.
  const canRecreate = isProviderNode && !hits.some(h => h.direct)

  return (
    <Dialog onOpenChange={value => !value && handleClose()} open={Boolean(target)}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="truncate pr-6">
            {showCorpus ? t.starmap.sourceTitle : t.starmap.sessionsTitle}
          </DialogTitle>
        </DialogHeader>

        <div className="truncate text-xs text-muted-foreground">{target?.label}</div>

        {showCorpus ? (
          <div className="max-h-[55vh] min-h-24 space-y-3 overflow-y-auto pr-1">
            {corpusLoading ? (
              <p className="text-sm text-muted-foreground">{t.starmap.sourceLoading}</p>
            ) : !corpus || corpus.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t.starmap.sourceEmpty}</p>
            ) : (
              corpus.map((m, i) => (
                <div className="rounded-md border border-(--ui-stroke-secondary) p-2" key={i}>
                  <div className="mb-1 flex items-baseline justify-between gap-2 text-[0.68rem] text-muted-foreground">
                    <span className="font-medium">{m.peer || '—'}</span>
                    {m.timestamp ? <span className="tabular-nums">{fmtTs(m.timestamp)}</span> : null}
                  </div>
                  <div className="whitespace-pre-wrap text-xs">{m.content}</div>
                </div>
              ))
            )}
          </div>
        ) : (
          <div className="max-h-[55vh] min-h-24 space-y-1 overflow-y-auto pr-1">
            {loading ? (
              <p className="text-sm text-muted-foreground">{t.starmap.sessionsLoading}</p>
            ) : hits.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                {isProviderNode ? t.starmap.sessionsProviderOnly : t.starmap.sessionsEmpty}
              </p>
            ) : (
              <>
                {!hits[0]?.direct && (
                  <p className="pb-1 text-[0.68rem] text-muted-foreground">{t.starmap.sessionsSearchNote}</p>
                )}
                {hits.map(h => (
                  <button
                    className="block w-full cursor-pointer rounded-md border border-transparent px-2 py-1.5 text-left hover:border-(--ui-stroke-secondary) hover:bg-(--ui-control-active-background)"
                    key={h.id}
                    onClick={() => onOpenSession(h.id)}
                    type="button"
                  >
                    <div className="truncate text-xs">{h.title}</div>
                    <div className="flex items-baseline justify-between text-[0.65rem] text-muted-foreground">
                      <span className="truncate">{h.id}</span>
                      {h.started ? <span className="shrink-0 pl-2 tabular-nums">{fmtTs(h.started)}</span> : null}
                    </div>
                  </button>
                ))}
              </>
            )}
          </div>
        )}

        <DialogFooter className="items-center gap-2">
          {showCorpus ? (
            <Button onClick={() => setShowCorpus(false)} type="button" variant="ghost">
              {t.starmap.backToSessions}
            </Button>
          ) : isProviderNode ? (
            <Button onClick={() => void openCorpus()} type="button" variant="outline">
              {t.starmap.viewSource}
            </Button>
          ) : null}
          {canRecreate ? (
            <Button disabled={recreating || loading} onClick={() => void recreate()} type="button" variant="outline">
              {recreating ? t.starmap.recreating : t.starmap.recreate}
            </Button>
          ) : null}
          <Button onClick={handleClose} type="button" variant="ghost">
            {t.starmap.close}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
