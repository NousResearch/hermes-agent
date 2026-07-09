import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { deleteSession, listAllProfileSessions, setSessionArchived } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { Archive, ArchiveOff, FolderOpen, Loader2, Trash2 } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import { untombstoneSessions } from '@/store/projects'
import { applyConfiguredDefaultProjectDir, ensureDefaultWorkspaceCwd, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { EmptyState, ListRow, LoadingState, SectionHeading, SettingsContent } from './primitives'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

const ARCHIVED_FETCH_LIMIT = 200

function workspaceLabel(cwd: null | string | undefined): string {
  const path = cwd?.trim()

  if (!path) {
    return ''
  }

  return (
    path
      .replace(/[/\\]+$/, '')
      .split(/[/\\]/)
      .filter(Boolean)
      .pop() ?? path
  )
}

export function SessionsSettings() {
  const { t } = useI18n()
  const s = t.settings.sessions
  const [sessions, setLocalSessions] = useState<SessionInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)

    try {
      const result = await listAllProfileSessions(ARCHIVED_FETCH_LIMIT, 0, 'only')
      setLocalSessions(result.sessions)
    } catch (err) {
      notifyError(err, s.failedLoad)
    } finally {
      setLoading(false)
    }
  }, [s.failedLoad])

  useEffect(() => {
    void load()
  }, [load])

  const unarchive = useCallback(
    async (session: SessionInfo) => {
      setBusyId(session.id)

      try {
        await setSessionArchived(session.id, false, session.profile)
        setLocalSessions(prev => prev.filter(s => s.id !== session.id))
        // Surface it again in the sidebar without waiting for a full refresh, and
        // lift any optimistic eviction so the grouped tree shows it again too.
        untombstoneSessions([session.id, session._lineage_root_id])
        setSessions(prev => [{ ...session, archived: false }, ...prev.filter(s => s.id !== session.id)])
        triggerHaptic('selection')
        notify({ durationMs: 2_000, kind: 'success', message: s.restored })
      } catch (err) {
        notifyError(err, s.unarchiveFailed)
      } finally {
        setBusyId(null)
      }
    },
    [s]
  )

  const remove = useCallback(
    async (session: SessionInfo) => {
      if (!window.confirm(s.deleteConfirm(sessionTitle(session)))) {
        return
      }

      setBusyId(session.id)

      try {
        await deleteSession(session.id, session.profile)
        setLocalSessions(prev => prev.filter(s => s.id !== session.id))
        triggerHaptic('warning')
      } catch (err) {
        notifyError(err, s.deleteFailed)
      } finally {
        setBusyId(null)
      }
    },
    [s]
  )

  useDeepLinkHighlight({
    elementId: id => `archived-session-${id}`,
    param: 'session',
    ready: id => !loading && sessions.some(session => session.id === id)
  })

  if (loading) {
    return <LoadingState label={s.loading} />
  }

  return (
    <SettingsContent>
      <DefaultProjectDirSetting />

      <SectionHeading
        icon={Archive}
        meta={sessions.length ? String(sessions.length) : undefined}
        title={s.archivedTitle}
      />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {s.archivedIntro}
      </p>

      {sessions.length === 0 ? (
        <EmptyState description={s.emptyArchivedDesc} title={s.emptyArchivedTitle} />
      ) : (
        <div className="grid gap-1">
          {sessions.map(session => {
            const label = workspaceLabel(session.cwd)
            const busy = busyId === session.id

            return (
              <div className="scroll-mt-6 rounded-lg" id={`archived-session-${session.id}`} key={session.id}>
                <ListRow
                  action={
                    <div className="flex items-center gap-1.5">
                      <Button
                        disabled={busy}
                        onClick={() => void unarchive(session)}
                        size="sm"
                        type="button"
                        variant="textStrong"
                      >
                        {busy ? <Loader2 className="size-3.5 animate-spin" /> : <ArchiveOff className="size-3.5" />}
                        <span>{s.unarchive}</span>
                      </Button>
                      <Tip label={s.deletePermanently}>
                        <Button
                          aria-label={s.deletePermanently}
                          className="text-muted-foreground hover:text-destructive"
                          disabled={busy}
                          onClick={() => void remove(session)}
                          size="icon"
                          type="button"
                          variant="ghost"
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </Tip>
                    </div>
                  }
                  description={session.preview || undefined}
                  hint={label ? `${label} · ${s.messages(session.message_count)}` : s.messages(session.message_count)}
                  title={sessionTitle(session)}
                />
              </div>
            )
          })}
        </div>
      )}
    </SettingsContent>
  )
}

// Lets the user pin the default cwd for new sessions. Without this, packaged
// builds on Windows used to spawn sessions in the install dir (`win-unpacked`
// / Program Files), which buried any files IX Agency wrote there.
function DefaultProjectDirSetting() {
  const { t } = useI18n()
  const s = t.settings.sessions
  const [dir, setDir] = useState<null | string>(null)
  const [fallback, setFallback] = useState<string>('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    // The bridge is only present when running inside Electron. In a Vitest
    // / Storybook / non-Electron context `window.hermesDesktop` is
    // undefined, so guard the WHOLE call chain rather than chaining
    // `?.settings.getDefaultProjectDir().then(...)` (the latter would
    // short-circuit to `undefined.then(...)` and throw at runtime).
    const settings = window.hermesDesktop?.settings

    if (!settings) {
      return
    }

    let alive = true

    void settings.getDefaultProjectDir().then(result => {
      if (!alive) {
        return
      }

      setDir(result.dir)
      setFallback(result.defaultLabel)
      applyConfiguredDefaultProjectDir(result.dir)
    })

    return () => {
      alive = false
    }
  }, [])

  const choose = useCallback(async () => {
    const settings = window.hermesDesktop?.settings

    if (!settings) {
      return
    }

    setBusy(true)

    try {
      const picked = await settings.pickDefaultProjectDir()

      if (picked.canceled || !picked.dir) {
        return
      }

      const result = await settings.setDefaultProjectDir(picked.dir)
      setDir(result.dir)
      applyConfiguredDefaultProjectDir(result.dir)
      notify({ durationMs: 4_000, kind: 'success', message: s.defaultDirUpdated })
    } catch (err) {
      notifyError(err, s.updateDirFailed)
    } finally {
      setBusy(false)
    }
  }, [s])

  const clear = useCallback(async () => {
    const settings = window.hermesDesktop?.settings

    if (!settings) {
      return
    }

    setBusy(true)

    try {
      await settings.setDefaultProjectDir(null)
      setDir(null)
      applyConfiguredDefaultProjectDir(null)
      await ensureDefaultWorkspaceCwd()
    } catch (err) {
      notifyError(err, s.clearDirFailed)
    } finally {
      setBusy(false)
    }
  }, [s])

  return (
    <div className="mb-6">
      <SectionHeading icon={FolderOpen} title={s.defaultDirTitle} />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {s.defaultDirDesc}
      </p>
      <ListRow
        action={
          <div className="flex items-center gap-3">
            <Button disabled={busy} onClick={() => void choose()} size="sm" type="button" variant="textStrong">
              <FolderOpen className="size-3.5" />
              <span>{dir ? s.change : s.choose}</span>
            </Button>
            {dir && (
              <Button disabled={busy} onClick={() => void clear()} size="sm" type="button" variant="text">
                {s.clear}
              </Button>
            )}
          </div>
        }
        description={dir || s.defaultsTo(fallback || '~')}
        title={dir ? dir : s.notSet}
      />
    </div>
  )
}
