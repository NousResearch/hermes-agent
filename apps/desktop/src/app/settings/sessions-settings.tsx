import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { deleteSession, listSessions, setSessionArchived } from '@/hermes'
import { useTranslation } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { Archive, ArchiveOff, FolderOpen, Loader2, Trash2 } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import { setSessions } from '@/store/session'
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
  const t = useTranslation()
  const [sessions, setLocalSessions] = useState<SessionInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)

    try {
      const result = await listSessions(ARCHIVED_FETCH_LIMIT, 0, 'only')
      setLocalSessions(result.sessions)
    } catch (err) {
      notifyError(err, t('settings.sessions.loadError'))
    } finally {
      setLoading(false)
    }
  }, [t])

  useEffect(() => {
    void load()
  }, [load])

  const unarchive = useCallback(
    async (session: SessionInfo) => {
      setBusyId(session.id)

      try {
        await setSessionArchived(session.id, false)
        setLocalSessions(prev => prev.filter(s => s.id !== session.id))
        // Surface it again in the sidebar without waiting for a full refresh.
        setSessions(prev => [{ ...session, archived: false }, ...prev.filter(s => s.id !== session.id)])
        triggerHaptic('selection')
        notify({ durationMs: 2_000, kind: 'success', message: t('settings.sessions.restored') })
      } catch (err) {
        notifyError(err, t('settings.sessions.unarchiveFailed'))
      } finally {
        setBusyId(null)
      }
    },
    [t]
  )

  const remove = useCallback(
    async (session: SessionInfo) => {
      if (!window.confirm(t('settings.sessions.confirmDelete', { title: sessionTitle(session) }))) {
        return
      }

      setBusyId(session.id)

      try {
        await deleteSession(session.id)
        setLocalSessions(prev => prev.filter(s => s.id !== session.id))
        triggerHaptic('warning')
      } catch (err) {
        notifyError(err, t('settings.sessions.deleteFailed'))
      } finally {
        setBusyId(null)
      }
    },
    [t]
  )

  useDeepLinkHighlight({
    elementId: id => `archived-session-${id}`,
    param: 'session',
    ready: id => !loading && sessions.some(session => session.id === id)
  })

  if (loading) {
    return <LoadingState label={t('settings.sessions.loading')} />
  }

  return (
    <SettingsContent>
      <DefaultProjectDirSetting />

      <SectionHeading
        icon={Archive}
        meta={sessions.length ? String(sessions.length) : undefined}
        title={t('settings.sessions.title')}
      />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {t('settings.sessions.description')}
      </p>

      {sessions.length === 0 ? (
        <EmptyState description={t('settings.sessions.empty.description')} title={t('settings.sessions.empty.title')} />
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
                        <span>{t('settings.sessions.unarchive')}</span>
                      </Button>
                      <Button
                        aria-label={t('settings.sessions.deletePermanently')}
                        className="text-muted-foreground hover:text-destructive"
                        disabled={busy}
                        onClick={() => void remove(session)}
                        size="icon"
                        title={t('settings.sessions.deletePermanently')}
                        type="button"
                        variant="ghost"
                      >
                        <Trash2 className="size-3.5" />
                      </Button>
                    </div>
                  }
                  description={session.preview || undefined}
                  hint={
                    label
                      ? t('settings.sessions.rowHintWithWorkspace', { count: session.message_count, workspace: label })
                      : t('settings.sessions.rowHint', { count: session.message_count })
                  }
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
// / Program Files), which buried any files Hermes wrote there.
function DefaultProjectDirSetting() {
  const t = useTranslation()
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
      notify({ durationMs: 2_000, kind: 'success', message: t('settings.sessions.defaultProject.updated') })
    } catch (err) {
      notifyError(err, t('settings.sessions.defaultProject.updateFailed'))
    } finally {
      setBusy(false)
    }
  }, [t])

  const clear = useCallback(async () => {
    const settings = window.hermesDesktop?.settings

    if (!settings) {
      return
    }

    setBusy(true)

    try {
      await settings.setDefaultProjectDir(null)
      setDir(null)
    } catch (err) {
      notifyError(err, t('settings.sessions.defaultProject.clearFailed'))
    } finally {
      setBusy(false)
    }
  }, [t])

  return (
    <div className="mb-6">
      <SectionHeading icon={FolderOpen} title={t('settings.sessions.defaultProject.title')} />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {t('settings.sessions.defaultProject.description')}
      </p>
      <ListRow
        action={
          <div className="flex items-center gap-3">
            <Button disabled={busy} onClick={() => void choose()} size="sm" type="button" variant="textStrong">
              <FolderOpen className="size-3.5" />
              <span>
                {dir ? t('settings.sessions.defaultProject.change') : t('settings.sessions.defaultProject.choose')}
              </span>
            </Button>
            {dir && (
              <Button disabled={busy} onClick={() => void clear()} size="sm" type="button" variant="text">
                {t('settings.sessions.defaultProject.clear')}
              </Button>
            )}
          </div>
        }
        description={dir || t('settings.sessions.defaultProject.defaultsTo', { path: fallback || '~/hermes-projects' })}
        title={dir ? dir : t('settings.sessions.defaultProject.notSet')}
      />
    </div>
  )
}
