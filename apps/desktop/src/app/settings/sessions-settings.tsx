import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { t } from '@/store/i18n'
import { deleteSession, listSessions, setSessionArchived } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { useTranslation } from '@/hooks/use-translation'
import { triggerHaptic } from '@/lib/haptics'
import { Archive, ArchiveOff, FolderOpen, Loader2, Trash2 } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import { applyConfiguredDefaultProjectDir, ensureDefaultWorkspaceCwd, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { EmptyState, ListRow, LoadingState, SectionHeading, SettingsContent } from './primitives'
import type { SearchProps } from './types'

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

export function SessionsSettings({ query }: SearchProps) {
  const { t } = useTranslation()
  const [sessions, setLocalSessions] = useState<SessionInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)

    try {
      const result = await listSessions(ARCHIVED_FETCH_LIMIT, 0, 'only')
      setLocalSessions(result.sessions)
    } catch (err) {
      notifyError(err, t('sessions.loadFailed'))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const unarchive = useCallback(async (session: SessionInfo) => {
    setBusyId(session.id)

    try {
      await setSessionArchived(session.id, false, session.profile)
      setLocalSessions(prev => prev.filter(s => s.id !== session.id))
      // Surface it again in the sidebar without waiting for a full refresh.
      setSessions(prev => [{ ...session, archived: false }, ...prev.filter(s => s.id !== session.id)])
      triggerHaptic('selection')
      notify({ durationMs: 2_000, kind: 'success', message: t('sessions.restored') })
    } catch (err) {
      notifyError(err, t('sessions.unarchiveFailed'))
    } finally {
      setBusyId(null)
    }
  }, [s])

  const remove = useCallback(async (session: SessionInfo) => {
    if (!window.confirm(s.deleteConfirm(sessionTitle(session)))) {
      return
    }

    setBusyId(session.id)

    try {
      await deleteSession(session.id, session.profile)
      setLocalSessions(prev => prev.filter(s => s.id !== session.id))
      triggerHaptic('warning')
    } catch (err) {
      notifyError(err, t('sessions.deleteFailed'))
    } finally {
      setBusyId(null)
    }
  }, [s])

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()

    if (!needle) {
      return sessions
    }

    return sessions.filter(session =>
      [sessionTitle(session), session.preview ?? '', session.cwd ?? ''].join(' ').toLowerCase().includes(needle)
    )
  }, [query, sessions])

  if (loading) {
    return <LoadingState label={t('sessions.loadingArchived')} />
  }

  return (
    <SettingsContent>
      <DefaultProjectDirSetting />

      <SectionHeading
        icon={Archive}
        meta={sessions.length ? String(sessions.length) : undefined}
        title={t('sessions.archivedTitle')}
      />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {s.archivedIntro}
      </p>

      {filtered.length === 0 ? (
        <EmptyState
          description={query.trim() ? t('sessions.noSearchMatch') : t('sessions.archiveHint')}
          title={t('sessions.nothingArchivedTitle')}
        />
      ) : (
        <div className="divide-y divide-border/30">
          {filtered.map(session => {
            const label = workspaceLabel(session.cwd)
            const busy = busyId === session.id

            return (
              <ListRow
                action={
                  <div className="flex items-center gap-1.5">
                    <Button
                      disabled={busy}
                      onClick={() => void unarchive(session)}
                      size="sm"
                      type="button"
                      variant="outline"
                    >
                      {busy ? <Loader2 className="size-3.5 animate-spin" /> : <ArchiveOff className="size-3.5" />}
                      <span>{t('sessions.unarchive')}</span>
                    </Button>
                    <Button
                      aria-label={t('sessions.deletePermanentlyTitle')}
                      className="text-muted-foreground hover:text-destructive"
                      disabled={busy}
                      onClick={() => void remove(session)}
                      size="icon"
                      title={t('sessions.deletePermanentlyTitle')}
                      type="button"
                      variant="ghost"
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </div>
                }
                description={session.preview || undefined}
                hint={label ? `${label} · ${session.message_count} messages` : `${session.message_count} messages`}
                key={session.id}
                title={sessionTitle(session)}
              />
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
      if (!alive) return
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

    if (!settings) return

    setBusy(true)

    try {
      const picked = await settings.pickDefaultProjectDir()

      if (picked.canceled || !picked.dir) {
        return
      }

      const result = await settings.setDefaultProjectDir(picked.dir)
      setDir(result.dir)
      notify({ durationMs: 2_000, kind: 'success', message: t('sessions.projectDirUpdated') })
    } catch (err) {
      notifyError(err, t('sessions.updateDirFailed'))
    } finally {
      setBusy(false)
    }
  }, [s])

  const clear = useCallback(async () => {
    const settings = window.hermesDesktop?.settings

    if (!settings) return

    setBusy(true)

    try {
      await settings.setDefaultProjectDir(null)
      setDir(null)
      applyConfiguredDefaultProjectDir(null)
      await ensureDefaultWorkspaceCwd()
    } catch (err) {
      notifyError(err, t('sessions.clearDirFailed'))
    } finally {
      setBusy(false)
    }
  }, [s])

  return (
    <div className="mb-6">
      <SectionHeading icon={FolderOpen} title={t('sessions.defaultProjectDirTitle')} />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {s.defaultDirDesc}
      </p>
      <ListRow
        action={
          <div className="flex items-center gap-3">
            <Button disabled={busy} onClick={() => void choose()} size="sm" type="button" variant="textStrong">
              <FolderOpen className="size-3.5" />
              <span>{dir ? t('sessions.change') : t('sessions.choose')}</span>
            </Button>
            {dir && (
              <Button disabled={busy} onClick={() => void clear()} size="sm" type="button" variant="ghost">
                Clear
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
