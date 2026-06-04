import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { deleteSession, listSessions, setSessionArchived } from '@/hermes'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { Archive, ArchiveOff, FolderOpen, Loader2, Trash2 } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import { setSessions } from '@/store/session'
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
  const [sessions, setLocalSessions] = useState<SessionInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)

    try {
      const result = await listSessions(ARCHIVED_FETCH_LIMIT, 0, 'only')
      setLocalSessions(result.sessions)
    } catch (err) {
      notifyError(err, 'Could not load archived sessions')
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
      await setSessionArchived(session.id, false)
      setLocalSessions(prev => prev.filter(s => s.id !== session.id))
      // Surface it again in the sidebar without waiting for a full refresh.
      setSessions(prev => [{ ...session, archived: false }, ...prev.filter(s => s.id !== session.id)])
      triggerHaptic('selection')
      notify({ durationMs: 2_000, kind: 'success', message: 'Restored' })
    } catch (err) {
      notifyError(err, 'Unarchive failed')
    } finally {
      setBusyId(null)
    }
  }, [])

  const remove = useCallback(async (session: SessionInfo) => {
    if (!window.confirm(`Permanently delete "${sessionTitle(session)}"? This cannot be undone.`)) {
      return
    }

    setBusyId(session.id)

    try {
      await deleteSession(session.id)
      setLocalSessions(prev => prev.filter(s => s.id !== session.id))
      triggerHaptic('warning')
    } catch (err) {
      notifyError(err, 'Delete failed')
    } finally {
      setBusyId(null)
    }
  }, [])

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
    return <LoadingState label="Loading archived sessions…" />
  }

  return (
    <SettingsContent>
      <DefaultProjectDirSetting />

      <SectionHeading
        icon={Archive}
        meta={sessions.length ? String(sessions.length) : undefined}
        title="Archived sessions"
      />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        Archived chats are hidden from the sidebar but keep all their messages. Ctrl/⌘-click a chat in the sidebar to
        archive it.
      </p>

      {filtered.length === 0 ? (
        <EmptyState
          description={query.trim() ? 'No archived chats match your search.' : 'Archive a chat to hide it here.'}
          title="Nothing archived"
        />
      ) : (
        <div className="grid gap-1">
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
                      <span>Unarchive</span>
                    </Button>
                    <Button
                      aria-label="Delete permanently"
                      className="text-muted-foreground hover:text-destructive"
                      disabled={busy}
                      onClick={() => void remove(session)}
                      size="icon"
                      title="Delete permanently"
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
      notify({ durationMs: 2_000, kind: 'success', message: 'Default project directory updated' })
    } catch (err) {
      notifyError(err, 'Could not update default directory')
    } finally {
      setBusy(false)
    }
  }, [])

  const clear = useCallback(async () => {
    const settings = window.hermesDesktop?.settings

    if (!settings) return

    setBusy(true)

    try {
      await settings.setDefaultProjectDir(null)
      setDir(null)
    } catch (err) {
      notifyError(err, 'Could not clear default directory')
    } finally {
      setBusy(false)
    }
  }, [])

  return (
    <div className="mb-6">
      <SectionHeading icon={FolderOpen} title="Default project directory" />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        New sessions start in this folder unless you pick another. Leave it unset to use your home directory.
      </p>
      <ListRow
        action={
          <div className="flex items-center gap-1.5">
            <Button disabled={busy} onClick={() => void choose()} size="sm" type="button" variant="outline">
              <FolderOpen className="size-3.5" />
              <span>{dir ? 'Change' : 'Choose'}</span>
            </Button>
            {dir && (
              <Button disabled={busy} onClick={() => void clear()} size="sm" type="button" variant="ghost">
                Clear
              </Button>
            )}
          </div>
        }
        description={dir || `Defaults to ${fallback || '~/hermes-projects'}.`}
        title={dir ? dir : 'Not set'}
      />
    </div>
  )
}
