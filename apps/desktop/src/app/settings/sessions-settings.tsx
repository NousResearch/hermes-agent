import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { deleteSession, listSessions, setSessionArchived } from '@/hermes'
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
  const [sessions, setLocalSessions] = useState<SessionInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)

    try {
      const result = await listSessions(ARCHIVED_FETCH_LIMIT, 0, 'only')
      setLocalSessions(result.sessions)
    } catch (err) {
      notifyError(err, '보관된 대화를 불러오지 못했습니다')
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
      notify({ durationMs: 2_000, kind: 'success', message: '복원됨' })
    } catch (err) {
      notifyError(err, '보관 취소 실패')
    } finally {
      setBusyId(null)
    }
  }, [])

  const remove = useCallback(async (session: SessionInfo) => {
    if (!window.confirm(`"${sessionTitle(session)}"을(를) 영구적으로 삭제하시겠습니까? 이 작업은 취소할 수 없습니다.`)) {
      return
    }

    setBusyId(session.id)

    try {
      await deleteSession(session.id, session.profile)
      setLocalSessions(prev => prev.filter(s => s.id !== session.id))
      triggerHaptic('warning')
    } catch (err) {
      notifyError(err, '삭제 실패')
    } finally {
      setBusyId(null)
    }
  }, [])

  useDeepLinkHighlight({
    elementId: id => `archived-session-${id}`,
    param: 'session',
    ready: id => !loading && sessions.some(session => session.id === id)
  })

  if (loading) {
    return <LoadingState label="보관된 대화 불러오는 중…" />
  }

  return (
    <SettingsContent>
      <DefaultProjectDirSetting />

      <SectionHeading
        icon={Archive}
        meta={sessions.length ? String(sessions.length) : undefined}
        title="보관된 대화"
      />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        보관된 대화는 사이드바에서 숨겨지지만 모든 메시지는 유지됩니다. 사이드바의 대화를 Ctrl/⌘-클릭하여 보관할 수 있습니다.
      </p>

      {sessions.length === 0 ? (
        <EmptyState description="대화를 보관하여 여기에 숨깁니다." title="보관된 항목 없음" />
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
                        <span>보관 취소</span>
                      </Button>
                      <Tip label="영구 삭제">
                        <Button
                          aria-label="영구 삭제"
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
                  hint={label ? `${label} · ${session.message_count}개 메시지` : `${session.message_count}개 메시지`}
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
      notify({ durationMs: 2_000, kind: 'success', message: '기본 프로젝트 디렉토리가 업데이트되었습니다' })
    } catch (err) {
      notifyError(err, '기본 디렉토리를 업데이트하지 못했습니다')
    } finally {
      setBusy(false)
    }
  }, [])

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
      notifyError(err, '기본 디렉토리를 지우지 못했습니다')
    } finally {
      setBusy(false)
    }
  }, [])

  return (
    <div className="mb-6">
      <SectionHeading icon={FolderOpen} title="기본 프로젝트 디렉토리" />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        다른 폴더를 선택하지 않으면 새 대화는 이 폴더에서 시작됩니다. 홈 디렉토리를 사용하려면 설정하지 않은 상태로 두세요.
      </p>
      <ListRow
        action={
          <div className="flex items-center gap-3">
            <Button disabled={busy} onClick={() => void choose()} size="sm" type="button" variant="textStrong">
              <FolderOpen className="size-3.5" />
              <span>{dir ? '변경' : '선택'}</span>
            </Button>
            {dir && (
              <Button disabled={busy} onClick={() => void clear()} size="sm" type="button" variant="text">
                지우기
              </Button>
            )}
          </div>
        }
        description={dir || `기본값: ${fallback || '~/hermes-projects'}.`}
        title={dir ? dir : '설정 안 됨'}
      />
    </div>
  )
}
