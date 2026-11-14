import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useId, useState } from 'react'

import { restoreListedSession } from '@/app/session/hooks/use-session-actions/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { deleteSession, setSessionArchived } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import { pathLeaf } from '@/lib/display-path'
import { triggerHaptic } from '@/lib/haptics'
import { hostHas } from '@/lib/host-capabilities'
import { Archive, ArchiveOff, FolderOpen, Loader2, Trash2 } from '@/lib/icons'
import { purgeInFlightTurnJournals } from '@/lib/inflight-turn-journal'
import { formatCombo } from '@/lib/keybinds/combo'
import { confirm } from '@/store/confirm'
import { bindingsFor } from '@/store/keybinds'
import { notify, notifyError } from '@/store/notifications'
import { $connection, applyConfiguredDefaultProjectDir, ensureDefaultWorkspaceCwd } from '@/store/session'
import { untombstoneSessions } from '@/store/session-removal'
import { forgetSessionUnread } from '@/store/session-unread'
import { usePortalShell } from '@/store/shell-variant'
import { listEveryArchivedSession } from '@/store/sidebar-archive'
import type { SessionInfo } from '@/types/hermes'

import { AUTO_ARCHIVE_PERIODS, useAutoArchiveConfig } from './auto-archive'
import { EmptyState, ListRow, SectionHeading, SettingsContent, SettingsSkeleton, ToggleRow } from './primitives'
import {
  SettingsCard,
  SettingsDisclosure,
  SettingsGroup,
  SettingsOverview,
  SettingsPageDescription
} from './settings-card'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { settingsPagesCopy } from './settings-pages-copy'
import { useDeepLinkHighlight } from './use-deep-link-highlight'
import { useSettingDeepLink } from './use-setting-deep-link'

// The toast names the chord the user actually has: a browser tab and a remap
// both differ from ⌘/Ctrl+N.
function newSessionShortcut(): null | string {
  const combo = bindingsFor('session.new')[0]

  return combo ? formatCombo(combo) : null
}

interface SessionsSettingsProps {
  subpage?: string
}

export function SessionsSettings({ subpage }: SessionsSettingsProps = {}) {
  useSettingDeepLink('sessions', page => subpage === undefined || page === subpage)
  const redesign = usePortalShell()

  if (subpage === 'default-directory') {
    return (
      <SettingsContent>
        <DefaultProjectDirSetting />
      </SettingsContent>
    )
  }

  return <ArchivedSessionsSettings includeDefaultDirectory={redesign || subpage === undefined} />
}

function ArchivedSessionsSettings({ includeDefaultDirectory }: { includeDefaultDirectory: boolean }) {
  const { t, locale } = useI18n()
  const pages = settingsPagesCopy(locale)
  const s = t.settings.sessions
  const redesign = usePortalShell()
  const [sessions, setLocalSessions] = useState<SessionInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [archiveOpen, setArchiveOpen] = useState(false)
  const archiveListId = useId()

  const load = useCallback(async () => {
    setLoading(true)

    try {
      setLocalSessions(await listEveryArchivedSession())
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
        restoreListedSession({ ...session, archived: false })
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
      const ok = await confirm({
        confirmLabel: s.deletePermanently,
        destructive: true,
        title: s.deleteConfirm(sessionTitle(session))
      })

      if (!ok) {
        return
      }

      setBusyId(session.id)

      try {
        await deleteSession(session.id, session.profile)
        // Permanent delete bypasses removeSession, so retire the persisted
        // unread state here too rather than leaving it to rot.
        forgetSessionUnread([session.id, session._lineage_root_id], session.profile)
        // Same for the journaled in-flight tail: it holds this session's
        // prompt and tool calls in localStorage, and a deleted session must
        // not leave that copy behind to age out on its own. Both ids — the
        // stored tip and the durable lineage root — the journal keys on the
        // stored id and the row may carry either.
        purgeInFlightTurnJournals([session.id, session._lineage_root_id])
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

  const linkedSession = useDeepLinkHighlight({
    elementId: id => `archived-session-${id}`,
    param: 'session',
    ready: id => !loading && (archiveOpen || !redesign) && sessions.some(session => session.id === id)
  })

  // A link to an archived chat opens the redesign's collapsed archive so the row can be shown.
  if (linkedSession && redesign && !archiveOpen) {
    setArchiveOpen(true)
  }

  const archivedList =
    sessions.length === 0 ? (
      <EmptyState description={s.emptyArchivedDesc} title={s.emptyArchivedTitle} />
    ) : (
      <div className="grid gap-1">
        {sessions.map(session => {
          const label = pathLeaf(session.cwd)
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
    )

  if (loading) {
    return <SettingsSkeleton sections={[{ rows: 1 }, { heading: true, rows: 4 }]} />
  }

  return (
    <SettingsContent>
      <SettingsPageDescription>{pages.intro.sessions}</SettingsPageDescription>
      <SettingsOverview>
        {includeDefaultDirectory && !redesign && <DefaultProjectDirSetting />}

        <SettingsGroup title={pages.archive}>
          {redesign ? (
            <>
              <SettingsCard>
                <AutoArchivePeriodRow />
                <ListRow
                  action={
                    <Button
                      aria-controls={archiveListId}
                      aria-expanded={archiveOpen}
                      onClick={() => setArchiveOpen(open => !open)}
                      variant="secondary"
                    >
                      {archiveOpen ? pages.archiveChats.hide : pages.archiveChats.view}
                    </Button>
                  }
                  description={pages.archiveChats.archivedDesc}
                  title={pages.archiveChats.archived}
                />
              </SettingsCard>
              {archiveOpen && <SettingsCard id={archiveListId}>{archivedList}</SettingsCard>}
            </>
          ) : (
            <>
              <SettingsCard classicClassName="">
                <AutoArchiveSetting />
              </SettingsCard>

              <SettingsDisclosure
                classicHeading={
                  <SectionHeading
                    icon={Archive}
                    meta={sessions.length ? String(sessions.length) : undefined}
                    title={s.archivedTitle}
                  />
                }
                count={sessions.length}
                title={s.archivedTitle}
              >
                <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  {s.archivedIntro}
                </p>

                {archivedList}
              </SettingsDisclosure>
            </>
          )}
        </SettingsGroup>
        {includeDefaultDirectory && redesign && (
          <SettingsGroup title={pages.newSessions}>
            <SettingsCard>
              <DefaultProjectDirSetting compact />
            </SettingsCard>
          </SettingsGroup>
        )}
      </SettingsOverview>
    </SettingsContent>
  )
}

function AutoArchiveSetting() {
  const { t } = useI18n()
  const s = t.settings.sessions
  const archive = useAutoArchiveConfig(s.autoArchiveFailed)

  if (!archive) {
    return null
  }

  return (
    <div className="mb-6">
      <ToggleRow
        checked={archive.enabled}
        description={s.autoArchiveDesc}
        id={settingElementId(SETTING_IDS.sessions.autoArchive)}
        label={s.autoArchiveTitle}
        onChange={on => archive.save(on, archive.days)}
      />
      {archive.enabled && (
        <ListRow
          action={
            <div className="flex items-center gap-2">
              <Input
                aria-label={s.autoArchiveDaysLabel}
                className="w-20"
                min={1}
                onBlur={() => archive.save(true, archive.days)}
                onChange={e => archive.setDays(Math.max(1, Math.round(Number(e.target.value) || 1)))}
                type="number"
                value={archive.days}
              />
              <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {s.autoArchiveDaysUnit}
              </span>
            </div>
          }
          title={s.autoArchiveDaysLabel}
        />
      )}
    </div>
  )
}

/** The redesign's auto-archive row (856:5981): one period select, "Never" turning the sweep off. */
function AutoArchivePeriodRow() {
  const { t, locale } = useI18n()
  const s = t.settings.sessions
  const copy = settingsPagesCopy(locale).archiveChats
  const archive = useAutoArchiveConfig(s.autoArchiveFailed)

  if (!archive) {
    return null
  }

  const periods = [...new Set([...AUTO_ARCHIVE_PERIODS, archive.days])].sort((a, b) => a - b)

  return (
    <ListRow
      action={
        <Select
          onValueChange={value =>
            value === 'off' ? archive.save(false, archive.days) : archive.save(true, Number(value))
          }
          value={archive.enabled ? String(archive.days) : 'off'}
        >
          <SelectTrigger aria-label={copy.period} className="min-w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="off">{copy.off}</SelectItem>
            {periods.map(days => (
              <SelectItem key={days} value={String(days)}>
                {copy.afterDays(days)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
      description={s.autoArchiveDesc}
      id={settingElementId(SETTING_IDS.sessions.autoArchive)}
      title={s.autoArchiveTitle}
    />
  )
}

// Lets the user pin the default cwd for new sessions. Without this, packaged
// builds on Windows used to spawn sessions in the install dir (`win-unpacked`
// / Program Files), which buried any files Hermes wrote there.
function DefaultProjectDirSetting({ compact = false }: { compact?: boolean } = {}) {
  const { t } = useI18n()
  const s = t.settings.sessions
  const profile = useStore($connection)?.profile ?? ''
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

    // The folder on screen belongs to the profile this effect loaded. Drop it
    // on a switch so the next paint and the next save both use the profile
    // that is open now.
    if (hostHas('originScopedPrefs')) {
      setDir(null)
    }

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
  }, [profile])

  const choose = useCallback(async () => {
    const settings = window.hermesDesktop?.settings

    if (!settings) {
      return
    }

    setBusy(true)

    try {
      // A browser tab has no native dialog for the server's disk: pick through the server
      // folder browser Projects already use.
      const picked = !hostHas('nativeFolderPicker')
        ? { dir: (await selectDesktopPaths({ defaultPath: dir ?? undefined, directories: true }))[0] ?? null }
        : await settings.pickDefaultProjectDir()

      if (!picked.dir) {
        return
      }

      const result = await settings.setDefaultProjectDir(picked.dir)
      setDir(result.dir)
      applyConfiguredDefaultProjectDir(result.dir)
      notify({ durationMs: 4_000, kind: 'success', message: s.defaultDirUpdated(newSessionShortcut()) })
    } catch (err) {
      notifyError(err, s.updateDirFailed)
    } finally {
      setBusy(false)
    }
  }, [dir, s])

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
      {!compact && (
        <>
          <SectionHeading icon={FolderOpen} page title={s.defaultDirTitle} />
          <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {s.defaultDirDesc}
          </p>
        </>
      )}
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
        title={compact ? s.defaultDirTitle : dir || s.notSet}
      />
    </div>
  )
}
