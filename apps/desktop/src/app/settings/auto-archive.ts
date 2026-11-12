import { useCallback, useEffect, useState } from 'react'

import { getHermesConfigRecord, peekConfigReadOrigin, retainConfigReadOrigin, saveHermesConfig } from '@/hermes'
import { notifyError } from '@/store/notifications'
import type { HermesConfigRecord } from '@/types/hermes'

const DEFAULT_AUTO_ARCHIVE_DAYS = 3

/** The periods the redesign's select offers; a configured value outside them is offered as well. */
export const AUTO_ARCHIVE_PERIODS = [1, 3, 7, 14, 30, 90] as const

export interface AutoArchiveConfig {
  enabled: boolean
  days: number
  setDays: (days: number) => void
  /** Optimistically applies and saves both keys; a failed save surfaces `failedMessage`. */
  save: (enabled: boolean, days: number) => void
}

// Opt-in retention: soft-hide chats untouched for N days. The policy itself
// (last-activity sweep, pin exemption) lives in the backend
// (sessions.auto_archive in config.yaml + SessionDB.maybe_auto_archive); this
// only edits the config keys, so CLI / gateway / Desktop all honour one
// setting. Pins are exempt on the backend, so pinned chats survive regardless.
// Null until the config is read: the control stays unmounted if it cannot be.
export function useAutoArchiveConfig(failedMessage: string): AutoArchiveConfig | null {
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const [enabled, setEnabled] = useState(false)
  const [days, setDays] = useState(DEFAULT_AUTO_ARCHIVE_DAYS)

  useEffect(() => {
    // Config REST is only reachable through the Electron bridge; skip in
    // non-Electron contexts (tests/storybook) rather than throwing.
    if (!window.hermesDesktop) {
      return
    }

    let alive = true

    void getHermesConfigRecord()
      .then(record => {
        if (!alive) {
          return
        }

        const sessions = (record.sessions ?? {}) as Record<string, unknown>
        const parsedDays = Number(sessions.auto_archive_days)
        setConfig(record)
        setEnabled(Boolean(sessions.auto_archive))
        setDays(Number.isFinite(parsedDays) && parsedDays > 0 ? Math.round(parsedDays) : DEFAULT_AUTO_ARCHIVE_DAYS)
      })
      .catch(() => {
        // Leave the control unmounted if config can't be read.
      })

    return () => {
      alive = false
    }
  }, [])

  const save = useCallback(
    (autoArchive: boolean, archiveDays: number) => {
      if (!config) {
        return
      }

      setEnabled(autoArchive)
      setDays(archiveDays)

      const sessions = {
        ...((config.sessions ?? {}) as Record<string, unknown>),
        auto_archive: autoArchive,
        auto_archive_days: archiveDays
      }

      // Read the route at save time from the record itself, and carry it onto
      // the replacement snapshot so the next save still targets the gateway
      // that served the original GET.
      const writeScope = peekConfigReadOrigin(config)

      setConfig(retainConfigReadOrigin({ ...config, sessions }, config))

      // Sparse patch: PUT /api/config deep-merges, and echoing the cached
      // snapshot would overwrite keys other surfaces changed since it loaded.
      saveHermesConfig({ sessions: { auto_archive: autoArchive, auto_archive_days: archiveDays } }, writeScope).catch(
        err => notifyError(err, failedMessage)
      )
    },
    [config, failedMessage]
  )

  return config ? { enabled, days, setDays, save } : null
}
