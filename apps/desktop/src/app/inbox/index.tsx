import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { MINUTE, relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import {
  $inboxItems,
  $inboxLoaded,
  type InboxItem,
  markInboxItemRead,
  refreshInbox
} from '@/store/inbox'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { forgetSessionOwnerHintsForSession, requestSessionResume } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { openCronRun } from '../cron/open-cron-run'
import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { openSession } from '../open-session'
import { PageSearchShell } from '../page-search-shell'

type InboxRun = Pick<SessionInfo, 'ended_at' | 'id' | 'last_active' | 'profile' | 'scheduler_owned' | 'source'>

type InboxViewProps = React.ComponentProps<'section'>

/** The SessionInfo-shaped row `openCronRun` judges: a running item stays
 *  scheduler-owned (read-only), a finished one is closed (writable). */
export function inboxRun(item: InboxItem, profile: string): InboxRun {
  return {
    ended_at: item.running ? null : item.delivered_at,
    id: item.session_id,
    last_active: item.delivered_at,
    profile,
    scheduler_owned: item.running,
    source: 'cron'
  }
}

function useMinuteClock(): number {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), MINUTE)

    return () => window.clearInterval(id)
  }, [])

  return now
}

export function InboxView(props: InboxViewProps) {
  const { locale, t } = useI18n()
  const navigate = useNavigate()
  const copy = t.inbox
  const items = useStore($inboxItems)
  const loaded = useStore($inboxLoaded)
  const profile = useStore($activeGatewayProfile)
  const now = useMinuteClock()
  const [refreshing, setRefreshing] = useState(false)

  const refresh = async () => {
    setRefreshing(true)

    try {
      await refreshInbox()
    } finally {
      setRefreshing(false)
    }
  }

  useRefreshHotkey(() => void refresh())

  useEffect(() => {
    void refreshInbox()
  }, [])

  // The sidebar's door for an untagged (active-profile) row: an explicit
  // resume, then focus it wherever it already is or load it into main.
  const open = (item: InboxItem) => {
    openCronRun(inboxRun(item, profile), id => {
      forgetSessionOwnerHintsForSession(id)
      requestSessionResume(id)
      openSession(id, navigate)
    })

    if (item.unread) {
      markInboxItemRead(item.session_id).catch(error => notifyError(error, copy.markReadFailed))
    }
  }

  return (
    <PageSearchShell
      {...props}
      searchHidden={items.length === 0}
      searchTrailingAction={
        <Tip label={refreshing ? copy.refreshing : copy.refresh}>
          <Button
            aria-label={refreshing ? copy.refreshing : copy.refresh}
            className="text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            disabled={refreshing}
            onClick={() => void refresh()}
            size="icon-titlebar"
            variant="ghost"
          >
            {refreshing ? <TitlebarIcon name="loading" spinning /> : <TitlebarIcon name="refresh" />}
          </Button>
        </Tip>
      }
      tabs={[{ id: 'all', label: copy.title, meta: loaded ? items.length : null }]}
    >
      {!loaded ? (
        <PageLoader label={copy.loading} />
      ) : items.length === 0 ? (
        <div className="grid h-full place-items-center px-6 text-center">
          <div>
            <div className="text-sm font-medium">{copy.emptyTitle}</div>
            <div className="mt-1 text-xs text-muted-foreground">{copy.emptyDesc}</div>
          </div>
        </div>
      ) : (
        <div className="h-full overflow-y-auto [scrollbar-gutter:stable]">
          <ul className="mx-auto flex max-w-3xl flex-col gap-px px-3 pb-2">
            {items.map(item => (
              <li key={item.session_id}>
                <RowButton
                  className="flex w-full items-start gap-2.5 rounded-md px-2.5 py-2 text-left hover:bg-(--chrome-action-hover) focus-visible:bg-(--chrome-action-hover) focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring/40"
                  onClick={() => open(item)}
                >
                  <span
                    aria-label={item.unread ? copy.unread : undefined}
                    className={cn(
                      'mt-1.5 size-1.5 shrink-0 rounded-full',
                      item.unread ? 'bg-(--ui-success)' : 'bg-transparent'
                    )}
                    role={item.unread ? 'status' : undefined}
                  />
                  <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span className="flex min-w-0 items-center gap-2">
                      <span
                        className={cn(
                          'min-w-0 truncate text-[0.8125rem]',
                          item.unread ? 'font-semibold text-foreground' : 'font-medium text-(--ui-text-secondary)'
                        )}
                      >
                        {item.job_name}
                      </span>
                      {item.running && (
                        <Badge size="xs" variant="warn">
                          {copy.running}
                        </Badge>
                      )}
                      <span className="ml-auto shrink-0 text-[0.6875rem] text-(--ui-text-tertiary) tabular-nums">
                        {relativeTime(item.delivered_at * 1000, now, locale)}
                      </span>
                    </span>
                    <span className="line-clamp-2 text-xs leading-relaxed text-(--ui-text-tertiary)">
                      {item.preview || copy.noPreview}
                    </span>
                  </span>
                </RowButton>
              </li>
            ))}
          </ul>
        </div>
      )}
    </PageSearchShell>
  )
}
