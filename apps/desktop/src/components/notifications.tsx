import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AlertCircle, AlertTriangle, CheckCircle2, type IconComponent, Info } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $notifications,
  type AppNotification,
  clearNotifications,
  dismissNotification,
  type NotificationKind
} from '@/store/notifications'

type ToneVariant = 'default' | 'destructive' | 'warning' | 'success'

const tone: Record<NotificationKind, { icon: IconComponent; iconClass: string; variant: ToneVariant }> = {
  error: { icon: AlertCircle, iconClass: 'text-destructive', variant: 'destructive' },
  warning: { icon: AlertTriangle, iconClass: 'text-primary', variant: 'warning' },
  info: { icon: Info, iconClass: 'text-muted-foreground', variant: 'default' },
  success: { icon: CheckCircle2, iconClass: 'text-primary', variant: 'success' }
}

const STACK_SURFACE = 'pointer-events-auto border border-(--stroke-nous) bg-popover/95 shadow-nous backdrop-blur-md'

function partitionNotifications(notifications: AppNotification[]) {
  const defaultStack: AppNotification[] = []
  const bottomRightStack: AppNotification[] = []

  for (const notification of notifications) {
    if (notification.placement === 'bottom-right') {
      bottomRightStack.push(notification)
    } else {
      defaultStack.push(notification)
    }
  }

  return { bottomRightStack, defaultStack }
}

export function NotificationStack() {
  const notifications = useStore($notifications)
  const { bottomRightStack, defaultStack } = partitionNotifications(notifications)
  const { t } = useI18n()
  const lastNotificationIdRef = useRef<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const copy = t.notifications

  useEffect(() => {
    if (defaultStack.length <= 1) {
      setExpanded(false)
    }
  }, [defaultStack.length])

  useEffect(() => {
    const latest = notifications[0]

    if (!latest || latest.id === lastNotificationIdRef.current) {
      return
    }

    lastNotificationIdRef.current = latest.id

    if (latest.kind === 'success') {
      triggerHaptic('success')
    } else if (latest.kind === 'error') {
      triggerHaptic('error')
    } else if (latest.kind === 'warning') {
      triggerHaptic('warning')
    }
  }, [notifications])

  return (
    <>
      {defaultStack.length > 0 && (
        <TopCenterStack
          copy={copy}
          expanded={expanded}
          notifications={defaultStack}
          onToggleExpanded={() => setExpanded(v => !v)}
        />
      )}
      {bottomRightStack.length > 0 && <BottomRightStack copy={copy} notifications={bottomRightStack} />}
    </>
  )
}

// Portaled to <body> with a z above the Radix dialog layer (overlay z-[120],
// content z-[130]) — see the top-center variant below for why.
const REGION_BASE = 'pointer-events-none fixed z-[200] flex gap-2'

// Primary stack: top-center, collapsed to the latest toast with a "+N more"
// expander + clear-all — the noisy/important surface (errors, warnings,
// action toasts). Without the portal it lives inside the React root subtree,
// which any body-level dialog/overlay portal paints over — so a toast fired
// while a dialog is open was invisible.
function TopCenterStack({
  copy,
  expanded,
  notifications,
  onToggleExpanded
}: {
  copy: ReturnType<typeof useI18n>['t']['notifications']
  expanded: boolean
  notifications: AppNotification[]
  onToggleExpanded: () => void
}) {
  const [latest, ...older] = notifications

  return createPortal(
    <div
      aria-label={copy.region}
      className={cn(
        REGION_BASE,
        'left-1/2 top-[calc(var(--titlebar-height,34px)+0.75rem)] w-[min(32rem,calc(100%-2rem))] -translate-x-1/2 flex-col'
      )}
      role="region"
    >
      <NotificationItem notification={latest} />
      {expanded && older.map(n => <NotificationItem key={n.id} notification={n} />)}
      {older.length > 0 && (
        <div className={cn(STACK_SURFACE, 'flex min-h-8 items-center justify-between rounded-lg px-3 text-xs')}>
          <Button className="-ml-2" onClick={onToggleExpanded} size="xs" type="button" variant="text">
            {expanded ? copy.hide : copy.show} {copy.more(older.length)}
          </Button>
          <Button className="-mr-2" onClick={clearNotifications} size="xs" type="button" variant="text">
            {copy.clearAll}
          </Button>
        </div>
      )}
    </div>,
    document.body
  )
}

// Ambient stack: bottom-right, every toast shown at once (routine confirmations
// rarely queue up), newest on top, no expand/clear-all chrome.
function BottomRightStack({
  copy,
  notifications
}: {
  copy: ReturnType<typeof useI18n>['t']['notifications']
  notifications: AppNotification[]
}) {
  return createPortal(
    <div
      aria-label={copy.region}
      className={cn(REGION_BASE, 'right-4 bottom-4 w-[min(24rem,calc(100%-2rem))] flex-col-reverse')}
      role="region"
    >
      {notifications.map(n => (
        <NotificationItem key={n.id} notification={n} />
      ))}
    </div>,
    document.body
  )
}

function NotificationItem({ notification }: { notification: AppNotification }) {
  const styles = tone[notification.kind]
  const Icon = styles.icon
  const hasDetail = Boolean(notification.detail && notification.detail !== notification.message)
  const { t } = useI18n()
  const copy = t.notifications

  return (
    <Alert
      aria-live={notification.kind === 'error' ? 'assertive' : 'polite'}
      className={cn(STACK_SURFACE, 'grid-cols-[auto_minmax(0,1fr)_auto] pr-2.5')}
      role={notification.kind === 'error' ? 'alert' : 'status'}
      variant={styles.variant}
    >
      {notification.icon ? (
        <Codicon className={styles.iconClass} name={notification.icon} size="1rem" />
      ) : (
        <Icon className={styles.iconClass} />
      )}
      <div className="col-start-2 min-w-0">
        {notification.title && <AlertTitle className="col-start-auto">{notification.title}</AlertTitle>}
        <AlertDescription className="col-start-auto">
          <p className="m-0">{notification.message}</p>
          {hasDetail && <NotificationDetail detail={notification.detail || ''} />}
          {notification.action && (
            <Button
              className="mt-1.5"
              onClick={() => {
                notification.action?.onClick()
                dismissNotification(notification.id)
              }}
              size="xs"
              type="button"
              variant="textStrong"
            >
              {notification.action.label}
            </Button>
          )}
        </AlertDescription>
      </div>
      <Button
        aria-label={copy.dismiss}
        className="col-start-3 -mr-1 text-muted-foreground"
        onClick={() => dismissNotification(notification.id)}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        <Codicon name="close" size="0.875rem" />
      </Button>
    </Alert>
  )
}

function NotificationDetail({ detail }: { detail: string }) {
  const { t } = useI18n()
  const copy = t.notifications

  return (
    <details className="mt-2 text-xs text-muted-foreground">
      <summary className="select-none font-medium text-muted-foreground hover:text-foreground">{copy.details}</summary>
      <div className="mt-1 rounded-md bg-background/65 p-2">
        <pre
          className="max-h-32 whitespace-pre-wrap wrap-break-word font-mono text-[0.6875rem] leading-relaxed"
          data-selectable-text="true"
        >
          {detail}
        </pre>
        <CopyButton
          appearance="inline"
          className="mt-1 rounded px-1.5 py-0.5 text-[0.6875rem]"
          errorMessage={copy.copyDetailFailed}
          iconClassName="size-3"
          label={copy.copyDetail}
          text={detail}
        >
          {copy.copyDetail}
        </CopyButton>
      </div>
    </details>
  )
}

export function InlineNotice({
  kind = 'info',
  title,
  children,
  className
}: {
  kind?: NotificationKind
  title?: string
  children: ReactNode
  className?: string
}) {
  const styles = tone[kind]
  const Icon = styles.icon

  return (
    <Alert className={cn('min-w-0', className)} role={kind === 'error' ? 'alert' : 'status'} variant={styles.variant}>
      <Icon />
      {title && <AlertTitle>{title}</AlertTitle>}
      <AlertDescription className={cn(!title && 'row-start-1')}>{children}</AlertDescription>
    </Alert>
  )
}
