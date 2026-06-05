import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
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

const STACK_SURFACE = 'pointer-events-auto border-border/80 bg-popover/95 shadow-lg shadow-black/5 backdrop-blur-md'
const GHOST_BTN = 'bg-transparent text-muted-foreground hover:text-foreground'

export function NotificationStack() {
  const notifications = useStore($notifications)
  const { t } = useI18n()
  const lastNotificationIdRef = useRef<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const copy = t.notifications

  useEffect(() => {
    if (notifications.length <= 1) {
      setExpanded(false)
    }
  }, [notifications.length])

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

  if (notifications.length === 0) {
    return null
  }

  const [latest, ...olderNotifications] = notifications
  const overflowCount = olderNotifications.length

  // Portaled to <body> with a z above the Radix dialog layer (overlay z-[120],
  // content z-[130]). Without the portal the stack lives inside the React root
  // subtree, which any body-level dialog/overlay portal paints over — so a
  // success toast fired while a dialog is open (or over an OverlayView page)
  // was invisible. The titlebar-height var only exists inside the app shell
  // scope, so fall back to its constant (34px) when mounted on <body>.
  return createPortal(
    <div
      aria-label={copy.region}
      className="pointer-events-none fixed left-1/2 top-[calc(var(--titlebar-height,34px)+0.75rem)] z-[200] flex w-[min(32rem,calc(100%-2rem))] -translate-x-1/2 flex-col gap-2"
      role="region"
    >
      <NotificationItem notification={latest} />
      {expanded && olderNotifications.map(n => <NotificationItem key={n.id} notification={n} />)}
      {overflowCount > 0 && (
        <div className={cn(STACK_SURFACE, 'flex min-h-8 items-center justify-between rounded-lg px-3 text-xs')}>
          <button className={cn(GHOST_BTN, 'font-medium')} onClick={() => setExpanded(v => !v)} type="button">
            {expanded ? copy.hide : copy.show} {copy.more(overflowCount)}
          </button>
          <button className={GHOST_BTN} onClick={clearNotifications} type="button">
            {copy.clearAll}
          </button>
        </div>
      )}
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
      variant="default"
    >
      <Icon className={styles.iconClass} />
      <div className="col-start-2 min-w-0">
        {notification.title && <AlertTitle className="col-start-auto">{notification.title}</AlertTitle>}
        <AlertDescription className="col-start-auto">
          <p className="m-0">{notification.message}</p>
          {hasDetail && <NotificationDetail detail={notification.detail || ''} />}
          {notification.action && (
            <button
              className="mt-1.5 inline-flex items-center rounded-md bg-primary/15 px-2 py-1 text-xs font-medium text-primary transition-colors hover:bg-primary/25"
              onClick={() => {
                notification.action?.onClick()
                dismissNotification(notification.id)
              }}
              type="button"
            >
              {notification.action.label}
            </button>
          )}
        </AlertDescription>
      </div>
      <button
        aria-label={copy.dismiss}
        className="col-start-3 -mr-1 grid size-6 place-items-center rounded-md bg-transparent text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        onClick={() => dismissNotification(notification.id)}
        type="button"
      >
        <Codicon name="close" size="0.875rem" />
      </button>
    </Alert>
  )
}

function NotificationDetail({ detail }: { detail: string }) {
  const { t } = useI18n()
  const copy = t.notifications

  return (
    <details className="mt-2 text-xs text-muted-foreground">
      <summary className="select-none font-medium text-muted-foreground hover:text-foreground">{copy.details}</summary>
      <div className="mt-1 rounded-md border border-border/70 bg-background/65 p-2">
        <pre className="max-h-32 whitespace-pre-wrap wrap-break-word font-mono text-[0.6875rem] leading-relaxed">
          {detail}
        </pre>
        <CopyButton
          appearance="inline"
          className="mt-1 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[0.6875rem] text-muted-foreground hover:bg-accent hover:text-foreground"
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
