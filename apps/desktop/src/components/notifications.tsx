import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useRef, useState } from 'react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { CopyButton } from '@/components/ui/copy-button'
import { triggerHaptic } from '@/lib/haptics'
import { AlertCircle, AlertTriangle, CheckCircle2, Info, type LucideIcon, X } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $desktopLanguage } from '@/store/language'
import {
  $notifications,
  type AppNotification,
  clearNotifications,
  dismissNotification,
  type NotificationKind
} from '@/store/notifications'

type ToneVariant = 'default' | 'destructive' | 'warning' | 'success'

const tone: Record<NotificationKind, { icon: LucideIcon; iconClass: string; variant: ToneVariant }> = {
  error: { icon: AlertCircle, iconClass: 'text-destructive', variant: 'destructive' },
  warning: { icon: AlertTriangle, iconClass: 'text-primary', variant: 'warning' },
  info: { icon: Info, iconClass: 'text-muted-foreground', variant: 'default' },
  success: { icon: CheckCircle2, iconClass: 'text-primary', variant: 'success' }
}

const STACK_SURFACE = 'pointer-events-auto border-border/80 bg-popover/95 shadow-lg shadow-black/5 backdrop-blur-md'
const GHOST_BTN = 'bg-transparent text-muted-foreground hover:text-foreground'

export function NotificationStack() {
  const notifications = useStore($notifications)
  const language = useStore($desktopLanguage)
  const lastNotificationIdRef = useRef<string | null>(null)
  const [expanded, setExpanded] = useState(false)

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

  return (
    <div
      aria-label={language === 'zh' ? '通知' : 'Notifications'}
      className="pointer-events-none absolute left-1/2 top-[calc(var(--titlebar-height)+0.75rem)] z-1050 flex w-[min(32rem,calc(100%-2rem))] -translate-x-1/2 flex-col gap-2"
      role="region"
    >
      <NotificationItem language={language} notification={latest} />
      {expanded && olderNotifications.map(n => <NotificationItem key={n.id} language={language} notification={n} />)}
      {overflowCount > 0 && (
        <div className={cn(STACK_SURFACE, 'flex min-h-8 items-center justify-between rounded-lg px-3 text-xs')}>
          <button className={cn(GHOST_BTN, 'font-medium')} onClick={() => setExpanded(v => !v)} type="button">
            {language === 'zh'
              ? `${expanded ? '隐藏' : '显示'}另外 ${overflowCount} 条通知`
              : `${expanded ? 'Hide' : 'Show'} ${overflowCount} more ${overflowCount === 1 ? 'notification' : 'notifications'}`}
          </button>
          <button className={GHOST_BTN} onClick={clearNotifications} type="button">
            {language === 'zh' ? '全部清除' : 'Clear all'}
          </button>
        </div>
      )}
    </div>
  )
}

function NotificationItem({ notification, language }: { notification: AppNotification; language: 'zh' | 'en' }) {
  const styles = tone[notification.kind]
  const Icon = styles.icon
  const hasDetail = Boolean(notification.detail && notification.detail !== notification.message)

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
        {hasDetail && <NotificationDetail detail={notification.detail || ''} language={language} />}
        </AlertDescription>
      </div>
      <button
        aria-label={language === 'zh' ? '关闭通知' : 'Dismiss notification'}
        className="col-start-3 -mr-1 grid size-6 place-items-center rounded-md bg-transparent text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        onClick={() => dismissNotification(notification.id)}
        type="button"
      >
        <X className="size-3.5" />
      </button>
    </Alert>
  )
}

function NotificationDetail({ detail, language }: { detail: string; language: 'zh' | 'en' }) {
  return (
    <details className="mt-2 text-xs text-muted-foreground">
      <summary className="cursor-pointer select-none font-medium text-muted-foreground hover:text-foreground">
        {language === 'zh' ? '详情' : 'Details'}
      </summary>
      <div className="mt-1 rounded-md border border-border/70 bg-background/65 p-2">
        <pre className="max-h-32 whitespace-pre-wrap wrap-break-word font-mono text-[0.6875rem] leading-relaxed">
          {detail}
        </pre>
        <CopyButton
          appearance="inline"
          className="mt-1 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[0.6875rem] text-muted-foreground hover:bg-accent hover:text-foreground"
          errorMessage={language === 'zh' ? '无法复制通知详情' : 'Could not copy notification detail'}
          iconClassName="size-3"
          label={language === 'zh' ? '复制详情' : 'Copy detail'}
          text={detail}
        >
          {language === 'zh' ? '复制详情' : 'Copy detail'}
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
