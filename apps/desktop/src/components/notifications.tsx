import { useStore } from '@nanostores/react'
import { type CSSProperties, type ReactNode, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { CardStack, type CardStackAction } from '@/components/ui/card-stack'
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

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
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
      <TopCenterStack
        copy={copy}
        expanded={expanded}
        notifications={defaultStack}
        onToggleExpanded={() => setExpanded(v => !v)}
      />
      <BottomRightStack copy={copy} notifications={bottomRightStack} />
    </>
  )
}

// Portaled to <body> on the over-modal rung so a toast clears an open dialog —
// see the top-center variant below for why.
const REGION_BASE = 'pointer-events-none fixed z-(--z-over-modal) flex gap-2'

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
  const older = notifications.slice(1)

  return createPortal(
    <div
      aria-label={copy.region}
      className={cn(
        REGION_BASE,
        'left-1/2 top-[calc(var(--titlebar-height,34px)+0.75rem)] w-[min(28rem,calc(100%-2rem))] -translate-x-1/2 flex-col p-1',
        expanded && 'max-h-[70vh] overflow-y-auto overscroll-contain'
      )}
      role="region"
    >
      <NotificationDeck expanded={expanded} notifications={notifications} />
      {older.length > 0 && (
        <div className="pointer-events-auto flex min-h-8 items-center justify-between px-3 text-xs">
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

// Ambient confirmations use the same bounded depth, rising from the corner.
function BottomRightStack({
  copy,
  notifications
}: {
  copy: ReturnType<typeof useI18n>['t']['notifications']
  notifications: AppNotification[]
}) {
  const [expanded, setExpanded] = useState(false)
  const older = notifications.slice(1)

  useEffect(() => {
    if (!older.length) {
      setExpanded(false)
    }
  }, [older.length])

  return createPortal(
    <div
      aria-label={copy.region}
      className={cn(
        REGION_BASE,
        'right-4 bottom-4 w-[min(24rem,calc(100%-2rem))] flex-col-reverse p-1',
        expanded && 'max-h-[70vh] overflow-y-auto overscroll-contain'
      )}
      role="region"
    >
      {older.length > 0 && (
        <div className="pointer-events-auto flex items-center justify-between px-2 text-xs">
          <Button onClick={() => setExpanded(value => !value)} size="xs" variant="text">
            {expanded ? copy.hide : copy.show} {copy.more(older.length)}
          </Button>
          <Button
            onClick={() => notifications.forEach(notification => dismissNotification(notification.id))}
            size="xs"
            variant="text"
          >
            {copy.clearAll}
          </Button>
        </div>
      )}
      <NotificationDeck expanded={expanded} notifications={notifications} />
    </div>,
    document.body
  )
}

export function NotificationDeck({
  notifications,
  expanded = false
}: {
  notifications: AppNotification[]
  expanded?: boolean
}) {
  return (
    <CardStack
      expanded={expanded}
      getKey={notification => notification.id}
      items={notifications}
      onSwipe={(notification, _side, action) => {
        void action.depart(() => dismissNotification(notification.id))
      }}
      surfaceClassName={cn(
        STACK_SURFACE,
        'rounded-lg',
        // A collapsed card must never outgrow the viewport: an oversized
        // message (model-switch confirmations run ~800 chars) previously blew
        // the card past the screen bottom, putting its action buttons
        // unreachable. Clamped + scrollable while collapsed; expanded the
        // stack itself scrolls, so the cap is unnecessary there.
        !expanded && 'notification-collapsed-clamp'
      )}
      swipeDirections={['left', 'right']}
    >
      {(notification, action) => <NotificationItem notification={notification} stack={action} />}
    </CardStack>
  )
}

// Emphasize only the leading money figure ("$16.00" — the amount used) with the
// accent color (semibold), leaving the rest of the line in its default muted
// tone. No accent, or no figure in the message → render the text untouched.
function renderMessage(message: string, accent?: string): ReactNode {
  const match = accent ? /\$\d+(?:\.\d{2})?/.exec(message) : null

  if (!match) {
    return message
  }

  const start = match.index
  const end = start + match[0].length

  return (
    <>
      {message.slice(0, start)}
      <span className="font-semibold" style={{ color: accent }}>
        {match[0]}
      </span>
      {message.slice(end)}
    </>
  )
}

// AlertTitle defaults to a single-line clamp. Toast errors are often a full
// sentence, so the toast wraps — then caps height and scrolls instead of
// growing down the chat or clipping with an ellipsis.
export function toastTitleClassName() {
  return 'col-start-auto line-clamp-none max-h-[4.5em] overflow-y-auto overscroll-contain whitespace-normal wrap-break-word'
}

function NotificationItem({ notification, stack }: { notification: AppNotification; stack: CardStackAction }) {
  const styles = tone[notification.kind]
  const Icon = styles.icon
  const hasDetail = Boolean(notification.detail && notification.detail !== notification.message)
  const { t } = useI18n()
  const copy = t.notifications

  // Nudge the icon down to sit on the first text line, in `ch` so it tracks the
  // toast's font size instead of a fixed rem. `accentColor` (when set) tints the
  // icon + message as a severity ramp, overriding the kind's default color.
  const accent = notification.accentColor
  const iconStyle: CSSProperties = { marginTop: '0.42ch', ...(accent ? { color: accent } : {}) }

  return (
    <Alert
      aria-live={!stack.active ? 'off' : notification.kind === 'error' ? 'assertive' : 'polite'}
      className="grid-cols-[auto_minmax(0,1fr)_auto] border-0 bg-transparent pr-2.5 shadow-none"
      role={notification.kind === 'error' ? 'alert' : 'status'}
      variant={styles.variant}
    >
      {notification.icon ? (
        <Codicon className={styles.iconClass} name={notification.icon} size="1rem" style={iconStyle} />
      ) : (
        <Icon className={styles.iconClass} style={iconStyle} />
      )}
      <div className="col-start-2 min-w-0">
        {notification.title && (
          <AlertTitle className={toastTitleClassName()} title={notification.title}>
            {notification.title}
          </AlertTitle>
        )}
        <AlertDescription className="col-start-auto">
          <p className="m-0 wrap-break-word">{renderMessage(notification.message, accent)}</p>
          {notification.meta && <p className="m-0 text-xs text-muted-foreground tabular-nums">{notification.meta}</p>}
          {hasDetail && <NotificationDetail detail={notification.detail || ''} />}
          {(notification.action || notification.secondaryAction) && (
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {notification.action && (
                <Button
                  disabled={stack.busy || !stack.active}
                  onClick={() => {
                    void stack.depart(() => {
                      notification.action?.onClick()
                      dismissNotification(notification.id)
                    })
                  }}
                  size="sm"
                  type="button"
                  variant="default"
                >
                  {notification.action.label}
                </Button>
              )}
              {notification.secondaryAction && (
                <Button
                  disabled={stack.busy || !stack.active}
                  onClick={() => {
                    void stack.depart(() => {
                      notification.secondaryAction?.onClick()
                      dismissNotification(notification.id)
                    })
                  }}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  {notification.secondaryAction.label}
                </Button>
              )}
            </div>
          )}
        </AlertDescription>
      </div>
      <Button
        aria-label={copy.dismiss}
        className="col-start-3 -mr-1 text-muted-foreground"
        disabled={stack.busy || !stack.active}
        onClick={() => {
          void stack.depart(() => dismissNotification(notification.id))
        }}
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
