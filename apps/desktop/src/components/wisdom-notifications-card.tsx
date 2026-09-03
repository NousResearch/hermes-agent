import { useState } from 'react'

import { Button } from '@/components/ui/button'
import type { WisdomNotification } from '@/hermes'
import { useI18n } from '@/i18n'
import type { WisdomTranslations } from '@/i18n/types'
import { openExternalLink } from '@/lib/external-link'
import { cn } from '@/lib/utils'

function notificationText(event: WisdomNotification, copy: WisdomTranslations): string {
  const skill = event.skill_name
  const version = event.version ? `v${event.version}` : undefined
  const skillVersion = version ? `${skill} ${version}` : skill

  if (event.category === 'publication_decision') {
    if (event.state === 'published' || event.state === 'approved') {
      return copy.decisionPublished(skillVersion)
    }

    if (event.state === 'changes_requested') {
      const note = event.moderation_note?.trim()

      return `${copy.decisionChanges(skillVersion)}${note ? ` ${note}` : ''}`
    }

    if (event.state === 'declined' || event.state === 'rejected') {
      return copy.decisionDeclined(skillVersion)
    }

    return copy.decisionChanged(skillVersion, copy.draftState(event.state || 'updated'))
  }

  if (event.category === 'installed') {
    return copy.installedNotice(skill, version)
  }

  if (event.category === 'updated') {
    return copy.updatedNotice(skill, version)
  }

  if (event.category === 'update_available') {
    return copy.updateNotice(skill, version)
  }

  if (event.category === 'new_skill') {
    return copy.newSkillNotice(skillVersion)
  }

  return copy.unavailableNotice(skillVersion)
}

export function WisdomNotificationsCard({
  className,
  events,
  onMarkAllRead,
  onPlanAction
}: {
  className?: string
  events: WisdomNotification[]
  onMarkAllRead: () => Promise<void>
  onPlanAction?: (action: 'install' | 'update', event: WisdomNotification) => Promise<void> | void
}) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [clearing, setClearing] = useState(false)
  const [planning, setPlanning] = useState<null | string>(null)

  if (events.length === 0) {
    return null
  }

  return (
    <section
      aria-label={copy.notifications}
      className={cn(
        'rounded-[4px] border border-(--ui-stroke-secondary) bg-(--ui-widget-surface-background) px-3 py-2.5',
        className
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-xs font-semibold">{copy.notifications}</h2>
          <p className="mt-0.5 text-[0.65rem] text-muted-foreground">{copy.activityReady(events.length)}</p>
        </div>
        <Button
          disabled={clearing}
          onClick={async () => {
            setClearing(true)

            try {
              await onMarkAllRead()
            } finally {
              setClearing(false)
            }
          }}
          size="inline"
          variant="text"
        >
          {copy.markSeen}
        </Button>
      </div>
      <ul className="mt-2 divide-y divide-(--ui-stroke-tertiary)">
        {events.slice(0, 8).map(event => (
          <li
            className="flex min-w-0 items-center justify-between gap-3 py-2 first:pt-0 last:pb-0"
            key={event.event_id}
          >
            <p className="min-w-0 text-[0.68rem] leading-4 text-(--ui-text-secondary)">
              {notificationText(event, copy)}
            </p>
            <div className="flex shrink-0 items-center gap-3">
              {event.portal_url ? (
                <Button onClick={() => openExternalLink(event.portal_url || '')} size="inline" variant="text">
                  {copy.viewSkill}
                </Button>
              ) : null}
              {onPlanAction && (event.category === 'new_skill' || event.category === 'update_available') ? (
                <Button
                  disabled={planning === event.event_id}
                  onClick={async () => {
                    setPlanning(event.event_id)

                    try {
                      await onPlanAction(event.category === 'new_skill' ? 'install' : 'update', event)
                    } finally {
                      setPlanning(null)
                    }
                  }}
                  size="inline"
                  variant="textStrong"
                >
                  {event.category === 'new_skill' ? copy.install : copy.reviewUpdate}
                </Button>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
