import { useEffect, useState } from 'react'

import { WisdomNotificationsCard } from '@/components/wisdom-notifications-card'
import {
  acknowledgeWisdomNotifications,
  getWisdomInstallations,
  type ProfileScope,
  type WisdomNotification
} from '@/hermes'
import { notifyError } from '@/store/notifications'

export function WisdomNoticeCard({ profile }: { profile?: ProfileScope }) {
  const [events, setEvents] = useState<WisdomNotification[]>([])

  useEffect(() => {
    let active = true

    const refresh = async () => {
      try {
        const result = await getWisdomInstallations(profile)

        if (active) {
          setEvents(result.notifications)
        }
      } catch {
        // The notice is an enhancement to the transcript. An unavailable or
        // unconfigured Wisdom plane must not make ordinary chat unusable.
        if (active) {
          setEvents([])
        }
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), 30_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [profile])

  if (events.length === 0) {
    return null
  }

  return (
    <WisdomNotificationsCard
      className="mb-(--conversation-turn-gap) bg-(--ui-chat-surface-background)"
      events={events}
      onMarkAllRead={async () => {
        try {
          await acknowledgeWisdomNotifications(profile)
          setEvents([])
        } catch (error) {
          notifyError(error, 'Could not acknowledge Wisdom notifications')
        }
      }}
      onPlanAction={(action, event) => {
        const params = new URLSearchParams({
          tab: 'collective',
          wisdomAction: action,
          wisdomSkillId: event.skill_id
        })

        window.location.hash = `#/skills?${params.toString()}`
      }}
    />
  )
}
