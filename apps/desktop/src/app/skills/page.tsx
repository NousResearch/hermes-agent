import type { ComponentProps } from 'react'
import { useNavigate } from 'react-router'

import { $selectedStoredSessionId } from '@/store/session'

import { NEW_CHAT_ROUTE, sessionRoute } from '../routes'

import { SkillsView } from './index'

interface SkillsPageProps extends Pick<ComponentProps<typeof SkillsView>, 'setStatusbarItemGroup'> {}

/** The workspace page owns navigation; embedded capability views do not. */
export function SkillsPage(props: SkillsPageProps) {
  const navigate = useNavigate()

  const close = () => {
    const sessionId = $selectedStoredSessionId.get()
    navigate(sessionId ? sessionRoute(sessionId) : NEW_CHAT_ROUTE, { replace: true })
  }

  return <SkillsView {...props} onClose={close} />
}
