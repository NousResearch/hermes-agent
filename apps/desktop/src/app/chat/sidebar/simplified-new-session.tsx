import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { RowButton } from '@/components/ui/row-button'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { recordAction } from '@/store/desktop-metrics'
import { $activeGatewayProfile, $profileColors, $profiles } from '@/store/profile'
import { $simplifiedAgentMode } from '@/store/simplified-agent-mode'
import { SIMPLIFIED_NEW_SESSION_EVENT } from '@/store/simplified-new-session'

import { useNewSessionCopy } from './simplified-new-session-copy'
import { rankNewSessionAgents, startSimplifiedAgentSession } from './simplified-new-session-model'
import { useSimplifiedAgents } from './use-simplified-agents'

export function SimplifiedNewSession({ sessions, onFallback }: { sessions: SessionInfo[]; onFallback: () => void }) {
  const { t } = useI18n()
  const c = useNewSessionCopy()
  const { agents, owner } = useSimplifiedAgents([])
  const profile = useStore($activeGatewayProfile)
  const profiles = useStore($profiles)
  const colors = useStore($profileColors)
  const [open, setOpen] = useState(false)
  const ranked = rankNewSessionAgents(agents, sessions, owner, profile)
  const defaultProfile = profiles.find(item => item.is_default)?.name ?? 'default'

  function request() {
    if (agents.length > 1) {
      setOpen(true)
    } else if (agents.length === 1) {
      startSimplifiedAgentSession(agents[0])
    } else {
      onFallback()
    }
  }

  useEffect(() => {
    const onRequest = (event: Event) => {
      if (!$simplifiedAgentMode.get()) {
        return
      }

      event.preventDefault()
      request()
    }

    window.addEventListener(SIMPLIFIED_NEW_SESSION_EVENT, onRequest)

    return () => window.removeEventListener(SIMPLIFIED_NEW_SESSION_EVENT, onRequest)
  })

  const button = (
    <RowButton
      className="flex min-h-8.5 w-full items-center gap-2 rounded-sm px-2 text-left text-xs font-medium hover:bg-(--chrome-action-hover) focus-visible:outline-2 focus-visible:outline-ring"
      onClick={() => {
        recordAction('session.new', 'click')

        if (agents.length <= 1) {
          request()
        }
      }}
    >
      <span className="grid size-4 shrink-0 place-items-center">
        <Codicon name="add" />
      </span>
      {t.sidebar.nav['new-session']}
    </RowButton>
  )

  if (agents.length <= 1) {
    return button
  }

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <DropdownMenuTrigger asChild>{button}</DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        <DropdownMenuLabel>{c.newSessionIn}</DropdownMenuLabel>
        {ranked.map(agent => (
          <DropdownMenuItem
            aria-label={`${agent.label} · ${agent.gateway}`}
            key={agent.id}
            onSelect={() => startSimplifiedAgentSession(agent)}
          >
            <ProfileGlyph
              color={resolveProfileColor(agent.profile, colors)}
              isDefault={agent.profile === 'default'}
              name={agent.label}
            />
            <span className="min-w-0 flex-1 truncate">{agent.label}</span>
            <span className="max-w-28 truncate text-xs text-(--ui-text-tertiary)">{agent.gateway}</span>
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={() => startSimplifiedAgentSession({ connectionId: owner, profile: defaultProfile })}
        >
          <Codicon name="circle-outline" />
          <span className="flex flex-col">
            <span>{c.blank}</span>
            <span className="text-xs text-(--ui-text-tertiary)">{c.blankHint}</span>
          </span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
