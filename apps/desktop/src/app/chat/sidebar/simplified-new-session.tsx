import { useStore } from '@nanostores/react'
import { Select as SelectPrimitive } from 'radix-ui'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectSeparator } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { useKeybindHint } from '@/lib/keybinds/use-keybind-hint'
import { resolveProfileColor } from '@/lib/profile-color'
import { recordAction } from '@/store/desktop-metrics'
import { recordIaAction } from '@/store/ia-metrics'
import { $activeGatewayProfile, $profileColors, $profiles } from '@/store/profile'
import { $focusedStoredSessionId } from '@/store/session-focus'
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
  const focusedSessionId = useStore($focusedStoredSessionId)
  const shortcut = useKeybindHint('session.new')
  const [open, setOpen] = useState(false)
  const picked = useRef(false)
  const context = JSON.stringify([owner, profile, focusedSessionId])
  const [choice, setChoice] = useState<{ context: string; id: string } | null>(null)
  const ranked = rankNewSessionAgents(agents, sessions, owner, profile)
  const defaultProfile = profiles.find(item => item.is_default)?.name ?? 'default'

  // A picker choice belongs to the chat being viewed, never a different gateway or session.
  const choiceId = choice?.context === context ? choice.id : null
  const blank = choiceId === 'blank'

  const selected =
    agents.find(agent => agent.id === choiceId) ??
    agents.find(agent => agent.connectionId === owner && agent.profile === profile) ??
    ranked[0]

  const value = blank ? 'blank' : selected?.id
  const targetLabel = blank ? c.blank : selected ? `${selected.label} · ${selected.gateway}` : c.newSessionIn

  function changeOpen(next: boolean) {
    if (next && !open) {
      picked.current = false

      recordIaAction('ia.new_session.sidebar')

      recordIaAction('ia.new_session.picker_opened')

      if (!$focusedStoredSessionId.get()) {
        recordIaAction('ia.new_session.picker_on_new')
      }
    } else if (!next && open && !picked.current) {
      recordIaAction('ia.new_session.dismissed')
    }

    setOpen(next)
  }

  function request() {
    changeOpen(false)

    if (blank) {
      recordIaAction('ia.new_session.default')
      startSimplifiedAgentSession({ connectionId: owner, profile: defaultProfile })
    } else if (selected) {
      recordIaAction('ia.new_session.specific')
      startSimplifiedAgentSession(selected)
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
      recordIaAction('ia.new_session.command')
      request()
    }

    window.addEventListener(SIMPLIFIED_NEW_SESSION_EVENT, onRequest)

    return () => window.removeEventListener(SIMPLIFIED_NEW_SESSION_EVENT, onRequest)
  })

  return (
    <div className="flex w-full items-stretch">
      <Button
        aria-label={t.sidebar.nav['new-session']}
        className={
          agents.length > 1 ? 'min-h-8.5 min-w-0 flex-1 justify-start rounded-r-none' : 'min-h-8.5 w-full justify-start'
        }
        onClick={() => {
          recordAction('session.new', 'click')
          recordIaAction('ia.new_session.sidebar')
          request()
        }}
      >
        <Codicon name="add" />
        <span className="truncate">{t.sidebar.nav['new-session']}</span>
        {shortcut && (
          <span aria-hidden className="ml-auto shrink-0 text-[0.625rem] opacity-70">
            {shortcut}
          </span>
        )}
      </Button>
      {agents.length > 1 && (
        <Select
          onOpenChange={changeOpen}
          onValueChange={id => {
            picked.current = true
            setChoice({ context, id })
          }}
          open={open}
          value={value}
        >
          <Tip label={`${c.newSessionIn}: ${targetLabel}`}>
            <SelectPrimitive.Trigger asChild>
              <Button
                aria-label={`${c.newSessionIn}: ${targetLabel}`}
                className="rounded-l-none border-l border-primary-foreground/20"
              >
                {blank ? (
                  <Codicon name="circle-outline" />
                ) : (
                  selected && (
                    <span className="rounded-sm bg-background p-0.5">
                      <ProfileGlyph
                        aria-hidden
                        color={resolveProfileColor(selected.profile, colors)}
                        isDefault={selected.profile === 'default'}
                        name={selected.label}
                      />
                    </span>
                  )
                )}
                <Codicon name={open ? 'chevron-up' : 'chevron-down'} size={12} />
              </Button>
            </SelectPrimitive.Trigger>
          </Tip>
          <SelectContent align="end" className="w-72">
            <SelectGroup>
              <SelectLabel>{c.newSessionIn}</SelectLabel>
              {ranked.map(agent => (
                <SelectItem
                  aria-label={`${agent.label} · ${agent.gateway}`}
                  aria-labelledby={undefined}
                  key={agent.id}
                  value={agent.id}
                >
                  <span className="flex items-center gap-2">
                    <ProfileGlyph
                      aria-hidden
                      color={resolveProfileColor(agent.profile, colors)}
                      isDefault={agent.profile === 'default'}
                      name={agent.label}
                    />
                    <span className="flex min-w-0 flex-col">
                      <span className="truncate">{agent.label}</span>
                      <span className="truncate text-xs text-(--ui-text-tertiary)">{agent.gateway}</span>
                    </span>
                  </span>
                </SelectItem>
              ))}
              <SelectSeparator />
              <SelectItem value="blank">
                <span className="flex items-center gap-2">
                  <Codicon name="circle-outline" />
                  <span className="flex flex-col">
                    <span>{c.blank}</span>
                    <span className="text-xs text-(--ui-text-tertiary)">{c.blankHint}</span>
                  </span>
                </span>
              </SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
