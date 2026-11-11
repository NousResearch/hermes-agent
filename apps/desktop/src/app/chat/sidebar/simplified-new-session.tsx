import { useStore } from '@nanostores/react'
import { Select as SelectPrimitive } from 'radix-ui'
import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectSeparator } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { useI18n } from '@/i18n'
import { useKeybindHint } from '@/lib/keybinds/use-keybind-hint'
import { resolveProfileColor } from '@/lib/profile-color'
import { recordAction } from '@/store/desktop-metrics'
import { recordIaAction } from '@/store/ia-metrics'
import { $activeGatewayProfile, $profileColors, $profiles } from '@/store/profile'
import { knownSessionOwner } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'
import { claimSimplifiedNewSession } from '@/store/simplified-new-session'

import { tileOwnerRoute } from '../session-tile-owner'

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
  const tiles = useStore($sessionTiles)

  const viewedOwner = focusedSessionId
    ? (tileOwnerRoute(tiles, sessions, focusedSessionId) ?? knownSessionOwner(sessions, focusedSessionId))
    : undefined

  const viewedRoute = typeof viewedOwner === 'string' ? { connectionId: owner, profile: viewedOwner } : viewedOwner
  const viewedProfile = viewedRoute?.profile ?? profile
  const viewedConnection = viewedRoute?.connectionId ?? owner
  const shortcut = useKeybindHint('session.new')
  const [open, setOpen] = useState(false)
  const splitButton = useRef<HTMLDivElement>(null)
  const [pickerWidth, setPickerWidth] = useState<number>()

  const measure = useCallback((entries: readonly ResizeObserverEntry[]) => {
    if (entries[0]) {
      setPickerWidth(entries[0].contentRect.width)
    }
  }, [])

  useResizeObserver(measure, splitButton)
  const picked = useRef(false)
  const context = JSON.stringify([owner, profile, focusedSessionId])
  const [choice, setChoice] = useState<{ context: string; id: string } | null>(null)
  const ranked = rankNewSessionAgents(agents, sessions, viewedConnection, viewedProfile)
  const defaultProfile = profiles.find(item => item.is_default)?.name ?? 'default'

  // A picker choice belongs to the chat being viewed, never a different gateway or session.
  const choiceId = choice?.context === context ? choice.id : null
  const blank = choiceId === 'blank'

  const selected =
    agents.find(agent => agent.id === choiceId) ??
    agents.find(agent => agent.connectionId === viewedConnection && agent.profile === viewedProfile) ??
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
    picked.current = true
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

  const onCommand = useEffectEvent(() => {
    recordIaAction('ia.new_session.command')
    request()
  })

  useEffect(() => claimSimplifiedNewSession(() => onCommand()), [])

  return (
    <div className="@container/new-session flex min-h-9.5 w-full items-stretch" data-slot="new-session-split" ref={splitButton}>
      <Button
        aria-label={t.sidebar.nav['new-session']}
        className={agents.length > 1 ? 'min-w-0 flex-1 justify-start rounded-r-none' : 'w-full justify-start'}
        onClick={() => {
          recordAction('session.new', 'click')
          recordIaAction('ia.new_session.sidebar')
          request()
        }}
      >
        <Codicon name="add" />
        <span className="truncate">{t.sidebar.nav['new-session']}</span>
        {shortcut && (
          <span
            aria-hidden
            className="ml-auto hidden shrink-0 rounded bg-primary-foreground/15 px-1 py-0.5 text-[0.625rem] @min-[15rem]/new-session:inline-flex"
            data-slot="new-session-shortcut"
          >
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
          <Tip className={open ? 'hidden' : undefined} label={`${c.newSessionIn}: ${targetLabel}`}>
            <SelectPrimitive.Trigger asChild>
              <Button
                aria-label={`${c.newSessionIn}: ${targetLabel}`}
                className="rounded-l-none border-l border-primary-foreground/20 data-[state=open]:bg-primary/85"
              >
                {blank ? (
                  <Codicon name="circle-outline" />
                ) : (
                  selected && (
                    <span className="rounded-sm bg-background p-0.5" data-slot="new-session-avatar">
                      <FigmaAgentIcon color={resolveProfileColor(selected.profile, colors)} />
                    </span>
                  )
                )}
                <Codicon name={open ? 'chevron-up' : 'chevron-down'} size={12} />
              </Button>
            </SelectPrimitive.Trigger>
          </Tip>
          <SelectContent align="end" className="new-session-picker w-72 max-w-[calc(100vw-1rem)]" style={{ width: pickerWidth }}>
            <SelectGroup>
              <SelectLabel>{c.newSessionIn}</SelectLabel>
              {ranked.map(agent => (
                <SelectItem
                  aria-label={`${agent.label} · ${agent.gateway}`}
                  aria-labelledby={undefined}
                  className="min-h-11 [&>span:last-child]:min-w-0 [&>span:last-child]:flex-1 data-[state=checked]:bg-accent data-[state=checked]:text-accent-foreground"
                  key={agent.id}
                  textValue={agent.label}
                  value={agent.id}
                >
                  <span className="flex items-center gap-2">
                    <FigmaAgentIcon color={resolveProfileColor(agent.profile, colors)} />
                    <span className="flex min-w-0 flex-col">
                      <span className="truncate">{agent.label}</span>
                      <span className="truncate text-xs text-(--ui-text-tertiary)">{agent.gateway}</span>
                    </span>
                  </span>
                </SelectItem>
              ))}
              <SelectSeparator />
              <SelectItem
                className="min-h-11 [&>span:last-child]:min-w-0 [&>span:last-child]:flex-1 data-[state=checked]:bg-accent data-[state=checked]:text-accent-foreground"
                textValue={c.blank}
                value="blank"
              >
                <span className="flex items-center gap-2">
                  <span aria-hidden className="grid size-6 shrink-0 place-items-center" data-slot="new-session-option-avatar">
                    <Codicon name="circle-outline" />
                  </span>
                  <span className="flex min-w-0 flex-col">
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
