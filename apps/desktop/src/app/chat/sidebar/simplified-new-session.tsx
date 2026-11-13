import { useStore } from '@nanostores/react'
import { Select as SelectPrimitive } from 'radix-ui'
import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { KbdCombo } from '@/components/ui/kbd'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectSeparator } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { useI18n } from '@/i18n'
import { ChevronDown, Message as IconMessage } from '@/lib/icons'
import { recordAction } from '@/store/desktop-metrics'
import { recordIaAction } from '@/store/ia-metrics'
import { $bindings, bindingsFor } from '@/store/keybinds'
import { $activeGatewayProfile, $newChatRoute, $profiles } from '@/store/profile'
import { knownSessionOwner } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'
import { claimSimplifiedNewSession } from '@/store/simplified-new-session'

import { tileOwnerRoute } from '../session-tile-owner'

import { useNewSessionCopy } from './simplified-new-session-copy'
import { rankNewSessionAgents, startSimplifiedAgentSession } from './simplified-new-session-model'
import { useSimplifiedAgents } from './use-simplified-agents'

/** Figma's three-letter agent chip ("DEF", "OPS"): identity without a saturated avatar. */
function AgentTag({ label }: { label: string }) {
  const tag = label
    .replace(/[^\p{L}\p{N}]/gu, '')
    .slice(0, 3)
    .toLocaleUpperCase()

  return (
    <span aria-hidden className="new-session-tag">
      {tag || '·'}
    </span>
  )
}

export function SimplifiedNewSession({ sessions, onFallback }: { sessions: SessionInfo[]; onFallback: () => void }) {
  const { t } = useI18n()
  const c = useNewSessionCopy()
  const { agents, owner } = useSimplifiedAgents([])
  const profile = useStore($activeGatewayProfile)
  const profiles = useStore($profiles)
  const focusedSessionId = useStore($focusedStoredSessionId)
  const tiles = useStore($sessionTiles)

  // A draft on screen is owned by the agent pinned for it, not by the ambient profile.
  const draft = useStore($newChatRoute)

  const viewedOwner = focusedSessionId
    ? (tileOwnerRoute(tiles, sessions, focusedSessionId) ?? knownSessionOwner(sessions, focusedSessionId))
    : (draft ?? undefined)

  const viewedRoute = typeof viewedOwner === 'string' ? { connectionId: owner, profile: viewedOwner } : viewedOwner
  const viewedProfile = viewedRoute?.profile ?? profile
  const viewedConnection = viewedRoute?.connectionId ?? owner
  const bindings = useStore($bindings)
  const shortcut = bindingsFor('session.new', bindings)[0]
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
  const ranked = rankNewSessionAgents(agents, sessions, viewedConnection, viewedProfile)
  const defaultProfile = profiles.find(item => item.is_default)?.name ?? 'default'

  // The button's own target: the agent of the chat being viewed (so, after a pick, the agent just
  // used), else the most recently used one.
  const selected =
    agents.find(agent => agent.connectionId === viewedConnection && agent.profile === viewedProfile) ?? ranked[0]

  const targetLabel = selected ? `${selected.label} · ${selected.gateway}` : c.newSessionIn

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

    if (selected) {
      recordIaAction('ia.new_session.specific')
      startSimplifiedAgentSession(selected)
    } else {
      onFallback()
    }
  }

  // A pick IS the action: it opens a new chat with that agent (or a blank one) at once, and never
  // re-targets the button behind the user's back.
  function pick(id: string) {
    picked.current = true
    const agent = agents.find(item => item.id === id)

    if (agent) {
      recordIaAction('ia.new_session.specific')
      startSimplifiedAgentSession(agent)
    } else {
      recordIaAction('ia.new_session.default')
      startSimplifiedAgentSession({ connectionId: owner, profile: defaultProfile })
    }
  }

  const onCommand = useEffectEvent(() => {
    recordIaAction('ia.new_session.command')
    request()
  })

  useEffect(() => claimSimplifiedNewSession(() => onCommand()), [])

  return (
    <div className="portal-new-session" data-slot="new-session-split" ref={splitButton}>
      <Button
        aria-label={t.sidebar.nav['new-session']}
        className="portal-new-session-main"
        onClick={() => {
          recordAction('session.new', 'click')
          recordIaAction('ia.new_session.sidebar')
          request()
        }}
        variant="ghost"
      >
        <IconMessage aria-hidden className="portal-new-session-icon" />
        <span className="portal-new-session-label">{t.sidebar.nav['new-session']}</span>
        {shortcut && <KbdCombo aria-hidden className="portal-kbd" combo={shortcut} data-slot="new-session-shortcut" />}
      </Button>
      {agents.length > 1 && (
        <Select
          onOpenChange={changeOpen}
          onValueChange={pick}
          open={open}
          // Never a held selection: picking the button's current agent must fire too.
          value=""
        >
          <Tip className={open ? 'hidden' : undefined} label={`${c.newSessionIn}: ${targetLabel}`}>
            <SelectPrimitive.Trigger asChild>
              <Button
                aria-label={`${c.newSessionIn}: ${targetLabel}`}
                className="portal-new-session-picker"
                variant="ghost"
              >
                {selected && <AgentTag label={selected.label} />}
                <ChevronDown aria-hidden className="portal-section-chevron" />
              </Button>
            </SelectPrimitive.Trigger>
          </Tip>
          <SelectContent align="end" className="new-session-picker" style={{ width: pickerWidth }}>
            <SelectGroup>
              <SelectLabel className="new-session-picker-label">{c.newSessionIn}</SelectLabel>
              {ranked.map(agent => (
                <SelectItem
                  aria-label={`${agent.label} · ${agent.gateway}`}
                  aria-labelledby={undefined}
                  className="new-session-option"
                  key={agent.id}
                  textValue={agent.label}
                  value={agent.id}
                >
                  <AgentTag label={agent.label} />
                  <span className="new-session-option-name">{agent.label}</span>
                  <span className="new-session-option-meta">{agent.gateway}</span>
                </SelectItem>
              ))}
              <SelectSeparator />
              <SelectItem className="new-session-option" textValue={c.blank} value="blank">
                <span aria-hidden className="new-session-option-tile">
                  <Codicon name="circle-outline" />
                </span>
                <span className="new-session-option-name">
                  {c.blank}
                  <span className="new-session-option-hint">{c.blankHint}</span>
                </span>
              </SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
