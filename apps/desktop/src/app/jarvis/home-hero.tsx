import './home-orbit.css'

import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { ImageIcon, LayoutDashboard, Mic, Newspaper, Search, Sparkles, Square } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $character } from '@/store/character'
import { requestBriefing } from '@/store/composer'
import { $liveVoiceChoice, $voiceEngine } from '@/store/voice-prefs'

import { requestComposerInsert } from '../chat/composer/focus'

import { greetingFor } from './characters'
import { JarvisCore } from './core'
import { DesktopOrbToggle } from './desktop-orb-toggle'
import { $jarvisRailVisible } from './focus-mode'
import { plasmaTone } from './plasma'
import { JarvisQuickAccess } from './quick-access'
import { $jarvisUi } from './store'
import type { JarvisVoiceState } from './types'
import { useLiveAutostart } from './use-live-autostart'

type IconComponent = React.ComponentType<{ className?: string }>

type HomeAction = 'analyze' | 'automate' | 'generate' | 'plan'

/** The four ways in under the orb; each one starts a request in the composer. */
const HOME_ACTIONS: readonly { icon: IconComponent; id: HomeAction }[] = [
  { icon: Sparkles, id: 'plan' },
  { icon: Search, id: 'analyze' },
  { icon: ImageIcon, id: 'generate' },
  { icon: LayoutDashboard, id: 'automate' }
]

/** Inline, the quick-access list keeps to the design's three entries. */
const INLINE_QUICK_ACCESS = 3

const FOCUS_RING =
  'outline-none focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--ui-accent)'

export interface JarvisHomeHeroProps {
  className?: string
  connected: boolean
  listening: boolean
  onStartListening: () => void
  onStopListening?: () => void
  profileDisplayName?: string
}

/**
 * The orb follows backend and microphone state. Action chips fill the composer
 * for review; quick access lives in the rail or here when there is no rail.
 */
export function JarvisHomeHero({
  className,
  connected,
  listening,
  onStartListening,
  onStopListening,
  profileDisplayName
}: JarvisHomeHeroProps) {
  useLiveAutostart(connected)
  const { t } = useI18n()
  const copy = t.jarvisShell.home
  const briefingCopy = t.jarvisShell.briefing
  const state = useStore($jarvisUi)
  const railVisible = useStore($jarvisRailVisible)
  // "default" is the machine's unnamed profile, not a person: greet without it.
  const rawName = profileDisplayName?.trim()
  const name = rawName && rawName.toLowerCase() !== 'default' ? rawName : undefined
  const character = useStore($character)
  const greeting = greetingFor(character, new Date(), copy.greetings)
  // The conversation owns the orb's voice state (`publishJarvisVoiceState`):
  // Live voice reports listening/speaking from the renderer, so masking the
  // state with `listening` would freeze the orb while Gemini actually speaks.
  // The fallback still shows a just-opened conversation as listening.
  const orbVoice: JarvisVoiceState = listening && state.voice === 'idle' ? 'listening' : state.voice
  const tone = plasmaTone(orbVoice, state.task.phase)

  const hint = !connected
    ? copy.offline
    : tone === 'working' && state.task.phase === 'planning'
      ? copy.orbStatus.thinking
      : copy.orbStatus[tone]

  const engine = useStore($voiceEngine)
  const live = useStore($liveVoiceChoice)

  const voiceName =
    engine === 'realtime' ? `${copy.voiceEngine[live.provider]} · ${live.model}` : copy.voiceEngine.classic

  return (
    // Laid out by the chat column's width, not the window's: the sidebar, the
    // rail and a split pane all eat into it.
    <div className="@container flex w-full justify-center">
      <section
        aria-labelledby="jarvis-home-title"
        className={cn(
          'jarvis-home relative flex w-full max-w-4xl flex-col items-center gap-2 px-3 pt-3 pb-24 text-center [--jarvis-hero-size:min(468px,42vh,88cqw)] @2xl:[--jarvis-hero-size:min(550px,46vh,70cqw)]',
          className
        )}
        data-testid="jarvis-home-hero"
      >
        <div className="jarvis-home__caption flex flex-col items-center gap-2 rounded-2xl px-5 py-3">
          <h1
            className="text-2xl font-semibold leading-tight tracking-tight text-(--ui-text-primary) @2xl:text-3xl"
            id="jarvis-home-title"
          >
            {name ? `${name}, ${greeting.charAt(0).toLowerCase()}${greeting.slice(1)}` : greeting}
          </h1>
          <p className="text-base text-(--ui-text-secondary) @2xl:text-lg">{copy.subtitle}</p>
        </div>

        <div aria-label={copy.actionsLabel} className="jarvis-home__actions flex flex-wrap justify-center gap-2" role="group">
          {HOME_ACTIONS.map(({ icon: Icon, id }) => (
            <button
              className={cn(
                'jarvis-glass jarvis-glass-hover flex min-h-11 items-center gap-2 rounded-full px-3 text-xs font-medium text-(--ui-text-primary)',
                FOCUS_RING
              )}
              key={id}
              onClick={() => requestComposerInsert(copy.actions[id].prompt, { mode: 'prefix', target: 'main' })}
              type="button"
            >
              <Icon className="size-4 text-(--ui-accent)" />
              {copy.actions[id].label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-center gap-2">
          <DesktopOrbToggle compact />
          <Button
            aria-pressed={listening}
            className={cn('min-h-12 rounded-full px-7 text-base font-semibold', !listening && 'jarvis-cta')}
            disabled={!connected}
            onClick={() => (listening ? onStopListening?.() : onStartListening())}
            type="button"
            variant={listening ? 'secondary' : 'default'}
          >
            {listening ? <Square /> : <Mic />}
            {listening ? copy.stopTalking : copy.talk}
          </Button>
          <Button
            className="min-h-11 jarvis-glass jarvis-glass-hover rounded-full px-4 text-(--ui-text-primary)"
            disabled={!connected}
            onClick={() => requestBriefing({ speak: true })}
            title={briefingCopy.buttonHint}
            type="button"
            variant="secondary"
          >
            <Newspaper />
            {briefingCopy.button}
          </Button>
        </div>

        <div
          className="jarvis-home__orb relative my-2 grid w-(--jarvis-hero-size) max-w-full shrink-0 place-items-center"
          data-tone={tone}
        >
          <span aria-hidden="true" className="jarvis-home__orbit" />
          <JarvisCore live taskPhase={state.task.phase} variant="hero" voice={orbVoice} />
        </div>

        <div
          aria-live="polite"
          className="jarvis-home__caption flex items-center gap-2 rounded-full px-3 py-1 text-xs text-(--ui-text-secondary)"
          data-testid="jarvis-home-status"
        >
          <span
            aria-hidden="true"
            className={cn(
              'size-2 rounded-full',
              !connected
                ? 'bg-(--ui-text-tertiary)'
                : tone === 'idle'
                  ? 'bg-emerald-400'
                  : 'animate-pulse bg-(--ui-accent)'
            )}
          />
          {hint}
        </div>
        <p
          className="jarvis-home__caption rounded-full px-3 py-1 text-xs text-(--ui-text-secondary)"
          data-testid="jarvis-home-voice-engine"
          title={copy.voiceEngine.hint}
        >
          {copy.voiceEngine.label}: <span className="text-(--ui-text-secondary)">{voiceName}</span>
        </p>

        {railVisible ? null : (
          <div className="flex w-full max-w-sm flex-col gap-2 pt-2 text-left">
            <p className="text-xs font-medium uppercase tracking-[0.2em] text-(--ui-text-tertiary)">
              {copy.shortcutsLabel}
            </p>
            <JarvisQuickAccess connected={connected} label={copy.shortcutsLabel} limit={INLINE_QUICK_ACCESS} />
          </div>
        )}
      </section>
    </div>
  )
}
