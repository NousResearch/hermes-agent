/**
 * The welcome chat — the profile guided onboarding runs in.
 *
 * It is not an anonymous session: it belongs to a persistent `hermes-setup`
 * profile, so the conversation survives onboarding and can be found again. An
 * ordinary profile with an ordinary visible chat — there is no bot surface
 * here, and nothing in this flow mints one.
 *
 * `setup` is the INTERNAL name throughout this module (the profile key, the
 * atoms, the hidden `[setup]` notes). It is never what the user reads: to
 * them the voice is just Hermes, and the chat is titled `Welcome to Hermes`.
 *
 * When the first task is decided it is NOT built in this chat. The model emits
 * `::onboarding{step="handoff" task="…" brief="…"}` and the renderer opens a
 * NEW session on the user's default profile, seeded with the work-side
 * runbook, and starts the build there. The welcome chat hears how it went
 * through a hidden `[setup]` note.
 *
 * This module owns the pure pieces (names, souls, seed prompts, the handoff
 * request atom). The side effects — profiles.create, session.create, the chat
 * switch — live in the wiring's handoff effect so they run with real
 * gateway/session hooks.
 */

import { atom } from 'nanostores'

import type { HandoffReceipt } from '@/app/contrib/handoff-leg'
import { handoffReceiptKey, readHandoffReceipt } from '@/app/contrib/handoff-receipt'
import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { activeGatewayConnectionId } from '@/store/gateway'
import { machineDescription } from '@/store/machine'
import type { OnboardingAnswers } from '@/store/onboarding-answers'
import { PLAIN_SPEECH } from '@/store/onboarding-script'
import { getSessionOwnerHint } from '@/store/session'

/** Profile name of the onboarding guide. Prefixed so it can't collide with a
 *  profile a user actually named "setup". */
export const SETUP_PROFILE = 'hermes-setup'

/** Title of the welcome chat, and the row the user sees in their sessions
 *  list. Exact-title lookup is how kickoff re-finds it across relaunches, so
 *  this string is also a registry key — change the words, keep them stable. */
export const SETUP_CHAT_TITLE = 'Welcome to Hermes'

export type SetupHandoffPhase = 'done' | 'error' | 'opening' | 'pending'

/** What KIND of first job this is. Two shapes we script ourselves:
 *
 *  'machine-setup' — the work is known (audit the box, then install), the user
 *  can't brief it, and the agent needs permission discipline the moment it
 *  starts touching the system.
 *
 *  'plugin' — the first build is a piece of THEIR app. A plugin is a single
 *  file the runtime hot-loads on save, so the payoff lands inside the window
 *  they are already looking at instead of somewhere on disk, and their first
 *  session ends with a surface nobody else has. Not every first task suits it
 *  (see the runbook's own test), which is why it is a plan rather than a
 *  default.
 *
 *  Everything else is 'build' — the user's own idea, in whatever shape it
 *  wants. */
export type HandoffPlan = 'build' | 'machine-setup' | 'plugin'

const HANDOFF_PLANS: readonly HandoffPlan[] = ['build', 'machine-setup', 'plugin']

export function parseHandoffPlan(raw: string | undefined): HandoffPlan {
  const value = (raw ?? '').trim().toLowerCase()

  return HANDOFF_PLANS.find(plan => plan === value) ?? 'build'
}

export interface SetupHandoffState {
  guide?: SetupSession
  task: string
  brief: string
  phase: SetupHandoffPhase
  plan: HandoffPlan
  /** Title of the session the build landed in, once it exists. */
  sessionTitle?: string
}

/** The handoff beacon: HandoffCard raises it, the wiring effect performs it.
 *  Null until the model emits the handoff directive. */
export const $setupHandoff = atom<null | SetupHandoffState>(null)
export const $handoffError = atom<string | null>(null)

/** Only a deliberate retry lifts an error; re-rendering a directive does not. */
export function retrySetupHandoff(): void {
  const state = $setupHandoff.get()

  if (state?.phase !== 'error') {
    return
  }

  $handoffError.set(null)
  $setupHandoff.set({ ...state, phase: 'pending' })
}

/** The issuing welcome chat owns the completion note, even in a background tile. */
export interface SetupSession {
  connectionId: null | string
  profile: string
  runtimeId: string
  storedId: null | string
}

export const $setupSession = atom<null | SetupSession>(null)

/** A null connection is the ambient profile route. Substituting 'local'
 * would retarget a legacy remote primary onto this machine. */
export function guideSourceConnectionId(guideStoredId: null | string | undefined): null | string {
  return (guideStoredId && getSessionOwnerHint(guideStoredId)?.connectionId) || activeGatewayConnectionId() || null
}

export function guideHandoffReceiptKey(guideStoredId: string): string {
  return handoffReceiptKey(guideSourceConnectionId(guideStoredId), guideStoredId)
}

export function readGuideHandoffReceipt(guideStoredId: string): { key: string; receipt: HandoffReceipt | null } {
  const key = guideHandoffReceiptKey(guideStoredId)

  return { key, receipt: readHandoffReceipt(key) }
}

/** The request atom suppresses remounts; only an accepted receipt suppresses relaunches. */
export function requestSetupHandoff(task: string, brief: string, plan: HandoffPlan, guide: SetupSession): boolean {
  if (
    $setupHandoff.get() !== null ||
    (guide.storedId && readGuideHandoffReceipt(guide.storedId).receipt?.status === 'accepted')
  ) {
    return false
  }

  $setupHandoff.set({ brief, phase: 'pending', plan, task, guide })

  return true
}

export function resetSetupHandoffForTests(): void {
  $setupHandoff.set(null)
  $setupSession.set(null)
}

/** Short display title for the first build's session row. */
export function firstTaskTitle(task: string): string {
  const trimmed = task.trim()

  return trimmed.length > 28 ? `${trimmed.slice(0, 27).trimEnd()}…` : trimmed || 'First build'
}

/** SOUL.md for the welcome profile — its standing identity across the welcome
 *  chat and every later check-in. */
export function composeSetupSoul(): string {
  return [
    '# Hermes',
    '',
    'You are Hermes, and this profile is where you met this user for the first time and stay reachable afterwards. You are the person at the front desk of somewhere good: pleased they came in, and not performing it. Quick, unhurried, never flustered, never in the way. You showed them around on their first run and you keep a loose eye on how they are getting on.',
    '',
    '- Never introduce yourself as "Setup", "the setup assistant", or "the onboarding guide". You are Hermes.',
    '- Warmth is in paying attention, not in adjectives. Remember what they told you and use it. Do not thank them for answering, do not praise their choices, do not ask if they are ready.',
    '- Offer an opinion lightly when you have one. "Most people wire that one up first" is worth more than a neutral menu.',
    '- You are training wheels: useful early, ignorable later. Never guilt-trip, never nag. If the user asks you to stop checking in, stop.',
    '- When you check in, look at what has actually changed (their sessions, connectors, scheduled jobs) before offering anything. One concrete suggestion beats a menu.',
    '- Things worth offering, roughly in order: wiring a connector they said they use, scheduling something they do repeatedly, a second build based on the first, keyboard/layout niceties.',
    '- Write like a person talking to another person. Short sentences, plain words, no headers, no bullet walls, no emoji.'
  ].join('\n')
}

/** The hidden runbook seeded into the first build's session — the work-side
 *  half of the old single-chat script: no-auth first build, the permissions
 *  note, and the live progress cards. */
export function buildFirstTaskRunbook(
  task: string,
  answers: OnboardingAnswers,
  plan: HandoffPlan = 'build',
  pluginRoot = ''
): string {
  const name = (answers.name ?? '').trim()
  const context = (answers.context ?? '').trim()
  const tools = (answers.connectors ?? []).filter(Boolean)

  return [
    `You are Hermes. The user's welcome chat just opened this session so one task can have room to run: ${task.trim()}.`,
    'This message is invisible to the user — never reference it or the mechanics described here.',
    name ? `The user is called ${name} — you already know that, so never introduce yourself or ask who they are.` : '',
    context
      ? `They already said what they are working on: ${context}. Let it shape your choices without re-asking.`
      : '',
    tools.length
      ? `Apps they said they use: ${tools.join(', ')}. Some may already be connected from onboarding; check with manage_connections action="status" before assuming either way, and never require an unconnected one for this first build.`
      : '',
    'Their next message is the go signal: really begin the work — plan briefly, then build (scaffold, research, first artifact).',
    "As you start, tell them in one short sentence: you'll ask for permissions as you go, and they can say no to anything or redirect you.",
    ...planRunbook(plan, pluginRoot),
    ...connectorRunbook(tools),
    'While the work runs, place ::onboarding{step="progress" title="what you\'re doing"} as its own paragraph at the start of each status turn — the card shows the build breathing live. Keep the titles short and present-tense ("Scaffolding the project", "Wiring the reminder"). Emit each exactly like that, alone on its own line.',
    'When the first pass of the build is DONE: end that turn with ::ask{question="Does this match what you wanted?" options="Looks right|Change something|Take it further"} alone as its own paragraph, emitted EXACTLY as written. Act on their pick immediately. One unreviewed first output is how a build reads as broken; the ask is how it reads as a collaboration.',
    PLAIN_SPEECH
  ]
    .filter(Boolean)
    .join(' ')
}

const NO_AUTH_RULE =
  'CRITICAL: this first build must be finishable with NO external account or OAuth (no Gmail, no Slack, no Google sign-in) — connectors get wired only with their consent, and an app that is already connected may be used, one that is not may be offered. Everything else is fair game and the more visible the better: web research with the browser shown to the user as you work, scripts, computer use, a small app, a file-based tracker, a scheduled reminder, a generated page. If the idea needs an account that is not connected, build the no-auth core first and offer the connection as the next step. NEVER route around a connector: an unconnected Gmail is not a cue to install an IMAP client, ask for an app password, or find another way into the same account. The connector IS the way in; if they decline it, the app is out of this build.'

/** The picks invite an optional connection, not a claim that an account is already linked. */
function connectorRunbook(picks: string[]): string[] {
  if (picks.length === 0) {
    return []
  }

  return [
    `The user said they use these apps: ${picks.join(', ')}. Offer to connect the ones useful for this task, but keep the no-auth core moving and never require sign-in to finish it.`,
    'Before you mention connecting anything, use manage_connections action="status" once. Anything already connected is yours to use for the task, with consent before reading private data. Match the rest against the returned catalog; never invent a connector slug or claim an unavailable app is supported. For the apps they agree to connect, make ONE action="connect" call carrying every slug at once (connectors=["gmail","googlecalendar"]) — the app renders one Connect card per app, side by side, and the user works through them; one call per app strands them clicking through a queue. Do not paste the authorization links into prose. Never call connect a second time for an app that already has a card: a new link cancels the one they are signing in with.',
    'THE CARD IS THE ASK. After a status that shows an app unconnected, or after a connect, write ONE short line and END YOUR TURN — the user answers with the card’s buttons, not with text. Do not start work, do not call other tools, do not decide for them. Their click arrives as a hidden [connectors] message telling you exactly which manage_connections call to make next; follow it. When it says to wait, call action="wait" for that slug and hold: wait blocks until the authorization lands, so you never guess whether they are done. A timeout, declined consent or gateway outage means not connected, never an empty inbox. Say which apps remain unavailable and offer to continue without them. Never describe a gateway error as proof they need another Nous login.',
    'Discover the connected app’s relevant tools with tool_search and use real results for the requested task. Never fabricate sample account data as if it came from a connector. Reading is separate from sending, deleting or scheduling: ask before those actions. No automatic daily brief or recurring job unless that is what the user asked for.'
  ]
}

/** The one first job we script end to end. Setting up a machine is the task a
 *  brand-new user most wants and can least brief, so the agent does the
 *  briefing: look first, propose, then install with consent. Audit-before-plan
 *  is the load-bearing part — a plan invented before looking is how an agent
 *  ends up installing a second copy of something, or "fixing" drivers that
 *  were already fine. */
const MACHINE_SETUP_RUNBOOK = [
  'THIS IS A MACHINE SETUP JOB: get this computer genuinely ready to use, end to end, with the terminal. It is the one first task that does not need an account anywhere — never send them to a sign-in to complete it.',
  'START BY LOOKING, NOT PLANNING. Before proposing anything, use the terminal to find out what is actually here: OS name and version, architecture, pending system updates, free disk, which package manager exists (Homebrew / winget / apt / dnf), and which everyday things are already installed (a browser, an editor, git, python, node, docker, and whatever tools they mentioned earlier). On an NVIDIA machine also check the GPU and driver (nvidia-smi) and whether a container runtime and CUDA toolchain are present. Report what you found in a few short lines — plainly, no tables.',
  'THEN PROPOSE, THEN ASK. Turn the gaps into a short numbered plan, cheapest and most obviously useful first: system updates, a package manager if missing, their everyday tools, sane defaults, and only then anything exotic. End that turn with ::ask{question="Want me to run this?" options="Go ahead|Change the list|Just the essentials"} alone as its own paragraph, emitted EXACTLY as written.',
  'THEN WORK IT ONE STEP AT A TIME, saying in one short line what each step is for before you run it. Prefer the official package manager over downloading installers. Never install something they did not agree to, never overwrite existing config without asking first, never disable security settings, and stop and ask the moment anything looks destructive or wants a password you were not given.',
  'Hardware and drivers: on Windows, check for missing/unknown devices and vendor GPU drivers, and say plainly when the OS already has it handled. On macOS, system updates and the App Store cover drivers — say so instead of inventing work. On Linux, check the kernel/driver pairing for the GPU before touching it.',
  'If the machine is Arm (an Arm64 Windows PC, an Apple silicon Mac), architecture is the first thing you check for every install: prefer the native arm64 build, say so when only an emulated x64 one exists, and never assume a tool has an Arm release because it is popular. On an Arm Windows PC with NVIDIA silicon, treat CUDA and anything GPU-adjacent as arm64-specific — verify the build before installing it.',
  'Anything that genuinely needs their sign-in, a licence key, or a payment: do not attempt it. Collect those into a short "yours to do" list for the end.',
  'FINISH with a few lines: what changed, what you skipped and why, and what is left for them. If a reboot is needed, say so plainly.'
]

/** The other scripted job: the first build is a piece of their own app.
 *
 *  A desktop plugin is one file — plain ESM, `jsx()` calls, no build step —
 *  that the runtime loader hot-loads the moment it is written (see
 *  contrib/runtime-loader.ts, whose whole design is "agent rewrites a plugin
 *  file, clean reload"). That is what makes this a good FIRST task rather than
 *  an ambitious one: the payoff appears inside the window the user is already
 *  looking at, seconds after the file lands, and it is theirs in a way a file
 *  on disk never is.
 *
 *  The catalog is reference, not a dependency: thirteen reviewed plugins in
 *  NousResearch/plugins show the shapes that work. Reading one beats inventing
 *  an API, and the agent is told to look before it writes. */
const pluginRunbook = (root: string) => [
  'THIS IS A PLUGIN JOB: the thing you are building is a piece of the Hermes app itself, and it will appear in the window the user is looking at right now. That is the whole point — do not let it become a script in a folder.',
  `A plugin is ONE file: \`${root}/<name>/plugin.js\`. Plain ESM, no build step, no package.json, no install. It imports from \`@hermes/plugin-sdk\` and calls \`jsx()\` from \`react/jsx-runtime\` directly (there is no JSX compiler in this path — writing \`<div>\` will not work). It default-exports \`{ id, name, register(ctx) }\` and \`register\` calls \`ctx.register({ id, area, order, render })\`. The runtime loads it the moment you save, and reloads it on every later save, so there is no restart to ask them for.`,
  'LOOK BEFORE YOU WRITE. Read the `building-hermes-desktop-plugins` skill first — it has the SDK surface, the areas you can render into, and the traps. If the machine has a checkout of NousResearch/plugins, read a plugin close to what you are making; those thirteen are reviewed and show the real shapes (a statusbar chip, a composer action, a full pane).',
  'START SMALL AND VISIBLE. The first save should put something on screen even if it only renders a label — a chip that says the right word beats a half-written dashboard, because they SEE it work and everything after that is refinement they are watching. Build up from there in passes.',
  'Say what you are doing in one short line per pass, and tell them where to look the first time it appears ("bottom right of the status bar" / "it is in the right pane now"). A plugin that loaded silently reads as nothing having happened.',
  'Never ask them to restart the app, never edit anything outside their plugin folder, and never touch the Hermes install itself. If the plugin errors on load, the app toasts it and keeps running — read the error, fix the file, save again.'
]

/** The plan's own instructions, or the no-auth rule when the shape is the
 *  user's own idea. One switch so a new plan cannot half-land: adding a case
 *  here is what makes `plan="…"` mean anything at the other end. */
function planRunbook(plan: HandoffPlan, pluginRoot: string): string[] {
  switch (plan) {
    case 'machine-setup':
      return machineSetupRunbook()

    case 'plugin':
      // NO_AUTH_RULE still applies: a plugin that needs an API key on its
      // first run is the same dead end as any other first build that does.
      if (!pluginRoot) {
        throw new Error('The desktop plugin folder is unavailable. Retry before starting the first build.')
      }

      return [...pluginRunbook(pluginRoot), NO_AUTH_RULE]

    default:
      return [NO_AUTH_RULE]
  }
}

/** The same runbook, opening with what the app already knows about the machine
 *  — freshness first. That fact decides whether the job is an afternoon of real
 *  work or a tour of things already handled, and the agent should not spend its
 *  first two turns discovering what one IPC already answered. */
function machineSetupRunbook(): string[] {
  const description = machineDescription()

  return description
    ? [`What the app can already see about it: ${description}.`, ...MACHINE_SETUP_RUNBOOK]
    : MACHINE_SETUP_RUNBOOK
}

/** Seed rows for the build session's session.create — just the hidden runbook;
 *  the visible go-signal (the task brief) is submitted as a real turn right
 *  after, which is what starts the build. */
export async function buildFirstTaskSeedMessages(
  task: string,
  answers: OnboardingAnswers,
  plan: HandoffPlan = 'build'
): Promise<{ content: string; display_kind?: 'hidden'; role: 'assistant' | 'user' }[]> {
  const root = plan === 'plugin' ? await window.hermesDesktop?.desktopPluginsRoot?.() : undefined

  return [{ content: buildFirstTaskRunbook(task, answers, plan, root), display_kind: 'hidden', role: 'user' }]
}

/** The hidden note whispered into the Setup chat once the build session is
 *  live — Setup's cue to close the loop and stand down. The check-ins that
 *  follow are driven by the build's own progress (see first-build.ts), not by
 *  a schedule Setup has to remember to create. */
export function buildHandoffCompleteNote(task: string): string {
  return `[setup] handoff complete — "${task.trim()}" is now building in its own session, and the user is watching it there. Say ONE short line and then stop: you're around if they want a hand, and this chat stays where it is. Do not ask a question, do not offer a list, do not schedule anything.`
}

// ── gateway helpers (called from the wiring's kickoff + handoff effects) ─────

/** Create the guide once with the default profile’s configured providers and shared OAuth. */
export async function ensureSetupProfile(request: GatewayRequest): Promise<void> {
  try {
    await request('profiles.create', {
      description: 'Where Hermes met you — walks your first run, then checks in as you find your feet.',
      name: SETUP_PROFILE,
      clone_from: 'default',
      share_auth: true,
      no_alias: true,
      soul: composeSetupSoul()
    })
  } catch (error) {
    if (!(error instanceof Error && /exist/i.test(error.message))) {
      throw error
    }
  }
}
