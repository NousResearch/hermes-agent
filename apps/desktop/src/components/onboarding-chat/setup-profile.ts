/**
 * The welcome chat that guided onboarding runs in, and the seed prompts for the first build session.
 *
 * The chat belongs to a persistent `hermes-setup` profile, so it survives onboarding and can be found again. `setup`
 * is the internal name throughout this module (the profile key, the atoms, the hidden `[setup]` notes); the user sees
 * only Hermes and the title `Welcome to Hermes`.
 *
 * This module holds the pure pieces: names, souls, seed prompts, and the handoff request atom. The side effects
 * (profiles.create, session.create, the chat switch) run in the wiring's kickoff and handoff effects, which hold the
 * gateway and session hooks.
 */

import { atom } from 'nanostores'

import type { HandoffReceipt } from '@/app/contrib/handoff-leg'
import { handoffReceiptKey, readHandoffReceipt } from '@/app/contrib/handoff-receipt'
import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { CONNECTOR_LEAD_ORDER } from '@/components/onboarding-chat/options'
import { connectorTitle } from '@/lib/connector-tools'
import { activeGatewayConnectionId } from '@/store/gateway'
import { machineDescription } from '@/store/machine'
import type { OnboardingAnswers } from '@/store/onboarding-answers'
import { PLAIN_SPEECH } from '@/store/onboarding-script'
import { getSessionOwnerHint } from '@/store/session'

/** Profile name of the onboarding guide. Prefixed so it cannot collide with a profile the user named "setup". */
export const SETUP_PROFILE = 'hermes-setup'

/** Title of the welcome chat, and the row the user sees in the sessions list. Kickoff re-finds the chat by exact
 *  title after a relaunch, so this string is also a lookup key. */
export const SETUP_CHAT_TITLE = 'Welcome to Hermes'

export type SetupHandoffPhase = 'done' | 'error' | 'opening' | 'pending'

/** Which runbook planRunbook() selects for the first build session. Set from the plan attribute on the model's
 *  handoff directive. */
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
  sessionTitle?: string
}

/** Set by HandoffCard, or restored from a saved receipt by the wiring's recovery effect. The wiring's handoff effect
 *  then advances phase. Null until the model emits the handoff directive. */
export const $setupHandoff = atom<null | SetupHandoffState>(null)
export const $handoffError = atom<string | null>(null)

/** Called only by the Retry control in HandoffCard and by the "Retry first build" toast, so a re-rendered handoff
 *  directive cannot clear the error. */
export function retrySetupHandoff(): void {
  const state = $setupHandoff.get()

  if (state?.phase !== 'error') {
    return
  }

  $handoffError.set(null)
  $setupHandoff.set({ ...state, phase: 'pending' })
}

/** Identifies the welcome chat that issued the handoff. The handoff wiring submits the completion note to this
 *  session, not to whichever session is active when the build starts. */
export interface SetupSession {
  connectionId: null | string
  profile: string
  runtimeId: string
  storedId: null | string
}

export const $setupSession = atom<null | SetupSession>(null)

/** Returns null for the ambient profile route. Returning 'local' instead would retarget a legacy remote primary onto
 * this machine. */
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

export function firstTaskTitle(task: string): string {
  const trimmed = task.trim()

  return trimmed.length > 28 ? `${trimmed.slice(0, 27).trimEnd()}…` : trimmed || 'First build'
}

/** SOUL.md for the welcome profile. It applies to the welcome chat and to every later check-in. */
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

export function buildFirstTaskRunbook(
  task: string,
  answers: OnboardingAnswers,
  plan: HandoffPlan = 'build',
  pluginRoot = ''
): string {
  const name = (answers.name ?? '').trim()
  const context = (answers.context ?? '').trim()
  const tools = (answers.connectors ?? []).filter(slug => CONNECTOR_LEAD_ORDER.includes(slug))
  // Machine setup needs no account anywhere; every other plan connects the
  // picked apps before it does anything else (D85).
  const connectFirst = tools.length > 0 && plan !== 'machine-setup'

  return [
    `You are Hermes. The user's welcome chat just opened this session so one task can have room to run: ${task.trim()}.`,
    'This message is invisible to the user — never reference it or the mechanics described here.',
    name ? `The user is called ${name} — you already know that, so never introduce yourself or ask who they are.` : '',
    context
      ? `They already said what they are working on: ${context}. Let it shape your choices without re-asking.`
      : '',
    tools.length && !connectFirst
      ? `Apps they said they use: ${tools.map(connectorTitle).join(', ')}. Some may already be connected from onboarding; check with manage_connections action="status" before assuming either way, and never require an unconnected one for this first build.`
      : '',
    connectFirst
      ? 'Their next message is the go signal. Before any plan and before any other tool, connect their apps as the CONNECT FIRST section says; the work itself starts the moment the wait returns or they tell you to start.'
      : 'Their next message is the go signal: really begin the work — plan briefly, then build (scaffold, research, first artifact).',
    "As you start, tell them in one short sentence: you'll ask for permissions as you go, and they can say no to anything or redirect you.",
    ...planRunbook(plan, pluginRoot, connectFirst),
    ...(connectFirst ? connectFirstRunbook(tools) : []),
    'While the work runs, place ::onboarding{step="progress" title="what you\'re doing"} as its own paragraph at the start of each status turn — the card shows the build breathing live. Keep the titles short and present-tense ("Scaffolding the project", "Wiring the reminder"). Emit each exactly like that, alone on its own line.',
    'When the first pass of the build is DONE: end that turn with ::ask{question="Does this match what you wanted?" options="Looks right|Change something|Take it further"} alone as its own paragraph, emitted EXACTLY as written. Act on their pick immediately. One unreviewed first output is how a build reads as broken; the ask is how it reads as a collaboration.',
    PLAIN_SPEECH
  ]
    .filter(Boolean)
    .join(' ')
}

const NO_AUTH_RULE =
  'CRITICAL: this first build must be finishable with NO external account or OAuth (no Gmail, no Slack, no Google sign-in) — connectors get wired only with their consent, and an app that is already connected may be used, one that is not may be offered. Everything else is fair game and the more visible the better: web research with the browser shown to the user as you work, scripts, computer use, a small app, a file-based tracker, a scheduled reminder, a generated page. If the idea needs an account that is not connected, build the no-auth core first and offer the connection as the next step. NEVER route around a connector: an unconnected Gmail is not a cue to install an IMAP client, ask for an app password, or find another way into the same account. The connector IS the way in; if they decline it, the app is out of this build.'

/** The picks are gateway slugs the user chose during setup. The agent, rather than the app, waits for the connection
 *  result, as decided in D85. */
function connectFirstRunbook(picks: string[]): string[] {
  const named = picks.map(slug => `${slug} (${connectorTitle(slug)})`).join(', ')

  return [
    `CONNECT FIRST. During setup the user picked these apps, given here as exact gateway slugs: ${named}. Your first action in this session, before any plan and before any other tool call, is ONE manage_connections call with action="connect" and connectors set to every one of those slugs. Do not call action="status" first; the slugs are exact and the catalog check is already done.`,
    'If every result comes back already active, there is nothing to wait for: begin the task at once.',
    "The app opens every sign-in from that result in the user's browser and shows one row per app, so never paste the links. In the same turn say one short line: which apps are being connected and, in a clause each, what this task gets from each one. Then end the turn.",
    'Then call manage_connections action="wait" with the same slugs and timeout_seconds=120, and say nothing until it returns. If the wait comes back as pending because the links were minted moments ago, end your turn: the app sends a hidden note that begins with "[setup] links opened" once your turn ends and the sign-ins are open, and that note is your cue to call the same wait again. A note that arrives after you have already waited needs no reply.',
    'The user can start early. A message from them that begins with "Start with" or "Start without" names the apps that are connected and the ones they skipped; treat it as the go signal and begin with the connected apps only.',
    'When the wait returns with every app connected, begin the task at once. When it returns with apps still pending, stop and ask in one line: which apps did not connect, and whether they want you to continue without them or try connecting again (a fresh action="connect" mints new links). Wait for their answer. If they choose to continue without an app, build the version of the task that needs no account for that part and say in one line what the connection would have added.',
    'Account data comes from the connected apps first. Tools already signed in on this machine, like a logged-in gh, are fair to use when the task benefits; say so in one line when you do.',
    "Discover a connected app's tools with tool_search and use real results for the task; never fabricate account data. Reading is separate from sending, deleting or scheduling: ask before those. No recurring job unless that is what they asked for.",
    'Make the result something they can open: a single HTML page when the idea allows it, and at least one real reading or action through a connected app.'
  ]
}

/** The machine-setup runbook. The audit comes before the plan because a plan written before looking is how an agent
 *  installs a second copy of something, or "fixes" drivers that were already correct. */
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

/** The plugin runbook. The save-time reload it promises is implemented in src/contrib/runtime-loader.ts. */
const pluginRunbook = (root: string) => [
  'THIS IS A PLUGIN JOB: the thing you are building is a piece of the Hermes app itself, and it will appear in the window the user is looking at right now. That is the whole point — do not let it become a script in a folder.',
  `A plugin is ONE file: \`${root}/<name>/plugin.js\`. Plain ESM, no build step, no package.json, no install. It imports from \`@hermes/plugin-sdk\` and calls \`jsx()\` from \`react/jsx-runtime\` directly (there is no JSX compiler in this path — writing \`<div>\` will not work). It default-exports \`{ id, name, register(ctx) }\` and \`register\` calls \`ctx.register({ id, area, order, render })\`. The runtime loads it the moment you save, and reloads it on every later save, so there is no restart to ask them for.`,
  'LOOK BEFORE YOU WRITE. Read the `building-hermes-desktop-plugins` skill first — it has the SDK surface, the areas you can render into, and the traps. If the machine has a checkout of NousResearch/plugins, read a plugin close to what you are making; those thirteen are reviewed and show the real shapes (a statusbar chip, a composer action, a full pane).',
  'START SMALL AND VISIBLE. The first save should put something on screen even if it only renders a label — a chip that says the right word beats a half-written dashboard, because they SEE it work and everything after that is refinement they are watching. Build up from there in passes.',
  'Say what you are doing in one short line per pass, and tell them where to look the first time it appears ("bottom right of the status bar" / "it is in the right pane now"). A plugin that loaded silently reads as nothing having happened.',
  'Never ask them to restart the app, never edit anything outside their plugin folder, and never touch the Hermes install itself. If the plugin errors on load, the app toasts it and keeps running — read the error, fix the file, save again.'
]

/** A new HandoffPlan takes effect only once it has a case here. */
function planRunbook(plan: HandoffPlan, pluginRoot: string, connectFirst: boolean): string[] {
  switch (plan) {
    case 'machine-setup':
      return machineSetupRunbook()

    case 'plugin':
      if (!pluginRoot) {
        throw new Error('The desktop plugin folder is unavailable. Retry before starting the first build.')
      }

      // With no picks NO_AUTH_RULE still applies: a plugin that needs an API key on its first run is as
      // unfinishable as any other first build that needs an account.
      return connectFirst ? pluginRunbook(pluginRoot) : [...pluginRunbook(pluginRoot), NO_AUTH_RULE]

    default:
      return connectFirst ? [] : [NO_AUTH_RULE]
  }
}

/** Prefixes MACHINE_SETUP_RUNBOOK with machineDescription(), so the agent does not spend its first turns finding out
 *  what the app already reports. */
function machineSetupRunbook(): string[] {
  const description = machineDescription()

  return description
    ? [`What the app can already see about it: ${description}.`, ...MACHINE_SETUP_RUNBOOK]
    : MACHINE_SETUP_RUNBOOK
}

/** Seed rows for the build session's session.create: the hidden runbook only. The task brief is submitted as a real
 *  turn right after, and that is what starts the build. */
export async function buildFirstTaskSeedMessages(
  task: string,
  answers: OnboardingAnswers,
  plan: HandoffPlan = 'build'
): Promise<{ content: string; display_kind?: 'hidden'; role: 'assistant' | 'user' }[]> {
  const root = plan === 'plugin' ? await window.hermesDesktop?.desktopPluginsRoot?.() : undefined

  return [{ content: buildFirstTaskRunbook(task, answers, plan, root), display_kind: 'hidden', role: 'user' }]
}

/** The hidden note sent to the welcome chat once the build session is live. The check-ins after it come from the
 *  build's own progress, in first-build.ts. */
export function buildHandoffCompleteNote(task: string): string {
  return `[setup] handoff complete — "${task.trim()}" is now building in its own session on the default profile, and the user is watching it there. The app is showing them a short tour of the profile rail and the sessions list right now, so do not describe either. Say ONE short line and then stop: you're around if they want a hand, and this chat stays where it is. Do not ask a question, do not offer a list, do not schedule anything.`
}

/** Creates the guide profile. The catch treats an already-existing profile as success, so kickoff can call this on
 *  every run. */
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
