/**
 * The plugin plan — the first build as a piece of the user's own app.
 *
 * `plan="plugin"` has to survive the whole trip: the model writes it on the
 * handoff directive, the renderer parses it, and the build session's runbook
 * changes because of it. A plan that parses but does not reach the runbook is
 * the failure worth pinning — it looks like it works and silently builds the
 * generic thing.
 */

import { describe, expect, it } from 'vitest'

import { buildFirstTaskRunbook, buildFirstTaskSeedMessages, parseHandoffPlan } from './setup-profile'

const ANSWERS = {
  accent: '',
  connectors: ['Linear'],
  context: 'shipping a release this week',
  focus: [],
  layout: '',
  name: 'Alex'
} as never

describe('parseHandoffPlan', () => {
  it('reads the plans the script is allowed to emit', () => {
    expect(parseHandoffPlan('plugin')).toBe('plugin')
    expect(parseHandoffPlan('machine-setup')).toBe('machine-setup')
    expect(parseHandoffPlan('build')).toBe('build')
  })

  it('tolerates the shapes a model actually writes', () => {
    expect(parseHandoffPlan(' PLUGIN ')).toBe('plugin')
    expect(parseHandoffPlan('Machine-Setup')).toBe('machine-setup')
  })

  it('falls back to a plain build for anything unknown', () => {
    // An invented plan must not strand the user: the generic runbook still
    // builds their idea, it just is not scripted.
    expect(parseHandoffPlan('make-a-website')).toBe('build')
    expect(parseHandoffPlan(undefined)).toBe('build')
    expect(parseHandoffPlan('')).toBe('build')
  })
})

describe('the plugin runbook', () => {
  const runbook = buildFirstTaskRunbook('A panel with my open tickets', ANSWERS, 'plugin')

  it('names the one file and the no-build-step shape', () => {
    // The traps that produce a broken first plugin: a build step that does not
    // exist, and JSX that never compiles on this path.
    expect(runbook).toContain('desktop-plugins')
    expect(runbook).toContain('plugin.js')
    expect(runbook).toContain('react/jsx-runtime')
    expect(runbook).toContain('no build step')
  })

  it('sends the agent to read before it writes', () => {
    expect(runbook).toContain('building-hermes-desktop-plugins')
    expect(runbook).toContain('NousResearch/plugins')
  })

  it('keeps the no-auth rule — a plugin needing a key is the same dead end', () => {
    expect(runbook).toContain('NO external account')
  })

  it('still carries the shared beats every first build gets', () => {
    expect(runbook).toContain('::onboarding{step="progress"')
    expect(runbook).toContain('Does this match what you wanted?')
    expect(runbook).toContain('Alex')
    expect(runbook).toContain('shipping a release this week')
  })

  it('does not leak into an ordinary build', () => {
    const plain = buildFirstTaskRunbook('Draft the release notes', ANSWERS, 'build')

    expect(plain).not.toContain('desktop-plugins')
    expect(plain).toContain('NO external account')
  })

  it('does not leak into a machine setup', () => {
    const machine = buildFirstTaskRunbook('Set up this Mac', ANSWERS, 'machine-setup')

    expect(machine).not.toContain('desktop-plugins')
    expect(machine).toContain('START BY LOOKING, NOT PLANNING')
  })

  it('reaches the session as the seeded runbook, not just as a parsed value', () => {
    const [seed] = buildFirstTaskSeedMessages('A panel with my open tickets', ANSWERS, 'plugin')

    expect(seed.display_kind).toBe('hidden')
    expect(seed.content).toContain('THIS IS A PLUGIN JOB')
  })
})
