import { readFileSync } from 'node:fs'
import { load } from 'js-yaml'
import { expect, it } from 'vitest'

const action = path => load(readFileSync(new URL(path, import.meta.url), 'utf8'))
const setup = action('../.github/actions/setup-pm/action.yml')

// Evaluates the Actions expression subset these gates use, so the tests pin
// how a gate behaves rather than how it is spelled. Status functions default
// to a healthy, uncancelled run; `status` overrides them.
const value = (expression, { status = {}, ...context } = {}) => {
  const lookup = path => path.split('.').reduce(
    (value, key) => (value !== null && typeof value === 'object' ? value[key] : undefined), context) ?? ''
  const statuses = { always: true, success: true, failure: false, cancelled: false, ...status }
  const body = String(expression).trim().replace(/^\$\{\{/, '').replace(/\}\}$/, '')
    .replace(/\b(?:inputs|steps|needs|github)(?:\.[\w-]+)+/g, path => JSON.stringify(lookup(path)))
    .replace(/\b(always|success|failure|cancelled)\(\)/g, (_, name) => String(statuses[name]))
  return Function(`"use strict"; return (${body})`)()
}
const evaluate = (expression, context) => Boolean(value(expression, context))

it('pull requests restore the caches main saves and never write their own', () => {
  const step = id => setup.runs.steps.find(candidate => candidate.id === id)
  const base = { toolchain: 'all', cache: 'true', 'cache-python': 'true', 'cache-node': 'true', 'test-environment': 'false', extras: 'voice' }
  for (const [kind, saving, restoring] of [
    ['tools', 'tools-cache', 'tools-cache-restore'],
    ['python', 'python-cache', 'python-cache-restore'],
    ['node', 'node-cache', 'node-cache-restore'],
  ]) {
    const input = `save-${kind}-cache`
    const save = step(saving)
    const restore = step(restoring)
    // A PR-scoped entry is restorable only by that PR, so PRs never save.
    for (const [event_name, saves] of [['pull_request', false], ['push', true], ['workflow_dispatch', true]]) {
      const inputs = { ...base, [input]: value(setup.inputs[input].default, { github: { event_name } }) }
      expect(evaluate(save.if, { inputs }), `${kind} on ${event_name}`).toBe(saves)
      expect(evaluate(restore.if, { inputs }), `${kind} on ${event_name}`).toBe(!saves)
    }
    // Restore-only reads exactly what the saving step writes on main.
    expect(restore.uses.split('@')[0]).toBe('actions/cache/restore')
    expect(restore.with.key).toBe(save.with.key)
    expect(restore.with['restore-keys']).toBe(save.with['restore-keys'])
    expect(restore.with.path).toBe(save.with.path)
  }
})

it('only dependency-carrying callers touch the uv cache', () => {
  const cached = setup.runs.steps.find(step => step.id === 'python-cache')
  // Tool-only jobs must not freeze an empty cache under the production key
  // (a stub exact-hit blocks real saves).
  const saving = { toolchain: 'all', 'cache-python': 'true', 'save-python-cache': 'true', 'test-environment': 'false' }
  expect(evaluate(cached.if, { inputs: { ...saving, extras: '' } })).toBe(false)
  expect(evaluate(cached.if, { inputs: { ...saving, extras: 'voice' } })).toBe(true)
  expect(evaluate(cached.if, { inputs: { ...saving, extras: '', 'test-environment': 'true' } })).toBe(true)
  // A suffix-only namespace isolates smoke runs from production's prefix.
  const namespace = "${{ inputs.cache-suffix || 'production' }}"
  expect(cached.with.key.replace(namespace, 'smoke-42-1').startsWith('setup-pm-uv-v3-production-')).toBe(false)
  expect(cached.with['restore-keys']).toContain("inputs.cache-suffix == ''")
})

// Key isolation, offline wheel/receipt relocation and the transport action are
// covered by tests/scripts/test_desktop_build_cache.py. These declarations guard
// the caller seam: a cache hit never replaces preparation.
const desktop = action('../.github/workflows/desktop-bundled-release.yml')
const payload = action('../.github/workflows/pm-bundle.yml')
const BUILD_CACHE = './.github/actions/desktop-build-cache'
// Jobs are found by the cache action they use, not by id: build legs get
// split and renamed while this contract stays the same.
const cacheUsers = workflow => Object.entries(workflow.jobs)
  .filter(([, job]) => (job.steps ?? []).some(step => step.uses === BUILD_CACHE))
const targetOf = job => job.strategy.matrix.target.map(row => row.label)
const desktopLegs = cacheUsers(desktop)

const SHA = 'a'.repeat(40)
const dispatches = {
  tag: { build_commit: '', channel: '', 'release-phase': '', upload_release: true },
  commit: { build_commit: SHA, channel: '', 'release-phase': '', upload_release: false },
  channel: { build_commit: SHA, channel: 'preview', 'release-phase': '', upload_release: false },
}
const admitted = labels => ({
  validate: { result: 'success', outputs: Object.fromEntries(labels.map(label => [label, 'true'])) },
})
// Payload snapshots are shared by every later build; only a trusted main push writes them.
const payloadEvents = {
  trusted: { event_name: 'push', ref: 'refs/heads/main', sha: SHA },
  pullRequest: { event_name: 'pull_request', ref: 'refs/heads/main', sha: SHA },
  branch: { event_name: 'push', ref: 'refs/heads/feature', sha: SHA },
}

it('finds a release and a commit leg for every native target, and the payload producer', () => {
  const legs = desktopLegs.map(([, job]) => `${targetOf(job)}:${job['cache-mode']}`).sort()
  const targets = ['darwin-arm64', 'darwin-x64', 'win32-arm64', 'win32-x64']
  expect(legs).toEqual(targets.flatMap(target => [`${target}:read`, `${target}:write`]).sort())
  expect(cacheUsers(payload)).toHaveLength(1)
})

it.each([
  ...desktopLegs.map(([id, job]) => [id, job, 'desktop', 'scripts/bundles/desktop.py']),
  ...cacheUsers(payload).map(([id, job]) => [id, job, 'payload-test', 'scripts/bundles/native_build.py']),
])('%s restores, admits and saves candidates before consuming them', (id, job, producer, driver) => {
  const cacheMode = job['cache-mode']
  if (cacheMode) {
    expect(job.needs).toEqual(['validate'])
    // Only the release branch writes the shared cache; commit and channel
    // builds of unreviewed inputs run the read-only leg.
    const needs = admitted(targetOf(job))
    expect(evaluate(job.if, { inputs: dispatches.tag, needs })).toBe(cacheMode === 'write')
    expect(evaluate(job.if, { inputs: dispatches.commit, needs })).toBe(cacheMode === 'read')
    expect(evaluate(job.if, { inputs: dispatches.channel, needs })).toBe(cacheMode === 'read')
  }
  const cacheSteps = job.steps.filter(step => step.uses === BUILD_CACHE)
  expect(cacheSteps.map(step => step.with.phase)).toEqual(['restore', 'save'])
  const [restore, upload] = cacheSteps
  const prepare = job.steps.find(step => step.id === 'prepare')
  const builds = job.steps.filter(step => step.run?.includes(driver) && step.run.includes('--prepared '))
  expect(prepare).toBeDefined()
  expect(builds).not.toHaveLength(0)
  expect(prepare.run).toContain(driver)
  expect(prepare.run).toContain('--prepare-only')
  // Admission runs on warm hits too; a failed preparation must fail the job.
  for (const step of [restore, prepare, ...builds]) {
    expect(step.if).toBeUndefined()
    expect(step['continue-on-error']).toBeUndefined()
  }
  expect(job.steps.indexOf(prepare)).toBeGreaterThan(job.steps.indexOf(restore))
  expect(job.steps.indexOf(upload)).toBeGreaterThan(job.steps.indexOf(prepare))
  for (const build of builds) {
    expect(job.steps.indexOf(build)).toBeGreaterThan(job.steps.indexOf(upload))
    expect(build.run).not.toContain('--prepare-only')
  }
  expect(restore.with.producer).toBe(producer)
  expect(restore.with.source).toBe('${{ github.workspace }}')
  expect(upload.with).toEqual({
    ...restore.with,
    phase: 'save',
    key: `\${{ steps.${restore.id}.outputs.cache-key }}`,
  })
  // No cache-hit gate: immutable underfilled snapshots must be replaceable.
  // A failed/cancelled preparation cannot publish; later build failure cannot
  // discard an already-saved candidate snapshot.
  const saves = (context, outcome = 'success') => evaluate(upload.if, {
    ...context, steps: { prepare: { outcome } }, status: { failure: outcome !== 'success' },
  })
  const trusted = cacheMode
    ? { inputs: dispatches.tag }
    : { inputs: { ref: '' }, github: payloadEvents.trusted }
  expect(saves(trusted)).toBe(true)
  expect(saves(trusted, 'failure')).toBe(false)
  expect(evaluate(upload.if, { ...trusted, steps: { prepare: { outcome: 'success' } },
    status: { cancelled: true, success: false } })).toBe(false)
  const untrusted = cacheMode
    ? [{ inputs: dispatches.commit }, { inputs: dispatches.channel }]
    : [
        { inputs: { ref: '' }, github: payloadEvents.pullRequest },
        { inputs: { ref: '' }, github: payloadEvents.branch },
        { inputs: { ref: 'b'.repeat(40) }, github: payloadEvents.trusted },
      ]
  for (const context of untrusted) {
    expect(saves(context)).toBe(false)
  }
})

it('each selection gate joins exactly one target\'s two trust branches after admission', () => {
  const legs = Object.fromEntries(desktopLegs.map(([id, job]) => [id, job]))
  const gates = Object.values(desktop.jobs).filter(job => job.env?.SELECTED_BUILD_SUCCEEDED)
  expect(gates).toHaveLength(4)
  for (const gate of gates) {
    const [admission, ...branches] = gate.needs
    expect(admission).toBe('validate')
    expect(branches.map(id => legs[id]?.['cache-mode']).sort()).toEqual(['read', 'write'])
    const [label] = new Set(branches.flatMap(id => targetOf(legs[id])))
    expect(new Set(branches.flatMap(id => targetOf(legs[id]))).size).toBe(1)
    // The gate runs always() to judge a skipped branch, so it must still
    // refuse a failed admission itself.
    const needs = admitted([label])
    expect(evaluate(gate.if, { inputs: dispatches.tag, needs })).toBe(true)
    expect(evaluate(gate.if, { inputs: dispatches.tag,
      needs: { validate: { ...needs.validate, result: 'failure' } } })).toBe(false)
  }
})
