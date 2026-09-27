import { readFileSync } from 'node:fs'
import { load } from 'js-yaml'
import { expect, it } from 'vitest'

const action = path => load(readFileSync(new URL(path, import.meta.url), 'utf8'))
const setup = action('../.github/actions/setup-pm/action.yml')
const save = action('../.github/actions/save-pm-cache/action.yml')

// Evaluates the Actions expression subset these gates use, so the tests pin
// how a gate behaves rather than how it is spelled. Status functions default
// to a healthy, uncancelled run; `status` overrides them.
const evaluate = (expression, { status = {}, ...context } = {}) => {
  const lookup = path => path.split('.').reduce(
    (value, key) => (value !== null && typeof value === 'object' ? value[key] : undefined), context) ?? ''
  const statuses = { always: true, success: true, failure: false, cancelled: false, ...status }
  const body = String(expression).trim().replace(/^\$\{\{/, '').replace(/\}\}$/, '')
    .replace(/\b(?:inputs|steps|needs|github)(?:\.[\w-]+)+/g, path => JSON.stringify(lookup(path)))
    .replace(/\b(always|success|failure|cancelled)\(\)/g, (_, name) => String(statuses[name]))
  return Boolean(Function(`"use strict"; return (${body})`)())
}

it('restores compatible wheels without freezing a partial build under its dependency key', () => {
  const cached = setup.runs.steps.find(step => step.id === 'python-cache')
  const restored = setup.runs.steps.find(step => step.id === 'python-cache-restore')
  const prefixes = restored.with['restore-keys'].trim().split('\n')
  // Prefer this dependency set before falling back across dependency changes.
  const rollingPrefix = prefixes[0]
  expect(restored.with.key).toBe(`${rollingPrefix}\${{ github.run_id }}-\${{ github.run_attempt }}-\${{ github.job }}`)
  expect(rollingPrefix).toBe(`${cached.with.key}-`)
  expect(prefixes[1].trim()).toBe(cached.with['restore-keys'])
  for (const boundary of ['target', 'os-version', 'python-version']) {
    expect(prefixes[1]).toContain(`steps.prepare.outputs.${boundary}`)
  }
  expect(prefixes[1]).toContain("inputs.cache-suffix == ''")
  expect(prefixes[1]).not.toContain('hashFiles')
  // Only dependency-carrying callers save; tool-only jobs must not freeze an
  // empty cache under the production key (a stub exact-hit blocks real saves).
  const saving = { toolchain: 'all', 'cache-python': 'true', 'save-python-cache': 'true', 'test-environment': 'false' }
  expect(evaluate(cached.if, { inputs: { ...saving, extras: '' } })).toBe(false)
  expect(evaluate(cached.if, { inputs: { ...saving, extras: 'voice' } })).toBe(true)
  expect(evaluate(cached.if, { inputs: { ...saving, extras: '', 'test-environment': 'true' } })).toBe(true)
  // A caller either saves the exact key or restores the rolling one, never both.
  for (const saveCache of ['true', 'false']) {
    const inputs = { ...saving, extras: 'voice', 'save-python-cache': saveCache }
    expect(evaluate(restored.if, { inputs })).toBe(!evaluate(cached.if, { inputs }))
  }
  expect(setup.outputs['python-cache-key'].value).toContain('steps.python-cache-restore.outputs.cache-primary-key')

  // A suffix-only namespace isolates smoke reads but still lets production
  // restore smoke writes through its broad dependency fallback.
  const namespace = "${{ inputs.cache-suffix || 'production' }}"
  for (const template of [cached.with.key, restored.with.key, rollingPrefix]) {
    const production = template.replace(namespace, 'production')
    const smoke = template.replace(namespace, 'smoke-42-1')
    expect(production.startsWith('setup-pm-uv-v3-production-')).toBe(true)
    expect(smoke.startsWith('setup-pm-uv-v3-smoke-42-1-')).toBe(true)
    expect(smoke.startsWith('setup-pm-uv-v3-production-')).toBe(false)
    expect(production.startsWith('setup-pm-uv-v3-smoke-42-1-')).toBe(false)
  }
})

it('explicit PM saves prune to the lock and do not prune during cancellation', () => {
  const [prune, upload] = save.runs.steps
  expect(evaluate(prune.if)).toBe(true)
  expect(evaluate(prune.if, { status: { failure: true, success: false } })).toBe(true)
  expect(evaluate(prune.if, { status: { cancelled: true, success: false } })).toBe(false)
  expect(prune.run.split(/\s+/)).toEqual(expect.arrayContaining(['pm.build_env', '--exact-lock', '--cache', '--lock-source']))
  expect(prune.env.PM_PYTHON).toBe('${{ inputs.python }}')
  expect(prune.env.PM_CACHE).toBe('${{ inputs.path }}')
  expect(prune.env.PM_LOCK_SOURCE).toBe('${{ github.workspace }}')
  const pruned = outcome => ({ steps: { [prune.id]: { outcome } } })
  expect(evaluate(upload.if, pruned('success'))).toBe(true)
  expect(evaluate(upload.if, pruned('failure'))).toBe(false)
  expect(evaluate(upload.if, { ...pruned('success'), status: { cancelled: true, success: false } })).toBe(false)
  expect(upload.uses.split('@')[0]).toBe('actions/cache/save')
  expect(upload.with).toEqual({ path: '${{ inputs.path }}', key: '${{ inputs.key }}' })
})

it('explicit npm snapshots remain replaceable and isolated from toolchain-only producers', () => {
  const restored = setup.runs.steps.find(step => step.id === 'node-cache-restore')
  expect(restored).toBeDefined()
  const prefix = restored.with['restore-keys'].trim()
  expect(restored.with.key).toBe(`${prefix}\${{ github.run_id }}-\${{ github.run_attempt }}`)
  // A toolchain-only job must not shadow the desktop consumer's warm snapshot.
  for (const boundary of ['github.job', 'node-cache-dependency-path', 'cache-suffix', 'target', 'npm-version']) {
    expect(prefix).toContain(boundary)
  }
  expect(setup.outputs['node-cache-key'].value).toContain('steps.node-cache-restore.outputs.cache-primary-key')
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
