import { readFileSync } from 'node:fs'
import { load } from 'js-yaml'
import { expect, it } from 'vitest'

const action = path => load(readFileSync(new URL(path, import.meta.url), 'utf8'))
const setup = action('../.github/actions/setup-pm/action.yml')
const save = action('../.github/actions/save-pm-cache/action.yml')

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
  expect(prefixes[1]).toContain('inputs.prune-python-cache')
  expect(prefixes[1]).not.toContain('hashFiles')
  expect(restored.uses.split('@')[0]).toBe('actions/cache/restore')
  expect(cached.if).toContain("inputs.save-python-cache == 'true'")
  expect(restored.if).toContain("inputs.save-python-cache == 'false'")
  expect(setup.outputs['python-cache-key'].value).toContain('steps.python-cache-restore.outputs.cache-primary-key')

  // A suffix-only namespace isolates smoke reads but still lets production
  // restore smoke writes through its broad dependency fallback.
  const namespace = "${{ inputs.cache-suffix || 'production' }}"
  for (const template of [cached.with.key, restored.with.key, rollingPrefix]) {
    const production = template.replace(namespace, 'production')
    const smoke = template.replace(namespace, 'smoke-42-1')
    expect(production.startsWith('setup-pm-uv-v2-production-')).toBe(true)
    expect(smoke.startsWith('setup-pm-uv-v2-smoke-42-1-')).toBe(true)
    expect(smoke.startsWith('setup-pm-uv-v2-production-')).toBe(false)
    expect(production.startsWith('setup-pm-uv-v2-smoke-42-1-')).toBe(false)
  }
})

it('explicit PM saves preserve downloaded offline wheels and do not prune during cancellation', () => {
  const [prune, upload] = save.runs.steps
  expect(prune.if).toBe('${{ !cancelled() }}')
  expect(prune.run).toBe('"$PM_PYTHON" -m pm.build_env --prune-cache --cache "$PM_CACHE"')
  expect(prune.env.PM_PYTHON).toBe('${{ inputs.python }}')
  expect(prune.env.PM_CACHE).toBe('${{ inputs.path }}')
  expect(upload.if).toBe(`\${{ !cancelled() && steps.${prune.id}.outcome == 'success' }}`)
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
const desktopSaveGate = "${{ !cancelled() && steps.prepare.outcome == 'success' && inputs.build_commit == '' }}"
const payloadSaveGate = "${{ !cancelled() && steps.prepare.outcome == 'success' && github.event_name != 'pull_request' && github.ref == 'refs/heads/main' && (inputs.ref == '' || inputs.ref == github.sha) }}"

it.each([
  ['build-win32-release', desktop, 'desktop', 'write', 'scripts/bundles/desktop.py', desktopSaveGate],
  ['build-win32-commit', desktop, 'desktop', 'read', 'scripts/bundles/desktop.py', desktopSaveGate],
  ['build-darwin-release', desktop, 'desktop', 'write', 'scripts/bundles/desktop.py', desktopSaveGate],
  ['build-darwin-commit', desktop, 'desktop', 'read', 'scripts/bundles/desktop.py', desktopSaveGate],
  ['bundle', payload, 'payload-test', undefined, 'scripts/bundles/native_build.py', payloadSaveGate],
])('%s restores, admits and saves candidates before consuming them', (id, workflow, producer, cacheMode, driver, saveGate) => {
  const job = workflow.jobs[id]
  expect(job).toBeDefined()
  if (cacheMode) {
    expect(job['cache-mode']).toBe(cacheMode)
    expect(job.needs).toEqual(['validate', 'archive-inputs'])
    expect(job.if).toContain(`inputs.build_commit ${cacheMode === 'read' ? '!=' : '=='} ''`)
  }
  const cacheSteps = job.steps.filter(step => step.uses === './.github/actions/desktop-build-cache')
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
  expect(upload.if).toBe(saveGate)
})

it.each(['win32', 'darwin'])('%s publication requires the selected build to succeed, not merely skip', platform => {
  const gate = desktop.jobs[`build-${platform}`]
  expect(gate.needs).toEqual(['validate', 'archive-inputs', `build-${platform}-release`, `build-${platform}-commit`])
  expect(gate.if).toContain("needs.validate.result == 'success' && needs.archive-inputs.result == 'success'")
  expect(gate.env.SELECTED_BUILD_SUCCEEDED.replace(/\s+/g, ' ').trim()).toBe(
    `\${{ (inputs.build_commit == '' && needs.build-${platform}-release.result == 'success' && needs.build-${platform}-commit.result == 'skipped') || (inputs.build_commit != '' && needs.build-${platform}-commit.result == 'success' && needs.build-${platform}-release.result == 'skipped') }}`,
  )
  expect(desktop.jobs[`publish-${platform}-updater`].needs).toContain(`build-${platform}`)
})
