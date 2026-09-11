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
  // An exact legacy entry wins over prefix matching. Prefer rolling entries
  // first so the next attempt can use additions from the last failed build.
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
  const namespace = "${{ inputs.cache-suffix != '' && 'isolated-' || '' }}"
  for (const template of [cached.with.key, restored.with.key, rollingPrefix]) {
    const production = template.replace(namespace, '')
    const smoke = template.replace(namespace, 'isolated-')
    expect(production.startsWith('setup-pm-uv-v1-')).toBe(true)
    expect(smoke.startsWith('setup-pm-uv-isolated-v1-')).toBe(true)
    expect(smoke.startsWith('setup-pm-uv-v1-')).toBe(false)
    expect(production.startsWith('setup-pm-uv-isolated-v1-')).toBe(false)
  }
})

it('all bundle consumers save on failure, after building, without discarding offline wheels', () => {
  const workflows = [
    action('../.github/workflows/desktop-bundled-release.yml'),
    action('../.github/workflows/pm-bundle.yml'),
  ]
  const jobs = workflows.flatMap(workflow => Object.values(workflow.jobs)).filter(job =>
    job.steps?.some(step => ['Build and package', 'Stage the payload'].includes(step.name)))
  expect(jobs.length).toBeGreaterThan(0)
  for (const job of jobs) {
    const setupIndex = job.steps.findIndex(step => step.uses === './.github/actions/setup-pm')
    const saveIndex = job.steps.findIndex(step => step.uses === './.github/actions/save-pm-cache')
    const setupStep = job.steps[setupIndex]
    const saveStep = job.steps[saveIndex]
    expect(setupStep.with['save-python-cache']).toBe(false)
    expect(saveIndex).toBeGreaterThan(setupIndex)
    expect(job.steps.slice(setupIndex + 1, saveIndex).some(step => step.run?.includes('bundle'))).toBe(true)
    expect(saveStep.if).toBe(`\${{ !cancelled() && steps.${setupStep.id}.outcome == 'success' }}`)
    expect(saveStep.with).toEqual({
      uv: `\${{ steps.${setupStep.id}.outputs.uv-path }}`,
      path: `\${{ steps.${setupStep.id}.outputs.uv-cache-path }}`,
      key: `\${{ steps.${setupStep.id}.outputs.python-cache-key }}`,
    })
  }
  const [prune, upload] = save.runs.steps
  expect(prune.if).toBe('${{ !cancelled() }}')
  expect(prune.run).toBe('"$PM_UV" cache prune')
  expect(prune.env.PM_UV).toBe('${{ inputs.uv }}')
  expect(prune.env.UV_CACHE_DIR).toBe('${{ inputs.path }}')
  expect(upload.if).toBe(`\${{ !cancelled() && steps.${prune.id}.outcome == 'success' }}`)
  expect(upload.uses.split('@')[0]).toBe('actions/cache/save')
  expect(upload.with).toEqual({ path: '${{ inputs.path }}', key: '${{ inputs.key }}' })
})
