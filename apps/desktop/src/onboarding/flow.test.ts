import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { answers, FIXTURES } from './fixtures.test-util'
import { type Facts, FLOW, stepOptions, visibleSteps } from './flow'

const copy = en.questionnaire

const ids = (facts: Facts, patch = {}) => visibleSteps(facts, answers(patch)).map(step => step.id)

function optionIds(facts: Facts, stepId: string, patch = {}) {
  const step = FLOW.find(candidate => candidate.id === stepId)

  return step ? stepOptions(step, facts, answers(patch)).map(option => option.id) : []
}

describe('visible steps per machine', () => {
  it.each([
    ['spark', ['name', 'accent', 'layout', 'local', 'apps', 'connectors', 'task', 'tour']],
    ['windowsRtx', ['name', 'accent', 'layout', 'local', 'apps', 'connectors', 'task', 'tour']],
    ['windowsLaptop', ['name', 'accent', 'layout', 'connectors', 'task', 'tour']],
    ['mac', ['name', 'accent', 'layout', 'local', 'apps', 'connectors', 'task', 'tour']]
  ] as const)('%s', (machine, expected) => {
    expect(ids(FIXTURES[machine])).toEqual(expected)
  })

  it('drops the connectors step when the list is unavailable and keeps it while loading', () => {
    expect(ids({ ...FIXTURES.mac, connectors: { status: 'unavailable' } })).not.toContain('connectors')
    expect(ids({ ...FIXTURES.mac, connectors: { status: 'loading' } })).toContain('connectors')
  })

  it('drops the local step when nothing fits and the apps step when presence failed', () => {
    const steps = ids({ ...FIXTURES.spark, local: null, plugins: [] })

    expect(steps).not.toContain('local')
    expect(steps).not.toContain('apps')
  })
})

describe('plugin apps', () => {
  it('offers only installed apps, running or not', () => {
    expect(optionIds(FIXTURES.spark, 'apps')).toEqual(['blender', 'nvidia-app'])
    expect(optionIds(FIXTURES.windowsRtx, 'apps')).toEqual(['nvidia-app', 'nvidia-broadcast'])
    expect(optionIds(FIXTURES.windowsLaptop, 'apps')).toEqual([])
  })

  it('offers nothing for an app the backend dropped for this OS', () => {
    expect(optionIds(FIXTURES.mac, 'apps')).toEqual(['blender'])
  })

  it('names the plugin by catalog name and carries the disclosure', () => {
    const step = FLOW.find(candidate => candidate.id === 'apps')!
    const [blender] = stepOptions(step, FIXTURES.spark, answers())

    expect(blender.before?.[0]).toContain('catalog name: blender')
    expect(blender.disclosure).toBe('Blender disclosure.')
  })

  it('labels an app that is installed but closed', () => {
    const step = FLOW.find(candidate => candidate.id === 'apps')!
    const broadcast = stepOptions(step, FIXTURES.windowsRtx, answers()).find(option => option.id === 'nvidia-broadcast')

    expect(broadcast?.detail?.(copy)).toBe(copy.apps.pluginNotRunning)
  })
})

describe('task step', () => {
  it('builds options from the picks and never offers machine setup on a Spark', () => {
    const tasks = optionIds(FIXTURES.spark, 'task', { apps: ['blender', 'nvidia-app'], connectors: ['gmail'] })

    expect(tasks).toEqual(['blender', 'nvidia', 'brief', 'tidy'])
    expect(optionIds(FIXTURES.spark, 'task')).toEqual(['tidy'])
  })

  it('caps the list at four', () => {
    const tasks = optionIds(FIXTURES.windowsRtx, 'task', {
      apps: ['blender', 'nvidia-app', 'nvidia-broadcast'],
      connectors: ['github']
    })

    expect(tasks).toHaveLength(4)
  })

  it('takes free text on name and task only; the tour is yes or no', () => {
    const withOther = FLOW.filter(step => step.other).map(step => step.id)

    expect(withOther).toEqual(['name', 'task'])
    expect(optionIds(FIXTURES.mac, 'tour')).toEqual(['quick', 'none'])
  })
})

describe('name step', () => {
  it('suggests the first name from the OS account and nothing without one', () => {
    expect(optionIds(FIXTURES.mac, 'name')).toEqual(['Sid'])
    expect(optionIds({ ...FIXTURES.mac, machine: null }, 'name')).toEqual([])
  })
})
