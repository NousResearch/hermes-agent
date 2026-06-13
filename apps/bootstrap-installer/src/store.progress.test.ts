import { beforeEach, describe, expect, it } from 'vitest'
import { $bootstrap, $progress, applyBootstrapEvent } from './store'

const stages = [
  {
    name: 'update',
    title: 'Updating Hermes',
    category: 'update',
    needs_user_input: false
  },
  {
    name: 'rebuild',
    title: 'Rebuilding the desktop app',
    category: 'update',
    needs_user_input: false
  },
  {
    name: 'install',
    title: 'Installing the updated app',
    category: 'update',
    needs_user_input: false
  }
]

describe('bootstrap progress', () => {
  beforeEach(() => {
    applyBootstrapEvent({
      type: 'manifest',
      stages,
      protocolVersion: null
    })
  })

  it('reports the active stage as the current step', () => {
    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'running'
    })

    expect($progress.get()).toEqual({
      done: 0,
      current: 1,
      total: 3,
      fraction: 1 / 3
    })

    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'succeeded'
    })
    applyBootstrapEvent({
      type: 'stage',
      name: 'rebuild',
      state: 'running'
    })

    expect($bootstrap.get().currentStage).toBe('rebuild')
    expect($progress.get()).toMatchObject({
      done: 1,
      current: 2,
      total: 3
    })
  })

  it('clears stale active stages when a stage finishes', () => {
    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'running'
    })
    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'succeeded'
    })

    expect($bootstrap.get().currentStage).toBeNull()
    expect($progress.get()).toMatchObject({
      done: 1,
      current: 1,
      total: 3
    })
  })

  it('does not reset progress when a duplicate manifest arrives for the active run', () => {
    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'succeeded',
      durationMs: 1200
    })
    applyBootstrapEvent({
      type: 'stage',
      name: 'rebuild',
      state: 'running'
    })
    applyBootstrapEvent({
      type: 'log',
      stage: 'rebuild',
      line: 'packaging...',
      stream: 'stdout'
    })

    applyBootstrapEvent({
      type: 'manifest',
      stages,
      protocolVersion: null
    })

    expect($bootstrap.get().currentStage).toBe('rebuild')
    expect($bootstrap.get().logs).toHaveLength(1)
    expect($bootstrap.get().stages.update).toMatchObject({
      state: 'succeeded',
      durationMs: 1200
    })
    expect($bootstrap.get().stages.rebuild).toMatchObject({
      state: 'running'
    })
    expect($progress.get()).toMatchObject({
      done: 1,
      current: 2,
      total: 3
    })
  })

  it('marks unfinished stages complete when the backend reports completion', () => {
    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'succeeded'
    })

    applyBootstrapEvent({
      type: 'complete',
      installRoot: '/tmp/hermes-agent',
      marker: null
    })

    expect($bootstrap.get().status).toBe('completed')
    expect($bootstrap.get().currentStage).toBeNull()
    expect($progress.get()).toEqual({
      done: 3,
      current: 3,
      total: 3,
      fraction: 1
    })
  })
})
