import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { setCloudAgentStarred, starredCloudAgentIds } from './cloud-agent-stars'

describe('cloud agent star metadata', () => {
  test('sanitizes persisted ids and drops empty or duplicate entries', () => {
    assert.deepEqual(starredCloudAgentIds([' agent-a ', '', null, 'agent-a', 'agent-b']), ['agent-a', 'agent-b'])
    assert.deepEqual(starredCloudAgentIds('not-an-array'), [])
  })

  test('adds and removes one id while preserving unrelated connection document fields', () => {
    const config = {
      mode: 'cloud',
      profiles: { work: { mode: 'remote', url: 'https://work.example' } },
      remote: { url: 'https://cloud.example' },
      starredCloudAgentIds: ['agent-a'],
      unrelated: { keep: true }
    }

    const starred = setCloudAgentStarred(config, 'agent-b', true)
    assert.deepEqual(starred.starredCloudAgentIds, ['agent-a', 'agent-b'])
    assert.deepEqual(starred.unrelated, config.unrelated)
    assert.deepEqual(starred.remote, config.remote)
    assert.deepEqual(starred.profiles, config.profiles)

    assert.deepEqual(setCloudAgentStarred(starred, 'agent-a', false).starredCloudAgentIds, ['agent-b'])
  })
})
