import { expect, it } from 'vitest'

import type { CatalogEntry, ConnectionRequest, ConnectionTarget } from '@/store/connection-request'
import { pluginOutcomesFrom } from '@/store/onboarding-plugin-outcomes'

const row = (
  name: string,
  state: ConnectionTarget['state'],
  kind: ConnectionTarget['kind'] = 'plugin'
): ConnectionTarget => ({
  action: 'install',
  connectionId: '',
  connectUrl: null,
  detail: state === 'failed' ? 'nope' : '',
  discoveryError: null,
  instructions: null,
  kind,
  name,
  requiredEnv: [],
  state,
  tools: state === 'connected' ? ['mcp__x__y'] : []
})

const catalog = (skill: string): CatalogEntry => ({
  appState: null,
  description: '',
  display: '',
  hasDesktopHalf: false,
  platforms: [],
  repo: null,
  requirements: [],
  scan: null,
  sha: null,
  skill,
  subdir: null,
  targetProfile: 'default',
  tier: null
})

const request = (settled: boolean, targets: ConnectionTarget[]): ConnectionRequest => ({
  deadlineAt: 1,
  opId: 'op',
  seq: 1,
  sessionId: 's',
  settled,
  settledBy: settled ? 'continue' : null,
  targets,
  toolCallId: 't'
})

it('maps a settled card to installed / failed / skipped per plugin row, whether or not the user acted', () => {
  const targets = [
    { ...row('a', 'connected'), catalog: catalog('agent-plugin-a:a') },
    row('b', 'failed'),
    row('c', 'pending'),
    row('gmail', 'connected', 'connector')
  ]

  expect(pluginOutcomesFrom(request(false, targets))).toBeNull()
  expect(pluginOutcomesFrom(request(true, targets))).toEqual({
    a: { detail: '', skill: 'agent-plugin-a:a', state: 'installed', tools: ['mcp__x__y'] },
    b: { detail: 'nope', skill: '', state: 'failed', tools: [] },
    c: { detail: '', skill: '', state: 'skipped', tools: [] }
  })
})
