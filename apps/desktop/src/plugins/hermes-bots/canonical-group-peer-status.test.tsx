import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { answerGroupRequests, localGatewaySdkMock, PEER_HOME, peerRosterRow } from './canonical-group-contract-test-utils'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $peerJoinErrors } from './canonical-group-peers'
import { $canonicalGroupBindings } from './canonical-group-registry'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import type * as DataModule from './data'
import wire from './fixtures/peer-control-responses.json'
import { $groupChats } from './group-chat'
import { translateBotsIn } from './i18n-test-helper'

const { request, locale, union } = vi.hoisted(() => ({ request: vi.fn(), locale: { value: 'en' }, union: { profiles: [] as unknown[] } }))
vi.mock('@hermes/plugin-sdk', async importOriginal =>
  localGatewaySdkMock(await importOriginal<typeof HermesSdk>(), request, () => translateBotsIn(locale.value)))
vi.mock('./data', async importOriginal => ({ ...await importOriginal<typeof DataModule>(), cachedUnionRoster: () => union }))

const binding = { connectionId: 'local', profile: 'default', roomId: 'linked' }
const locales = Object.keys(CANONICAL_GROUP_LOCALES) as (keyof typeof CANONICAL_GROUP_LOCALES)[]
const cases = locales.flatMap(language => (['offline', 'notJoined'] as const).map(state => ({ language, state, connection: 'absent' })))
cases.push(...['matching', 'another computer', 'unreachable', 'contradictory route'].map(connection => ({ language: 'en' as const, state: 'offline' as const, connection })))

beforeEach(() => {
  $canonicalGroupBindings.set({})
  $groupChats.set({})
  $peerJoinErrors.set({})
  union.profiles = []
})
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear() })

it.each(cases)('shows the gateway’s $state peer with a $connection connection ($language)', async ({ language, state, connection }) => {
  locale.value = language
  const labels = CANONICAL_GROUP_LOCALES[language]
  const snapshot = JSON.parse(wire[state]).result
  const member = snapshot.room.members.find((row: { member_id: string }) => row.member_id === 'reviewer')

  if (connection !== 'absent') {
    union.profiles = [{ ...peerRosterRow('mini', 'Member computer'),
      installId: connection === 'another computer' ? 'another-installation' : member.target.installation_id.slice('install:'.length),
      sourceReachable: connection !== 'unreachable', ...(connection === 'contradictory route' ? { connectionId: 'other' } : {}) }]
  }

  answerGroupRequests(request, { local: method => ({
    'groups.capabilities': PEER_HOME, 'groups.state': snapshot, 'groups.log': { events: [] }
  } as Record<string, unknown>)[method] })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const text = (state === 'offline' ? labels.peerAway : labels.peerNotJoined).replace('{name}', 'reviewer')
  const row = (await screen.findByText(text)).closest('[role="status"]') as HTMLElement
  const roster = within(screen.getByRole('list', { name: labels.members }))
  expect(roster.getByText(state === 'offline' ? labels.peerOfflineBadge : labels.peerNotJoinedBadge)).toBeTruthy()
  expect(roster.queryByText(state === 'offline' ? labels.peerNotJoinedBadge : labels.peerOfflineBadge)).toBeNull()
  expect(Boolean(within(row).queryByRole('button', { name: 'Retry' }))).toBe(state === 'notJoined' || connection === 'matching')

  if (language === 'en' && state === 'offline') {expect(row.textContent).toContain('The next message to it will try again.')}
})
