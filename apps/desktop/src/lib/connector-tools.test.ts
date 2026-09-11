import { expect, it } from 'vitest'

import { connectionRows, connectorAuthorizationUrl, connectorCalls, connectorTitle } from './connector-tools'

it('reads the real batched contract and preserves distinct app outcomes', () => {
  expect(
    connectionRows(
      { connectors: ['gmail', 'slack'] },
      JSON.stringify({
        results: [
          { connector: 'gmail', status: 'initiated', connect_url: 'https://connect.example/link' },
          { connector: 'slack', connected: true }
        ]
      })
    )
  ).toEqual([{ connector: 'gmail' }, { connector: 'slack', connected: true }])
  expect(connectorTitle('googlecalendar')).toBe('Google Calendar')
  expect(
    connectorCalls('tool_call', { calls: [{ name: 'connectors__gmail__SEND_EMAIL', arguments: { subject: 'test' } }] })
  ).toEqual([{ name: 'connectors__gmail__SEND_EMAIL', arguments: { subject: 'test' } }])
  expect(
    connectionRows(
      { connectors: [{ connector: 'gmail', connected: true, name: 'Mail' }] },
      { connectors: [{ connector: 'gmail', connected: 'false', name: 42, enabled: false }] }
    )
  ).toEqual([{ connector: 'gmail', connected: true, name: 'Mail', enabled: false }])
})

it('refuses unsafe authorization URLs without rewriting identifiers', () => {
  for (const url of [
    'javascript:alert(1)',
    'file:///tmp/x',
    'http://connect.example/x',
    'https://user:secret@connect.example/x'
  ]) {
    expect(connectorAuthorizationUrl(url)).toBeNull()
  }

  expect(connectorAuthorizationUrl('https://connect.example/link?id=original')).toBe(
    'https://connect.example/link?id=original'
  )
})
