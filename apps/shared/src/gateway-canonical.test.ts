// A plain-HTTP page (the Webapp opened from another machine) has `crypto.getRandomValues` but no
// `crypto.randomUUID`: session.create still gets a request_id there.
import { afterEach, expect, it, vi } from 'vitest'

import { canonicalRequest } from './gateway-canonical'

afterEach(() => {
  vi.unstubAllGlobals()
})

it('mints the session.create request_id outside a secure context', () => {
  vi.stubGlobal('crypto', { getRandomValues: (bytes: Uint8Array) => bytes.fill(7) })

  const request = canonicalRequest('session.create', {}, { sources: ['tui'], parameters: ['source', 'request_id'] })

  expect(request.params).toEqual({ source: 'tui', request_id: '07070707-0707-4707-8707-070707070707' })
})

// The launch identity: no identity is main's request and wording; another source or a client name is
// honored when the owner advertises it and refused in neutral words when it does not.
const owner = { sources: ['cli', 'tui'], parameters: ['source', 'request_id', 'model', 'client_env'] }

const refusal = (run: () => unknown) => { try { run() } catch (error) { return (error as Error).message } }

it('launches as tui with main\'s params and wording when no identity is given', () => {
  const create = (params: Record<string, unknown>, contract = owner) => canonicalRequest('session.create', params, contract)

  expect(Object.entries(create({ model: 'm', cols: 80, request_id: 'r' }).params))
    .toEqual([['model', 'm'], ['request_id', 'r'], ['source', 'tui']])
  expect(create({ request_id: 'r' }, { ...owner, parameters: [...owner.parameters, 'client'] }).params)
    .toEqual({ request_id: 'r', source: 'tui' })
  expect(refusal(() => create({}, { sources: ['cli'], parameters: [] })))
    .toBe('gateway does not support tui session policy; update/restart the gateway')
  expect(refusal(() => create({ yolo: true }))).toBe('gateway does not support TUI launch options: yolo')
  expect(canonicalRequest('session.create', { request_id: 'r' }, owner, undefined, { source: 'tui' }).params)
    .toEqual({ request_id: 'r', source: 'tui' })
})

it('launches with another advertised source and refuses an unadvertised one in neutral words', () => {
  expect(canonicalRequest('session.create', { request_id: 'r', source: 'gui' }, owner, undefined, { source: 'cli' }).params)
    .toEqual({ request_id: 'r', source: 'cli' })
  expect(refusal(() => canonicalRequest('session.create', {}, owner, undefined, { source: 'tool' })))
    .toBe('gateway does not support tool session policy; update/restart the gateway')
  expect(refusal(() => canonicalRequest('session.create', { yolo: true }, owner, undefined, { source: 'cli' })))
    .toBe('gateway does not support these session options: yolo')
})

it('sends the client name only to an owner that lists client', () => {
  const client = { name: 'example-client', version: '1.2.0' }
  const takes = { ...owner, parameters: [...owner.parameters, 'client'] }

  expect(canonicalRequest('session.create', { request_id: 'r' }, takes, undefined, { client }).params)
    .toEqual({ request_id: 'r', source: 'tui', client })
  expect(canonicalRequest('session.create', { request_id: 'r' }, owner, undefined, { client }).params)
    .toEqual({ request_id: 'r', source: 'tui' })
  expect(refusal(() => canonicalRequest('session.create', { yolo: true }, owner, undefined, { client })))
    .toBe('gateway does not support these session options: yolo')
  expect(refusal(() => canonicalRequest('session.create', {}, { sources: ['cli'], parameters: [] }, undefined, { client })))
    .toBe('gateway does not support tui session policy; update/restart the gateway')
  // The option replaces a client already in the params, and neither reaches an owner that doesn't take it.
  const stale = { request_id: 'r', client: { name: 'other' } }
  expect(canonicalRequest('session.create', stale, takes, undefined, { client }).params)
    .toEqual({ request_id: 'r', source: 'tui', client })
  expect(canonicalRequest('session.create', stale, owner, undefined, { client }).params)
    .toEqual({ request_id: 'r', source: 'tui' })
})
