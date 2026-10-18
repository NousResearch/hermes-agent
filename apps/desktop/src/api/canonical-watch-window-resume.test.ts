// @vitest-environment node
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test } from 'vitest'

import { primaryResumeParams } from '@/app/session/hooks/use-session-actions/resume-owner'

import { CanonicalDesktopProtocol } from './canonical-protocol'

// N93: every Desktop resume frame (watch window included) carries only keys the owner's generated
// resume contract admits; a watch window, which skips the REST prefetch, asks for the transcript.
const doc = JSON.parse(readFileSync(join(process.cwd(), '..', 'shared', 'src', 'gateway-contract.openrpc.json'), 'utf8'))
const resumeRef = doc['x-canonical-methods'].find((m: { name: string }) => m.name === 'session.resume').params[0].schema.$ref
const RESUME_KEYS = new Set(Object.keys(doc.components.schemas[resumeRef.split('/').at(-1)].properties))

test.each([
  ['watch window', true, undefined],
  ['cold resume', false, undefined],
  ['authoritative snapshot', false, true]
] as const)('%s: the canonical resume frame fits the owner contract', (_label, watchWindow, snapshot) => {
  const frame = new CanonicalDesktopProtocol().prepare('session.resume', primaryResumeParams('s', watchWindow, snapshot, 'work'))
  expect(Object.keys(frame).filter(key => !RESUME_KEYS.has(key))).toEqual([])
  expect(frame).toMatchObject({ session_id: 's', profile: 'work' })

  if (watchWindow) { expect(frame).toMatchObject({ omit_messages: false, defer_history: false }) }
})
