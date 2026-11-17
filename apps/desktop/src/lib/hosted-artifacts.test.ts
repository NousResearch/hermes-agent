import { describe, expect, it } from 'vitest'

import {
  HOSTED_ARTIFACT_TICKET_TTL_SECONDS,
  hostedArtifactFrameFor,
  isHostedArtifactCardTool,
  MAX_UI_MESSAGE_LENGTH,
  parseHostedArtifactResult,
  UI_MESSAGE_THROTTLE_MS,
  uiMessageToPrompt
} from './hosted-artifacts'

const ID = '01j9z8y7x6w5v4t3s2r1q0p9n8'
const CONTENT_URL = `http://${ID}.artifacts.localhost:8787/__av?t=ticket`
const VIEWER_URL = `https://portal.example/a/${ID}`
const PAYLOAD = { content_url: CONTENT_URL, id: ID, title: 'Chart', version: 3, viewer_url: VIEWER_URL }

const EXPECTED = { contentUrl: CONTENT_URL, id: ID, title: 'Chart', version: 3, viewerUrl: VIEWER_URL }

describe('hosted artifact tool names', () => {
  it('renders create, update and show; list, get and share stay activity rows', () => {
    for (const tool of ['artifact_create', 'artifact_update', 'artifact_show']) {
      expect(isHostedArtifactCardTool(`mcp__hermes_artifacts__${tool}`)).toBe(true)
    }

    for (const tool of ['artifact_get', 'artifact_list', 'artifact_share']) {
      expect(isHostedArtifactCardTool(`mcp__hermes_artifacts__${tool}`)).toBe(false)
    }

    expect(isHostedArtifactCardTool('mcp__other__artifact_show')).toBe(false)
    expect(isHostedArtifactCardTool(undefined)).toBe(false)
  })
})

describe('parseHostedArtifactResult', () => {
  it('reads the bare JSON text, the handler envelope and structuredContent', () => {
    expect(parseHostedArtifactResult(JSON.stringify(PAYLOAD))).toEqual(EXPECTED)
    expect(parseHostedArtifactResult(JSON.stringify({ result: JSON.stringify(PAYLOAD) }))).toEqual(EXPECTED)
    expect(parseHostedArtifactResult({ result: 'created', structuredContent: PAYLOAD })).toEqual(EXPECTED)
    expect(parseHostedArtifactResult(PAYLOAD)).toEqual(EXPECTED)
  })

  it('treats a create result without a title as untitled', () => {
    const { title: _title, ...created } = PAYLOAD

    expect(parseHostedArtifactResult(JSON.stringify(created))?.title).toBe('')
  })

  it.each([
    ['a stale-base error', JSON.stringify({ error: 'stale_base', head_version: 4, message: 'stale' })],
    ['an enveloped error', JSON.stringify({ result: JSON.stringify({ error: 'not_found', message: 'gone' }) })],
    ['a handler error', JSON.stringify({ error: 'MCP tool returned an error' })],
    ['an error envelope that still carries a payload', { error: 'tool failed', structuredContent: PAYLOAD }],
    ['a payload flagged as an error', JSON.stringify({ ...PAYLOAD, error: 'stale_base' })],
    [
      'an enveloped payload flagged as an error',
      JSON.stringify({ result: JSON.stringify({ ...PAYLOAD, error: 'stale_base' }) })
    ],
    ['an id that is not an artifact id', JSON.stringify({ ...PAYLOAD, id: '../etc' })],
    ['a non-integer version', JSON.stringify({ ...PAYLOAD, version: '3' })],
    ['a non-web viewer URL', JSON.stringify({ ...PAYLOAD, viewer_url: 'javascript:alert(1)' })],
    ['prose', 'Created the artifact.'],
    ['nothing', undefined]
  ])('returns null for %s', (_label, result) => {
    expect(parseHostedArtifactResult(result)).toBeNull()
  })
})

describe('hostedArtifactFrameFor', () => {
  const artifact = parseHostedArtifactResult(PAYLOAD)!
  const dev = { dev: true }

  it('frames a fresh ticket on the artifact origin', () => {
    expect(hostedArtifactFrameFor(artifact, { completedAt: 1000, nowSeconds: 1010, policy: dev })).toBe(
      `http://${ID}.artifacts.localhost:8787`
    )
    expect(hostedArtifactFrameFor(artifact, { policy: dev })).toBe(`http://${ID}.artifacts.localhost:8787`)
  })

  it('links instead once the ticket has expired', () => {
    expect(
      hostedArtifactFrameFor(artifact, {
        completedAt: 1000,
        nowSeconds: 1000 + HOSTED_ARTIFACT_TICKET_TTL_SECONDS,
        policy: dev
      })
    ).toBeNull()
  })

  it('links when this build may not frame the origin', () => {
    expect(hostedArtifactFrameFor(artifact, { policy: { dev: false } })).toBeNull()
  })
})

describe('uiMessageToPrompt', () => {
  const message = (content: unknown[], role = 'user') => ({ content, role })

  it('turns user text blocks into one prompt', () => {
    expect(uiMessageToPrompt(message([{ text: ' Make it blue ', type: 'text' }]), 5000, 0)).toEqual({
      kind: 'submit',
      text: 'Make it blue'
    })
    expect(
      uiMessageToPrompt(
        message([
          { text: 'one', type: 'text' },
          { text: 'two', type: 'text' }
        ]),
        5000,
        0
      )
    ).toEqual({ kind: 'submit', text: 'one\n\ntwo' })
  })

  it.each([
    ['a non-user role', message([{ text: 'hi', type: 'text' }], 'assistant')],
    ['an image block', message([{ data: 'xx', mimeType: 'image/png', type: 'image' }])],
    [
      'text mixed with an image',
      message([
        { text: 'hi', type: 'text' },
        { data: 'xx', type: 'image' }
      ])
    ],
    ['blank text', message([{ text: '   ', type: 'text' }])],
    ['no content', { role: 'user' }],
    ['not an object', 'hi']
  ])('refuses %s', (_label, params) => {
    expect(uiMessageToPrompt(params, 5000, 0)).toEqual({ kind: 'reject', reason: 'invalid' })
  })

  it('refuses an over-length message instead of cutting it', () => {
    expect(
      uiMessageToPrompt(message([{ text: 'x'.repeat(MAX_UI_MESSAGE_LENGTH + 1), type: 'text' }]), 5000, 0)
    ).toEqual({
      kind: 'reject',
      reason: 'too_long'
    })
  })

  it('throttles to human speed', () => {
    const params = message([{ text: 'again', type: 'text' }])

    expect(uiMessageToPrompt(params, 5000, 5000 - UI_MESSAGE_THROTTLE_MS + 1)).toEqual({
      kind: 'reject',
      reason: 'throttled'
    })
    expect(uiMessageToPrompt(params, 5000, 5000 - UI_MESSAGE_THROTTLE_MS)).toEqual({ kind: 'submit', text: 'again' })
  })
})
