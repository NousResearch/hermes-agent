import { describe, expect, it } from 'vitest'

import {
  coerceDraft,
  draftAgentFromIntent,
  draftPrompt,
  parseDraftJson
} from './agent-wizard'

describe('parseDraftJson', () => {
  it('parses a clean JSON object', () => {
    const draft = parseDraftJson('{"name":"fiscal-bot","title":"Analista"}')

    expect(draft?.name).toBe('fiscal-bot')
  })

  it('extracts JSON from a fenced or prosed response', () => {
    const raw = 'Here is your agent:\n```json\n{"name":"auditor","title":"Auditor"}\n```\nHope it helps!'

    expect(parseDraftJson(raw)?.name).toBe('auditor')
  })

  it('returns null on prose-only or malformed output', () => {
    expect(parseDraftJson('I cannot help with that.')).toBeNull()
    expect(parseDraftJson('{"name": broken')).toBeNull()
    expect(parseDraftJson('')).toBeNull()
  })
})

describe('coerceDraft', () => {
  it('slugifies and clamps every field', () => {
    const draft = coerceDraft({
      color: '#FF0000',
      description: 'x'.repeat(500),
      groupMembers: ['a', 42, 'b'],
      name: 'Fiscal Bot!',
      soul: 'y'.repeat(5000),
      suggestsGroup: 'yes',
      title: 'z'.repeat(200)
    } as never)

    expect(draft.name).toBe('fiscal-bot')
    expect(draft.description.length).toBe(200)
    expect(draft.soul.length).toBe(2000)
    expect(draft.title.length).toBe(80)
    // Color not on the whitelist -> null (the dialog keeps its swatch pick).
    expect(draft.color).toBeNull()
    // Non-string members dropped; suggestsGroup coerced to boolean.
    expect(draft.groupMembers).toEqual(['a', 'b'])
    expect(draft.suggestsGroup).toBe(false)
  })

  it('accepts a whitelisted color and a valid slug', () => {
    const draft = coerceDraft({ color: '#38BDF8', name: 'legal-checker' })

    expect(draft.color).toBe('#38bdf8')
    expect(draft.name).toBe('legal-checker')
  })

  it('returns the empty draft on null input', () => {
    expect(coerceDraft(null)).toEqual({
      color: null,
      description: '',
      groupMembers: [],
      name: '',
      soul: '',
      suggestsGroup: false,
      title: ''
    })
  })
})

describe('draftAgentFromIntent', () => {
  it('runs one cli.exec -z turn and coerces the JSON reply', async () => {
    const calls: Array<{ method: string; argv: string[] }> = []
    const request = async (method: string, params: { argv?: string[] }) => {
      calls.push({ argv: params.argv ?? [], method })

      return { code: 0, output: '{"name":"nf-reader","title":"Leitor de NF","soul":"Você lê notas.","color":"#22c55e","groupMembers":[],"suggestsGroup":false}' }
    }

    const draft = await draftAgentFromIntent(request, 'um bot que le notas fiscais')

    expect(calls[0].method).toBe('cli.exec')
    expect(calls[0].argv[0]).toBe('-z')
    expect(calls[0].argv[1]).toContain('notas fiscais')
    expect(draft.name).toBe('nf-reader')
    expect(draft.title).toBe('Leitor de NF')
    expect(draft.color).toBe('#22c55e')
  })

  it('degrades to the empty draft when the exec fails', async () => {
    const request = async () => {
      throw new Error('gateway down')
    }

    const draft = await draftAgentFromIntent(request, 'qualquer coisa')

    expect(draft.name).toBe('')
    expect(draft.soul).toBe('')
  })

  it('short-circuits on an empty intent (no LLM call)', async () => {
    let called = false
    const request = async () => {
      called = true

      return { code: 0, output: '{}' }
    }

    await draftAgentFromIntent(request, '   ')

    expect(called).toBe(false)
  })
})

describe('draftPrompt', () => {
  it('carries the user intent and the roster names', () => {
    const prompt = draftPrompt('monitora preços', ['atlas', 'dev'])

    expect(prompt).toContain('monitora preços')
    expect(prompt).toContain('atlas, dev')
    expect(prompt).toContain('"name"')
  })
})
