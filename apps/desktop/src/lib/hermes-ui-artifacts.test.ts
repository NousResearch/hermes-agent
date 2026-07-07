import { describe, expect, it } from 'vitest'

import { type HermesUiChecklistArtifact, parseHermesUiArtifact, stableArtifactStorageKey } from './hermes-ui-artifacts'

const validChecklist = {
  description: 'Interactive checklist rendered inline by Hermes Desktop.',
  id: 'Obsidian Source Truth',
  items: [
    { id: 'profile-vault', label: 'Active profile: office-work' },
    { id: 'source-truth', label: 'Obsidian is the source of truth.' }
  ],
  title: 'Obsidian source-of-truth policy',
  type: 'checklist'
}

const obsidianPolicyChecklist = {
  description: 'Operational source-of-truth checklist for Obsidian-backed durable context.',
  id: 'obsidian-source-of-truth-policy',
  items: [
    {
      id: 'obsidian-profile-vault',
      label:
        'Active profile: office-work; canonical vault: /media/endlessblink/data/app-data/sync/Dropbox/OBSIDIAN_SYNCED; visible workspace: /media/endlessblink/data/app-data/sync/Dropbox/OBSIDIAN_SYNCED/MAIN VULT'
    },
    {
      id: 'obsidian-source-truth',
      label: 'Obsidian is the source of truth for durable context. Built-in memory and conversation summaries are only pointers/caches.'
    },
    {
      id: 'obsidian-routing-policy',
      label: 'Routing policy note: MAIN VULT/_System/Hermes Governance/Hermes Vault Routing Policy.md.'
    },
    {
      id: 'obsidian-no-hermes-memory',
      label: 'Never create or write notes under Hermes Memory/; route durable notes under MAIN VULT/.'
    },
    {
      id: 'obsidian-create-edit-routing',
      label:
        'Create/edit routing: _System/ for Hermes governance/reports; _System/Hermes Knowledge Graph/ for internal agent/profile context; 🚀 My Projects/, 💼 Work/, 📦 My Stuff/ only for user-facing/project-facing content.'
    },
    {
      id: 'obsidian-read-relevant-note',
      label:
        'For project/profile questions, continuation after long chats, setup/MCP/tooling details, or durable decisions: read the relevant Obsidian note before answering.'
    },
    {
      id: 'obsidian-update-durable-knowledge',
      label:
        'If this turn creates/changes durable knowledge, update/create an Obsidian note before final response; keep Hermes memory compact.'
    },
    {
      id: 'obsidian-turn-ledgers',
      label: 'Turn ledgers go to MAIN VULT/_System/Hermes Turn Logs; curated facts still belong in project/profile notes.'
    },
    {
      id: 'obsidian-start-indexes',
      label:
        'Start indexes: MAIN VULT/_System/INDEX.md; MAIN VULT/_System/Hermes Knowledge Graph/Hermes Knowledge Graph.md; MAIN VULT/_System/Hermes Governance/Legacy Hermes Memory Index.md; MAIN VULT/_System/Hermes Governance/Hermes Vault Routing Policy.md.'
    }
  ],
  title: 'Obsidian source-of-truth policy',
  type: 'checklist'
}

describe('parseHermesUiArtifact', () => {
  it('parses a valid checklist artifact', () => {
    const result = parseHermesUiArtifact(JSON.stringify(validChecklist))

    expect(result.ok).toBe(true)
    expect(result.ok && result.artifact.items).toHaveLength(2)
  })

  it('parses the Obsidian source-of-truth policy checklist', () => {
    const result = parseHermesUiArtifact(JSON.stringify(obsidianPolicyChecklist))

    expect(result.ok).toBe(true)
    expect(result.ok && result.artifact.items).toHaveLength(9)
    expect(result.ok && result.artifact.items[4]?.label).toContain('🚀 My Projects')
  })

  it('preserves an explicit rtl direction for Hebrew artifacts', () => {
    const result = parseHermesUiArtifact(
      JSON.stringify({
        ...validChecklist,
        direction: 'rtl',
        items: [{ id: 'hebrew-item', label: 'פרופיל פעיל: office-work' }],
        title: 'מדיניות מקור האמת'
      })
    )

    expect(result.ok).toBe(true)
    expect(result.ok && result.artifact.direction).toBe('rtl')
  })

  it('rejects invalid JSON', () => {
    expect(parseHermesUiArtifact('{ nope').ok).toBe(false)
  })

  it('rejects unsupported artifact types', () => {
    const result = parseHermesUiArtifact(JSON.stringify({ type: 'html', value: '<button>Run</button>' }))

    expect(result.ok).toBe(false)
  })

  it('rejects duplicate item ids', () => {
    const result = parseHermesUiArtifact(
      JSON.stringify({ ...validChecklist, items: [{ id: 'same', label: 'A' }, { id: 'same', label: 'B' }] })
    )

    expect(result.ok).toBe(false)
    expect(!result.ok && result.error).toContain('Duplicate')
  })

  it('rejects excessive item counts and string lengths', () => {
    const tooManyItems = Array.from({ length: 101 }, (_, index) => ({ id: `item-${index}`, label: 'Item' }))
    expect(parseHermesUiArtifact(JSON.stringify({ ...validChecklist, items: tooManyItems })).ok).toBe(false)

    expect(parseHermesUiArtifact(JSON.stringify({ ...validChecklist, title: 'x'.repeat(161) })).ok).toBe(false)
    expect(
      parseHermesUiArtifact(
        JSON.stringify({ ...validChecklist, items: [{ id: 'item', label: 'x'.repeat(801) }] })
      ).ok
    ).toBe(false)
  })

  it('uses a stable normalized storage key when an id is present', () => {
    const result = parseHermesUiArtifact(JSON.stringify(validChecklist))

    expect(result.ok && stableArtifactStorageKey(result.artifact as HermesUiChecklistArtifact)).toBe(
      'hermes-ui:checklist:obsidian-source-truth'
    )
  })

  it('uses a stable hash when no id is present', () => {
    const first = { ...validChecklist, id: undefined }
    const second = { items: first.items, title: first.title, description: first.description, type: first.type }
    const firstResult = parseHermesUiArtifact(JSON.stringify(first))
    const secondResult = parseHermesUiArtifact(JSON.stringify(second))

    expect(firstResult.ok && secondResult.ok).toBe(true)
    expect(
      firstResult.ok &&
        secondResult.ok &&
        stableArtifactStorageKey(firstResult.artifact) === stableArtifactStorageKey(secondResult.artifact)
    ).toBe(true)
  })
})
