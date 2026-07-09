/**
 * Tests for electron/ix-skills.ts — user-level SKILL.md drafts and the
 * publish-to-portal flow.
 *
 * Run with: node --test electron/ix-skills.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: drafts must round-trip through the SKILL.md frontmatter
 * (so local Hermes and the desktop agree on title/description/publish state),
 * and publishing must only mark a skill published when the portal REALLY
 * accepted it and returned a document id.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import {
  deleteUserSkill,
  IX_SKILL_TEMPLATES,
  listUserSkills,
  parseSkillMd,
  publishUserSkill,
  readUserSkill,
  saveUserSkill,
  skillSlug,
  userSkillsDir
} from './ix-skills'

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'ix-skills-'))
}

/* ── slug + frontmatter round-trip ───────────────────────────────────────── */

test('skillSlug produces safe folder names', () => {
  assert.equal(skillSlug('Weekly Notifuse Report!'), 'weekly-notifuse-report')
  assert.equal(skillSlug('  --Ünïcode?? '), 'n-code')
  assert.equal(skillSlug('a'.repeat(100)).length, 64)
})

test('save → read round-trips title, description, content and publish state', () => {
  const dir = tmpDir()

  try {
    const saved = saveUserSkill(dir, {
      title: 'Weekly report',
      description: 'Numbers, "quoted"',
      content: '# Skill\n\nDo the thing.'
    })

    assert.equal(saved.id, 'weekly-report')
    assert.equal(saved.publishedId, null)

    const read = readUserSkill(dir, 'weekly-report')

    assert.ok(read)
    assert.equal(read.title, 'Weekly report')
    assert.equal(read.description, 'Numbers, "quoted"')
    assert.equal(read.content, '# Skill\n\nDo the thing.')
    assert.equal(read.publishedId, null)
    assert.ok(read.updatedAt > 0)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('saveUserSkill validates title and content', () => {
  const dir = tmpDir()

  try {
    assert.throws(() => saveUserSkill(dir, { title: '', content: 'x' }), /title/)
    assert.throws(() => saveUserSkill(dir, { title: 'ok', content: '  ' }), /content/i)
    assert.throws(() => saveUserSkill(dir, { title: 'ok', content: 'y'.repeat(20_001) }), /too long/)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('editing a published skill keeps its publishedId', () => {
  const dir = tmpDir()

  try {
    saveUserSkill(dir, { title: 'Keep', content: 'v1' })

    // Simulate an earlier publish by rewriting with a publishedId.
    const md = fs.readFileSync(path.join(dir, 'keep', 'SKILL.md'), 'utf8')

    fs.writeFileSync(
      path.join(dir, 'keep', 'SKILL.md'),
      md.replace('---\n\n', '    publishedId: "adminSkill.keep"\n---\n\n')
    )
    assert.equal(readUserSkill(dir, 'keep')?.publishedId, 'adminSkill.keep')

    const edited = saveUserSkill(dir, { id: 'keep', title: 'Keep', content: 'v2' })

    assert.equal(edited.publishedId, 'adminSkill.keep')
    assert.equal(readUserSkill(dir, 'keep')?.content, 'v2')
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('listUserSkills returns skills newest-first and skips junk folders', () => {
  const dir = tmpDir()

  try {
    saveUserSkill(dir, { title: 'Old', content: 'a' })
    fs.mkdirSync(path.join(dir, 'not-a-skill'))
    fs.writeFileSync(path.join(dir, 'stray.txt'), 'x')

    const newer = saveUserSkill(dir, { title: 'New', content: 'b' })
    // Force distinct ordering regardless of clock resolution.
    const bumped = { ...newer, updatedAt: Date.now() + 1000 }

    fs.writeFileSync(
      path.join(dir, 'new', 'SKILL.md'),
      fs
        .readFileSync(path.join(dir, 'new', 'SKILL.md'), 'utf8')
        .replace(/updatedAt: \d+/, `updatedAt: ${bumped.updatedAt}`)
    )

    const skills = listUserSkills(dir)

    assert.deepEqual(skills.map(s => s.id), ['new', 'old'])
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('deleteUserSkill removes real skills and refuses junk ids', () => {
  const dir = tmpDir()

  try {
    saveUserSkill(dir, { title: 'Gone', content: 'x' })
    assert.equal(deleteUserSkill(dir, 'gone'), true)
    assert.equal(fs.existsSync(path.join(dir, 'gone')), false)
    assert.equal(deleteUserSkill(dir, 'never-existed'), false)
    assert.equal(deleteUserSkill(dir, ''), false)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

/* ── publish flow ────────────────────────────────────────────────────────── */

test('publishUserSkill POSTs to /api/admin/skills and stamps the returned id', async () => {
  const dir = tmpDir()

  try {
    saveUserSkill(dir, { title: 'Launch runbook', description: 'Blurb', content: '# Steps' })

    let seenUrl = ''
    let seenBody: Record<string, unknown> = {}

    const fakeFetch = (async (url: string, init: RequestInit) => {
      seenUrl = url
      seenBody = JSON.parse(String(init.body))

      return new Response(JSON.stringify({ ok: true, id: 'adminSkill.launch-runbook' }), { status: 200 })
    }) as unknown as typeof fetch

    const published = await publishUserSkill(dir, 'launch-runbook', 'https://portal.example.com', fakeFetch, 'me@ix.ai')

    assert.equal(seenUrl, 'https://portal.example.com/api/admin/skills')
    assert.deepEqual(seenBody, {
      label: 'Launch runbook',
      blurb: 'Blurb',
      content: '# Steps',
      updatedBy: 'me@ix.ai'
    })
    assert.equal(published.publishedId, 'adminSkill.launch-runbook')
    // Persisted — a fresh read sees the publish state.
    assert.equal(readUserSkill(dir, 'launch-runbook')?.publishedId, 'adminSkill.launch-runbook')

    // Re-publish targets the SAME portal document.
    await publishUserSkill(dir, 'launch-runbook', 'https://portal.example.com', fakeFetch)
    assert.equal((seenBody as { id?: string }).id, 'adminSkill.launch-runbook')
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('publishUserSkill surfaces portal errors and never marks published', async () => {
  const dir = tmpDir()

  try {
    saveUserSkill(dir, { title: 'Rejected', content: 'x' })

    const failing = (async () =>
      new Response(JSON.stringify({ error: 'content too long' }), { status: 400 })) as unknown as typeof fetch

    await assert.rejects(
      () => publishUserSkill(dir, 'rejected', 'https://portal.example.com', failing),
      /content too long/
    )
    assert.equal(readUserSkill(dir, 'rejected')?.publishedId, null)

    await assert.rejects(
      () => publishUserSkill(dir, 'missing', 'https://portal.example.com', failing),
      /not found/
    )
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

/* ── templates + paths ───────────────────────────────────────────────────── */

test('templates cover the SKILL.md anatomy the team should copy', () => {
  assert.ok(IX_SKILL_TEMPLATES.length >= 3)

  for (const template of IX_SKILL_TEMPLATES) {
    assert.ok(template.title && template.description, template.id)
    assert.match(template.content, /## Goal/)
    assert.match(template.content, /## Steps/)
    assert.match(template.content, /## Output format/)
  }
})

test('userSkillsDir nests under skills/ix-user and parse tolerates no frontmatter', () => {
  assert.equal(userSkillsDir('/home/x/.hermes'), path.join('/home/x/.hermes', 'skills', 'ix-user'))

  const bare = parseSkillMd('raw', '# Just markdown')

  assert.equal(bare.title, 'raw')
  assert.equal(bare.content, '# Just markdown')
  assert.equal(bare.publishedId, null)
})
