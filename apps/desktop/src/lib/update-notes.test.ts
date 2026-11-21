import { describe, expect, it } from 'vitest'

import { extractReleaseHighlights, latestMergedPrTitles, updateNotesSource } from './update-notes'

describe('update notes', () => {
  it('summarizes a release as its highlight headlines, and falls back to the first bullet list', () => {
    const major = [
      '# Hermes Agent v0.22.0',
      '> The Threshold Release.',
      '## ✨ Highlights',
      '- **A first launch that sets itself up** — Open the desktop app ([#1](https://x/pull/1) — @a)',
      '- **Connectors: Hermes works inside your apps** — The agent can now act.',
      '- Plain [linked](https://x) bullet with `code` ([#2](https://x/pull/2))',
      '## ⚠️ Upgrade notes',
      '- **Python 3.14 is the runtime** for managed installs.'
    ].join('\n')

    const patch = [
      '**What this patch ships:**',
      '- **Updates now follow tagged releases.** A source install updates to releases.',
      '- **Reasoning survives resume.** Signed thinking replays.',
      '## Updating',
      '- Desktop app: updates itself.'
    ].join('\n')

    expect(extractReleaseHighlights(major)).toEqual([
      'A first launch that sets itself up',
      'Connectors: Hermes works inside your apps',
      'Plain linked bullet with code'
    ])
    expect(extractReleaseHighlights(patch)).toEqual(['Updates now follow tagged releases', 'Reasoning survives resume'])
    expect(
      extractReleaseHighlights(
        '## Updating\r\n- Docker: pull :stable\r\n## Security Fixes\r\n**Auth**\r\n- Spoofed headers reset limits (#1)'
      )
    ).toEqual(['Spoofed headers reset limits'])
    // The dialog clamps the display; the item keeps its whole text for hover.
    expect(extractReleaseHighlights(`## Highlights\n- **${'word '.repeat(60).trim()}**`)[0]).toHaveLength(299)
  })

  it('lists the newest merged PRs, pinned to the update when its commit list is complete', () => {
    const pulls = [
      { title: 'closed unmerged', merged_at: null, merge_commit_sha: 'aaaaaaa1' },
      { title: 'older', merged_at: '2026-10-14T01:00:00Z', merge_commit_sha: 'bbbbbbb1' },
      { title: 'newest', merged_at: '2026-10-14T03:00:00Z', merge_commit_sha: 'ccccccc1' },
      { title: 'installed already', merged_at: '2026-10-13T00:00:00Z', merge_commit_sha: 'ddddddd1' }
    ]

    expect(latestMergedPrTitles(pulls, [], null)).toEqual(['newest', 'older', 'installed already'])
    // Backend rows carry 7-char SHAs; the update holds only these two commits.
    expect(latestMergedPrTitles(pulls, [{ sha: 'ccccccc' }, { sha: 'bbbbbbb' }], 2)).toEqual(['newest', 'older'])
    // An incomplete list (more behind than listed) cannot exclude anything.
    expect(latestMergedPrTitles(pulls, [{ sha: 'ccccccc' }], 300)).toHaveLength(3)

    expect(updateNotesSource({ channel: 'stable', sourceVersion: '0.22.0' })).toEqual({
      kind: 'release',
      tag: 'v0.22.0'
    })
    expect(updateNotesSource({ branch: 'main' })).toEqual({ kind: 'prs' })
    expect(updateNotesSource({ branch: 'my-feature' })).toBeNull()
    expect(updateNotesSource({ branch: 'my-feature', channel: 'main' })).toBeNull()
    expect(updateNotesSource({ channel: 'main' })).toEqual({ kind: 'prs' })
    expect(updateNotesSource({ channel: 'stable', rollback: true, sourceVersion: '0.21.0' })).toBeNull()
  })
})
