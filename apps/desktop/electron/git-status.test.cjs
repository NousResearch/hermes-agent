const assert = require('node:assert/strict')
const test = require('node:test')

const { parseGitStatusPorcelainV2 } = require('./git-status.cjs')

test('parses branch, staged, worktree, untracked, rename, and conflict states', () => {
  const output = [
    '# branch.oid abc123',
    '# branch.head feature/sidebar',
    '# branch.upstream origin/feature/sidebar',
    '# branch.ab +2 -1',
    '1 M. N... 100644 100644 100644 a a src/staged.ts',
    '1 .D N... 100644 100644 000000 a a src/deleted.ts',
    '1 MM N... 100644 100644 100644 a b src/both.ts',
    '2 R. N... 100644 100644 100644 a b R100 src/new name.ts',
    'src/old name.ts',
    'u UU N... 100644 100644 100644 100644 a b c src/conflict.ts',
    '? src/new.ts',
    ''
  ].join('\0')

  const result = parseGitStatusPorcelainV2(output, '/repo')

  assert.deepEqual(result.branch, {
    ahead: 2,
    behind: 1,
    detached: false,
    name: 'feature/sidebar',
    oid: 'abc123',
    upstream: 'origin/feature/sidebar'
  })
  assert.equal(result.entries[0].index, 'modified')
  assert.equal(result.entries[1].worktree, 'deleted')
  assert.equal(result.entries[2].index, 'modified')
  assert.equal(result.entries[2].worktree, 'modified')
  assert.equal(result.entries[3].originalPath, 'src/old name.ts')
  assert.equal(result.entries[4].conflicted, true)
  assert.equal(result.entries[5].untracked, true)
})
