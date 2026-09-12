import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const desktop = fileURLToPath(new URL('../', import.meta.url))
const clarifySourceFiles = [
  'src/app/chat/session-tile.tsx',
  'src/app/chat/session-tile.test.tsx',
  'src/app/chat/session-view.tsx',
  'src/app/chat/session-view.test.ts',
  'src/app/session/hooks/use-session-actions.test.tsx',
  'src/app/session/hooks/use-session-actions/index.ts',
  'src/app/session/hooks/use-session-actions/utils.ts',
  'src/app/session/hooks/use-session-actions/utils.test.ts',
  'src/app/session/hooks/use-session-actions/restore-pending-clarify.test.ts',
  'src/app/session/hooks/use-session-actions/restore-pending-clarify.ts',
  'src/app/session/hooks/use-session-actions/transcript-provenance.test.ts',
  'src/app/session/hooks/use-session-actions/transcript-provenance.ts',
  'src/app/session/hooks/use-session-state-cache.test.tsx',
  'src/app/session/hooks/use-session-state-cache.ts',
  'src/components/assistant-ui/thread/clarify-stream-visibility.test.tsx',
  'src/store/session-states.ts',
  'src/store/session-transcript-view.ts',
  'src/store/session-transcript-view.test.ts'
]

function run(args) {
  const result = spawnSync(process.execPath, args, {
    cwd: desktop,
    shell: false,
    encoding: 'utf8',
    timeout: 240_000,
    maxBuffer: 16 * 1024 * 1024
  })
  if (result.stdout) process.stdout.write(result.stdout)
  if (result.stderr) process.stderr.write(result.stderr)
  assert.ifError(result.error)
  assert.equal(result.signal, null)
  assert.equal(result.status, 0, 'Child process must finish successfully')
}

test('pending clarify navigation full regressions', () => {
  run([
    fileURLToPath(new URL('../../../node_modules/vitest/vitest.mjs', import.meta.url)),
    'run', '--project=ui', '--maxWorkers=1',
    'src/app/session/hooks/use-session-actions.test.tsx',
    'src/app/session/hooks/use-session-actions/utils.test.ts',
    'src/app/session/hooks/use-session-state-cache.test.tsx',
    'src/app/session/hooks/use-session-actions/restore-pending-clarify.test.ts',
    'src/app/session/hooks/use-session-actions/transcript-provenance.test.ts',
    'src/app/session/hooks/use-message-stream/clarify-hydration.test.tsx',
    'src/store/clarify.test.ts',
    'src/store/session-transcript-view.test.ts',
    'src/store/session-states.test.ts',
    'src/app/chat/session-view.test.ts',
    'src/app/chat/session-tile.test.tsx',
    'src/components/assistant-ui/clarify-tool.test.tsx',
    'src/components/assistant-ui/thread/clarify-stream-visibility.test.tsx'
  ])
})

test('desktop TypeScript check', () => {
  run([
    fileURLToPath(new URL('../../../node_modules/typescript/bin/tsc', import.meta.url)),
    '-p', 'tsconfig.json', '--noEmit', '--pretty', 'false'
  ])
})

test('clarify navigation targeted ESLint check', () => {
  run([
    fileURLToPath(new URL('../../../node_modules/eslint/bin/eslint.js', import.meta.url)),
    ...clarifySourceFiles
  ])
})
