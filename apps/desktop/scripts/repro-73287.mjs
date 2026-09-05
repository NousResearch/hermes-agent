// Repro #73287: a hidden tile must not fire the 5s process.list safety-net
// poll. Runs the hidden-pane test before (guard stashed) and after the fix.
import { execSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
const root = fileURLToPath(new URL('.', import.meta.url))
const fix = '../src/app/chat/composer/status-stack/index.tsx'
const test = 'src/app/chat/composer/status-stack/polling-guard.test.tsx -t "hidden tile"'
const check = () => {
  try {
    execSync(`npx vitest run --project ui ${test}`, { cwd: root, stdio: 'pipe' })
    return 'PASS (hidden tile: 1 process.list call, mount seed only)'
  } catch (e) {
    return `FAIL (${(String(e.stdout ?? '') + String(e.stderr ?? '')).replace(/\[[0-9;]*m/g, '').match(/expected \d+ to be \d+/)?.[0] ?? 'see output'})`
  }
}
execSync(`git stash push -q -- ${fix}`, { cwd: root })
console.log('before (no guard):', check())
execSync('git stash pop -q', { cwd: root })
console.log('after (guard):     ', check())
