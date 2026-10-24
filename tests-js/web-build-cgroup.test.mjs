import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { afterEach, expect, test } from 'vitest'
import { cgroupMemoryLimitMb } from '../scripts/build/web.mjs'

const temporary = []
afterEach(() => { for (const dir of temporary.splice(0)) rmSync(dir, { recursive: true, force: true }) })

// Build a fake cgroup tree (plus a fake /proc/self/cgroup inside it) and read the
// limit the way the build helper does.
function limitMb(files) {
  const root = mkdtempSync(path.join(tmpdir(), 'web-build-cgroup '))
  temporary.push(root)
  for (const [name, content] of Object.entries(files)) {
    const file = path.join(root, name)
    mkdirSync(path.dirname(file), { recursive: true })
    writeFileSync(file, content)
  }
  return cgroupMemoryLimitMb({
    cgroupRoot: root,
    procCgroup: path.join(root, 'proc/self/cgroup'),
  })
}

const gib = n => `${n * 1024 ** 3}\n`

// A systemd unit's MemoryHigh=/MemoryMax= lands on the unit's own cgroup while the
// cgroup root reads `max`, so a root-only read called a capped service unconstrained
// and handed Node the 4 GB ceiling — then the cgroup OOM-killed the build.
test('own cgroup limit wins over an unconstrained root', () => {
  expect(limitMb({
    'memory.max': 'max\n',
    'proc/self/cgroup': '0::/user.slice/user-1000.slice/hermes.service\n',
    'user.slice/user-1000.slice/hermes.service/memory.max': gib(2),
  })).toBe(2048)
})

test('own memory.high beats own memory.max', () => {
  expect(limitMb({
    'proc/self/cgroup': '0::/user.slice/hermes.service\n',
    'user.slice/hermes.service/memory.high': gib(3),
    'user.slice/hermes.service/memory.max': gib(8),
  })).toBe(3072)
})

test('root limit still applies when the process sits at the root', () => {
  expect(limitMb({
    'memory.max': gib(2),
    'proc/self/cgroup': '0::/\n',
  })).toBe(2048)
})

test('cgroup v1 limit is the last resort', () => {
  expect(limitMb({ 'memory/memory.limit_in_bytes': gib(4) })).toBe(4096)
})

test('unconstrained when nothing is readable', () => {
  expect(limitMb({ 'proc/self/cgroup': '0::/user.slice/hermes.service\n' })).toBeNull()
})
