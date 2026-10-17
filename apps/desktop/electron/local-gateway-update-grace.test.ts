import { EventEmitter } from 'node:events'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('node:child_process', () => ({ spawn: vi.fn() }))
import { spawn } from 'node:child_process'
import { runGatewayEnsure } from './local-gateway'

afterEach(() => { vi.useRealTimers(); vi.clearAllMocks() })

function start(options: { timeoutMs?: number; label?: string } = {}) {
  vi.useFakeTimers()
  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn()
  })
  vi.mocked(spawn).mockReturnValue(child as never)
  const result = runGatewayEnsure({ command: 'unused', args: [] }, process.cwd(), process.cwd(), {}, options)

  return { child, result }
}

test.each([
  'finishing an interrupted source update',
  'completing source-update dependencies'
])('an owed update (%s) survives the ordinary ensure deadline and returns its verdict', async banner => {
  const { child, result } = start()
  child.stderr.emit('data', `hermes: ${banner}...\n`)
  await vi.advanceTimersByTimeAsync(75_000)
  expect(child.kill).not.toHaveBeenCalled()
  child.stdout.emit('data', '{"state":"ready"}')
  child.emit('close', 0)
  expect(await result).toMatchObject({ code: 0, stdout: '{"state":"ready"}' })
  await vi.advanceTimersByTimeAsync(30 * 60_000)
  expect(child.kill).not.toHaveBeenCalled()
})

test('split banners survive bounded diagnostic retention; repeated banners cannot extend the total cap', async () => {
  const { child, result } = start()
  await vi.advanceTimersByTimeAsync(60_000)
  child.stderr.emit('data', 'hermes: finishing an interrupted ')
  child.stderr.emit('data', `source update...\n${'x'.repeat(5000)}`)
  await vi.advanceTimersByTimeAsync(29 * 60_000 - 1)
  expect(child.kill).not.toHaveBeenCalled()
  child.stderr.emit('data', 'hermes: completing source-update dependencies...\n')
  await vi.advanceTimersByTimeAsync(1)
  expect(child.kill).toHaveBeenCalledTimes(1)
  child.emit('close', 0)
  expect(await result).toMatchObject({ code: 7, stdout: '', stderr: expect.stringContaining('while completing a source update') })
})

test.each(['ordinary stderr', 'stdout banner'])('%s does not extend a caller-selected deadline', async input => {
  const { child, result } = start({ timeoutMs: 100, label: 'restart client' })
  if (input === 'ordinary stderr') {child.stderr.emit('data', 'still importing\n')}
  else {child.stdout.emit('data', 'finishing an interrupted source update')}
  await vi.advanceTimersByTimeAsync(100)
  expect(child.kill).toHaveBeenCalledTimes(1)
  child.emit('close', 0)
  expect(await result).toMatchObject({ code: 7, stderr: expect.stringContaining('restart client timed out') })
})

test('an update never shortens a longer caller-selected deadline', async () => {
  const { child, result } = start({ timeoutMs: 40 * 60_000 })
  child.stderr.emit('data', 'completing source-update dependencies')
  await vi.advanceTimersByTimeAsync(30 * 60_000)
  expect(child.kill).not.toHaveBeenCalled()
  child.emit('close', 3)
  expect((await result).code).toBe(3)
})

test('a failed launch clears the extended deadline', async () => {
  const { child, result } = start()
  const failed = expect(result).rejects.toThrow('Could not run hermes gateway ensure')
  child.stderr.emit('data', 'completing source-update dependencies')
  child.emit('error', new Error('spawn failed'))
  await failed
  await vi.advanceTimersByTimeAsync(31 * 60_000)
  expect(child.kill).not.toHaveBeenCalled()
})

test('update progress cannot disable the stdout size guard or resurrect an expired client', async () => {
  const { child, result } = start()
  const failed = expect(result).rejects.toThrow('output exceeded limit')
  child.stderr.emit('data', 'finishing an interrupted source update')
  child.stdout.emit('data', 'x'.repeat(65537))
  child.stderr.emit('data', 'completing source-update dependencies')
  await vi.advanceTimersByTimeAsync(1000)
  await failed
  expect(child.kill.mock.calls).toEqual([[], ['SIGKILL']])
})

test('a real child emits an update banner before a delayed protocol verdict', async () => {
  const actual = await vi.importActual<typeof import('node:child_process')>('node:child_process')
  vi.mocked(spawn).mockImplementation(actual.spawn)
  const fs = await import('node:fs/promises')
  const os = await import('node:os')
  const path = await import('node:path')
  const home = await fs.mkdtemp(path.join(os.tmpdir(), 'ensure-update-'))
  const env = { ...process.env, HOME: home, USERPROFILE: home, HERMES_HOME: home, APPDATA: home, LOCALAPPDATA: home }

  try {
    const result = await runGatewayEnsure({
      command: process.execPath,
      args: ['-e', 'process.stderr.write("hermes: finishing an interrupted source update...\\n"); setTimeout(() => process.stdout.write(JSON.stringify({state:"ready"})), 2300)']
    }, home, home, env, { timeoutMs: 2000 })
    expect(result).toMatchObject({ code: 0, stdout: '{"state":"ready"}' })
  } finally {
    await fs.rm(home, { recursive: true, force: true })
  }
}, 10_000)
