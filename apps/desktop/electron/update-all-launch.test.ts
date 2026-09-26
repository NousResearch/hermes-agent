import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { createUpdateAllLaunch, takeUpdateAllLaunchArg } from './update-all-launch'

const directories: string[] = []

function requestFile(id = 'a'.repeat(32)) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-all-'))
  directories.push(directory)
  const file = path.join(directory, 'request.json')
  const request = { version: 1, id, state: 'pending', launcher_pids: [123], expires_at: Date.now() + 5_000 }
  fs.writeFileSync(file, JSON.stringify(request))

  return { file, request }
}

afterEach(() => {
  vi.useRealTimers()

  for (const directory of directories.splice(0)) {
    fs.rmSync(directory, { recursive: true, force: true })
  }
})

describe('Desktop launch update intent', () => {
  it('consumes the switch, waits for commit/launcher exit and renderer, and coalesces without replay', async () => {
    vi.useFakeTimers()
    let alive = true
    const deliver = vi.fn()
    const reportError = vi.fn()
    const controller = createUpdateAllLaunch({ deliver, reportError, isPidAlive: () => alive })
    const first = requestFile()
    const argv = ['Hermes', '--other', `--hermes-update-all-request=${first.file}`]
    expect(takeUpdateAllLaunchArg(argv)).toBe(first.file)
    expect(takeUpdateAllLaunchArg(argv)).toBeNull()
    expect(argv).toEqual(['Hermes', '--other'])
    const pending = controller.request(first.file)
    expect(JSON.parse(fs.readFileSync(first.file, 'utf8')).state).toBe('accepted')
    const duplicate = requestFile('b'.repeat(32))
    await controller.request(duplicate.file)
    expect(JSON.parse(fs.readFileSync(duplicate.file, 'utf8')).state).toBe('coalesced')
    fs.writeFileSync(first.file, JSON.stringify({ ...first.request, state: 'committed' }))
    await vi.advanceTimersByTimeAsync(100)
    expect(deliver).not.toHaveBeenCalled()
    alive = false
    await vi.advanceTimersByTimeAsync(100)
    await pending
    expect(deliver).not.toHaveBeenCalled()
    controller.ready()
    expect(deliver).toHaveBeenCalledExactlyOnceWith(first.request.id)
    controller.rendererGone()
    controller.ready()
    expect(deliver).toHaveBeenCalledTimes(1)
    controller.complete('wrong-request')
    const stillBusy = requestFile('c'.repeat(32))
    await controller.request(stillBusy.file)
    expect(JSON.parse(fs.readFileSync(stillBusy.file, 'utf8')).state).toBe('interrupted')
    controller.complete(first.request.id)
    const next = requestFile('d'.repeat(32))
    const later = controller.request(next.file)
    fs.writeFileSync(next.file, JSON.stringify({ ...next.request, state: 'committed' }))
    await vi.advanceTimersByTimeAsync(100)
    await later
    expect(deliver).toHaveBeenCalledTimes(2)
    expect(reportError).not.toHaveBeenCalled()
  })

  it('never dispatches an uncommitted, expired, or invalid file request', async () => {
    vi.useFakeTimers()
    const deliver = vi.fn()
    const reportError = vi.fn()
    const controller = createUpdateAllLaunch({ deliver, reportError, isPidAlive: () => false })
    controller.ready()
    const first = requestFile()
    const pending = controller.request(first.file)
    await vi.advanceTimersByTimeAsync(6_000)
    await pending
    expect(deliver).not.toHaveBeenCalled()
    const invalid = requestFile()
    fs.writeFileSync(invalid.file, JSON.stringify({ ...invalid.request, launcher_pids: [-1] }))
    await controller.request(invalid.file)
    expect(deliver).not.toHaveBeenCalled()
    expect(reportError).toHaveBeenCalledTimes(2)
    expect(JSON.parse(fs.readFileSync(invalid.file, 'utf8')).state).toBe('pending')
  })
})
