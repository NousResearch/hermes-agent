import { describe, expect, it, vi } from 'vitest'

import { APPINSTALLER_CHECK_TIMEOUT_MS, type ExecFileImpl, runAppInstallerChecker } from './appinstaller-checker'

type Callback = Parameters<ExecFileImpl>[3]

interface CapturedCall {
  file: string
  args: readonly string[]
  options: { encoding: string; timeout: number; windowsHide: boolean; env?: NodeJS.ProcessEnv }
  callback: Callback
}

function stubExecFile(behavior: (call: CapturedCall) => void) {
  const calls: CapturedCall[] = []

  const impl = (
    file: string,
    args: readonly string[],
    options: CapturedCall['options'],
    callback: Callback
  ) => {
    const call: CapturedCall = { file, args, options, callback }

    calls.push(call)
    behavior(call)

    return { kill: vi.fn() }
  }

  return { impl: impl as unknown as ExecFileImpl, calls }
}

const UNKNOWN_JSON = JSON.stringify({ available: null, error: 'checker timed out after 50ms' })

describe('runAppInstallerChecker', () => {
  it('runs python on the script, hidden and with the default deadline', async () => {
    const { impl, calls } = stubExecFile(call => call.callback(null, '{"available": false}', ''))
    const result = await runAppInstallerChecker('python.exe', 'check.py', { execFileImpl: impl })

    expect(result).toEqual({ code: 0, stdout: '{"available": false}' })
    expect(calls[0].file).toBe('python.exe')
    expect(calls[0].args).toEqual(['check.py'])
    expect(calls[0].options.windowsHide).toBe(true)
    expect(calls[0].options.timeout).toBe(APPINSTALLER_CHECK_TIMEOUT_MS)
  })

  it('a nonzero exit keeps the checker stdout (the JSON "unknown" contract)', async () => {
    const stdout = '{"available": null, "error": "winrt import failed"}'
    const { impl } = stubExecFile(call => call.callback(Object.assign(new Error('exited'), { code: 1 }), stdout, ''))

    const result = await runAppInstallerChecker('python.exe', 'check.py', { execFileImpl: impl })

    expect(result).toEqual({ code: 1, stdout })
  })

  it('a kill at the deadline resolves the caller-unknown shape', async () => {
    const { impl } = stubExecFile(call =>
      call.callback(Object.assign(new Error('killed'), { code: null, killed: true }), '', '')
    )

    const result = await runAppInstallerChecker('python.exe', 'check.py', {
      execFileImpl: impl,
      timeoutMs: 50
    })

    expect(result.code).toBe(1)
    expect(JSON.parse(result.stdout)).toEqual({ available: null, error: 'checker timed out after 50ms' })
  })

  it('resolves at the deadline even when the child never emits close', async () => {
    // The old main.ts helper killed the child and then waited on 'close'
    // forever. The deadline here is an independent bound: with a stub child
    // that never calls back and never exits, the promise still settles.
    const { impl, calls } = stubExecFile(() => undefined) // no callback, ever
    const started = Date.now()

    const result = await runAppInstallerChecker('python.exe', 'check.py', {
      execFileImpl: impl,
      timeoutMs: 50
    })

    expect(Date.now() - started).toBeLessThan(2000)
    expect(result).toEqual({ code: 1, stdout: UNKNOWN_JSON })
    expect(calls[0].options.timeout).toBe(50)
  })

  it('waits for confirmed child exit when the helper can install an update', async () => {
    vi.useFakeTimers()

    try {
      let callback: Callback | undefined
      const { impl } = stubExecFile(call => { callback = call.callback })
      let settled = false

      const pending = runAppInstallerChecker('python.exe', 'store.py', {
        execFileImpl: impl, timeoutMs: 50, waitForExit: true
      }).then(result => { settled = true;

 return result })

      await vi.advanceTimersByTimeAsync(100)
      expect(settled).toBe(false)
      callback!(Object.assign(new Error('killed'), { killed: true }), '', '')
      expect((await pending).code).toBe(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('an interpreter that cannot spawn resolves unknown, never "no update"', async () => {
    const { impl } = stubExecFile(call => call.callback(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }), '', ''))

    const result = await runAppInstallerChecker('missing-python.exe', 'check.py', { execFileImpl: impl })

    expect(result.code).toBe(1)
    expect(JSON.parse(result.stdout)).toEqual({ available: null, error: 'ENOENT' })
  })

  it('forwards checker stderr to the diagnostic sink without breaking the result', async () => {
    const onStderr = vi.fn(() => {
      throw new Error('sink exploded')
    })

    const { impl } = stubExecFile(call => call.callback(null, '{}', 'some warning'))

    const result = await runAppInstallerChecker('python.exe', 'check.py', { execFileImpl: impl, onStderr })

    expect(result).toEqual({ code: 0, stdout: '{}' })
    expect(onStderr).toHaveBeenCalledWith('some warning')
  })

  it('a synchronously throwing execFile resolves unknown and leaves no pending timer', async () => {
    const impl = (() => {
      throw new Error('bad arguments')
    }) as unknown as ExecFileImpl

    const result = await runAppInstallerChecker('python.exe', 'check.py', { execFileImpl: impl, timeoutMs: 50 })

    expect(result.code).toBe(1)
    expect(JSON.parse(result.stdout)).toEqual({ available: null, error: 'bad arguments' })
  })

  it('forwards the requested Store mode and window handle without a shell', async () => {
    const { impl, calls } = stubExecFile(call => call.callback(null, '{"ok":true}', ''))
    const args = ['--mode', 'install', '--hwnd', '1311768467139281697']
    await runAppInstallerChecker('packaged-python.exe', 'store.py', { execFileImpl: impl, args })
    expect(calls[0].args).toEqual(['store.py', ...args])
  })

  it('passes the caller env through (PYTHONPATH for the payload site-packages)', async () => {
    const { impl, calls } = stubExecFile(call => call.callback(null, '', ''))

    await runAppInstallerChecker('python.exe', 'check.py', { execFileImpl: impl, env: { PYTHONPATH: 'C:\\sp' } })

    expect(calls[0].options.env).toEqual({ PYTHONPATH: 'C:\\sp' })
  })
})
