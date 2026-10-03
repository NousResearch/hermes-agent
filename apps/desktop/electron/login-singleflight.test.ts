import { describe, expect, it } from 'vitest'
import { createLoginSingleFlight } from './login-singleflight'

describe('login flow coordination', () => {
  it('shares a pending flow and its result across concurrent callers', async () => {
    const run = createLoginSingleFlight<string>()
    let finish!: (value: string) => void
    let windows = 0
    const start = () => { windows++; return new Promise<string>(resolve => { finish = resolve }) }
    const first = run('partition-a|gateway-a|visible', start)
    const second = run('partition-a|gateway-a|visible', start)
    expect(first).toBe(second)
    await Promise.resolve()
    expect(windows).toBe(1)
    finish('authenticated')
    expect(await Promise.all([first, second])).toEqual(['authenticated', 'authenticated'])
  })

  it('keeps different credential partitions independent', async () => {
    const run = createLoginSingleFlight<string>()
    expect(await Promise.all([run('a', async () => 'a'), run('b', async () => 'b')])).toEqual(['a', 'b'])
  })

  it('releases failed and completed flows so a deliberate retry can run', async () => {
    const run = createLoginSingleFlight<string>()
    const failed = run('a', async () => { throw new Error('closed') })
    const joined = run('a', async () => 'unexpected')
    expect(joined).toBe(failed)
    await expect(failed).rejects.toThrow('closed')
    expect(await run('a', async () => 'retry')).toBe('retry')
    expect(await run('a', async () => 'later')).toBe('later')
  })
})
