import { expect, test } from 'vitest'

import { cleanupAfterOwnedElectron, finishOwnedElectronShutdown } from './owned-electron-cleanup'

for (const scenario of [
  { name: 'close completed but backend exit is unproven', launchAttempted: true, retained: true },
  { name: 'launch failed before returning a process handle', launchAttempted: true, retained: true },
  { name: 'setup failed before any launch attempt', launchAttempted: false, retained: false }
]) {
  test(`sandbox cleanup preserves ownership evidence when ${scenario.name}`, () => {
    const sandboxFiles = new Map([['desktop.log', 'Synthetic shutdown evidence']])

    const cleaned = cleanupAfterOwnedElectron(scenario, () => sandboxFiles.clear())

    expect(sandboxFiles.has('desktop.log')).toBe(scenario.retained)
    expect(cleaned).toBe(!scenario.retained)
  })
}

test('preserves close and exit-poll failures in their original order', async () => {
  const closeError = new Error('Synthetic close failure')
  const pollError = new Error('Synthetic exit-poll failure')

  const result = await finishOwnedElectronShutdown({
    close: async () => { throw closeError },
    killIfRunning: () => {},
    waitForExit: async () => { throw pollError }
  }).catch(error => error)

  expect(result).toBeInstanceOf(AggregateError)
  expect(result.errors).toEqual([closeError, pollError])
})

test('attempts the exit poll and preserves all errors when the owned kill also fails', async () => {
  const closeError = new Error('Synthetic close failure')
  const killError = new Error('Synthetic owned-kill failure')
  const pollError = new Error('Synthetic exit-poll failure')
  const steps: string[] = []

  const result = await finishOwnedElectronShutdown({
    close: async () => { steps.push('close'); throw closeError },
    killIfRunning: () => { steps.push('owned kill'); throw killError },
    waitForExit: async () => { steps.push('exit poll'); throw pollError }
  }).catch(error => error)

  expect(steps).toEqual(['close', 'owned kill', 'exit poll'])
  expect(result).toBeInstanceOf(AggregateError)
  expect(result.errors).toEqual([closeError, killError, pollError])
})

test('does not invoke the fallback kill after a successful close', async () => {
  let pollCompleted = false
  await finishOwnedElectronShutdown({
    close: async () => {},
    killIfRunning: () => { throw new Error('Unexpected fallback kill') },
    waitForExit: async () => { pollCompleted = true }
  })

  expect(pollCompleted).toBe(true)
})

test('retains an exit-poll failure after a successful close', async () => {
  const pollError = new Error('Synthetic exit-poll failure')

  const result = await finishOwnedElectronShutdown({
    close: async () => {},
    killIfRunning: () => {},
    waitForExit: async () => { throw pollError }
  }).catch(error => error)

  expect(result).toBeInstanceOf(AggregateError)
  expect(result.errors).toEqual([pollError])
})

test('does not discard a falsy close rejection after a successful exit poll', async () => {
  const result = await finishOwnedElectronShutdown({
    close: async () => { throw null },
    killIfRunning: () => {},
    waitForExit: async () => {}
  }).catch(error => error)

  expect(result).toBeInstanceOf(AggregateError)
  expect(result.errors).toEqual([null])
})
