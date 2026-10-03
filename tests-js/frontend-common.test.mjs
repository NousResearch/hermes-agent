import { expect, test, vi } from 'vitest'

// rmTree retries ENOTEMPTY from Finder/Spotlight dropping files into a
// directory mid-removal (#122803), and publishDirectory retries the EPERM
// Windows scanners hold on a freshly built tree (#128588). Node's rmSync /
// renameSync are the only seams that can reproduce those races deterministically,
// so replace them and keep the rest of node:fs real.
const { rmSync, renameSync } = vi.hoisted(() => ({ rmSync: vi.fn(), renameSync: vi.fn() }))

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal()
  return { ...actual, rmSync, renameSync }
})

const enotempty = () => { throw Object.assign(new Error('directory not empty'), { code: 'ENOTEMPTY' }) }
const eperm = () => { throw Object.assign(new Error('operation not permitted, rename'), { code: 'EPERM' }) }

test('rmTree retries ENOTEMPTY removals until the directory clears', async () => {
  const { rmTree } = await import('../scripts/build/frontend-common.mjs')
  rmSync.mockReset()
  rmSync.mockImplementationOnce(enotempty).mockImplementationOnce(enotempty).mockImplementationOnce(() => {})
  await expect(rmTree('/tmp/product-scratch')).resolves.toBeUndefined()
  expect(rmSync).toHaveBeenCalledTimes(3)
})

test('rmTree stops retrying after a bounded number of attempts', async () => {
  const { rmTree } = await import('../scripts/build/frontend-common.mjs')
  rmSync.mockReset()
  rmSync.mockImplementation(enotempty)
  await expect(rmTree('/tmp/product-scratch')).rejects.toThrow('directory not empty')
  expect(rmSync.mock.calls.length).toBeGreaterThanOrEqual(3)
  expect(rmSync.mock.calls.length).toBeLessThanOrEqual(5)
})

test('rmTree does not retry unrelated failures like ENOENT or EPERM', async () => {
  const { rmTree } = await import('../scripts/build/frontend-common.mjs')
  rmSync.mockReset()
  rmSync.mockImplementationOnce(() => { throw Object.assign(new Error('no such file'), { code: 'ENOENT' }) })
  await expect(rmTree('/tmp/product-scratch')).rejects.toThrow('no such file')
  expect(rmSync).toHaveBeenCalledTimes(1)
})

// #126914's class: `hermes update`'s TUI build died with `EPERM: operation not permitted,
// rename '…/.dist-OtWazq' -> '…/ui-tui/dist'` because Windows scanners hold a just-built
// product tree for a moment. The publication rename must ride that lock out, like rmTree does.
test('renameWithRetry rides out the transient EPERM a fresh Windows build tree carries', async () => {
  const { renameWithRetry } = await import('../scripts/build/frontend-common.mjs')
  renameSync.mockReset()
  renameSync.mockImplementationOnce(eperm).mockImplementationOnce(eperm).mockImplementationOnce(() => {})
  expect(() => renameWithRetry('/a/.dist-x', '/a/dist', { baseDelayMs: 0 })).not.toThrow()
  expect(renameSync).toHaveBeenCalledTimes(3)
  expect(renameSync).toHaveBeenLastCalledWith('/a/.dist-x', '/a/dist')
})

test('renameWithRetry gives up on a permanent EPERM and never retries other errors', async () => {
  const { renameWithRetry } = await import('../scripts/build/frontend-common.mjs')
  renameSync.mockReset()
  renameSync.mockImplementation(eperm)
  expect(() => renameWithRetry('/a/.dist-x', '/a/dist', { maxAttempts: 3, baseDelayMs: 0 }))
    .toThrow('operation not permitted, rename')
  expect(renameSync).toHaveBeenCalledTimes(3)

  renameSync.mockReset()
  renameSync.mockImplementationOnce(() => { throw Object.assign(new Error('not found'), { code: 'ENOENT' }) })
  expect(() => renameWithRetry('/a/.dist-x', '/a/dist', { baseDelayMs: 0 })).toThrow('not found')
  expect(renameSync).toHaveBeenCalledTimes(1)
})
