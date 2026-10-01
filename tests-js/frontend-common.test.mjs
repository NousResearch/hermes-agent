import { expect, test, vi } from 'vitest'

// rmTree retries ENOTEMPTY from Finder/Spotlight dropping files into a
// directory mid-removal (#122803). Node's rmSync is the only seam that can
// reproduce that race deterministically, so replace it and keep the rest of
// node:fs real.
const { rmSync } = vi.hoisted(() => ({ rmSync: vi.fn() }))

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal()
  return { ...actual, rmSync }
})

const enotempty = () => { throw Object.assign(new Error('directory not empty'), { code: 'ENOTEMPTY' }) }

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
