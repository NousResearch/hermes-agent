import { expect, test } from 'vitest'
import { boundedHeapMb } from '../scripts/build/web.mjs'

test('an uncontained small host gets a heap ceiling inside its available memory', () => {
  const heapMb = boundedHeapMb({ cgroupLimitMb: null, hostAvailableMb: 2048 })

  expect(heapMb).toBeGreaterThanOrEqual(1024)
  expect(heapMb).toBeLessThan(2048)
})

test.each([
  ['a cgroup limit still wins over host memory', { cgroupLimitMb: 2048, hostAvailableMb: 64 * 1024 }, 1536],
  ['nothing available gets the floor', { cgroupLimitMb: null, hostAvailableMb: 0 }, 1024],
  ['a large host keeps the ceiling', { cgroupLimitMb: null, hostAvailableMb: 64 * 1024 }, 4096],
  ['no host reading keeps the ceiling', { cgroupLimitMb: null, hostAvailableMb: null }, 4096]
])('%s', (_name, input, expected) => {
  expect(boundedHeapMb(input)).toBe(expected)
})
