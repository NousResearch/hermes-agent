import assert from 'node:assert/strict'

import { test } from 'vitest'

import { execText } from './backend-claim'

test('execText closes noninteractive stdin and trims output after EOF', async () => {
  const output = await execText(
    process.execPath,
    ['-e', "process.stdin.resume(); process.stdin.on('end', () => process.stdout.write('  eof  \\n'))"],
    { timeout: 2000 }
  )

  assert.equal(output, 'eof')
}, 10_000)

// Windows forcibly terminates children; a graceful SIGTERM handler is POSIX-only.
test.skipIf(process.platform === 'win32')(
  'execText rejects timed-out output even when SIGTERM exits zero',
  async () => {
    await assert.rejects(
      execText(
        process.execPath,
        [
          '-e',
          "process.on('SIGTERM', () => process.exit(0)); process.stdout.write('candidate'); setInterval(() => {}, 1000)"
        ],
        { timeout: 2000 }
      ),
      /timed out/i
    )
  },
  10_000
)
