/**
 * E2E contract: an audio file selected in the file browser PLAYS in the
 * preview rail.
 *
 * The bug this pins: a `.wav` selected in the tree showed the "This looks like
 * a binary file" refusal screen. The jsdom suite
 * (src/app/chat/right-rail/preview-file.audio.test.tsx) pins the branch logic,
 * but it cannot prove the thing the user actually reported — that the source
 * the player is handed is a REAL, playable stream rather than a data URL, a
 * `file://` path the app origin cannot load, or nothing at all.
 *
 * So this spec drives the real app, opens a real WAV through the real
 * classifier, and asserts on the real `<audio>` element: that it exists, that
 * its source is the `hermes-media://stream/…` protocol, and that Chromium
 * reports genuine audio metadata for it (a duration). A source that 404s or is
 * refused by the protocol handler yields no duration — which is exactly the
 * silent-broken-player failure this test exists to catch.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import { setupMockBackend, waitForAppReady } from './fixtures'

/** Seconds of tone to synthesize — long enough for a real duration, short
 *  enough to write instantly and stream trivially. */
const TONE_SECONDS = 1.5

/**
 * Write a real, valid WAV (PCM 16-bit mono) to `filePath`.
 *
 * A fake file with a .wav extension would prove nothing: the classifier is
 * extension-driven, but the PLAYER assertion depends on Chromium actually
 * decoding the bytes. Generating a true RIFF/WAVE header is what makes the
 * duration assertion meaningful.
 */
function writeSilentWav(filePath: string, seconds: number, sampleRate = 44_100): void {
  const sampleCount = Math.floor(sampleRate * seconds)
  const dataBytes = sampleCount * 2 // 16-bit mono
  const buffer = Buffer.alloc(44 + dataBytes)

  buffer.write('RIFF', 0, 'ascii')
  buffer.writeUInt32LE(36 + dataBytes, 4) // chunk size
  buffer.write('WAVE', 8, 'ascii')
  buffer.write('fmt ', 12, 'ascii')
  buffer.writeUInt32LE(16, 16) // PCM subchunk size
  buffer.writeUInt16LE(1, 20) // audio format = PCM
  buffer.writeUInt16LE(1, 22) // channels = mono
  buffer.writeUInt32LE(sampleRate, 24)
  buffer.writeUInt32LE(sampleRate * 2, 28) // byte rate
  buffer.writeUInt16LE(2, 32) // block align
  buffer.writeUInt16LE(16, 34) // bits per sample
  buffer.write('data', 36, 'ascii')
  buffer.writeUInt32LE(dataBytes, 40)
  // Samples stay zero (silence) — the point is decodable audio, not sound.

  fs.writeFileSync(filePath, buffer)
}

test.describe('audio preview in the file browser', () => {
  test('plays a .wav instead of refusing it as binary', async () => {
    const fixture = await setupMockBackend()

    try {
      const { app, page, sandbox } = fixture

      // The file browser roots itself at the project dir, which the main
      // process reads from this file in the app's own userData.
      const projectDir = path.join(sandbox.root, 'project')
      fs.mkdirSync(projectDir, { recursive: true })
      const wavPath = path.join(projectDir, 'take.wav')
      writeSilentWav(wavPath, TONE_SECONDS)

      // Write the workspace root BEFORE the app reads it, then reload so the
      // tree re-roots without tearing the sandbox down.
      fs.writeFileSync(
        path.join(sandbox.userDataDir, 'project-dir.json'),
        JSON.stringify({ dir: projectDir }, null, 2),
        'utf8',
      )

      await waitForAppReady(fixture)
      await app.evaluate(({ BrowserWindow }) => {
        BrowserWindow.getAllWindows()[0]?.reload()
      })

      // Open the file browser pane (the right side of the main zone).
      await page.keyboard.press('Control+j')

      // Select the file the way the user does — it is inside the project dir
      // we just rooted the tree at.
      const row = page.getByText('take.wav', { exact: true }).first()
      await row.waitFor({ state: 'visible', timeout: 30_000 })
      await row.dblclick()

      // The refusal screen is the reported bug: it must not appear.
      await expect(page.getByText(/looks like a binary file/i)).toHaveCount(0)

      // The player must mount, and must not be fed a data URL (which caps at
      // the read limit and cannot seek) or a bare file:// path.
      const audio = page.locator('audio[controls]')
      await expect(audio).toHaveCount(1)

      const src = await audio.getAttribute('src')
      expect(src).not.toBeNull()
      expect(src).toContain('hermes-media://stream/')

      // The real proof: Chromium decoded the stream and reported a duration.
      // A refused/missing source leaves readyState 0 and duration NaN.
      const decoded = await audio.evaluate(async element => {
        const player = element as HTMLAudioElement

        if (player.readyState < 1) {
          await new Promise<void>((resolve, reject) => {
            const timer = window.setTimeout(() => reject(new Error('audio metadata timed out')), 15_000)

            player.addEventListener(
              'loadedmetadata',
              () => {
                window.clearTimeout(timer)
                resolve()
              },
              { once: true },
            )
            player.addEventListener(
              'error',
              () => {
                window.clearTimeout(timer)
                reject(new Error('audio element errored before metadata'))
              },
              { once: true },
            )
          })
        }

        return { duration: player.duration, readyState: player.readyState }
      })

      expect(decoded.readyState).toBeGreaterThanOrEqual(1)
      expect(decoded.duration).toBeGreaterThan(0)
      // Duration must match the file we wrote, not merely be non-zero.
      expect(decoded.duration).toBeGreaterThan(TONE_SECONDS * 0.8)
      expect(decoded.duration).toBeLessThan(TONE_SECONDS * 1.2)
    } finally {
      await fixture.cleanup()
    }
  })
})
