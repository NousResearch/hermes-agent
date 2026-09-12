import { randomUUID } from 'node:crypto'
import * as fs from 'node:fs'
import path from 'node:path'

type ProvisionFs = Pick<typeof fs, 'mkdirSync' | 'lstatSync' | 'readlinkSync' | 'statSync' | 'symlinkSync' | 'renameSync' | 'unlinkSync'>

export function provisionCliLinks(
  commands: Readonly<Record<string, string>>,
  binDir: string,
  log: (message: string) => void,
  io: ProvisionFs = fs
): void {
  try {
    io.mkdirSync(binDir, { recursive: true })
  } catch (error) {
    log(`[payload] CLI PATH provision skipped: ${String(error)}`)

    return
  }

  let linked = 0

  for (const source of Object.values(commands)) {
    // The map keys are backend entrypoint identities, not public shell names.
    const name: string = path.basename(source)
    const target = path.join(binDir, name)

    try {
      const existing = io.lstatSync(target, { throwIfNoEntry: false })

      if (!existing) {
        io.symlinkSync(source, target)
        linked += 1

        continue
      }

      if (!existing.isSymbolicLink()) {
        continue
      }

      try {
        io.statSync(target)

        continue
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code

        if (code !== 'ENOENT' && code !== 'ENOTDIR') {
          throw error
        }
      }

      const destination = io.readlinkSync(target)
      const bin = path.dirname(destination)

      // Only bundled CLI links have this absolute destination shape.
      if (!path.isAbsolute(destination) || path.basename(destination) !== name ||
          path.basename(bin) !== 'bin' || path.basename(path.dirname(bin)) !== 'agent-payload') {
        continue
      }

      // Rename a staged link, never the payload command itself.
      const staged = `${target}.hermes-provision-${randomUUID()}`

      io.symlinkSync(source, staged)

      try {
        io.renameSync(staged, target)
      } catch (error) {
        try {
          io.unlinkSync(staged)
        } catch (cleanupError) {
          if ((cleanupError as NodeJS.ErrnoException).code !== 'ENOENT') {
            throw new AggregateError([error, cleanupError], `${String(error)}; temporary link cleanup failed: ${String(cleanupError)}`)
          }
        }

        throw error
      }

      linked += 1
    } catch (error) {
      log(`[payload] CLI PATH provision skipped for ${target}: ${String(error)}`)
    }
  }

  if (linked > 0) {
    log(`[payload] linked ${linked} CLI trampoline(s) into ${binDir}`)
  }
}
