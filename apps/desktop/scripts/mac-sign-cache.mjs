// Reuse Developer ID signatures for loose Mach-O payload files across release builds.
//
// codesign embeds the signature (with a secure timestamp) in the file, so signing
// is not reproducible: the cache stores the signed bytes, keyed by the unsigned
// bytes plus everything that shapes the signature. A restored file is verified
// with codesign before it replaces the unsigned one. Only loose files are
// cached: a bundle's signature spans its other files, so osx-sign keeps those.
import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

const exec = promisify(execFile)
// Thin and fat Mach-O magics, both byte orders (sign-nested-chromium.mjs reads the same set).
const MACHO_MAGICS = new Set([0xfeedface, 0xcefaedfe, 0xfeedfacf, 0xcffaedfe, 0xcafebabe, 0xbebafeca])

/** @param {string} file */
function isMachO(file) {
  const head = Buffer.alloc(4)
  const fd = fs.openSync(file, 'r')
  try {
    return fs.readSync(fd, head, 0, 4, 0) === 4 && MACHO_MAGICS.has(head.readUInt32BE(0))
  } finally {
    fs.closeSync(fd)
  }
}
const sha256 = (/** @type {Buffer | string} */ bytes) => createHash('sha256').update(bytes).digest('hex')
const BUNDLE = /\.(app|framework|bundle|plugin|xpc|appex)$/

/** @typedef {{ key: string, files: string[] }} SignGroup */
/** @typedef {{ pending: SignGroup[], restored: number }} SignPlan */
/** @typedef {(file: string, args: string[]) => Promise<void>} Codesign */
/** @typedef {(files: string[], teamId: string) => Promise<Set<string>>} Verify */

/**
 * Whether a file lives inside a code bundle below `app` (its seal covers it).
 * @param {string} app @param {string} file
 */
export function insideBundle(app, file) {
  const parts = path.relative(path.join(app, 'Contents'), path.dirname(file)).split(path.sep)
  return parts.some(part => BUNDLE.test(part))
}

/**
 * Loose Mach-O files under `app/Contents` that the cache may sign: never inside
 * a nested bundle, never the app's own executable.
 * @param {string} app @param {(file: string) => boolean} ignore
 * @returns {string[]}
 */
export function looseMachOFiles(app, ignore) {
  const main = path.join(app, 'Contents', 'MacOS')
  /** @type {string[]} */
  const found = []
  /** @param {string} dir */
  const walk = dir => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name)
      if (entry.isSymbolicLink()) continue
      if (entry.isDirectory()) {
        if (!BUNDLE.test(entry.name)) walk(file)
      } else if (entry.isFile() && path.dirname(file) !== main && !ignore(file) && isMachO(file)) {
        found.push(file)
      }
    }
  }
  walk(path.join(app, 'Contents'))
  return found.sort()
}

/**
 * Accept only files codesign verifies strictly and that carry this team's
 * Developer ID, a secure timestamp and the hardened runtime.
 * @type {Verify}
 */
export async function verifySigned(files, teamId) {
  const valid = new Set()
  for (const file of files) {
    try {
      await exec('codesign', ['--verify', '--strict', file])
      const { stderr } = await exec('codesign', ['-dvv', file])
      if (stderr.includes(`TeamIdentifier=${teamId}`) && /^Timestamp=/m.test(stderr) &&
          /flags=0x[0-9a-f]*\(runtime\)/.test(stderr)) valid.add(file)
    } catch {
      // Unsigned, tampered or the wrong team: not a cache entry.
    }
  }
  return valid
}

/**
 * The Developer ID team of the signing identity: the "(TEAMID)" suffix of the
 * certificate's common name, looked up by its SHA-1 hash in the job keychain.
 * @param {string} identity @param {string | undefined} keychain
 * @returns {Promise<string | undefined>}
 */
export async function identityTeam(identity, keychain) {
  try {
    const { stdout } = await exec('security', ['find-identity', '-v', '-p', 'codesigning', ...(keychain ? [keychain] : [])])
    const line = stdout.split('\n').find(row => row.includes(identity))
    return line?.match(/Developer ID Application: .*\(([A-Z0-9]{10})\)"/)?.[1]
  } catch {
    return undefined
  }
}

/** @type {Codesign} */
async function runCodesign(file, args) {
  await exec('codesign', [...args, file])
}

/**
 * @param {{ root?: string, identity?: string, keychain?: string, teamId?: string, entitlements: string,
 *   codesign?: Codesign, verify?: Verify, concurrency?: number }} options
 */
export function createMacSignCache({ root, identity, keychain, teamId, entitlements, codesign = runCodesign,
  verify = verifySigned, concurrency = 8 }) {
  if (!root || !teamId || !identity) return null
  // osx-sign's per-file arguments for a loose binary (sign.js signApplication).
  const args = ['--sign', identity, '--force', ...(keychain ? ['--keychain', keychain] : []),
    '--timestamp', '--options', 'runtime', '--entitlements', entitlements]
  // The keychain is a per-job temp path: it locates the key and never shapes the signature.
  const policy = sha256(JSON.stringify({ schema: 1, identity, teamId,
    flags: ['--force', '--timestamp', '--options', 'runtime'],
    entitlements: sha256(fs.readFileSync(entitlements)) }))
  const directory = path.join(root, policy)
  // codesign derives the identifier from the file name, so it is part of the key.
  const keyOf = (/** @type {string} */ file) => sha256(`${path.basename(file)}\0${sha256(fs.readFileSync(file))}`)
  const entry = (/** @type {string} */ key) => path.join(directory, key)

  return {
    /**
     * Restore every file whose signed bytes are cached and verify; group the
     * rest so each distinct input is signed once.
     * @param {string[]} files @returns {Promise<SignPlan>}
     */
    async prepare(files) {
      /** @type {Map<string, SignGroup>} */
      const groups = new Map()
      for (const file of files) {
        const key = keyOf(file)
        const group = groups.get(key)
        if (group) group.files.push(file)
        else groups.set(key, { key, files: [file] })
      }
      const candidates = [...groups.values()].filter(group => fs.existsSync(path.join(entry(group.key), 'signed')))
      const valid = await verify(candidates.map(group => path.join(entry(group.key), 'signed')), teamId)
      let restored = 0
      for (const group of candidates) {
        const signed = path.join(entry(group.key), 'signed')
        if (!valid.has(signed)) {
          fs.rmSync(entry(group.key), { recursive: true, force: true })
          continue
        }
        for (const file of group.files) fs.copyFileSync(signed, file)
        restored += group.files.length
        groups.delete(group.key)
      }
      return { pending: [...groups.values()], restored }
    },

    /**
     * Sign each pending group once, verify, store and copy to duplicates.
     * @param {SignPlan} plan
     */
    async sign(plan) {
      const queue = [...plan.pending]
      const worker = async () => {
        for (let group = queue.shift(); group; group = queue.shift()) {
          await codesign(group.files[0], args)
        }
      }
      await Promise.all(Array.from({ length: Math.min(concurrency, queue.length) }, worker))
      const firsts = plan.pending.map(group => group.files[0])
      const valid = await verify(firsts, teamId)
      for (const group of plan.pending) {
        const file = group.files[0]
        if (!valid.has(file)) throw new Error(`Signed payload failed verification: ${file}`)
        fs.mkdirSync(directory, { recursive: true })
        const temp = fs.mkdtempSync(path.join(directory, '.tmp-'))
        try {
          fs.copyFileSync(file, path.join(temp, 'signed'))
          // A complete entry appears in one rename, never half-written.
          if (!fs.existsSync(entry(group.key))) fs.renameSync(temp, entry(group.key))
        } finally {
          fs.rmSync(temp, { recursive: true, force: true })
        }
        for (const duplicate of group.files.slice(1)) fs.copyFileSync(file, duplicate)
      }
    }
  }
}
