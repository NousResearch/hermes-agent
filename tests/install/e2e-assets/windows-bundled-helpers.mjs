// Windows packaged-app acceptance helpers. Release Python owns descriptor XML;
// msix-shared supplies the expected publisher and content types. --msix-shared
// selects the checkout whose release module and native facts are under test.
//
// CLI (space-separated flag pairs — `node script.mjs -- --flag value` also
// works; see the repo AGENTS note on Node eating `--` args):
//   node windows-bundled-helpers.mjs validate-manifest --manifest <path> --msix-shared <path>
//   node windows-bundled-helpers.mjs descriptor --feed <dir> --base-url <url>
//        --identity <name> --publisher <subject> --version <4-part> --bundle <filename>
//        [--descriptor-filename update.appinstaller] [--msix-shared <path>]
//   node windows-bundled-helpers.mjs serve --feed <dir> --port-file <path>

import fs from 'node:fs'
import { execFileSync } from 'node:child_process'
import path from 'node:path'
import http from 'node:http'
import { pathToFileURL, fileURLToPath } from 'node:url'

import { validateDownloadedBundle } from './bundle-manifest.cjs'

export function validateBundledManifest(manifest, { expectedPublisher, arch = 'x64' }) {
  try {
    validateDownloadedBundle(manifest, 'windows', arch)
    for (const side of ['old', 'new']) {
      if (manifest[side].publisher !== expectedPublisher) {
        throw new Error(`${side}.publisher does not equal the production OUT_OF_STORE_PUBLISHER`)
      }
    }
    return { ok: true, errors: [] }
  } catch (error) {
    return { ok: false, errors: [error.message] }
  }
}

/** Load the production msix-shared module from an explicit path. */
export async function loadMsixShared(msixSharedPath) {
  const resolved = path.resolve(msixSharedPath)
  if (!fs.existsSync(resolved)) {
    throw new Error(`production msix-shared.mjs not found at ${resolved} — pass --msix-shared`)
  }
  return import(pathToFileURL(resolved).href)
}

// ── CLI ──────────────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const out = {}
  for (let i = 0; i < argv.length; i += 2) {
    out[String(argv[i]).replace(/^--/, '')] = argv[i + 1]
  }
  return out
}

async function main() {
  const argv = process.argv[2] === '--' ? process.argv.slice(3) : process.argv.slice(2)
  const cmd = argv[0]
  const flags = parseArgs(argv.slice(1))
  const msixShared = flags['msix-shared'] ||
    path.resolve(fileURLToPath(new URL('../../../scripts/msix-shared.mjs', import.meta.url)))

  if (cmd === 'validate-manifest') {
    const shared = await loadMsixShared(msixShared)
    const manifest = JSON.parse(fs.readFileSync(flags.manifest, 'utf8'))
    const result = validateBundledManifest(manifest, {
      expectedPublisher: shared.OUT_OF_STORE_PUBLISHER,
      arch: flags.arch || undefined
    })
    console.log(JSON.stringify(result))
    process.exit(result.ok ? 0 : 1)
  }

  if (cmd === 'descriptor') {
    const feed = path.resolve(flags.feed)
    const outPath = flags.out || path.join(feed, 'update.appinstaller')
    const base = flags['base-url'].replace(/\/+$/, '')
    execFileSync(process.env.HERMES_PYTHON || 'python', [
      '-m', 'scripts.bundles.release_artifacts', 'appinstaller', '--root', feed, '--out', outPath,
      '--identity', flags.identity, '--publisher', flags.publisher, '--version', flags.version,
      '--self-uri', `${base}/${flags['descriptor-filename'] || 'update.appinstaller'}`,
      '--artifact-uri', `${base}/${flags.bundle}`,
    ], { cwd: path.resolve(path.dirname(msixShared), '..'), stdio: 'inherit' })
    console.log(JSON.stringify({ ok: true, path: outPath }))
    return
  }

  if (cmd === 'serve') {
    const shared = await loadMsixShared(msixShared)
    const feed = path.resolve(flags.feed)
    fs.mkdirSync(feed, { recursive: true })
    const server = http.createServer((req, res) => {
      let urlPath
      try { urlPath = decodeURIComponent((req.url || '/').split('?')[0]) }
      catch { res.writeHead(400).end('bad path'); return }
      const rel = urlPath.replace(/^\/+/, '')
      const target = path.resolve(feed, rel)
      if (!target.startsWith(feed + path.sep) && target !== feed) {
        res.writeHead(403).end('forbidden')
        return
      }
      fs.stat(target, (err, stat) => {
        if (err || !stat.isFile()) {
          console.log(`[feed] 404 ${urlPath}`)
          res.writeHead(404).end('not found')
          return
        }
        const mime = shared.contentTypeFor(path.basename(target)) || 'application/octet-stream'
        res.writeHead(200, { 'Content-Type': mime, 'Content-Length': stat.size, 'Cache-Control': 'no-store' })
        const stream = fs.createReadStream(target)
        stream.on('error', () => res.destroy())
        res.on('close', () => stream.destroy())
        stream.pipe(res)
        console.log(`[feed] 200 ${urlPath} (${mime})`)
      })
    })
    server.listen(0, '127.0.0.1', () => {
      const url = `http://127.0.0.1:${server.address().port}`
      fs.writeFileSync(flags['port-file'], url)
      console.log(`[feed] serving ${feed} at ${url}`)
    })
    return
  }

  console.error('usage: validate-manifest | descriptor | serve (see file header)')
  process.exit(2)
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(err => {
    console.error(err && err.stack || String(err))
    process.exit(1)
  })
}
