// Rewrite payload-internal python links to RELATIVE symlinks.
//
// uv venv --relocatable on POSIX makes venv/bin/python* a symlink (or a
// hardlink) onto the store interpreter, with an ABSOLUTE target (the build
// runner's staging path). An absolute symlink dangles once the unpacked
// tree moves (first-boot smoke, or a user installing to a different path),
// and a materialized COPY loses the tree identity the interpreter needs to
// resolve its stdlib (sys.prefix falls back to the baked build prefix →
// "No module named 'encodings'").
//
// The correct relocatable form is a RELATIVE symlink: the target lives
// inside the bundle (resources/agent-payload/tools/<entry>/bin/python3),
// so venv/bin/python -> ../tools/<entry>/bin/python3 survives moving the
// whole tree, and the interpreter resolves its real prefix through the
// link (stdlib found, no /install fallback).
//
// Only venv/bin is rewritten. Do not walk the rest of the payload: a
// file-symlink at Foo.framework/Foo is the framework binary, and
// flattening it makes codesign report "bundle format is ambiguous".
//
// A symlink whose target points OUTSIDE the bundle is a build bug (the
// payload's own venv must never reference the builder's machine) — fail
// loudly rather than ship a dangling link.

import fs from 'node:fs'
import path from 'node:path'

export function findPackedPayload(appOutDir, platform) {
  if (!appOutDir) return null
  const candidates =
    platform === 'darwin'
      ? [
          path.join(appOutDir, 'Contents', 'Resources', 'agent-payload'),
          path.join(appOutDir, 'Hermes.app', 'Contents', 'Resources', 'agent-payload'),
          path.join(appOutDir, 'HermesBundled.app', 'Contents', 'Resources', 'agent-payload')
        ]
      : [path.join(appOutDir, 'resources', 'agent-payload')]
  return candidates.find(p => fs.existsSync(p)) || null
}

/**
 * Rewrite payload venv/bin symlinks to RELATIVE targets that resolve
 * inside the bundle. Returns the count rewritten.
 *
 * The single rule: a bundled venv's links must point at the payload's own
 * store. Links whose target already resolves inside the payload are
 * relativized (or left if already relative). Links whose target is the
 * BUILD staging path (build/agent-payload/tools/<entry>/...) — which
 * doesn't exist in the unpacked app — are rewritten to the matching
 * store entry under <payload>/tools/. Anything else fails the build.
 */
export function relativizePayloadLinks(root) {
  if (!root || !fs.existsSync(root)) return 0
  const bin = path.join(root, 'venv', 'bin')
  if (!fs.existsSync(bin)) return 0
  const payloadRoot = path.resolve(root)
  const toolsDir = path.join(payloadRoot, 'tools')
  let count = 0
  let entries
  try {
    entries = fs.readdirSync(bin, { withFileTypes: true })
  } catch {
    return 0
  }
  for (const ent of entries) {
    const p = path.join(bin, ent.name)
    let st
    try {
      st = fs.lstatSync(p)
    } catch {
      continue
    }
    if (!st.isSymbolicLink()) continue
    let target
    try {
      target = fs.readlinkSync(p)
    } catch {
      continue
    }
    const absTarget = path.resolve(bin, target)
    const insidePayload = absTarget !== payloadRoot && absTarget.startsWith(payloadRoot + path.sep)
    let resolvedTarget = absTarget
    if (!insidePayload) {
      // The BUILD staging form: build/agent-payload/tools/<entry>/... .
      // The tail names a store entry that must exist inside THIS payload.
      const m = target.match(/(?:^|\/)tools\/(.+)$/)
      if (!m) {
        throw new Error(
          `venv/bin/${ent.name} -> ${target} points outside the payload and does not name ` +
          'a store entry (tools/<entry>/...); a bundled venv must reference the payload store',
        )
      }
      const inPayload = path.join(toolsDir, m[1])
      if (!fs.existsSync(inPayload)) {
        throw new Error(
          `venv/bin/${ent.name} -> ${target}: store entry ${m[1]} is not present in the payload ` +
          `(${inPayload}); a bundled venv must reference the payload store`,
        )
      }
      resolvedTarget = inPayload
    }
    const rel = path.relative(bin, resolvedTarget)
    if (target === rel) continue
    fs.unlinkSync(p)
    fs.symlinkSync(rel, p)
    count += 1
  }
  return count
}
