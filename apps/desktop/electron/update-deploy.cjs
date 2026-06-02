const path = require('node:path')

// Decide how a freshly rebuilt POSIX desktop GUI actually reaches the user after
// `hermes update` + `hermes desktop --build-only` have run.
//
// `hermes desktop --build-only` always writes the rebuilt app into
// <updateRoot>/apps/desktop/release/. Whether that rebuild is ever *loaded*
// depends entirely on where the currently running binary lives:
//
//   - macOS .app bundle: swap the bundle in place and relaunch (the caller runs
//     the ditto/open script).
//   - Running binary already inside the rebuilt release/ tree (a CLI
//     `hermes desktop` install launches release/linux-unpacked): the rebuild
//     overwrites that very binary, so a plain relaunch loads it.
//   - A packaged install (Linux AppImage/deb/rpm, or anything else launched from
//     a system/mount path the rebuild never touches): the rebuilt artifact in
//     the checkout is NOT the binary the user launches, so restarting just
//     reruns the same stale shell. The desktop cannot self-update the GUI here,
//     so the honest outcome is "backend updated, reinstall to update the app".

// True when `child` is the same path as, or nested under, `parent`.
// Separator-aware and tolerant of trailing separators; never throws.
function isPathInside(child, parent) {
  if (!child || !parent) return false
  const rel = path.relative(path.resolve(parent), path.resolve(child))
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel))
}

// Returns one of:
//   { kind: 'swap-mac', src, dst } - swap the macOS bundle and relaunch
//   { kind: 'restart' }            - rebuild landed in place; a relaunch loads it
//   { kind: 'manual-gui' }         - packaged install; the GUI needs a reinstall
function resolvePosixGuiDeploy({ platform, execPath, releaseDir, macBundleSrc, macBundleDst } = {}) {
  if (platform === 'darwin') {
    if (macBundleSrc && macBundleDst) {
      return { kind: 'swap-mac', src: macBundleSrc, dst: macBundleDst }
    }
    // Dev run not launched from a packaged .app: nothing to swap in place, but
    // the rebuild is staged for the next launch.
    return { kind: 'restart' }
  }

  // Linux and any other POSIX target.
  if (isPathInside(execPath, releaseDir)) {
    return { kind: 'restart' }
  }
  return { kind: 'manual-gui' }
}

module.exports = { isPathInside, resolvePosixGuiDeploy }
