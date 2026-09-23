import path from 'node:path'

// The embedded dashboard and every native window must resolve the same
// renderer generation, including the asar-unpacked repair copy.
export function createDesktopRendererAssetsRuntime(deps: {
  APP_ROOT: string
  IS_PACKAGED: boolean
  directoryExists: (filePath: string) => boolean
  fileExists: (filePath: string) => boolean
  missingRendererAssets: (indexPath: string) => string[]
  rememberLog: (message: string) => void
  unpackedPathFor: (filePath: string) => string
}) {
  const {
    APP_ROOT,
    IS_PACKAGED,
    directoryExists,
    fileExists,
    missingRendererAssets,
    rememberLog,
    unpackedPathFor
  } = deps

  function resolveWebDist() {
    const override = process.env.HERMES_DESKTOP_WEB_DIST

    if (override && directoryExists(path.resolve(override))) {
      return path.resolve(override)
    }

    const unpackedDist = path.join(unpackedPathFor(APP_ROOT), 'dist')

    if (directoryExists(unpackedDist)) {
      return unpackedDist
    }

    // Final fallback: APP_ROOT/dist. When packaged with asar:true this lives
    // INSIDE app.asar — not a servable filesystem directory — so the embedded
    // dashboard backend 404s on static routes (see #41327, #39472). The durable
    // fix is unpacking dist/ (PR #41411 adds dist/** to asarUnpack so the tier-2
    // unpackedDist above resolves). If we still land here while packaged, log it
    // so the cause isn't silent.
    const fallback = path.join(APP_ROOT, 'dist')

    if (IS_PACKAGED && /app\.asar(?=$|[\\/])/.test(fallback) && !directoryExists(fallback)) {
      rememberLog(
        `[web-dist] dashboard frontend dir resolved to an asar-internal path that ` +
          `is not a real directory: ${fallback}. Static routes will 404. ` +
          `Ensure dist/** is unpacked (asarUnpack) or set HERMES_DESKTOP_WEB_DIST.`
      )
    }

    return fallback
  }

  // Same resolution as resolveRendererIndex, but also hands back the missing
  // asset list already computed for the copy it chose. The primary-window path
  // needs BOTH, and re-deriving the list means walking the whole renderer
  // generation a second time: missingRendererAssets follows index.html's
  // modulepreload refs and then every chunk's inline __vite__mapDeps table, so
  // on a release build it reads ~28 MiB across ~160 files synchronously on the
  // main thread — measured ~49 ms per walk, twice before loadWindowUrl().
  // Callers that only need the path keep using resolveRendererIndex below.
  function resolveRendererIndexWithMissing(): { index: string; missing: string[] } {
    const asarIndex = path.join(APP_ROOT, 'dist', 'index.html')
    const webDistIndex = path.join(resolveWebDist(), 'index.html')

    // A packaged build ships dist/ twice: inside app.asar AND — because
    // asarUnpack lists dist/** — beside it in app.asar.unpacked. Prefer the
    // unpacked tree, matching the resolveWebDist()/unpackedPathFor precedent:
    // it is the copy the embedded dashboard serves and the copy a repair
    // rewrites, while pointing the window at the asar-internal index.html is
    // exactly how lazy chunks end up fetched from a path that cannot serve
    // them (#93479). Every window loader shares this resolver (main, overlay,
    // quick), so the ordering fix covers all of them. Dev is unchanged:
    // unpackedPathFor is a no-op outside an asar, so both candidates collapse
    // to APP_ROOT/dist and the original order is preserved.
    const candidates = IS_PACKAGED ? [webDistIndex, asarIndex] : [asarIndex, webDistIndex]
    const present = [...new Set(candidates)].filter(fileExists)

    // index.html and the hashed chunks it names are one generation. An update
    // that replaces only one of the two shipped copies (app.asar vs
    // app.asar.unpacked) leaves a TORN copy: the window loads, then dies on the
    // first lazy import with "Failed to fetch dynamically imported module" and
    // every restart reloads the same torn copy. Prefer a copy whose modules are
    // all present, so the intact generation heals the boot by itself.
    // Remember the FIRST candidate's list: if every copy turns out to be torn we
    // load present[0], and its list is already in hand — recomputing it there
    // would reintroduce the very second walk this function exists to avoid.
    let firstMissing: string[] | null = null

    for (const candidate of present) {
      const missing = missingRendererAssets(candidate)

      if (missing.length === 0) {
        return { index: candidate, missing: [] }
      }

      if (firstMissing === null) {
        firstMissing = missing
      }

      rememberLog(
        `[renderer] skipping torn renderer bundle at ${candidate}: ` +
          `${missing.length} module file(s) named by index.html are missing ` +
          `(${missing.slice(0, 3).join(', ')}${missing.length > 3 ? ', …' : ''})`
      )
    }

    if (present.length > 0) {
      // Every copy is torn. Load the first one anyway — the boundary's error is
      // still better than a blank window — but say what is wrong and how to fix
      // it, because no amount of restarting repairs a torn bundle.
      rememberLog(
        `[renderer] every renderer bundle is incomplete (${present.join(', ')}). ` +
          `The last update replaced the app while its files were locked. ` +
          `Repair with: hermes desktop --force-build`
      )

      // present[0]'s own list, captured on the first loop iteration — never the
      // last candidate's, which would describe a bundle we are not loading.
      return { index: present[0], missing: firstMissing ?? [] }
    }

    // Nothing on disk. A packaged build with no renderer bundle blank-pages with
    // a bare ERR_FILE_NOT_FOUND and no clue why (see #39484). Surface the cause
    // and the fix before Electron loads the missing file.
    rememberLog(
      `[renderer] index.html not found — the desktop app was packaged without a ` +
        `renderer bundle. Tried: ${candidates.join(', ')}. ` +
        `Rebuild with: hermes desktop --force-build`
    )

    return { index: candidates[0], missing: [] }
  }

  // Path-only accessor: unchanged behaviour for the window loaders that do not
  // need the torn-asset list.
  function resolveRendererIndex() {
    return resolveRendererIndexWithMissing().index
  }

  return { resolveWebDist, resolveRendererIndexWithMissing, resolveRendererIndex }
}
