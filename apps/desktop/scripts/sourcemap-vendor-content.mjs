// Shipped sourcemaps keep app sources readable in DevTools but drop the
// embedded text of dependencies. Vendor sources (shiki grammars, mermaid, …)
// are most of a map's weight and nobody steps through them in a packaged app.
// Mappings stay intact, so vendor frames still resolve to file:line.
import { readFileSync, writeFileSync } from 'node:fs'

const VENDOR_SOURCE = /(^|[\\/])node_modules[\\/]/

export function stripVendorSourcesContent(map) {
  if (!Array.isArray(map.sourcesContent)) {
    return map
  }

  return {
    ...map,
    sourcesContent: map.sourcesContent.map((content, i) => (VENDOR_SOURCE.test(map.sources[i] ?? '') ? null : content))
  }
}

export function stripVendorSourcesContentFile(mapPath) {
  const map = JSON.parse(readFileSync(mapPath, 'utf8'))

  writeFileSync(mapPath, JSON.stringify(stripVendorSourcesContent(map)))
}
