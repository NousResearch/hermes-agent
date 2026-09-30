import type { CatalogCardVariant } from './catalog-card'
import type { CatalogEntry } from './catalog-data'

export const BENTO_COLUMNS = 4

export interface BentoTile {
  entry: CatalogEntry
  variant: CatalogCardVariant
  column: number
  row: number
  w: number
  h: number
}

const weight = (entry: CatalogEntry) =>
  (entry.imageUrl ? 8 : 0) + Math.min(entry.stars ?? 0, 1000) / 1000 + (entry.description ? 0.2 : 0)

/** Pack a section into a `columns`-wide grid that always closes: no holes and
 *  no full-width filler.
 *
 *  The mix is solved up front rather than discovered by trial placement. Over
 *  R rows there are N = columns * R cells, and k cards cover them with s 2x2
 *  showcases (pictured cards only), w two-wide cards and c single cells:
 *
 *      4s + 2w + c = N    and    s + w + c = k    =>    w = N - 3s - k
 *
 *  R is the fewest rows that hold every card, up to `maxRows`. Each two-row
 *  band can anchor one showcase when the spare cells pay for it (3 each); a
 *  section with more cards than cells trades cards for showcases instead and
 *  leaves the rest to "See all". Showcases sit on alternating edges, so every
 *  row's free run is contiguous and even; filling it wides-first then singles
 *  lands exactly on the edge. */
export function packBento(items: CatalogEntry[], columns = BENTO_COLUMNS, maxRows = 3, flip = false): BentoTile[] {
  const ranked = [...items].sort((a, b) => weight(b) - weight(a) || a.name.localeCompare(b.name))

  if (ranked.length < 2) {
    return ranked.map(entry => ({ entry, variant: 'compact', column: 0, row: 0, w: Math.min(2, columns), h: 1 }))
  }

  const rows = Math.min(maxRows, Math.ceil(ranked.length / columns))
  const cells = columns * rows
  const pictured = ranked.filter(entry => entry.imageUrl).length
  const bands = columns >= 4 ? Math.floor(rows / 2) : 0
  const spare = Math.max(0, cells - ranked.length)
  const showcases = Math.min(pictured, bands, ranked.length > cells ? bands : Math.floor(spare / 3))
  const shown = Math.min(ranked.length, cells - 3 * showcases)
  let wides = cells - 3 * showcases - shown

  const queue = ranked.slice(0, shown)
  const heroes = queue.filter(entry => entry.imageUrl).slice(0, showcases)
  const rest = queue.filter(entry => !heroes.includes(entry))
  const occupied = new Set<string>()
  const taken = (column: number, row: number) => occupied.has(`${column},${row}`)

  const claim = (column: number, row: number, w: number, h: number) => {
    for (let y = row; y < row + h; y++) {for (let x = column; x < column + w; x++) {occupied.add(`${x},${y}`)}}
  }

  const tiles: BentoTile[] = heroes.map((entry, band) => {
    const column = (band % 2 === 1) !== flip ? columns - 2 : 0
    claim(column, band * 2, 2, 2)

    return { entry, variant: 'showcase', column, row: band * 2, w: 2, h: 2 }
  })

  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns && rest.length; ) {
      if (taken(column, row)) {
        column++

        continue
      }

      let run = 0

      while (column + run < columns && !taken(column + run, row)) {run++}

      const w = wides > 0 && run >= 2 ? 2 : 1
      const entry = rest.shift()!

      if (w === 2) {wides--}
      claim(column, row, w, 1)
      tiles.push({ entry, variant: 'compact', column, row, w, h: 1 })
      column += w
    }
  }

  return tiles.sort((a, b) => a.row - b.row || a.column - b.column)
}
