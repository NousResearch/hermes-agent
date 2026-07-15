/**
 * FancyZones grid model — a faithful port of PowerToys'
 * `FancyZonesEditor/Models/GridLayoutModel.cs` + `FancyZonesEditor/GridData.cs`
 * (microsoft/PowerToys, MIT). Function/field names, algorithms, and invariants
 * follow the C# sources so behavior matches the original editor:
 *
 *  - A layout is rows x columns with percent tracks summing to MULTIPLIER
 *    (10000) and a cellChildMap assigning each cell to a zone; a zone spanning
 *    multiple cells appears as the same index in adjacent cells.
 *  - Zones are rectangles in the 0..10000 coordinate space (prefix sums).
 *  - Resizers are the shared edges between zones, derived from cellChildMap
 *    discontinuities; dragging one moves every zone touching that edge.
 *  - Merging computes the rectangular CLOSURE of the selection (extending it
 *    until no zone is partially cut) — the signature FancyZones merge feel.
 */

// The sum of row/column percents should be equal to this number.
export const MULTIPLIER = 10000

/** Minimum zone extent in model units (editor ergonomics; C# uses 1). */
export const MIN_ZONE_SIZE = 500

export interface GridLayout {
  rows: number
  columns: number
  rowPercents: number[]
  columnPercents: number[]
  /** cellChildMap[row][col] = zone index; spans = same index in adjacent cells. */
  cellChildMap: number[][]
}

export interface GridZone {
  index: number
  left: number
  top: number
  right: number
  bottom: number
}

export interface GridResizer {
  orientation: 'horizontal' | 'vertical'
  /** All zones to the left/up, in order. */
  negativeSideIndices: number[]
  /** All zones to the right/down, in order. */
  positiveSideIndices: number[]
}

// ---------------------------------------------------------------------------
// GridData helpers (verbatim)
// ---------------------------------------------------------------------------

/** result[k] is the sum of the first k elements of the given list. */
export function prefixSum(list: number[]): number[] {
  const result: number[] = [0]
  let sum = 0

  for (const value of list) {
    sum += value
    result.push(sum)
  }

  return result
}

/** Opposite of prefixSum: differences of consecutive elements. */
function adjacentDifference(list: number[]): number[] {
  if (list.length <= 1) {
    return []
  }

  const result: number[] = []

  for (let i = 0; i < list.length - 1; i++) {
    result.push(list[i + 1] - list[i])
  }

  return result
}

/** Contiguous-segment unique (order-preserving), as in GridData.Unique. */
function unique(list: number[]): number[] {
  const result: number[] = []

  if (list.length === 0) {
    return result
  }

  let last = list[0]
  result.push(last)

  for (let i = 1; i < list.length; i++) {
    if (list[i] !== last) {
      last = list[i]
      result.push(last)
    }
  }

  return result
}

// ---------------------------------------------------------------------------
// Model -> zones / resizers (GridData.ModelToZones / ModelToResizers)
// ---------------------------------------------------------------------------

export function modelToZones(model: GridLayout): GridZone[] | null {
  const { rows, columns: cols, cellChildMap } = model

  let zoneCount = 0

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      zoneCount = Math.max(zoneCount, cellChildMap[row][col])
    }
  }

  zoneCount++

  if (zoneCount > rows * cols) {
    return null
  }

  const indexCount = new Array<number>(zoneCount).fill(0)
  const indexRowLow = new Array<number>(zoneCount).fill(Number.MAX_SAFE_INTEGER)
  const indexRowHigh = new Array<number>(zoneCount).fill(0)
  const indexColLow = new Array<number>(zoneCount).fill(Number.MAX_SAFE_INTEGER)
  const indexColHigh = new Array<number>(zoneCount).fill(0)

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const index = cellChildMap[row][col]
      indexCount[index]++
      indexRowLow[index] = Math.min(indexRowLow[index], row)
      indexColLow[index] = Math.min(indexColLow[index], col)
      indexRowHigh[index] = Math.max(indexRowHigh[index], row)
      indexColHigh[index] = Math.max(indexColHigh[index], col)
    }
  }

  for (let index = 0; index < zoneCount; index++) {
    if (indexCount[index] === 0) {
      return null
    }

    // Each zone must occupy a full rectangle of cells.
    if (
      indexCount[index] !==
      (indexRowHigh[index] - indexRowLow[index] + 1) * (indexColHigh[index] - indexColLow[index] + 1)
    ) {
      return null
    }
  }

  if (
    model.rowPercents.length !== rows ||
    model.columnPercents.length !== cols ||
    model.rowPercents.some(x => x < 1) ||
    model.columnPercents.some(x => x < 1)
  ) {
    return null
  }

  const rowPrefixSum = prefixSum(model.rowPercents)
  const colPrefixSum = prefixSum(model.columnPercents)

  if (rowPrefixSum[rows] !== MULTIPLIER || colPrefixSum[cols] !== MULTIPLIER) {
    return null
  }

  const zones: GridZone[] = []

  for (let index = 0; index < zoneCount; index++) {
    zones.push({
      index,
      left: colPrefixSum[indexColLow[index]],
      right: colPrefixSum[indexColHigh[index] + 1],
      top: rowPrefixSum[indexRowLow[index]],
      bottom: rowPrefixSum[indexRowHigh[index] + 1]
    })
  }

  return zones
}

export function modelToResizers(model: GridLayout): GridResizer[] {
  const grid = model.cellChildMap
  const { rows, columns: cols } = model
  const resizers: GridResizer[] = []

  // Horizontal
  for (let row = 1; row < rows; row++) {
    for (let startCol = 0; startCol < cols; ) {
      if (grid[row - 1][startCol] !== grid[row][startCol]) {
        let endCol = startCol

        while (endCol + 1 < cols && grid[row - 1][endCol + 1] !== grid[row][endCol + 1]) {
          endCol++
        }

        const positive: number[] = []
        const negative: number[] = []

        for (let col = startCol; col <= endCol; col++) {
          negative.push(grid[row - 1][col])
          positive.push(grid[row][col])
        }

        resizers.push({
          orientation: 'horizontal',
          positiveSideIndices: unique(positive),
          negativeSideIndices: unique(negative)
        })

        startCol = endCol + 1
      } else {
        startCol++
      }
    }
  }

  // Vertical
  for (let col = 1; col < cols; col++) {
    for (let startRow = 0; startRow < rows; ) {
      if (grid[startRow][col - 1] !== grid[startRow][col]) {
        let endRow = startRow

        while (endRow + 1 < rows && grid[endRow + 1][col - 1] !== grid[endRow + 1][col]) {
          endRow++
        }

        const positive: number[] = []
        const negative: number[] = []

        for (let row = startRow; row <= endRow; row++) {
          negative.push(grid[row][col - 1])
          positive.push(grid[row][col])
        }

        resizers.push({
          orientation: 'vertical',
          positiveSideIndices: unique(positive),
          negativeSideIndices: unique(negative)
        })

        startRow = endRow + 1
      } else {
        startRow++
      }
    }
  }

  return resizers
}

// ---------------------------------------------------------------------------
// Zones -> model (GridData.ZonesToModel)
// ---------------------------------------------------------------------------

export function zonesToModel(zones: GridZone[]): GridLayout {
  const xCoords = [...new Set(zones.flatMap(z => [z.left, z.right]))].sort((a, b) => a - b)
  const yCoords = [...new Set(zones.flatMap(z => [z.top, z.bottom]))].sort((a, b) => a - b)

  const model: GridLayout = {
    rows: yCoords.length - 1,
    columns: xCoords.length - 1,
    rowPercents: adjacentDifference(yCoords),
    columnPercents: adjacentDifference(xCoords),
    cellChildMap: Array.from({ length: yCoords.length - 1 }, () => new Array<number>(xCoords.length - 1).fill(0))
  }

  for (let index = 0; index < zones.length; index++) {
    const zone = zones[index]
    const startRow = yCoords.indexOf(zone.top)
    const endRow = yCoords.indexOf(zone.bottom)
    const startCol = xCoords.indexOf(zone.left)
    const endCol = xCoords.indexOf(zone.right)

    for (let row = startRow; row < endRow; row++) {
      for (let col = startCol; col < endCol; col++) {
        model.cellChildMap[row][col] = index
      }
    }
  }

  return model
}

// ---------------------------------------------------------------------------
// Closure + merge (GridData.ComputeClosure / DoMerge)
// ---------------------------------------------------------------------------

function computeClosure(zones: GridZone[], indices: number[]): { indices: number[]; zone: GridZone } {
  let left = Number.MAX_SAFE_INTEGER
  let right = Number.MIN_SAFE_INTEGER
  let top = Number.MAX_SAFE_INTEGER
  let bottom = Number.MIN_SAFE_INTEGER

  if (indices.length === 0) {
    return { indices: [], zone: { index: -1, left, right, top, bottom } }
  }

  const extend = (zone: GridZone) => {
    left = Math.min(left, zone.left)
    right = Math.max(right, zone.right)
    top = Math.min(top, zone.top)
    bottom = Math.max(bottom, zone.bottom)
  }

  for (const index of indices) {
    extend(zones[index])
  }

  let possiblyBroken = true

  while (possiblyBroken) {
    possiblyBroken = false

    for (const zone of zones) {
      const area = (zone.bottom - zone.top) * (zone.right - zone.left)

      const cutLeft = Math.max(left, zone.left)
      const cutRight = Math.min(right, zone.right)
      const cutTop = Math.max(top, zone.top)
      const cutBottom = Math.min(bottom, zone.bottom)

      const newArea = Math.max(0, cutBottom - cutTop) * Math.max(0, cutRight - cutLeft)

      if (newArea !== 0 && newArea !== area) {
        // Bad intersection found, extend.
        extend(zone)
        possiblyBroken = true
      }
    }
  }

  const resultIndices = zones
    .filter(zone => left <= zone.left && zone.right <= right && top <= zone.top && zone.bottom <= bottom)
    .map(zone => zone.index)

  return { indices: resultIndices, zone: { index: -1, left, right, top, bottom } }
}

export function mergeClosureIndices(model: GridLayout, indices: number[]): number[] {
  const zones = modelToZones(model)

  return zones ? computeClosure(zones, indices).indices : []
}

export function doMerge(model: GridLayout, indices: number[]): GridLayout {
  if (indices.length === 0) {
    return model
  }

  const zones = modelToZones(model)

  if (!zones) {
    return model
  }

  const lowestIndex = Math.min(...indices)
  const closure = computeClosure(zones, indices)
  const closureIndices = new Set(closure.indices)

  const remaining = zones.filter(zone => !closureIndices.has(zone.index))
  remaining.splice(lowestIndex, 0, closure.zone)

  return zonesToModel(remaining)
}

// ---------------------------------------------------------------------------
// Split (GridData.CanSplit / Split)
// ---------------------------------------------------------------------------

export function canSplit(
  model: GridLayout,
  zoneIndex: number,
  position: number,
  orientation: 'horizontal' | 'vertical'
): boolean {
  const zones = modelToZones(model)

  if (!zones || !zones[zoneIndex]) {
    return false
  }

  const zone = zones[zoneIndex]

  if (orientation === 'horizontal') {
    return zone.top + MIN_ZONE_SIZE <= position && position <= zone.bottom - MIN_ZONE_SIZE
  }

  return zone.left + MIN_ZONE_SIZE <= position && position <= zone.right - MIN_ZONE_SIZE
}

export function splitZone(
  model: GridLayout,
  zoneIndex: number,
  position: number,
  orientation: 'horizontal' | 'vertical'
): GridLayout {
  if (!canSplit(model, zoneIndex, position, orientation)) {
    return model
  }

  const zones = modelToZones(model)!
  const zone = zones[zoneIndex]
  const zone1 = { ...zone }
  const zone2 = { ...zone }

  zones.splice(zoneIndex, 1)

  if (orientation === 'horizontal') {
    zone1.bottom = position
    zone2.top = position
  } else {
    zone1.right = position
    zone2.left = position
  }

  zones.splice(zoneIndex, 0, zone1)
  zones.splice(zoneIndex + 1, 0, zone2)

  return zonesToModel(zones)
}

// ---------------------------------------------------------------------------
// Resizer drag (GridData.CanDrag / Drag)
// ---------------------------------------------------------------------------

export function canDrag(model: GridLayout, resizerIndex: number, delta: number): boolean {
  const zones = modelToZones(model)
  const resizers = modelToResizers(model)
  const resizer = resizers[resizerIndex]

  if (!zones || !resizer) {
    return false
  }

  const getSize = (zoneIndex: number) => {
    const zone = zones[zoneIndex]

    return resizer.orientation === 'vertical' ? zone.right - zone.left : zone.bottom - zone.top
  }

  for (const zoneIndex of resizer.positiveSideIndices) {
    if (getSize(zoneIndex) - delta < MIN_ZONE_SIZE) {
      return false
    }
  }

  for (const zoneIndex of resizer.negativeSideIndices) {
    if (getSize(zoneIndex) + delta < MIN_ZONE_SIZE) {
      return false
    }
  }

  return true
}

export function dragResizer(model: GridLayout, resizerIndex: number, delta: number): GridLayout {
  if (!canDrag(model, resizerIndex, delta)) {
    return model
  }

  const zones = modelToZones(model)!
  const resizer = modelToResizers(model)[resizerIndex]

  for (const zoneIndex of resizer.positiveSideIndices) {
    const zone = zones[zoneIndex]

    if (resizer.orientation === 'horizontal') {
      zone.top += delta
    } else {
      zone.left += delta
    }
  }

  for (const zoneIndex of resizer.negativeSideIndices) {
    const zone = zones[zoneIndex]

    if (resizer.orientation === 'horizontal') {
      zone.bottom += delta
    } else {
      zone.right += delta
    }
  }

  return zonesToModel(zones)
}

// ---------------------------------------------------------------------------
// Templates + validation (GridLayoutModel)
// ---------------------------------------------------------------------------

/** Even track sizes that sum EXACTLY to MULTIPLIER (GridLayoutModel.InitRows note). */
function evenPercents(count: number): number[] {
  const out: number[] = []

  for (let i = 0; i < count; i++) {
    out.push(Math.floor((MULTIPLIER * (i + 1)) / count) - Math.floor((MULTIPLIER * i) / count))
  }

  return out
}

export function initColumns(count: number): GridLayout {
  return {
    rows: 1,
    columns: count,
    rowPercents: [MULTIPLIER],
    columnPercents: evenPercents(count),
    cellChildMap: [Array.from({ length: count }, (_, i) => i)]
  }
}

export function initRows(count: number): GridLayout {
  return {
    rows: count,
    columns: 1,
    rowPercents: evenPercents(count),
    columnPercents: [MULTIPLIER],
    cellChildMap: Array.from({ length: count }, (_, i) => [i])
  }
}

export function initGrid(zoneCount: number): GridLayout {
  let rows = 1

  while (Math.floor(zoneCount / rows) >= rows) {
    rows++
  }

  rows--
  let cols = Math.floor(zoneCount / rows)

  if (zoneCount % rows !== 0) {
    cols++
  }

  const model: GridLayout = {
    rows,
    columns: cols,
    rowPercents: evenPercents(rows),
    columnPercents: evenPercents(cols),
    cellChildMap: Array.from({ length: rows }, () => new Array<number>(cols).fill(0))
  }

  let index = 0

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      model.cellChildMap[row][col] = index++

      if (index === zoneCount) {
        index--
      }
    }
  }

  return model
}

/**
 * The "Priority Grid" template (GridLayoutModel._priorityData, decoded from
 * its byte format: percents are `hi * 256 + lo`). Falls back to initGrid for
 * counts beyond the table, as in the original.
 */
export function initPriorityGrid(zoneCount: number): GridLayout {
  if (zoneCount === 2) {
    return { rows: 1, columns: 2, rowPercents: [MULTIPLIER], columnPercents: [6667, 3333], cellChildMap: [[0, 1]] }
  }

  if (zoneCount === 3) {
    return {
      rows: 1,
      columns: 3,
      rowPercents: [MULTIPLIER],
      columnPercents: [2500, 5000, 2500],
      cellChildMap: [[0, 1, 2]]
    }
  }

  return initGrid(zoneCount)
}

/** GridLayoutModel.IsModelValid, extended with the rectangular-span check. */
export function isGridValid(model: unknown): model is GridLayout {
  if (!model || typeof model !== 'object') {
    return false
  }

  const m = model as GridLayout

  if (typeof m.rows !== 'number' || typeof m.columns !== 'number' || m.rows <= 0 || m.columns <= 0) {
    return false
  }

  if (
    !Array.isArray(m.rowPercents) ||
    !Array.isArray(m.columnPercents) ||
    m.rowPercents.length !== m.rows ||
    m.columnPercents.length !== m.columns ||
    m.rowPercents.some(x => typeof x !== 'number' || x < 1) ||
    m.columnPercents.some(x => typeof x !== 'number' || x < 1)
  ) {
    return false
  }

  if (
    !Array.isArray(m.cellChildMap) ||
    m.cellChildMap.length !== m.rows ||
    m.cellChildMap.some(r => !Array.isArray(r) || r.length !== m.columns || r.some(c => typeof c !== 'number'))
  ) {
    return false
  }

  const rowPrefix = prefixSum(m.rowPercents)
  const colPrefix = prefixSum(m.columnPercents)

  if (rowPrefix[m.rows] !== MULTIPLIER || colPrefix[m.columns] !== MULTIPLIER) {
    return false
  }

  return modelToZones(m) !== null
}
