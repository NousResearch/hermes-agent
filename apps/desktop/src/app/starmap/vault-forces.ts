export interface Point {
  vx: number
  vy: number
  x: number
  y: number
}

/** Beyond this distance two notes do not push each other. */
export const REPEL_RADIUS = 400
const RADIUS_SQ = REPEL_RADIUS * REPEL_RADIUS
/** Below this many bodies, checking every pair is cheaper than building a grid. */
const GRID_FROM = 120

function push(a: Point, b: Point, alpha: number, strength: number) {
  let dx = b.x - a.x
  let dy = b.y - a.y
  const d2 = dx * dx + dy * dy + 0.01

  if (d2 > RADIUS_SQ) {
    return
  }

  const d = Math.sqrt(d2)
  const force = (strength * alpha) / d2
  dx = (dx / d) * force
  dy = (dy / d) * force
  a.vx -= dx
  a.vy -= dy
  b.vx += dx
  b.vy += dy
}

/**
 * Charge repulsion between bodies. The all-pairs loop is O(n²) per frame, which
 * a large vault turns into millions of pair checks; since the force is cut off
 * at REPEL_RADIUS, bucketing bodies into REPEL_RADIUS-sized cells and comparing
 * only within a cell and its forward neighbours visits exactly the same pairs
 * at O(n) cost. `bruteForce` keeps the reference path reachable for tests.
 */
export function applyRepulsion(bodies: readonly Point[], alpha: number, strength = 3200, bruteForce = false) {
  const n = bodies.length

  if (bruteForce || n <= GRID_FROM) {
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        push(bodies[i]!, bodies[j]!, alpha, strength)
      }
    }

    return
  }

  const cells = new Map<string, Point[]>()
  const key = (cx: number, cy: number) => `${cx},${cy}`

  for (const body of bodies) {
    const k = key(Math.floor(body.x / REPEL_RADIUS), Math.floor(body.y / REPEL_RADIUS))
    const cell = cells.get(k)

    if (cell) {
      cell.push(body)
    } else {
      cells.set(k, [body])
    }
  }

  // Each unordered pair of cells is visited once: same cell, then E, SW, S, SE.
  const forward = [
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1]
  ] as const

  for (const [k, cell] of cells) {
    const [cx, cy] = k.split(',').map(Number) as [number, number]

    for (let i = 0; i < cell.length; i += 1) {
      for (let j = i + 1; j < cell.length; j += 1) {
        push(cell[i]!, cell[j]!, alpha, strength)
      }
    }

    for (const [dx, dy] of forward) {
      const other = cells.get(key(cx + dx, cy + dy))

      if (other) {
        for (const a of cell) {
          for (const b of other) {
            push(a, b, alpha, strength)
          }
        }
      }
    }
  }
}
