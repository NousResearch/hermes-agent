/**
 * Claw3D Navigation System — ported from iamlukethedev/Claw3D (MIT License).
 * Copyright (c) 2026 Luke The Dev
 * Copyright (c) 2026 zhuohoudeputao (Hermes integration)
 *
 * Three layers, exactly as Claw3D does it:
 *
 * 1. Nav grid builder — rasterises furniture into a Uint8Array grid where
 *    1 = blocked (desks, walls, bookshelves, etc.), 0 = walkable.
 * 2. A* pathfinder — finds a collision-aware waypoint path from A to B
 *    on the nav grid, with no corner-cutting on diagonal moves.
 * 3. Per-frame movement — advances the agent along path[0], shifting
 *    waypoints off the front as they're reached. When the path is empty,
 *    picks a new roam point and re-plans.
 *
 * All constants (CANVAS_W, GRID_CELL, etc.) come from our own constants.ts.
 */

import {
  CANVAS_W,
  WALK_SPEED,
  WALL_THICKNESS,
  DOOR_LENGTH,
  DOOR_THICKNESS,
} from "./constants";
import { LOCAL_OFFICE_CANVAS_HEIGHT } from "./constants";
import { ROAM_POINTS, DESK_POSITIONS } from "./furnitureDefaults";
import type { FurnitureItem, RenderAgent } from "./types";

// Mutable roam points — set by OfficeScene when layout loads from API.
// Defaults to Claw3D's hardcoded ROAM_POINTS until overridden.
let _customRoamPoints: typeof ROAM_POINTS | null = null;

export function setCustomRoamPoints(points: typeof ROAM_POINTS): void {
  _customRoamPoints = points;
}

// ---------------------------------------------------------------------------
// Item metadata — which furniture types block navigation
// ---------------------------------------------------------------------------

export const ITEM_FOOTPRINT: Record<string, [number, number]> = {
  wall: [80, WALL_THICKNESS],
  door: [DOOR_LENGTH, DOOR_THICKNESS],
  desk_cubicle: [100, 55],
  chair: [24, 24],
  round_table: [120, 120],
  executive_desk: [130, 65],
  couch: [100, 40],
  couch_v: [40, 80],
  bookshelf: [80, 120],
  plant: [24, 24],
  beanbag: [40, 40],
  pingpong: [100, 60],
  table_rect: [80, 40],
  coffee_machine: [32, 34],
  fridge: [40, 80],
  water_cooler: [20, 54],
  atm: [42, 38],
  sms_booth: [58, 54],
  phone_booth: [78, 72],
  whiteboard: [10, 60],
  cabinet: [200, 40],
  computer: [30, 20],
  lamp: [30, 30],
  printer: [40, 35],
  stove: [40, 40],
  microwave: [30, 20],
  wall_cabinet: [80, 20],
  sink: [40, 40],
  vending: [40, 60],
  server_rack: [45, 90],
  server_terminal: [42, 34],
  qa_terminal: [54, 38],
  kanban_board: [130, 65],
  device_rack: [70, 36],
  test_bench: [90, 42],
  treadmill: [70, 35],
  weight_bench: [90, 45],
  dumbbell_rack: [80, 28],
  exercise_bike: [45, 65],
  punching_bag: [28, 28],
  jukebox: [60, 40],
  rowing_machine: [90, 34],
  kettlebell_rack: [70, 26],
  yoga_mat: [70, 30],
  keyboard: [30, 14],
  mouse: [16, 10],
  trash: [20, 20],
  mug: [14, 14],
  clock: [20, 20],
};

export const ITEM_METADATA: Record<string, { blocksNavigation: boolean; navPadding?: number }> = {
  // ── structural ──
  wall:            { blocksNavigation: true  },
  door:            { blocksNavigation: false }, // passable
  // ── seating / lounge ──
  chair:           { blocksNavigation: false }, // passable / agents sit on them
  couch:           { blocksNavigation: true  },
  couch_v:         { blocksNavigation: true  },
  beanbag:         { blocksNavigation: true  }, // large floor seat
  // ── desks / workstations ──
  desk_cubicle:    { blocksNavigation: true, navPadding: 0 },
  executive_desk:  { blocksNavigation: true  },
  // ── tables ──
  round_table:     { blocksNavigation: true  },
  table_rect:      { blocksNavigation: true  },
  pingpong:        { blocksNavigation: true  },
  // ── storage / shelving ──
  bookshelf:       { blocksNavigation: true  },
  cabinet:         { blocksNavigation: true  },
  wall_cabinet:    { blocksNavigation: false }, // wall-mounted; agents walk under
  // ── kitchen appliances ──
  fridge:          { blocksNavigation: true  },
  stove:           { blocksNavigation: true  },
  microwave:       { blocksNavigation: false }, // counter-top / elevated
  dishwasher:      { blocksNavigation: true  }, // floor appliance
  sink:            { blocksNavigation: true  },
  coffee_machine:  { blocksNavigation: false }, // elevated on counter
  // ── office equipment ──
  printer:         { blocksNavigation: true  },
  vending:         { blocksNavigation: true  },
  atm:             { blocksNavigation: true  },
  whiteboard:      { blocksNavigation: true  },
  computer:        { blocksNavigation: false }, // desk item
  keyboard:        { blocksNavigation: false }, // desk decoration
  mouse:           { blocksNavigation: false }, // desk decoration
  // ── server room ──
  server_rack:     { blocksNavigation: true  },
  server_terminal: { blocksNavigation: true  }, // floor-standing terminal
  sms_booth:       { blocksNavigation: true  },
  phone_booth:     { blocksNavigation: true  },
  // ── QA lab ──
  qa_terminal:     { blocksNavigation: true  },
  kanban_board:    { blocksNavigation: true  },
  device_rack:     { blocksNavigation: true  },
  test_bench:      { blocksNavigation: true  },
  // ── gym ──
  treadmill:       { blocksNavigation: true  },
  weight_bench:    { blocksNavigation: true  },
  dumbbell_rack:   { blocksNavigation: true  },
  exercise_bike:   { blocksNavigation: true  },
  punching_bag:    { blocksNavigation: true  },
  jukebox:         { blocksNavigation: true  },
  rowing_machine:  { blocksNavigation: true  },
  kettlebell_rack: { blocksNavigation: true  },
  yoga_mat:        { blocksNavigation: true  },
  // ── art room ──
  easel:           { blocksNavigation: true  }, // floor-standing prop
  // ── water cooler ──
  water_cooler:    { blocksNavigation: true  }, // freestanding floor appliance
  // ── decorative / small ──
  plant:           { blocksNavigation: true  },
  lamp:            { blocksNavigation: false }, // floor lamp but thin; passable
  trash:           { blocksNavigation: false }, // small bin
  clock:           { blocksNavigation: false }, // wall-mounted
  mug:             { blocksNavigation: false }, // desk item
};

export const FURNITURE_ROTATION: Record<string, number> = {
  couch: Math.PI,
  couch_v: Math.PI / 2,
  executive_desk: -Math.PI / 2,
  whiteboard: Math.PI / 2,
};

export const getItemRotationRadians = (item: FurnitureItem) =>
  ((item.facing ?? 0) * Math.PI) / 180 +
  (FURNITURE_ROTATION[resolveItemTypeKey(item)] ?? 0);

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

const resolveItemTypeKey = (item: FurnitureItem) =>
  item.type === "couch" && item.vertical ? "couch_v" : item.type;

export const getItemBaseSize = (item: FurnitureItem) => {
  if (item.r !== undefined) {
    return { width: item.r * 2, height: item.r * 2 };
  }
  const [defaultWidth, defaultHeight] =
    ITEM_FOOTPRINT[resolveItemTypeKey(item)] ?? [item.w ?? 40, item.h ?? 40];
  return {
    width: item.w ?? defaultWidth,
    height: item.h ?? defaultHeight,
  };
};

export const getItemBounds = (item: FurnitureItem) => {
  const { width, height } = getItemBaseSize(item);
  const rotation = getItemRotationRadians(item);
  const absCos = Math.abs(Math.cos(rotation));
  const absSin = Math.abs(Math.sin(rotation));
  const boundsWidth = width * absCos + height * absSin;
  const boundsHeight = width * absSin + height * absCos;
  const centerX = item.x;
  const centerY = item.y;
  return {
    x: centerX - boundsWidth / 2,
    y: centerY - boundsHeight / 2,
    w: boundsWidth,
    h: boundsHeight,
  };
};

// ---------------------------------------------------------------------------
// Nav grid construction
// ---------------------------------------------------------------------------

const GRID_CELL = 25;
const GRID_COLS = Math.ceil(CANVAS_W / GRID_CELL);
const GRID_ROWS = Math.ceil(LOCAL_OFFICE_CANVAS_HEIGHT / GRID_CELL);

export type NavGrid = Uint8Array;

const itemBlocksNavigation = (type: string): boolean =>
  ITEM_METADATA[type]?.blocksNavigation ?? false;

export function buildNavGrid(furniture: FurnitureItem[]): NavGrid {
  const grid = new Uint8Array(GRID_COLS * GRID_ROWS);
  const defaultPad = GRID_CELL * 0.6;
  for (const item of furniture) {
    if (!itemBlocksNavigation(item.type)) continue;
    const itemPad = ITEM_METADATA[item.type]?.navPadding ?? defaultPad;
    const bounds = getItemBounds(item);
    const x1 = bounds.x - itemPad;
    const y1 = bounds.y - itemPad;
    const x2 = bounds.x + bounds.w + itemPad;
    const y2 = bounds.y + bounds.h + itemPad;
    const c1 = Math.max(0, Math.floor(x1 / GRID_CELL));
    const c2 = Math.min(GRID_COLS - 1, Math.floor(x2 / GRID_CELL));
    const r1 = Math.max(0, Math.floor(y1 / GRID_CELL));
    const r2 = Math.min(GRID_ROWS - 1, Math.floor(y2 / GRID_CELL));
    for (let row = r1; row <= r2; row += 1) {
      for (let column = c1; column <= c2; column += 1) {
        grid[row * GRID_COLS + column] = 1;
      }
    }
  }
  // Mark borders as blocked
  for (let column = 0; column < GRID_COLS; column += 1) {
    grid[column] = 1;
    grid[(GRID_ROWS - 1) * GRID_COLS + column] = 1;
  }
  for (let row = 0; row < GRID_ROWS; row += 1) {
    grid[row * GRID_COLS] = 1;
    grid[row * GRID_COLS + GRID_COLS - 1] = 1;
  }
  return grid;
}

// ---------------------------------------------------------------------------
// A* pathfinder
// ---------------------------------------------------------------------------

export function astar(
  sx: number,
  sy: number,
  ex: number,
  ey: number,
  grid: NavGrid,
): { x: number; y: number }[] {
  const clamp = (value: number, low: number, high: number) =>
    Math.min(high, Math.max(low, value));
  const toCell = (x: number, y: number) => ({
    c: clamp(Math.floor(x / GRID_CELL), 0, GRID_COLS - 1),
    r: clamp(Math.floor(y / GRID_CELL), 0, GRID_ROWS - 1),
  });
  const cellCx = (column: number) => column * GRID_CELL + GRID_CELL / 2;
  const cellCy = (row: number) => row * GRID_CELL + GRID_CELL / 2;

  const findFree = (column: number, row: number) => {
    if (!grid[row * GRID_COLS + column]) return { c: column, r: row };
    for (let distance = 1; distance < 10; distance += 1) {
      for (let rowOffset = -distance; rowOffset <= distance; rowOffset += 1) {
        for (let columnOffset = -distance; columnOffset <= distance; columnOffset += 1) {
          if (Math.abs(rowOffset) !== distance && Math.abs(columnOffset) !== distance) continue;
          const nextRow = row + rowOffset;
          const nextColumn = column + columnOffset;
          if (nextRow < 0 || nextRow >= GRID_ROWS || nextColumn < 0 || nextColumn >= GRID_COLS) continue;
          if (!grid[nextRow * GRID_COLS + nextColumn]) return { c: nextColumn, r: nextRow };
        }
      }
    }
    return null;
  };

  let { c: sc, r: sr } = toCell(sx, sy);
  let { c: ec, r: er } = toCell(ex, ey);
  const startFree = findFree(sc, sr);
  const endFree = findFree(ec, er);
  if (!startFree || !endFree) return [];
  sc = startFree.c; sr = startFree.r;
  ec = endFree.c; er = endFree.r;
  if (sc === ec && sr === er) return [{ x: ex, y: ey }];

  const nodeCount = GRID_COLS * GRID_ROWS;
  const gCost = new Float32Array(nodeCount).fill(Infinity);
  const parent = new Int32Array(nodeCount).fill(-1);
  const visited = new Uint8Array(nodeCount);
  const startIndex = sr * GRID_COLS + sc;
  const endIndex = er * GRID_COLS + ec;
  gCost[startIndex] = 0;

  // Binary min-heap open set
  const open: [number, number][] = [];
  const pushOpen = (entry: [number, number]) => {
    open.push(entry);
    let index = open.length - 1;
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      if (open[parentIndex][1] <= entry[1]) break;
      open[index] = open[parentIndex];
      index = parentIndex;
    }
    open[index] = entry;
  };
  const popOpen = (): [number, number] | null => {
    if (open.length === 0) return null;
    const first = open[0];
    const last = open.pop();
    if (!last || open.length === 0) return first;
    let index = 0;
    while (true) {
      const leftIndex = index * 2 + 1;
      const rightIndex = leftIndex + 1;
      if (leftIndex >= open.length) break;
      let smallestIndex = leftIndex;
      if (rightIndex < open.length && open[rightIndex][1] < open[leftIndex][1]) {
        smallestIndex = rightIndex;
      }
      if (open[smallestIndex][1] >= last[1]) break;
      open[index] = open[smallestIndex];
      index = smallestIndex;
    }
    open[index] = last;
    return first;
  };

  pushOpen([startIndex, Math.hypot(ec - sc, er - sr)]);
  const directions: [number, number, number][] = [
    [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
    [1, 1, 1.414], [1, -1, 1.414], [-1, 1, 1.414], [-1, -1, 1.414],
  ];

  while (open.length) {
    const next = popOpen();
    if (!next) break;
    const [current] = next;
    if (visited[current]) continue;
    visited[current] = 1;

    if (current === endIndex) {
      const path: { x: number; y: number }[] = [];
      let node = current;
      while (node !== startIndex) {
        path.push({
          x: cellCx(node % GRID_COLS),
          y: cellCy(Math.floor(node / GRID_COLS)),
        });
        node = parent[node];
      }
      path.reverse();
      if (path.length) path[path.length - 1] = { x: ex, y: ey };
      else path.push({ x: ex, y: ey });
      return path;
    }

    const currentColumn = current % GRID_COLS;
    const currentRow = Math.floor(current / GRID_COLS);
    for (const [columnOffset, rowOffset, cost] of directions) {
      const nextColumn = currentColumn + columnOffset;
      const nextRow = currentRow + rowOffset;
      if (nextColumn < 0 || nextColumn >= GRID_COLS || nextRow < 0 || nextRow >= GRID_ROWS) continue;
      const nextIndex = nextRow * GRID_COLS + nextColumn;
      if (visited[nextIndex] || grid[nextIndex]) continue;
      // No diagonal corner-cutting
      if (columnOffset !== 0 && rowOffset !== 0) {
        const orthogonalA = (currentRow + rowOffset) * GRID_COLS + currentColumn;
        const orthogonalB = currentRow * GRID_COLS + (currentColumn + columnOffset);
        if (grid[orthogonalA] || grid[orthogonalB]) continue;
      }
      const nextCost = gCost[current] + cost;
      if (nextCost < gCost[nextIndex]) {
        gCost[nextIndex] = nextCost;
        parent[nextIndex] = current;
        pushOpen([nextIndex, nextCost + Math.hypot(ec - nextColumn, er - nextRow)]);
      }
    }
  }
  return [];
}

// ---------------------------------------------------------------------------
// Roam point selection
// ---------------------------------------------------------------------------

export function pickRoamPoint(agentId: string) {
  const points = _customRoamPoints ?? ROAM_POINTS;
  return points[Math.floor(Math.random() * points.length)];
}

// ---------------------------------------------------------------------------
// Per-frame agent movement — exactly as Claw3D's useAgentTick does it
// ---------------------------------------------------------------------------

/**
 * Advance a single walking agent along its A* path.
 *
 * Mutates the agent in-place: updates x, y, facing, path, frame.
 * When the path is exhausted, picks a new roam point and plans a fresh
 * A* path to it — so the agent keeps patrolling the office indefinitely,
 * walking around furniture instead of through it.
 *
 * Returns true if the agent reached its final destination (path empty and
 * no new target was set — shouldn't happen for roaming agents).
 */
export function advanceAgent(
  agent: RenderAgent,
  delta: number,
  grid: NavGrid,
  now: number,
): void {
  const baseSpeed = agent.walkSpeed ?? WALK_SPEED;
  const speed = baseSpeed * 200 * delta; // canvas units per frame

  let path = agent.path ?? [];
  const wpX = path.length > 0 ? path[0].x : agent.x;
  const wpY = path.length > 0 ? path[0].y : agent.y;
  const dx = wpX - agent.x;
  const dy = wpY - agent.y;
  const dist = Math.hypot(dx, dy);

  if (dist > speed) {
    // Move toward first waypoint
    agent.x += (dx / dist) * speed;
    agent.y += (dy / dist) * speed;
    // atan2(dx, dy) = rotation.y for direction of travel
    agent.facing = Math.atan2(dx, dy);
    // Don't mutate agent.state — it's owned by the session-based useEffect.
    // Just advance position; the animation updater handles state transitions.
  } else {
    // Reached current waypoint
    agent.x = wpX;
    agent.y = wpY;
    if (path.length > 1) {
      // Advance to next waypoint
      agent.path = path.slice(1);
    } else {
      // Path exhausted — pick a new roam point and re-plan
      agent.path = [];
      const target = pickRoamPoint(agent.id);
      agent.targetX = target.x;
      agent.targetY = target.y;
      const newPath = astar(agent.x, agent.y, target.x, target.y, grid);
      agent.path = newPath;
      // If no path found, just stay in place — next frame will try again
      // with a different random roam point
    }
  }
  agent.frame += 1;
}
