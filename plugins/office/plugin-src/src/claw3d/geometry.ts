/**
 * Claw3D geometry helpers — ported from iamlukethedev/Claw3D (MIT License).
 * Copyright (c) 2026 Luke The Dev
 */

import {
  LOCAL_OFFICE_CANVAS_HEIGHT,
  CANVAS_W,
  SCALE,
  SNAP_GRID,
} from "./constants";

export const toWorld = (cx: number, cy: number): [number, number, number] => [
  cx * SCALE - CANVAS_W * SCALE * 0.5,
  0,
  cy * SCALE - LOCAL_OFFICE_CANVAS_HEIGHT * SCALE * 0.5,
];

export const snap = (value: number) =>
  Math.round(value / SNAP_GRID) * SNAP_GRID;

export const normalizeDegrees = (value: number) => {
  const normalized = value % 360;
  return normalized < 0 ? normalized + 360 : normalized;
};
