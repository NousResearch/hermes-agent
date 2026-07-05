/**
 * Claw3D constants — ported from iamlukethedev/Claw3D (MIT License).
 * Copyright (c) 2026 Luke The Dev
 *
 * Essential constants for the retro-office 3D scene.
 * Trimmed to only the values needed by the Hermes Office plugin.
 */

export const SCALE = 0.018;
export const CANVAS_W = 1800;
export const CANVAS_H = 1800;
export const WORLD_W = CANVAS_W * SCALE;
export const WORLD_H = CANVAS_H * SCALE;
export const AGENT_SCALE = 1.75;
export const WALK_SPEED = 0.3;
export const WALK_ANIM_SPEED = 0.15;
export const AGENT_RADIUS = 20;
export const SEPARATION_STRENGTH = 3;
export const WALL_THICKNESS = 8;
export const DOOR_THICKNESS = 8;
export const DOOR_LENGTH = 40;
export const SNAP_GRID = 10;
export const MIN_WALL_LENGTH = SNAP_GRID * 2;

// Local office dimensions (the main room)
export const LOCAL_OFFICE_CANVAS_WIDTH = 1800;
export const LOCAL_OFFICE_CANVAS_HEIGHT = 720;

// District zones
export const CITY_PATH_ZONE = {
  minX: 0,
  maxX: LOCAL_OFFICE_CANVAS_WIDTH,
  minY: 760,
  maxY: 980,
};

export const REMOTE_OFFICE_ZONE = {
  minX: 0,
  maxX: LOCAL_OFFICE_CANVAS_WIDTH,
  minY: 1020,
  maxY: 1020 + LOCAL_OFFICE_CANVAS_HEIGHT,
};

// East wing (gym/QA lab) — included for floor texture coordinates
export const EAST_WING_START_X = 1092;
export const EAST_WING_SIDE_MARGIN = 34;
export const EAST_WING_ROOM_TOP_Y = 40;
export const EAST_WING_ROOM_HEIGHT = 640;
export const EAST_HALL_WIDTH = 56;
export const EAST_WING_SPECIALTY_ROOM_WIDTH = 176;
export const GYM_ROOM_X = EAST_WING_START_X + EAST_WING_SIDE_MARGIN;
export const GYM_ROOM_WIDTH = EAST_WING_SPECIALTY_ROOM_WIDTH;
export const GYM_ROOM_END_X = GYM_ROOM_X + GYM_ROOM_WIDTH;
export const QA_LAB_X = GYM_ROOM_END_X + EAST_HALL_WIDTH;
export const QA_LAB_WIDTH = EAST_WING_SPECIALTY_ROOM_WIDTH;
export const QA_LAB_END_X = QA_LAB_X + QA_LAB_WIDTH;
export const EAST_WING_DOOR_Y = 260;

// Camera presets
export const DISTRICT_CAMERA_POSITION: [number, number, number] = [14, 16, 18];
export const DISTRICT_CAMERA_TARGET: [number, number, number] = [0, 0, 1];
export const DISTRICT_CAMERA_ZOOM = 34;
