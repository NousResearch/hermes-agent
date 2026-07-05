/**
 * Office furniture layout — adapted from Claw3D furnitureDefaults
 * (iamlukethedev/Claw3D, MIT License).
 * Copyright (c) 2026 Luke The Dev
 * Copyright (c) 2026 zhuohoudeputao (Hermes integration)
 *
 * Defines the default furniture layout for the Hermes office.
 * This is Claw3D's full DEFAULT_FURNITURE — 8 desks with computers/keyboards/
 * mice, full kitchen, dining area, lounge, server room, gym, QA lab, art room,
 * ATM, phone booth, and SMS booth.
 */

import {
  DOOR_LENGTH,
  DOOR_THICKNESS,
  EAST_WING_DOOR_Y,
  EAST_WING_ROOM_HEIGHT,
  EAST_WING_ROOM_TOP_Y,
  GYM_ROOM_END_X,
  GYM_ROOM_X,
  QA_LAB_END_X,
  QA_LAB_X,
  WALL_THICKNESS,
} from "./constants";
import type { FurnitureItem, FurnitureSeed } from "./types";

let uidCounter = 0;
const nextUid = () => `fi_${Date.now()}_${uidCounter++}`;

// ── Sub-layouts ──────────────────────────────────────────────────

const DEFAULT_PINGPONG_TABLE: FurnitureSeed = {
  type: "pingpong",
  x: 950,
  y: 600,
  w: 100,
  h: 60,
};

const DEFAULT_ATM_MACHINE: FurnitureSeed = {
  type: "atm",
  x: 430,
  y: 210,
  facing: 90,
};

const DEFAULT_PHONE_BOOTH: FurnitureSeed = {
  type: "phone_booth",
  x: 1050,
  y: 190,
  facing: 270,
};

const DEFAULT_SMS_BOOTH: FurnitureSeed = {
  type: "sms_booth",
  x: 700,
  y: 10,
  facing: 0,
};

const DEFAULT_KANBAN_BOARD: FurnitureSeed = {
  type: "kanban_board",
  x: 460,
  y: -60,
  facing: 180,
};

const DEFAULT_DINING_ITEMS: FurnitureSeed[] = [
  { type: "round_table", x: 890, y: 100, r: 50 },
  { type: "chair", x: 930, y: 100, facing: 0 },
  { type: "chair", x: 930, y: 180, facing: 180 },
  { type: "chair", x: 880, y: 130, facing: 90 },
  { type: "chair", x: 970, y: 130, facing: 270 },
];

const DEFAULT_SERVER_ROOM_ITEMS: FurnitureSeed[] = [
  { type: "wall", x: 0, y: 560, w: 230, h: WALL_THICKNESS },
  { type: "wall", x: 220, y: 560, w: WALL_THICKNESS, h: 60 },
  {
    type: "door",
    x: 210,
    y: 630,
    w: DOOR_LENGTH,
    h: DOOR_THICKNESS,
    facing: 90,
  },
  { type: "wall", x: 220, y: 660, w: WALL_THICKNESS, h: 60 },
  { type: "server_rack", x: 50, y: 595, facing: 0 },
  { type: "server_rack", x: 125, y: 595, facing: 0 },
  { type: "server_terminal", x: 110, y: 645, facing: 180 },
];

const EAST_WING_ROOM_BOTTOM_Y = EAST_WING_ROOM_TOP_Y + EAST_WING_ROOM_HEIGHT;
const EAST_WING_ROOM_BOTTOM_WALL_Y = EAST_WING_ROOM_BOTTOM_Y - WALL_THICKNESS;
const EAST_WING_DOOR_BOTTOM_Y = EAST_WING_DOOR_Y + DOOR_LENGTH;
const EAST_WING_TOP_WALL_HEIGHT = EAST_WING_DOOR_Y - EAST_WING_ROOM_TOP_Y;
const EAST_WING_BOTTOM_WALL_HEIGHT =
  EAST_WING_ROOM_BOTTOM_Y - EAST_WING_DOOR_BOTTOM_Y;

const DEFAULT_GYM_ITEMS: FurnitureSeed[] = [
  {
    type: "wall",
    x: GYM_ROOM_X,
    y: EAST_WING_ROOM_TOP_Y,
    w: WALL_THICKNESS,
    h: EAST_WING_ROOM_HEIGHT,
  },
  {
    type: "wall",
    x: GYM_ROOM_X,
    y: EAST_WING_ROOM_TOP_Y,
    w: GYM_ROOM_END_X - GYM_ROOM_X + WALL_THICKNESS,
    h: WALL_THICKNESS,
  },
  {
    type: "wall",
    x: GYM_ROOM_X,
    y: EAST_WING_ROOM_BOTTOM_WALL_Y,
    w: GYM_ROOM_END_X - GYM_ROOM_X + WALL_THICKNESS,
    h: WALL_THICKNESS,
  },
  {
    type: "wall",
    x: GYM_ROOM_END_X,
    y: EAST_WING_ROOM_TOP_Y,
    w: WALL_THICKNESS,
    h: 220,
  },
  {
    type: "door",
    x: 1280,
    y: 280,
    w: DOOR_LENGTH,
    h: DOOR_THICKNESS,
    facing: 90,
  },
  {
    type: "wall",
    x: GYM_ROOM_END_X,
    y: 300,
    w: WALL_THICKNESS,
    h: 380,
  },
  { type: "treadmill", x: 1142, y: 90, facing: 90 },
  { type: "weight_bench", x: 1204, y: 92, facing: 90 },
  { type: "dumbbell_rack", x: 1220, y: 160, facing: 180 },
  { type: "rowing_machine", x: 1140, y: 222, facing: 90 },
  { type: "kettlebell_rack", x: 1224, y: 248, facing: 180 },
  { type: "exercise_bike", x: 1146, y: 366, facing: 90 },
  { type: "punching_bag", x: 1266, y: 380, facing: 0 },
  { type: "yoga_mat", x: 1168, y: 542, facing: 0, color: "#0f766e" },
  { type: "plant", x: 1268, y: 82 },
  { type: "plant", x: 1268, y: 622 },
];

const DEFAULT_QA_LAB_ITEMS: FurnitureSeed[] = [
  {
    type: "wall",
    x: QA_LAB_X,
    y: EAST_WING_ROOM_TOP_Y,
    w: WALL_THICKNESS,
    h: 220,
  },
  {
    type: "door",
    x: 1340,
    y: 280,
    w: DOOR_LENGTH,
    h: DOOR_THICKNESS,
    facing: 90,
  },
  {
    type: "wall",
    x: QA_LAB_X,
    y: 300,
    w: WALL_THICKNESS,
    h: 380,
  },
  {
    type: "wall",
    x: QA_LAB_X,
    y: EAST_WING_ROOM_TOP_Y,
    w: QA_LAB_END_X - QA_LAB_X + WALL_THICKNESS,
    h: WALL_THICKNESS,
  },
  {
    type: "wall",
    x: QA_LAB_X,
    y: EAST_WING_ROOM_BOTTOM_WALL_Y,
    w: QA_LAB_END_X - QA_LAB_X + WALL_THICKNESS,
    h: WALL_THICKNESS,
  },
  {
    type: "wall",
    x: QA_LAB_END_X,
    y: EAST_WING_ROOM_TOP_Y,
    w: WALL_THICKNESS,
    h: EAST_WING_ROOM_HEIGHT,
  },
  { type: "qa_terminal", x: 1374, y: 92, facing: 90 },
  { type: "device_rack", x: 1454, y: 92, facing: 180 },
  { type: "device_rack", x: 1454, y: 204, facing: 180 },
  { type: "test_bench", x: 1372, y: 316, facing: 90 },
  { type: "test_bench", x: 1372, y: 450, facing: 90 },
  { type: "plant", x: 1496, y: 82 },
  { type: "plant", x: 1496, y: 622 },
];

const DEFAULT_ART_ROOM_ITEMS: FurnitureSeed[] = [
  { type: "wall", x: 260, y: 40, w: 8, h: 230 },
  { type: "wall", x: 260, y: 40, w: 178, h: 8 },
  { type: "wall", x: 260, y: 262, w: 178, h: 8 },
  { type: "wall", x: 430, y: 40, w: 8, h: 90 },
  { type: "door", x: 420, y: 150, w: 40, h: 8, facing: 90 },
  { type: "wall", x: 430, y: 170, w: 8, h: 100 },
  { type: "easel", x: 278, y: 84, facing: 90 },
  { type: "easel", x: 278, y: 158, facing: 90 },
  { type: "plant", x: 280, y: 60 },
  { type: "plant", x: 280, y: 240 },
];

// ── Main default furniture layout ────────────────────────────────
//
// Claw3D's full DEFAULT_FURNITURE — every item from the real office:
// 8 desks (desk_0–desk_7) with computers/keyboards/mice, full kitchen
// (fridge, stove, cabinets, microwave, sink, dishwasher, wall cabinets),
// dining area (round table + 4 chairs), lounge (couches, beanbags, ping
// pong), server room (walls, door, server racks, terminal), gym (treadmills,
// weight benches, dumbbell racks, etc.), QA lab (terminals, device racks,
// test benches), art room (easels), ATM, phone booth, SMS booth.

const FURNITURE_SEEDS: FurnitureSeed[] = [
  // ── Lounge / meeting area (top-left) ──
  { type: "round_table", x: 50, y: 50, r: 90 },
  { type: "chair", x: 130, y: 50, facing: 0 },
  { type: "chair", x: 200, y: 90, facing: 325 },
  { type: "chair", x: 180, y: 170, facing: 240 },
  { type: "chair", x: 120, y: 480, facing: 180 },
  { type: "chair", x: 50, y: 150, facing: 105 },
  { type: "chair", x: 60, y: 80, facing: 60 },
  { type: "chair", x: 550, y: 50, facing: 0 },
  { type: "bookshelf", x: 600, y: 30, w: 80, h: 120 },
  { type: "couch", x: 270, y: 90, w: 40, h: 80, vertical: true, facing: 180 },

  // ── Kitchen area (top-right) ──
  { type: "fridge", x: 1050, y: 20, w: 40, h: 80 },
  { type: "stove", x: 920, y: 20 },
  { type: "cabinet", x: 980, y: 30, w: 40, h: 40 },
  { type: "microwave", x: 1030, y: 10, facing: 0 },
  { type: "sink", x: 970, y: 20 },
  { type: "dishwasher", x: 950, y: 20, w: 40, h: 40 },
  { type: "cabinet", x: 840, y: 30, w: 80, h: 40, elevation: 0 },
  { type: "coffee_machine", x: 880, y: 30, elevation: 0.56 },
  { type: "wall_cabinet", x: 960, y: 10, w: 80, h: 20, elevation: 0.9 },
  { type: "wall_cabinet", x: 880, y: 10, w: 80, h: 20, elevation: 0.9 },

  // ── Dining area ──
  ...DEFAULT_DINING_ITEMS,

  // ── Vending / trash ──
  { type: "vending", x: 790, y: 10 },
  { type: "trash", x: 210, y: 20 },

  // ── Desk row 1 (y=300) — desk_0 through desk_3 ──
  { type: "desk_cubicle", x: 100, y: 300, id: "desk_0" },
  { type: "chair", x: 120, y: 290, facing: 180 },
  { type: "computer", x: 120, y: 287 },
  { type: "keyboard", x: 130, y: 295 },
  { type: "mouse", x: 152, y: 295 },
  { type: "trash", x: 170, y: 290 },
  { type: "desk_cubicle", x: 300, y: 300, id: "desk_1" },
  { type: "chair", x: 320, y: 290, facing: 180 },
  { type: "computer", x: 320, y: 287 },
  { type: "keyboard", x: 330, y: 295 },
  { type: "mouse", x: 352, y: 295 },
  { type: "trash", x: 370, y: 290 },
  { type: "desk_cubicle", x: 500, y: 300, id: "desk_2" },
  { type: "chair", x: 520, y: 290, facing: 180 },
  { type: "computer", x: 520, y: 287 },
  { type: "keyboard", x: 530, y: 295 },
  { type: "mouse", x: 552, y: 295 },
  { type: "trash", x: 570, y: 290 },
  { type: "desk_cubicle", x: 700, y: 300, id: "desk_3" },
  { type: "chair", x: 720, y: 290, facing: 180 },
  { type: "computer", x: 720, y: 287 },
  { type: "keyboard", x: 730, y: 295 },
  { type: "mouse", x: 752, y: 295 },
  { type: "trash", x: 770, y: 290 },

  // ── Desk row 2 (y=500) — desk_4 through desk_7 ──
  { type: "desk_cubicle", x: 100, y: 500, id: "desk_4" },
  { type: "computer", x: 120, y: 487 },
  { type: "keyboard", x: 130, y: 490 },
  { type: "mouse", x: 152, y: 495 },
  { type: "trash", x: 170, y: 490 },
  { type: "desk_cubicle", x: 300, y: 500, id: "desk_5" },
  { type: "chair", x: 310, y: 490, facing: 180 },
  { type: "computer", x: 320, y: 487 },
  { type: "keyboard", x: 330, y: 495 },
  { type: "mouse", x: 352, y: 495 },
  { type: "trash", x: 370, y: 500 },
  { type: "desk_cubicle", x: 500, y: 500, id: "desk_6" },
  { type: "chair", x: 520, y: 490, facing: 180 },
  { type: "computer", x: 520, y: 487 },
  { type: "keyboard", x: 530, y: 495 },
  { type: "mouse", x: 552, y: 495 },
  { type: "trash", x: 570, y: 490 },
  { type: "desk_cubicle", x: 700, y: 500, id: "desk_7" },
  { type: "chair", x: 720, y: 490, facing: 180 },
  { type: "computer", x: 720, y: 487 },
  { type: "keyboard", x: 730, y: 495 },
  { type: "mouse", x: 752, y: 495 },
  { type: "trash", x: 770, y: 490 },

  // ── Lounge furniture ──
  { type: "couch", x: 1000, y: 380, w: 100, h: 40, facing: 90 },
  { type: "couch", x: 390, y: 630, w: 100, h: 40 },
  { type: "table_rect", x: 980, y: 380, w: 60, h: 30, facing: 270 },
  { type: "pingpong", x: 950, y: 600, w: 100, h: 60 },
  { type: "beanbag", x: 1000, y: 330, color: "#e65100", facing: 90 },
  { type: "beanbag", x: 1000, y: 410, color: "#1565c0", facing: 90 },

  // ── Special items ──
  DEFAULT_ATM_MACHINE,
  DEFAULT_PHONE_BOOTH,
  DEFAULT_KANBAN_BOARD,
  DEFAULT_SMS_BOOTH,

  // ── Whiteboards, clock, lamps ──
  { type: "whiteboard", x: 40, y: 200, w: 10, h: 60 },
  { type: "clock", x: 550, y: 5 },
  { type: "lamp", x: 430, y: 100 },
  { type: "lamp", x: 980, y: 390 },

  // ── Trash ──
  { type: "trash", x: 830, y: 20 },

  // ── Plants ──
  { type: "plant", x: 40, y: 40 },
  { type: "plant", x: 660, y: 30 },
  { type: "plant", x: 340, y: 700 },
  { type: "plant", x: 450, y: 450 },
  { type: "plant", x: 1090, y: 310 },
  { type: "plant", x: 1100, y: 490 },
  { type: "plant", x: 530, y: 700 },

  // ── Server room ──
  ...DEFAULT_SERVER_ROOM_ITEMS,

  // ── Gym ──
  ...DEFAULT_GYM_ITEMS,

  // ── QA lab ──
  ...DEFAULT_QA_LAB_ITEMS,

  // ── Art room ──
  ...DEFAULT_ART_ROOM_ITEMS,

  // ── Extra chair ──
  { type: "chair", x: 100, y: 200, facing: 180 },
];

export const DEFAULT_FURNITURE: FurnitureItem[] = FURNITURE_SEEDS.map(
  (item, index) => ({ ...item, _uid: `office_${index}` }),
);

/** Desk positions for agents — mapped to desk IDs desk_0 through desk_7 */
export const DESK_POSITIONS = [
  { x: 150, y: 260, facing: 0, deskId: "desk_0" },
  { x: 350, y: 260, facing: 0, deskId: "desk_1" },
  { x: 550, y: 260, facing: 0, deskId: "desk_2" },
  { x: 750, y: 260, facing: 0, deskId: "desk_3" },
  { x: 150, y: 460, facing: 0, deskId: "desk_4" },
  { x: 350, y: 460, facing: 0, deskId: "desk_5" },
  { x: 550, y: 460, facing: 0, deskId: "desk_6" },
  { x: 750, y: 460, facing: 0, deskId: "desk_7" },
];

/** Roam points for walking agents — from Claw3D's navigation.ts */
export const ROAM_POINTS = [
  { x: 800, y: 200 },
  { x: 850, y: 500 },
  { x: 820, y: 580 },
  { x: 450, y: 420 },
  { x: 250, y: 420 },
  { x: 650, y: 420 },
  { x: 150, y: 620 },
];
