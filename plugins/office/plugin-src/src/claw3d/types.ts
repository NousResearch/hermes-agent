/**
 * Claw3D types — ported from iamlukethedev/Claw3D (MIT License).
 * Copyright (c) 2026 Luke The Dev
 *
 * Trimmed to only the types needed by the Hermes Office plugin.
 * Janitor, ping-pong, gym, QA lab, and server room types are omitted
 * as they're not relevant to Hermes profiles.
 */

import type { AgentAvatarProfile } from "./profile";

export type OfficeAgent = {
  id: string;
  name: string;
  subtitle?: string | null;
  status: "working" | "idle" | "error";
  color: string;
  item: string;
  avatarProfile?: AgentAvatarProfile | null;
};

export type RenderAgent = OfficeAgent & {
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  path: { x: number; y: number }[];
  facing: number;
  frame: number;
  walkSpeed: number;
  phaseOffset: number;
  state: "walking" | "sitting" | "standing" | "away";
  awayUntil?: number;
};

export type FurnitureItem = {
  _uid: string;
  type: string;
  x: number;
  y: number;
  w?: number;
  h?: number;
  r?: number;
  color?: string;
  id?: string;
  facing?: number;
  vertical?: boolean;
  elevation?: number;
};

export type FurnitureSeed = Omit<FurnitureItem, "_uid">;

export type CanvasPoint = {
  x: number;
  y: number;
};

export type FacingPoint = CanvasPoint & {
  facing: number;
};
