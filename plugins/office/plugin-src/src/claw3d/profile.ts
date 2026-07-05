/**
 * Claw3D avatar profile — ported from iamlukethedev/Claw3D (MIT License).
 * Copyright (c) 2026 Luke The Dev
 *
 * Procedural agent avatar appearance. Given a seed string (agent ID),
 * deterministically generates skin tone, hair style/color, clothing, etc.
 */

export type AgentAvatarHairStyle = "short" | "parted" | "spiky" | "bun";
export type AgentAvatarTopStyle = "tee" | "hoodie" | "jacket";
export type AgentAvatarBottomStyle = "pants" | "shorts" | "cuffed";
export type AgentAvatarHatStyle = "none" | "cap" | "beanie";

export type AgentAvatarProfile = {
  version: 1;
  seed: string;
  body: {
    skinTone: string;
  };
  hair: {
    style: AgentAvatarHairStyle;
    color: string;
  };
  clothing: {
    topStyle: AgentAvatarTopStyle;
    topColor: string;
    bottomStyle: AgentAvatarBottomStyle;
    bottomColor: string;
    shoesColor: string;
  };
  accessories: {
    glasses: boolean;
    headset: boolean;
    hatStyle: AgentAvatarHatStyle;
    backpack: boolean;
  };
};

type ColorOption = {
  id: string;
  label: string;
  color: string;
};

type EnumOption<T extends string> = {
  id: T;
  label: string;
};

export const AGENT_AVATAR_SKIN_TONE_OPTIONS: ColorOption[] = [
  { id: "fair", label: "Fair", color: "#f7d7c2" },
  { id: "light", label: "Light", color: "#f4c58a" },
  { id: "warm", label: "Warm", color: "#d8a06e" },
  { id: "tan", label: "Tan", color: "#b7794e" },
  { id: "deep", label: "Deep", color: "#8a5a3b" },
  { id: "rich", label: "Rich", color: "#5d3a24" },
];

export const AGENT_AVATAR_HAIR_STYLE_OPTIONS: EnumOption<AgentAvatarHairStyle>[] = [
  { id: "short", label: "Short" },
  { id: "parted", label: "Parted" },
  { id: "spiky", label: "Spiky" },
  { id: "bun", label: "Bun" },
];

export const AGENT_AVATAR_HAIR_COLOR_OPTIONS: ColorOption[] = [
  { id: "ink", label: "Ink", color: "#151515" },
  { id: "espresso", label: "Espresso", color: "#3e2723" },
  { id: "walnut", label: "Walnut", color: "#6b4f3a" },
  { id: "auburn", label: "Auburn", color: "#7b341e" },
  { id: "blonde", label: "Blonde", color: "#d6b56c" },
  { id: "violet", label: "Violet", color: "#7c3aed" },
  { id: "cyan", label: "Cyan", color: "#0891b2" },
  { id: "pink", label: "Pink", color: "#db2777" },
];

export const AGENT_AVATAR_TOP_STYLE_OPTIONS: EnumOption<AgentAvatarTopStyle>[] = [
  { id: "tee", label: "Tee" },
  { id: "hoodie", label: "Hoodie" },
  { id: "jacket", label: "Jacket" },
];

export const AGENT_AVATAR_BOTTOM_STYLE_OPTIONS: EnumOption<AgentAvatarBottomStyle>[] = [
  { id: "pants", label: "Pants" },
  { id: "shorts", label: "Shorts" },
  { id: "cuffed", label: "Cuffed" },
];

export const AGENT_AVATAR_HAT_STYLE_OPTIONS: EnumOption<AgentAvatarHatStyle>[] = [
  { id: "none", label: "None" },
  { id: "cap", label: "Cap" },
  { id: "beanie", label: "Beanie" },
];

export const AGENT_AVATAR_CLOTHING_COLOR_OPTIONS: ColorOption[] = [
  { id: "graphite", label: "Graphite", color: "#2d3748" },
  { id: "sky", label: "Sky", color: "#7090ff" },
  { id: "mint", label: "Mint", color: "#34d399" },
  { id: "amber", label: "Amber", color: "#f59e0b" },
  { id: "rose", label: "Rose", color: "#f43f5e" },
  { id: "violet", label: "Violet", color: "#8b5cf6" },
  { id: "cream", label: "Cream", color: "#f5f5f4" },
  { id: "slate", label: "Slate", color: "#64748b" },
];

export const AGENT_AVATAR_SHOE_COLOR_OPTIONS: ColorOption[] = [
  { id: "black", label: "Black", color: "#1a1a1a" },
  { id: "navy", label: "Navy", color: "#1e3a8a" },
  { id: "brown", label: "Brown", color: "#7c4a2d" },
  { id: "white", label: "White", color: "#e5e7eb" },
];

const AGENT_AVATAR_VERSION = 1 as const;

const hashSeed = (seed: string) => {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

const pick = <T,>(values: readonly T[], index: number) => values[index % values.length];

export const createAgentAvatarProfileFromSeed = (seed: string): AgentAvatarProfile => {
  const normalizedSeed = seed.trim() || "agent";
  const hash = hashSeed(normalizedSeed);
  const skinTone = pick(AGENT_AVATAR_SKIN_TONE_OPTIONS, hash).color;
  const hairStyle = pick(AGENT_AVATAR_HAIR_STYLE_OPTIONS, hash >>> 3).id;
  const hairColor = pick(AGENT_AVATAR_HAIR_COLOR_OPTIONS, hash >>> 5).color;
  const topStyle = pick(AGENT_AVATAR_TOP_STYLE_OPTIONS, hash >>> 7).id;
  const topColor = pick(AGENT_AVATAR_CLOTHING_COLOR_OPTIONS, hash >>> 9).color;
  const bottomStyle = pick(AGENT_AVATAR_BOTTOM_STYLE_OPTIONS, hash >>> 11).id;
  const bottomColor = pick(AGENT_AVATAR_CLOTHING_COLOR_OPTIONS, hash >>> 13).color;
  const shoesColor = pick(AGENT_AVATAR_SHOE_COLOR_OPTIONS, hash >>> 15).color;
  const hatStyle = pick(AGENT_AVATAR_HAT_STYLE_OPTIONS, hash >>> 17).id;

  return {
    version: AGENT_AVATAR_VERSION,
    seed: normalizedSeed,
    body: {
      skinTone,
    },
    hair: {
      style: hairStyle,
      color: hairColor,
    },
    clothing: {
      topStyle,
      topColor,
      bottomStyle,
      bottomColor,
      shoesColor,
    },
    accessories: {
      glasses: Boolean((hash >>> 19) % 2),
      headset: Boolean((hash >>> 20) % 2),
      hatStyle,
      backpack: Boolean((hash >>> 21) % 2),
    },
  };
};

export const createDefaultAgentAvatarProfile = (seed: string): AgentAvatarProfile =>
  createAgentAvatarProfileFromSeed(seed);
