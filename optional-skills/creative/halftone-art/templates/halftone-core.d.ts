/**
 * Type declarations for halftone-core.js (hand-written, no build step).
 * Keep this next to halftone-core.js so TypeScript projects resolve it.
 */

export type DotShape =
  | 'round'
  | 'square'
  | 'diamond'
  | 'ellipse'
  | 'line'
  | 'cross'
  | 'ring';

export type PlateCurve = 'full' | 'shadows' | 'midtone' | 'highlights' | 'liner';

export interface HalftoneOptions {
  cell?: number;
  angle?: number;
  shape?: DotShape;
  ink?: string;
  paper?: string;
  gain?: number;
  contrast?: number;
  brightness?: number;
  gamma?: number;
  levels?: number;
  invert?: boolean;
  grain?: number;
  seed?: number;
  density?: number;
  fit?: 'cover' | 'contain' | 'stretch';
  focal?: [number, number];
  zoom?: number;
  autoLevels?: boolean;
  levelsIn?: [number, number] | null;
  clipPercent?: number;
  mode?: 'mono' | 'duotone' | 'cmyk';
  inks?: string[] | null;
  plateAngles?: number[] | null;
  plateCurves?: PlateCurve[] | null;
  misregister?: number;
  /* cutout / sticker */
  cutout?: boolean;
  cutoutMode?: 'auto' | 'flood' | 'key' | 'alpha';
  cutoutTolerance?: number;
  cutoutKey?: string | null;
  cutoutAlpha?: number;
  cutoutFill?: 'paper' | 'none';
  cutoutOutline?: number;
  cutoutOutlineColor?: string | null;
  cutoutKeyTolerance?: number;
  cutoutClose?: number;
  cutoutKeepLargest?: boolean;
  plate?: 'luma' | 'r' | 'g' | 'b' | 'max' | 'min';
  invertB?: boolean;
  width?: number;
  height?: number;
  pixelRatio?: number;
}

export type ToneSource =
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement
  | ImageBitmap
  | OffscreenCanvas;

export interface CoverageMap {
  tone: Float32Array;
  mw: number;
  mh: number;
}

export function renderHalftone(
  canvas: HTMLCanvasElement,
  source: ToneSource,
  options?: HalftoneOptions,
): HTMLCanvasElement;

export function renderHalftoneField(
  canvas: HTMLCanvasElement,
  fn: (u: number, v: number, i: number, j: number) => number,
  options?: HalftoneOptions,
): HTMLCanvasElement;

export function plateCoverage(
  source: ToneSource | ((u: number, v: number, x: number, y: number) => number),
  width: number,
  height: number,
  options?: HalftoneOptions,
  kind?: 'luma' | 'r' | 'g' | 'b' | 'max' | 'min',
): CoverageMap;

export function cmykCoverage(
  source: ToneSource,
  width: number,
  height: number,
  options?: HalftoneOptions,
): CoverageMap[];

export function shapeTone(t: number, options?: HalftoneOptions): number;

/** 1 = subject (kept), 0 = backdrop (dropped). Alpha sources are read exactly;
 *  opaque sources are flooded from the frame corners. */
export function buildSubjectMask(
  source: ToneSource,
  width: number,
  height: number,
  options?: HalftoneOptions,
): CoverageMap;

export function halftoneToDataURL(source: ToneSource, options?: HalftoneOptions): string;

export const PRESETS: Record<string, HalftoneOptions>;

export const CURVES: Record<PlateCurve, (t: number) => number>;

export const DEFAULTS: HalftoneOptions;
