/**
 * Palette construction.
 *
 * Hue is the only value the seed controls. Lightness and chroma are authored
 * constants, which is what makes every blobatar look like it came from the same
 * designer rather than from a random number generator.
 *
 * Colors are resolved to hex rather than emitted as `oklch()`. Browsers handle
 * `oklch()` in SVG fine, but server-side rasterizers (resvg, librsvg, sharp)
 * largely do not — and blobatars get rasterized server-side constantly. Doing the
 * conversion here also means the contrast guarantee is enforced against real
 * sRGB luminance instead of assumed from OKLab lightness, which drifts by up to
 * ~1.4:1 between hues at equal L.
 */

export interface Oklch {
  l: number;
  c: number;
  h: number;
}

/** Every color slot a blobatar has. */
export type ColorKey = "bg" | "head" | "eye";
export type Palette = Partial<Record<ColorKey, string>>;

/** OKLCh -> linear-light sRGB. Components may fall outside [0,1] (out of gamut). */
function toLinear({ l, c, h }: Oklch): [number, number, number] {
  const r = (h * Math.PI) / 180;
  const a = c * Math.cos(r);
  const b = c * Math.sin(r);

  const l_ = l + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = l - 0.0894841775 * a - 1.291485548 * b;

  const L = l_ * l_ * l_;
  const M = m_ * m_ * m_;
  const S = s_ * s_ * s_;

  return [
    4.0767416621 * L - 3.3077115913 * M + 0.2309699292 * S,
    -1.2684380046 * L + 2.6097574011 * M - 0.3413193965 * S,
    -0.0041960863 * L - 0.7034186147 * M + 1.707614701 * S,
  ];
}

const inGamut = (rgb: number[]) =>
  rgb.every((v) => v >= -1e-4 && v <= 1 + 1e-4);

/**
 * Resolves to in-gamut linear sRGB, reducing chroma if needed.
 *
 * Chroma is the right axis to give up: lowering it desaturates, while clipping
 * channels shifts hue — a clipped vivid blue turns purple.
 */
function resolve(color: Oklch): [number, number, number] {
  let rgb = toLinear(color);
  if (!inGamut(rgb)) {
    let lo = 0;
    let hi = color.c;
    for (let i = 0; i < 12; i++) {
      const mid = (lo + hi) / 2;
      if (inGamut(toLinear({ ...color, c: mid }))) lo = mid;
      else hi = mid;
    }
    rgb = toLinear({ ...color, c: lo });
  }
  return rgb.map((v) => Math.min(1, Math.max(0, v))) as [
    number,
    number,
    number,
  ];
}

/**
 * WCAG relative luminance. The values coming out of `resolve` are already
 * linear-light sRGB, which is exactly what WCAG's piecewise transfer function
 * produces — so this needs no further linearization.
 */
function luminance(color: Oklch): number {
  const [r, g, b] = resolve(color);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a: Oklch, b: Oklch): number {
  const x = luminance(a);
  const y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

/**
 * Pushes `fg`'s lightness away from `bg` until the pair clears `min`.
 *
 * Walks in the direction it is already leaning first, so a dark ink on a light
 * head gets darker rather than flipping to light. If that direction runs out of
 * range, it tries the other way before giving up at pure black or white.
 */
export function ensureContrast(fg: Oklch, bg: Oklch, min: number): Oklch {
  if (contrast(fg, bg) >= min) return fg;

  const lean = fg.l >= bg.l ? 1 : -1;
  for (const dir of [lean, -lean]) {
    const probe = { ...fg };
    for (let i = 0; i < 60; i++) {
      probe.l = Math.min(1, Math.max(0, probe.l + dir * 0.02));
      if (contrast(probe, bg) >= min) return probe;
      if (probe.l === 0 || probe.l === 1) break;
    }
  }

  // Unreachable for the authored ramps, but a palette override could get here.
  const black = { ...fg, l: 0, c: 0 };
  const white = { ...fg, l: 1, c: 0 };
  return contrast(black, bg) >= contrast(white, bg) ? black : white;
}

export function toHex(color: Oklch): string {
  return (
    "#" +
    resolve(color)
      .map((v) => {
        const s =
          v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
        return Math.round(s * 255)
          .toString(16)
          .padStart(2, "0");
      })
      .join("")
  );
}

/**
 * sRGB hex → OKLCh. The inverse of `toLinear` plus `resolve`'s decode, and the
 * only way back into the color space from a palette that has already been
 * serialized.
 *
 * It exists because the tint (§ `hot` below) has to start from the colors that
 * are actually on screen, not from the ramp that produced them: `BlobatarOptions.palette`
 * lets a consumer override `head` or `eye` outright, and a hot pair derived from
 * the ramp instead would tint toward a color the blobatar never wore.
 */
export function fromHex(hex: string): Oklch {
  const n = parseInt(hex.slice(1), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const s = v / 255;
    return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  }) as [number, number, number];

  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);

  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  const B = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;

  return {
    l: 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    c: Math.hypot(A, B),
    h: (Math.atan2(B, A) * 180) / Math.PI,
  };
}

/**
 * Blend two colors in OKLab — `color-mix(in oklab, a, b t)`, done here.
 *
 * Here rather than in CSS because a hot pose has to resolve to a *finished*
 * color before it reaches the stylesheet: the hot endpoint would otherwise be a
 * custom property that vanishes the instant an expression is cleared, snapping
 * the fill back to base while the rest of the pose eases out over 360ms. See
 * `heatTint` in `src/expression.ts`.
 *
 * Interpolating in OKLab means lerping cartesian `a`/`b`, not the polar `c`/`h`
 * this module otherwise speaks — a hue lerp would swing a desaturated color
 * around the wheel and pick up chroma that is in neither endpoint.
 */
export function mix(a: Oklch, b: Oklch, t: number): Oklch {
  const rad = (v: number) => (v * Math.PI) / 180;
  const ax = a.c * Math.cos(rad(a.h));
  const ay = a.c * Math.sin(rad(a.h));
  const bx = b.c * Math.cos(rad(b.h));
  const by = b.c * Math.sin(rad(b.h));
  const x = ax + (bx - ax) * t;
  const y = ay + (by - ay) * t;
  return {
    l: a.l + (b.l - a.l) * t,
    c: Math.hypot(x, y),
    h: (Math.atan2(y, x) * 180) / Math.PI,
  };
}

/** `mix` between two serialized colors, serialized. */
export const mixHex = (a: string, b: string, t: number) =>
  toHex(mix(fromHex(a), fromHex(b), t));

/**
 * Where a tinting pose is heading.
 *
 * Four numbers rather than an authored colour, because the endpoint has to be
 * derived per seed — see `tinted` below. A `Tint` says *which way*, and the
 * blobatar's own palette says where that lands.
 *
 * This was three constants and a red until the roster wanted more than anger.
 * Generalising it is what makes a second tinting pose cost its numbers rather
 * than a second copy of the contrast walk — and the walk is the part that is
 * easy to get subtly wrong, so having exactly one of it is the point.
 */
export interface Tint {
  /** Hue the body arrives at, in degrees. Reached outright, not approached. */
  h: number;
  /** Lightness it heads toward. */
  l: number;
  /** How far of the way to `l` the body actually travels, 0–1. */
  pull: number;
  /** Chroma floor. The body never desaturates on the way. */
  c: number;
}

/**
 * Red, because every reference for anger is — and only 60% of the way there in
 * lightness, so the tone set survives the trip.
 */
export const HOT: Tint = { h: 27, l: 0.58, pull: 0.6, c: 0.18 };

/**
 * The rest of the targets, kept here beside `HOT` rather than in `expression.ts`
 * so that the module that owns the guarantee owns the full set of endpoints it
 * has to hold for. `test/color.test.ts` iterates this list; a target added over
 * there and not here would be a tint nothing verifies.
 *
 * `pull` is the dial that keeps them apart as much as `h` is. `BLUSH` travels
 * only 0.4 of the way and lands pale — a shy blobatar that goes as red as an
 * angry one is an angry one.
 */
export const ROSE: Tint = { h: 358, l: 0.72, pull: 0.55, c: 0.16 };
export const BLUSH: Tint = { h: 12, l: 0.84, pull: 0.4, c: 0.1 };
export const BILE: Tint = { h: 142, l: 0.66, pull: 0.6, c: 0.13 };

/** Every target the suite has to hold the contrast guarantee across. */
export const TINTS: [string, Tint][] = [
  ["hot", HOT],
  ["rose", ROSE],
  ["blush", BLUSH],
  ["bile", BILE],
];

/**
 * A hair over the 4.5:1 the suite asserts.
 *
 * The margin is for 8-bit quantization and nothing else: the pair is authored in
 * OKLCh, serialized to hex, and read back out of that hex by anything checking
 * it — including the browser. Rounding to a byte per channel moves the ratio in
 * the third decimal, so the floor is cleared by more than that rather than sat
 * exactly on.
 */
const TINT_FLOOR = 4.55;

/**
 * The palette a tinting pose heads toward, given the one it is tinting from.
 *
 * Derived per seed rather than being a single authored colour, and the reason is
 * polarity: `blob` flips its eye between near-black and near-white depending on
 * the body's lightness, and no fixed red clears 4.5:1 against both.
 *
 * So the tinted body meets its target **partway** rather than landing on it.
 * Holding the body's own lightness was the first attempt and it is too quiet — a
 * pastel goes pink rather than angry, because at L 0.86 there is no red to be
 * had. Travelling the whole way is the opposite failure: every blobatar in the
 * roster converges on one red and the tone set, which is most of what makes a
 * grid look like a crowd, disappears at the exact moment the grid is loudest.
 * `pull` keeps a pale blobatar recognisably pale and an ink one recognisably
 * dark while giving both somewhere to go.
 *
 * The eye endpoint is then pushed until **every point along the mix** clears the
 * floor, not merely both ends. A straight line in OKLab between two passing
 * pairs is not itself a passing pair: the body travels further than the eye, so
 * the two lightnesses can close on each other in the middle of a transition that
 * is legible at both stops. This walks the mix and fixes the worst point.
 *
 * Tree-shaken out of any bundle that imports no tinting expression — it is
 * reached only through `tint` on the expression value, the same indirection that
 * keeps `expression.ts` itself out of the core. One walk serves every target, so
 * a bundle with three tinting poses in it carries this once.
 */
export function tinted(
  head: string,
  eye: string,
  t: Tint,
): [string, string] {
  const base = fromHex(head);
  const baseEye = fromHex(eye);

  // Chroma is floored rather than replaced: a body that is already vivid should
  // not *lose* saturation on the way, and the pale neutral swatch has almost
  // none to keep.
  let hotHead: Oklch = {
    l: base.l + (t.l - base.l) * t.pull,
    c: Math.max(base.c, t.c),
    h: t.h,
  };
  // The body still has to be visible on a dark page at full heat, which is the
  // same floor the ramp enforces and for the same reason.
  hotHead = ensureContrast(hotHead, DARK_SURFACE, SURFACE_FLOOR);

  let hotEye = ensureContrast(baseEye, hotHead, TINT_FLOOR);

  // Walked rather than assumed, in the exact terms that ship — the hexes, mixed
  // the way `mixHex` mixes them — because "both ends pass" does not imply "every
  // point passes". It very nearly does here: holding the body's lightness makes
  // the body's luminance almost constant along the mix, which leaves contrast
  // monotone in the eye's and puts the minimum at an end. Almost, because
  // reddening moves chroma and chroma moves luminance a little. Eleven samples
  // catch that wobble; the loop normally exits on the first pass and pushes the
  // eye endpoint further from the body when it does not.
  const dir = hotEye.l >= hotHead.l ? 1 : -1;
  const headHex = toHex(hotHead);
  for (let pass = 0; pass < 40; pass++) {
    const eyeHex = toHex(hotEye);
    let worst = Infinity;
    for (let i = 0; i <= 10; i++) {
      const t = i / 10;
      worst = Math.min(
        worst,
        contrast(
          fromHex(mixHex(eye, eyeHex, t)),
          fromHex(mixHex(head, headHex, t)),
        ),
      );
    }
    if (worst >= TINT_FLOOR) return [headHex, eyeHex];
    const l = Math.min(1, Math.max(0, hotEye.l + dir * 0.02));
    if (l === hotEye.l) return [headHex, eyeHex];
    hotEye = { ...hotEye, l };
  }

  return [headHex, toHex(hotEye)];
}

/**
 * The tone set.
 *
 * This is the one place the seed is allowed to move lightness and chroma, not
 * just hue — a body vocabulary this varied looks monotonous in a single tone.
 * Letting the seed roam freely over L and C is what makes generated palettes
 * look generated, so instead it picks from six authored swatches: the same
 * discipline as a designer handing you a set, rather than a slider.
 *
 * Thresholds are cumulative, so pale and mid tones dominate and the near-black
 * body stays a rare find.
 */
const TONES: [number, { l: number; c: number }][] = [
  [0.2, { l: 0.86, c: 0.085 }], // pastel
  [0.36, { l: 0.9, c: 0.028 }], // pale neutral
  [0.62, { l: 0.73, c: 0.135 }], // mid
  [0.8, { l: 0.62, c: 0.165 }], // deep
  [0.93, { l: 0.87, c: 0.16 }], // bright
  // Dark, but not darker than a dark host surface. At l 0.17 this swatch scored
  // 1.03:1 against a near-black page and the body simply vanished, leaving two
  // floating eyes. l 0.34 still reads as the ink tone and clears both ends.
  [1.0, { l: 0.34, c: 0.035 }], // ink
];

const toneAt = (v: number) =>
  TONES.find(([edge]) => v < edge)?.[1] ?? TONES[0]![1];

/**
 * The darkest host surface a backdrop-less blob is expected to land on, and the
 * ratio it must clear against it.
 *
 * `FLOORS` can only relate colors that are in the palette, and the surface never
 * is: `blob` ships with its backdrop off, so the body sits directly on whatever
 * the page provides. Guaranteeing contrast against the palette's own light `bg`
 * says nothing about that case — which is how the ink tone came to render as a
 * near-invisible silhouette on a dark page.
 */
const DARK_SURFACE: Oklch = { l: 0.145, c: 0, h: 0 }; // ≈ #0a0a0b
const SURFACE_FLOOR = 1.5;

/** The authored ramp: the seed picks a hue and a tone, and everything follows. */
const RAMP = (h: number, tone: number): Record<string, Oklch> => {
  const t = toneAt(tone);
  const head = ensureContrast({ l: t.l, c: t.c, h }, DARK_SURFACE, SURFACE_FLOOR);
  return {
    bg: { l: 0.965, c: 0.01, h },
    head,
    // Polarity follows the body: dark eyes on a light body, light eyes on a
    // dark one. Without this the ink tone would render an invisible face.
    eye: head.l >= 0.5 ? { l: 0.17, c: 0.02, h } : { l: 0.97, c: 0.012, h },
  };
};

/**
 * Minimum contrast ratios as [foreground, background, ratio], applied in order.
 * Later pairs resolve against already-final earlier colors, so the chain
 * converges. `4.5` on the eyes is the WCAG text floor: they are small marks
 * that have to read at 24px.
 *
 * The body/backdrop floor is deliberately weak. The backdrop is off by default,
 * and the pale swatches are meant to sit quietly on a light surface — forcing
 * 1.6:1 there would darken exactly the tones the style exists for.
 */
const FLOORS: [string, string, number][] = [
  ["head", "bg", 1.25],
  ["eye", "head", 4.5],
];

export { FLOORS };

/** The palette in OKLCh, before hex encoding. The test suite asserts against this. */
export function ramp(
  hue: number,
  enforce = true,
  tone = 0,
): Record<string, Oklch> {
  const r = RAMP(hue, tone);
  if (enforce) {
    for (const [fg, bg, min] of FLOORS) {
      r[fg] = ensureContrast(r[fg]!, r[bg]!, min);
    }
  }
  return r;
}

export function palette(hue: number, enforce = true, tone = 0): Palette {
  const r = ramp(hue, enforce, tone);
  const out: Palette = {};
  for (const k in r) out[k as ColorKey] = toHex(r[k]!);
  return out;
}
