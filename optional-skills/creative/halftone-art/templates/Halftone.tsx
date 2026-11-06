/**
 * Halftone.tsx — React bindings for halftone-core.js (zero dependencies).
 *
 * Drop halftone-core.js next to this file. The core is framework-agnostic:
 * plain ES module, no build magic, ~16 KB unminified.
 *
 *   <HalftoneImage src="/portrait.jpg" options={{ cell: 8, mode: 'duotone' }} alt="Portrait" />
 *   <HalftoneField options={{ cell: 8, ink: '#15140f', paper: '#c8ff3d' }}
 *                 fn={(u, v) => 1 - Math.hypot(u - .5, v - .5) * 1.6} />
 *   <HalftoneType cell={5}>HALFTONE</HalftoneType>
 *
 * Rules baked in, learned the hard way:
 *  - renders only when the element actually has a box (ResizeObserver, not window.resize)
 *  - one render per frame at most (rAF-coalesced), so dragging a slider stays smooth
 *  - `prefers-reduced-motion` disables the animated field variant entirely
 *  - the canvas always gets device-pixel resolution; `cell` stays in CSS px
 */

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';

// halftone-core.d.ts ships alongside the core, so this is fully typed.
import { renderHalftone, renderHalftoneField } from './halftone-core.js';

export type DotShape =
  | 'round'
  | 'square'
  | 'diamond'
  | 'ellipse'
  | 'line'
  | 'cross'
  | 'ring';

export interface HalftoneOptions {
  /** cell size in CSS px — the single most important control */
  cell?: number;
  /** screen angle in degrees. 45 is canonical, 15/75 for process colour */
  angle?: number;
  shape?: DotShape;
  /** ink colour for mono, or the first ink for duotone */
  ink?: string;
  /** paper colour — never pure white if you want it to read as print */
  paper?: string;
  /** dot gain: 0.8 lighter / 1 heavier / 1.15 ink-spread poster look */
  gain?: number;
  contrast?: number;
  brightness?: number;
  gamma?: number;
  /** posterise tone into N steps (screenprint separations) */
  levels?: number;
  /** stretch the source's real tonal range before screening */
  autoLevels?: boolean;
  invert?: boolean;
  grain?: number;
  mode?: 'mono' | 'duotone' | 'cmyk';
  inks?: string[];
  plateAngles?: number[];
  plateCurves?: Array<'full' | 'shadows' | 'midtone' | 'highlights' | 'liner'>;
  misregister?: number;
  /** crop bias for object-fit cover, 0..1 each axis */
  focal?: [number, number];
  zoom?: number;
  density?: number;
  /* --- cutout / sticker --- */
  /** drop the backdrop and keep only the subject silhouette (transparent PNG) */
  cutout?: boolean;
  /** 'alpha' = exact (pre-cut PNG), 'key' = flood the backdrop from the corners */
  cutoutMode?: 'auto' | 'flood' | 'key' | 'alpha';
  /** 0..1 colour distance that counts as backdrop, for key/flood mode */
  cutoutTolerance?: number;
  /** force the backdrop colour instead of sampling the corners */
  cutoutKey?: string | null;
  /** alpha threshold when the source already has transparency */
  cutoutAlpha?: number;
  /** 'paper' keeps the stock inside the shape; 'none' is clear vinyl (dots only) */
  cutoutFill?: 'paper' | 'none';
  /** px of paper bleed around the silhouette — the die-cut rim */
  cutoutOutline?: number;
  cutoutOutlineColor?: string | null;
  cutoutKeyTolerance?: number;
  cutoutClose?: number;
  cutoutKeepLargest?: boolean;
}

export type ToneFn = (u: number, v: number, i: number, j: number) => number;

/* ---------------------------------------------------------------------------
 * Core hook: renders a source into a canvas, tracks size, coalesces frames.
 * ------------------------------------------------------------------------- */
function useHalftoneCanvas(
  draw: ((canvas: HTMLCanvasElement, w: number, h: number, dpr: number) => void) | null,
  deps: unknown[],
) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const frame = useRef(0);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !draw) return;

    let visible = true;
    let disposed = false;

    const paint = () => {
      if (disposed || !visible) return;
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return; // 0-size box = nothing to draw
      cancelAnimationFrame(frame.current);
      frame.current = requestAnimationFrame(() => {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        draw(canvas, rect.width, rect.height, dpr);
      });
    };

    paint();

    // ResizeObserver catches container changes that window.resize misses
    // (sidebars, grid reflow, font swap); IntersectionObserver stops work
    // for off-screen plates on long editorial pages.
    const ro = new ResizeObserver(paint);
    ro.observe(canvas);

    const io = new IntersectionObserver(
      (entries) => {
        visible = entries[0]?.isIntersecting ?? true;
        if (visible) paint();
      },
      { rootMargin: '200px' },
    );
    io.observe(canvas);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame.current);
      ro.disconnect();
      io.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}

/* ---------------------------------------------------------------------------
 * <HalftoneImage /> — photographic source → halftone print
 * ------------------------------------------------------------------------- */
export interface HalftoneImageProps {
  src: string;
  alt: string;
  options?: HalftoneOptions;
  className?: string;
  style?: CSSProperties;
  /** paint the flat paper colour before the image decodes */
  placeholder?: string;
}

export function HalftoneImage({
  src,
  alt,
  options,
  className,
  style,
  placeholder,
}: HalftoneImageProps) {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    const el = new Image();
    el.crossOrigin = 'anonymous'; // needed if src is a CDN and you export PNG
    el.decoding = 'async';
    el.onload = () => alive && setImg(el);
    el.onerror = () => alive && setFailed(true);
    el.src = src;
    return () => {
      alive = false;
    };
  }, [src]);

  const opts = useMemo(
    () => ({ density: 2, autoLevels: true, ...options }),
    // options is usually an inline object; stringify keeps the dep stable
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(options)],
  );

  const ref = useHalftoneCanvas(
    img
      ? (canvas, w, h, dpr) =>
          renderHalftone(canvas, img, {
            ...opts,
            width: Math.round(w),
            height: Math.round(h),
            pixelRatio: dpr,
          })
      : null,
    [img, opts],
  );

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={failed ? `${alt} (halftone render unavailable)` : alt}
      className={className}
      style={{
        display: 'block',
        width: '100%',
        height: '100%',
        background: placeholder ?? opts.paper ?? '#f2eee2',
        ...style,
      }}
    />
  );
}

/* ---------------------------------------------------------------------------
 * <HalftoneField /> — procedural dots (hero backgrounds, gradients, blobs)
 * ------------------------------------------------------------------------- */
export interface HalftoneFieldProps {
  fn: ToneFn;
  options?: HalftoneOptions;
  className?: string;
  style?: CSSProperties;
  /** aria-hidden by default: decorative */
  label?: string;
}

export function HalftoneField({ fn, options, className, style, label }: HalftoneFieldProps) {
  const opts = useMemo(
    () => ({ density: 1, ...options }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(options)],
  );

  const ref = useHalftoneCanvas(
    (canvas, w, h, dpr) =>
      renderHalftoneField(canvas, fn, {
        ...opts,
        width: Math.round(w),
        height: Math.round(h),
        pixelRatio: dpr,
      }),
    [fn, opts],
  );

  return (
    <canvas
      ref={ref}
      {...(label ? { role: 'img', 'aria-label': label } : { 'aria-hidden': true })}
      className={className}
      style={{ display: 'block', width: '100%', height: '100%', ...style }}
    />
  );
}

/* ---------------------------------------------------------------------------
 * <HalftoneType /> — display type cut out of a dot screen (pure CSS mask)
 * Never give the masked element a background: the mask eats background too,
 * so an inline box renders as a solid dot bar instead of letterforms.
 * ------------------------------------------------------------------------- */
export interface HalftoneTypeProps {
  children: ReactNode;
  /** dot pitch in px; 4–6 px reads as a fine screen at display sizes */
  cell?: number;
  /** dot radius as a fraction of the cell — 0.55–0.62 keeps stroke weight */
  radius?: number;
  className?: string;
  style?: CSSProperties;
  as?: 'span' | 'h1' | 'h2' | 'div';
}

export function HalftoneType({
  children,
  cell = 5,
  radius = 0.58,
  className,
  style,
  as: Tag = 'span',
}: HalftoneTypeProps) {
  const p = radius * 100;
  const mask = `radial-gradient(circle at 50% 50%, #000 ${p}%, transparent ${p + 2}%)`;
  return (
    <Tag
      className={className}
      style={{
        WebkitMaskImage: mask,
        maskImage: mask,
        WebkitMaskSize: `${cell}px ${cell}px`,
        maskSize: `${cell}px ${cell}px`,
        WebkitMaskRepeat: 'round',
        maskRepeat: 'round',
        ...style,
      }}
    >
      {children}
    </Tag>
  );
}

/* ---------------------------------------------------------------------------
 * <HalftoneSticker /> — die-cut cutout: transparent behind the subject
 * ------------------------------------------------------------------------- */
export interface HalftoneStickerProps extends Omit<HalftoneImageProps, 'placeholder'> {
  /** px of stock bleeding past the silhouette */
  outline?: number;
  outlineColor?: string;
  /** 'paper' = die-cut stock inside the shape, 'none' = clear vinyl */
  fill?: 'paper' | 'none';
}

export function HalftoneSticker({
  outline = 6,
  outlineColor = '#ffffff',
  fill = 'paper',
  options,
  ...rest
}: HalftoneStickerProps) {
  return (
    <HalftoneImage
      {...rest}
      options={{
        cutout: true,
        cutoutOutline: outline,
        cutoutOutlineColor: outlineColor,
        cutoutFill: fill,
        ...options,
      }}
    />
  );
}

/* ---------------------------------------------------------------------------
 * <HalftonePlate /> — static export helper: PNG data URL at any width
 * ------------------------------------------------------------------------- */
export function useExportHalftone(src: string, options: HalftoneOptions) {
  return async (exportWidth = 2000): Promise<string | null> => {
    if (typeof window === 'undefined') return null;
    const img = await new Promise<HTMLImageElement | null>((resolve) => {
      const el = new Image();
      el.crossOrigin = 'anonymous';
      el.onload = () => resolve(el);
      el.onerror = () => resolve(null);
      el.src = src;
    });
    if (!img) return null;
    const ratio = img.naturalHeight / img.naturalWidth || 1;
    const canvas = document.createElement('canvas');
    renderHalftone(canvas, img, {
      ...options,
      pixelRatio: 1,
      width: exportWidth,
      height: Math.round(exportWidth * ratio),
    });
    return canvas.toDataURL('image/png');
  };
}
