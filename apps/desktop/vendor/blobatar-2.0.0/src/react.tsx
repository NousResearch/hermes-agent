import { useMemo, type ImgHTMLAttributes, type SVGProps } from "react";
import type { Animate } from "./animate";
import { _parts, type BlobatarOptions } from "./blobatar";
import { blobatarUri } from "./uri";

/**
 * Two rendering modes, and the props follow the mode.
 *
 * Static blobatars render as an `<img>`: a list of a few hundred is exactly the
 * case where you do not want extra DOM nodes per screen, and nothing here uses
 * `currentColor`, so inline SVG would buy nothing.
 *
 * Animated blobatars cannot. Content inside an `<img>` is an isolated,
 * non-interactive document — `:hover` never fires inside it and host-page CSS
 * cannot reach the shapes — so `animate` switches to inline SVG and costs
 * roughly a dozen nodes per blobatar. That trade is the reason animation is
 * opt-in rather than a default.
 *
 * The union is deliberate: `onLoad` should stop type-checking the moment you
 * turn animation on, because it stops firing.
 */
type StaticProps = { animate?: false } & Omit<ImgHTMLAttributes<HTMLImageElement>, "src">;

type AnimatedProps = { animate: Animate } & Omit<
  SVGProps<SVGSVGElement>,
  "children" | "dangerouslySetInnerHTML" | "viewBox"
>;

export type BlobatarProps = {
  /**
   * Who the blobatar is for. A username, a display name, an email, a bot's
   * handle, a user id — any string, and the same string always renders the
   * same blobatar. The only required prop.
   *
   * Named for what the value is rather than for what the library does with it.
   * A blobatar always stands for somebody, and that somebody has a name; `seed`
   * describes the hashing step, which is this module's business and not the
   * call site's. Internally it is still a seed, and the docs that are about
   * derivation still call it one — see `CONTEXT.md`.
   */
  name: string;
} & BlobatarOptions &
  (StaticProps | AnimatedProps);

export function Blobatar({
  name: seed,
  size,
  background,
  palette,
  hue,
  tone,
  normalize,
  contrast,
  title,
  animate,
  expression,
  traits,
  ...rest
}: BlobatarProps) {
  // Pulled out explicitly like every other option, because what is left in
  // `rest` goes straight onto the DOM element — a `traits` object spread onto
  // an `<img>` is a React warning per blobatar.
  const opts = { size, background, palette, hue, tone, normalize, contrast, title, expression, traits };

  // Both branches are hooks-stable: `animate` changing swaps the element type,
  // which remounts anyway.
  //
  const dep = JSON.stringify([seed, opts, animate]);

  const src = useMemo(
    () => (animate ? "" : blobatarUri(seed, opts)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dep],
  );

  const parts = useMemo(
    () => (animate ? _parts(seed, { ...opts, animate }) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dep],
  );

  /**
   * The markup object, kept stable by identity and not just by value.
   *
   * Getting the varying class out of `parts.inner` is only half of what the
   * morph needs. React compares props by reference, and `dangerouslySetInnerHTML`
   * is a fresh `{__html}` literal on every render — so it re-assigns
   * `innerHTML` whenever anything else about the blobatar changes, even when the
   * string is byte-identical. That assignment destroys and rebuilds the whole
   * subtree, which is exactly the thing an expression change must not do: a
   * fresh element has no previous computed value, so no transition runs on it,
   * and every idle animation underneath restarts from phase zero and throws
   * away its seeded offset.
   *
   * Keyed on the string, so the object survives an expression change and
   * changes only when the markup genuinely does.
   */
  const html = useMemo(() => ({ __html: parts?.inner ?? "" }), [parts?.inner]);

  if (parts) {
    const { style, ...svgRest } = rest as SVGProps<SVGSVGElement>;
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 100 100"
        width={size}
        height={size}
        // With a `title` the markup carries a `<title>`, so this is a labelled
        // image; without one it is decoration and should be skipped entirely —
        // the same call `alt=""` makes on the `<img>` path. Never both: a
        // `role="img"` that is also `aria-hidden` just contradicts itself.
        role={title ? "img" : undefined}
        aria-hidden={title ? undefined : true}
        style={{ ...(parts.vars as React.CSSProperties), ...style }}
        {...svgRest}
      >
        {/*
          Three real children rather than one innerHTML blob, and the reason is
          the morph.

          Only the third varies at runtime — its class does, when the expression
          changes — and `dangerouslySetInnerHTML` is all-or-nothing: had the root
          `<g>` stayed inside that string, every expression change would replace
          the entire subtree. A fresh element has no previous computed value, so
          no transition runs on it and every idle animation under it restarts
          from phase zero. Split this way, React writes one attribute and the
          DOM below survives, which is what the transition needs to exist at all.

          The first two have to be siblings of the root rather than inside it:
          `<title>` names the element it is the first child of, so nesting it
          would label a `<g>` instead of this `<svg>`, and the backdrop must sit
          outside the hover-lift or the plate scales with the creature.
        */}
        {title ? <title>{title}</title> : null}
        {parts.bg ? <path d={parts.bg.d} fill={parts.bg.fill} /> : null}
        <g className={parts.cls} dangerouslySetInnerHTML={html} />
      </svg>
    );
  }

  const { alt, ...imgRest } = rest as ImgHTMLAttributes<HTMLImageElement>;
  return <img src={src} width={size} height={size} alt={alt ?? title ?? ""} {...imgRest} />;
}
