import { type ImgHTMLAttributes, type SVGProps } from "react";
import type { Animate } from "./animate";
import { type BlobatarOptions } from "./blobatar";
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
type StaticProps = {
    animate?: false;
} & Omit<ImgHTMLAttributes<HTMLImageElement>, "src">;
type AnimatedProps = {
    animate: Animate;
} & Omit<SVGProps<SVGSVGElement>, "children" | "dangerouslySetInnerHTML" | "viewBox">;
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
} & BlobatarOptions & (StaticProps | AnimatedProps);
export declare function Blobatar({ name: seed, size, background, palette, hue, tone, normalize, contrast, title, animate, expression, traits, ...rest }: BlobatarProps): import("react").JSX.Element;
export {};
//# sourceMappingURL=react.d.ts.map