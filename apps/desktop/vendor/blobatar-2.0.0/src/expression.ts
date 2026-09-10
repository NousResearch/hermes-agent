/**
 * Expressions — `blobatar/expression`. See docs/expression-spec.md.
 *
 * An expression is a named pose the consumer sets and the library holds. It is
 * a separate axis from the idle loop in `animate.ts`: idle motion is ambient and
 * gated on hover, an expression is triggered and gated on nothing. A blobatar can
 * be sad and still breathing.
 *
 * Pose-only, by definition — every channel below moves a part the blobatar already
 * has. Nothing here adds a mark, so a `blob` grows no mouth when it is happy.
 * That ceiling is what the roster has to live inside: two capsule eyes and a
 * soft body, with no brows to carry anger the easy way.
 *
 * **Each expression is passed in, not named.** `blobatar()` never imports this
 * module; it calls through whatever object it is handed, so a consumer who
 * imports `happy` ships `happy` and one who imports nothing ships nothing. That
 * is the same indirection keeping `animate.ts` out of static bundles, and it is
 * here for the same reason: as a plain string option, expressions charged every
 * caller ~420 B for a feature most will never use.
 *
 * It is also why the exports below are plain object literals over shared
 * function references rather than calls to a `make()` factory. A top-level
 * function call is not provably side-effect-free, so every expression would
 * survive tree-shaking whether or not it was imported — the same trap already
 * documented on `_parts` in `blobatar.ts`.
 */

import {
  BILE,
  BLUSH,
  HOT,
  ROSE,
  mixHex,
  tinted,
  type Palette,
  type Tint,
} from "./color";

/**
 * The channels a pose may touch, and nothing else.
 *
 * Petals are excluded on purpose: a sun's nine petals are silhouette, and moving
 * them independently reads as wind or as the creature coming apart. Path data is
 * excluded because interpolating it puts geometry on the main thread every frame.
 *
 * **The body deforms for nothing, so it no longer deforms.** `bsx`, `bsy` and
 * `skew` used to scale and lean the whole creature and have been removed. Three
 * things settled it. They rank fourth of five for legibility in the first place;
 * they are the only channels with no headroom, since frame containment binds at
 * roughly `bsx: 1.08` and a 3° lean puts a body outside the viewBox — so they
 * break the frame before they get loud enough to read; and in this variant the
 * silhouette *is* the identity. Six shapes, a seeded lopsidedness and a seeded
 * lean are what make a grid read as a crowd, and squashing that per-expression
 * is the one move that makes a blobatar stop looking like itself.
 *
 * `bdy` survives because it is a rigid translate. It moves the creature without
 * distorting it, which is what `happy`'s lift and `sad`'s sink actually needed.
 *
 * Units: scales are factors, `tilt` is degrees, offsets are viewBox units, and
 * `heat`, `shake` and `rock` are 0–1 amounts. `tilt` and `edx` are mirrored per
 * side.
 *
 * The `*2` channels are the **second eye's differential**, not its value: they
 * add to the shared channel on the right eye only, so an identity of 0 is a
 * symmetric face and every existing pose keeps emitting exactly what it did.
 * Expressed as deltas rather than as a second endpoint for precisely that
 * reason — a pair of endpoints would force every symmetric pose to state both.
 */
export interface Pose {
  /** Eye width, about each eye's own center. */
  esx: number;
  /** Eye height, about each eye's own center. */
  esy: number;
  /**
   * Eye tilt, mirrored per side: the left eye rotates by `-tilt`, the right by
   * `+tilt`.
   *
   * **What that reads as depends on the eye's orientation, and the sign flips
   * when it changes.** On a *portrait* capsule — the natural shape here, median
   * aspect 2.55:1 — a positive tilt leans both tops outward and brings the inner
   * edges down, which is the angry direction. Flatten the capsule past square
   * with `esy` and the same rotation raises the inner ends instead: on a
   * *landscape* bar, **negative** tilt is the angry `\ /` and positive is the
   * sad `/ \`.
   *
   * This is not a quirk of the implementation, it is what rotating a bar does,
   * and it is invisible to every test in the suite — clearance, containment and
   * the composition gate are all sign-blind. It cost a full roster of poses
   * wearing each other's brows to notice. **Look at the render.**
   */
  tilt: number;
  /** Eye pair offset, positive = down. */
  edy: number;
  /** Eye convergence, positive = apart. */
  edx: number;
  /** Extra width on the right eye only. */
  esx2: number;
  /** Extra height on the right eye only. */
  esy2: number;
  /** Extra tilt on the right eye only, before the per-side mirroring. */
  tilt2: number;
  /**
   * Extra vertical offset on the right eye only, positive = down.
   *
   * The differential the first three rosters never needed. There is one for
   * width, one for height and one for tilt, and none for *position*, because
   * every pose up to `thinking` said its asymmetry with shape — `wink` closes an
   * eye, `unsure` shrinks one. A pair of eyes at two **heights** is a sentence
   * none of those can make, and it is the one that reads as attention pointed
   * somewhere other than at you.
   *
   * It is also the static half of `rock`, and the reason a `thinking` blobatar
   * still reads with the stylesheet missing or the loop stopped for reduced
   * motion: what is left on the face is one frame of the seesaw rather than
   * nothing at all.
   */
  edy2: number;
  /**
   * How much of the **seeded** eye lean the pose overrides, 0–1. At 0 the pose's
   * `tilt` adds to whatever lean the seed drew; at 1 the seeded lean is cancelled
   * and the eyes sit at exactly the angle the pose names.
   *
   * It exists because `tilt` is a *brow*, and a brow is an absolute direction.
   * `styles/blob.ts` leans each eye by up to 12° in a single seeded direction —
   * identity, and good identity, on an idle face. Added to a pose it is not
   * identity, it is noise on the one channel that carries the meaning: `mad`'s
   * `\ /` at −33° meets a +12° seed and comes out `\ \`, a pair of parallel bars
   * that read as bored rather than angry, while the seed 12° the other way gets
   * an unusually furious `mad`. The reference renders are unambiguous about which
   * of those is wanted.
   *
   * So the loud poses take their tilt absolute and the seeded lean returns intact
   * the moment the expression clears — the identity is in the idle face, not in a
   * per-seed discount on the expression. This is the same rule `esx`/`esy` already
   * follow by being factors on a shape the pose flattens nearly out of existence:
   * every blobatar wears the same strength of a given expression.
   *
   * Interpolates like every other channel, so the lean eases out over the morph
   * rather than snapping.
   */
  lock: number;
  /** How far the palette shifts toward its hot pair, 0–1. */
  heat: number;
  /** Tremor amplitude, 0–1. Held, not fired — see `@keyframes mo-shake`. */
  shake: number;
  /**
   * Seesaw amplitude, 0–1. Held, not fired — see `@keyframes mo-rock`.
   *
   * The second channel whose content is a *duration* rather than a shape, and
   * built the way `shake` is for the same reason: an expression is set and held,
   * with no timers, no self-termination and no notion of firing again, so a
   * repeating motion has to be an amplitude on a loop that always runs and
   * resolves to the identity at 0. `shake` made that argument once; this is the
   * evidence it generalises past a tremor.
   *
   * What it drives is **antiphase**: the left eye rises as the right one falls,
   * on `--mo-wrap`'s sign, so the pair trades heights and the mean stays put. A
   * bounce — both eyes dipping together — would be `mo-bob` at a second period
   * and would beat against it. A trade cannot.
   *
   * The swing is `edy2` wide, which is what makes the static bake exactly frame
   * zero of the loop rather than an approximation of it. See `bakePose`.
   *
   * What the number means, precisely: it is how much of the stagger *takes part*
   * in the swing. At 1 the pair fully inverts every half cycle; at 0.7 it swings
   * through level and comes back out 40% inverted, about the same mean. So the
   * two extremes are not mirror images, and that is the intent — the pose the
   * consumer set stays the one the face spends most of its time near, and the
   * loop breathes around it rather than replacing it half the time.
   */
  rock: number;
  /** Whole-creature offset, positive = down. */
  bdy: number;
}

/**
 * The identity pose, and the key list — every channel's custom property is its
 * own name prefixed, so no lookup table is needed. Iterating this is what lets
 * `poseVars` skip channels a pose leaves alone, which is why `idle` emits
 * nothing at all.
 */
const IDENT: Pose = {
  esx: 1,
  esy: 1,
  tilt: 0,
  edy: 0,
  edx: 0,
  esx2: 0,
  esy2: 0,
  tilt2: 0,
  edy2: 0,
  lock: 0,
  heat: 0,
  shake: 0,
  rock: 0,
  bdy: 0,
};

/** What `bakePose` needs of a layout. Structural, so this module imports no variant. */
export interface Posable {
  eyes: { cx: number; cy: number; rx: number; ry: number; rot: number }[];
}

/**
 * A pose plus the two ways to apply it.
 *
 * The functions ride on the value rather than being imported by the renderer:
 * that is what makes the whole module reachable only from a consumer's own
 * import, and it is the difference between the core paying for expressions and
 * not.
 */
export interface Expression {
  readonly p: Pose;
  readonly vars: (p: Pose) => Record<string, string>;
  readonly bake: <L extends Posable>(l: L, p: Pose) => { l: L; wrap: string };
  /**
   * The palette this pose wears, for the poses that move it. Absent on the ones
   * that do not, which is what keeps the colour path out of every bundle that imports
   * only cool expressions — the same indirection `vars` and `bake` use to keep
   * the whole module out of the core.
   */
  readonly tint?: (pal: Palette, p: Pose) => Palette;
}

const r3 = (v: number) => String(Math.round(v * 1000) / 1000);

/**
 * The animated path: pose as registered custom properties.
 *
 * Only channels that actually move are emitted. Every property is registered in
 * `motion.css` with its identity as `initial-value`, so an omitted declaration
 * *is* the identity — which makes `idle` free, shortens every other pose's
 * output, and means clearing an expression transitions back toward those
 * initials rather than snapping.
 *
 * The morph is therefore not implemented anywhere. It is what a transition on
 * these numbers does.
 */
export function poseVars(p: Pose): Record<string, string> {
  const out: Record<string, string> = {};
  for (const k in IDENT) {
    // `heat` is the one channel the stylesheet never sees. Colour is not
    // composed in CSS — it is resolved here and emitted as the finished
    // `--mo-head`/`--mo-eye`, so a `--mo-heat` declaration would be a variable
    // that looks live and is read by nothing. See `heatTint`.
    if (k === "heat") continue;
    const v = p[k as keyof Pose];
    if (v !== IDENT[k as keyof Pose]) out["--mo-" + k] = r3(v);
  }
  return out;
}

/**
 * The palette a tinting pose wears: the resolved one, mixed `heat` of the way
 * toward the pair `tinted` derives for its target.
 *
 * **The mix happens here rather than in CSS, and that is the load-bearing
 * choice.** The obvious shape — one registered `--mo-heat` and a
 * `color-mix(in oklab, …)` in the stylesheet — morphs beautifully on the way in
 * and pops on the way out: the hot endpoint is a custom property the expression
 * emits, so clearing the expression deletes it in the same frame `--mo-heat`
 * begins easing back, and the fill snaps to base while everything else takes
 * 560ms. Resolving the colour here instead means the stylesheet only ever sees a
 * finished colour, and `transition: fill` carries it symmetrically in both
 * directions for free.
 *
 * The pair is derived per seed rather than authored as one colour, because
 * `blob` flips its eye between near-black and near-white with the body's
 * lightness and no fixed red clears 4.5:1 against both. `tinted()` keeps that
 * guarantee across the whole mix, not merely at its ends; `test/color.test.ts`
 * verifies it at every hue, tone, heat **and target**.
 *
 * The target is a parameter rather than a constant because the roster now has
 * more than one tinting pose. Every one of them goes through this single walk —
 * a second copy of it is the thing most likely to keep the guarantee by
 * accident, and stop keeping it without saying so.
 */
export function tintWith(pal: Palette, p: Pose, t: Tint): Palette {
  const [head, eye] = tinted(pal.head!, pal.eye!, t);
  return {
    ...pal,
    head: mixHex(pal.head!, head, p.heat),
    eye: mixHex(pal.eye!, eye, p.heat),
  };
}

/**
 * The red target, bound. An arrow rather than a `bind` or a factory call, for
 * the reason the module header gives: a top-level *call* is not provably
 * side-effect-free and would pin the whole colour path into every bundle,
 * imported or not. A function expression is.
 */
export const heatTint = (pal: Palette, p: Pose) => tintWith(pal, p, HOT);

/**
 * The static path: eye channels baked into geometry, body channels handed back
 * as one `transform` attribute for the caller to wrap.
 *
 * Baking is exact rather than approximate, because the CSS composes in the same
 * order the geometry does. `superellipse` scales by `rx`/`ry` and *then* rotates,
 * and `.mo-eye` applies the pose scale innermost (in `@keyframes mo-blink`) and
 * the tilt outside it (in `mo-wrap`'s `rotate`). The offsets commute with both,
 * since the rotation and scale are about each eye's own center —
 * `transform-box: fill-box` on the animated side, the eye's own `cx`/`cy` here.
 *
 * The body half of that attribute used to be a three-transform chain derived
 * from the CSS — a scale about (50, 50), an offset, and a lean — and needed a
 * `translate(50 50) … translate(-50 -50)` sandwich to put the origin in the
 * right place. With the deforming channels gone it is one rigid translate, which
 * commutes with everything and needs no origin at all. The whole apparatus that
 * kept it honest, including the divergence the composition gate measured, was in
 * service of the two channels that are no longer here.
 *
 * The eye channels stay out of that attribute: baking them costs nothing, where
 * a per-eye transform would need its own origin round-trip and ~85 B per eye.
 */
export function bakePose<L extends Posable>(
  l: L,
  p: Pose,
): { l: L; wrap: string } {
  return {
    l: {
      ...l,
      eyes: l.eyes.map((e, i) => ({
        ...e,
        // `--mo-wrap`'s sign, spelled out: -1 on the left eye, +1 on the right,
        // so a positive tilt leans both tops outward and brings the inner edges
        // down. That asymmetry is the entire brow vocabulary available here.
        cx: e.cx + p.edx * (i ? 1 : -1),
        // `edy2` lands on the right eye only, like every other differential —
        // and unlike them it has a moving counterpart, which is what decides the
        // shape of `--mo-ph` on the animated side rather than the other way
        // round. The seesaw swings the pair symmetrically about its own centre,
        // so its share of the differential is `(1 + wrap·phase) / 2`; that
        // expression *is* `--mo-sel` at phase +1, on both eyes, because `wrap`
        // is ±1. So the stagger baked here is exactly the loop's own extreme,
        // with no compensating term anywhere and nothing to keep in step by
        // hand. `probe-compose.ts` check A measures the two against each other.
        cy: e.cy + p.edy + (i ? p.edy2 : 0),
        // The `*2` differential lands on the right eye only, which is
        // `--mo-sel`'s job on the animated side. It is added *before* the
        // mirroring on `tilt`, matching `calc(var(--mo-t) * var(--mo-wrap))` —
        // adding it after would flip its sign on the left eye and turn a
        // one-sided brow into a symmetric one.
        rx: e.rx * (p.esx + (i ? p.esx2 : 0)),
        ry: e.ry * (p.esy + (i ? p.esy2 : 0)),
        // `lock` fades the seeded lean out rather than switching it off, which
        // is what lets it interpolate on the animated side. At 0 this is the
        // plain sum it always was.
        //
        // The animated path cannot bake it away — the lean is already in the
        // path's coordinates — so it subtracts the same amount on `.mo-eye`'s
        // `rotate` instead. Both resolve to R(tilt·wrap) · scale about the eye's
        // own centre; `probe-compose.ts` check A measures that they agree.
        rot: e.rot * (1 - p.lock) + (p.tilt + (i ? p.tilt2 : 0)) * (i ? 1 : -1),
      })),
    },
    wrap: p.bdy !== 0 ? `translate(0 ${r3(p.bdy)})` : "",
  };
}

/**
 * The roster. Frozen per major, exactly like the shape thresholds and the tone
 * set — a fifth expression added later is additive and safe, renaming one is not.
 *
 * These are the third tuning pass, and the loud one. The second pass established
 * *which* channels carry — eye height, then convergence, then vertical offset,
 * with tilt nearly free of signal on a tall capsule — and then set them
 * timidly: eyes scaled by 1.06–1.14 and squashed to 0.5–0.72. Measured against
 * the reference frames in `docs/references/`, that is roughly half the amplitude
 * a face needs to read as anything at a glance.
 *
 * The numbers below come off measurement rather than taste alone. Across 4000
 * seeds the capsule's natural aspect is 2.03 / 2.55 / 3.12 (p10 / median / p90,
 * `ry:rx`) — a *portrait* capsule — so a landscape bar needs `esx/esy` near 10,
 * not near 2. Containment was never the binding guard (worst corner reach is
 * ~0.95 of the 1.12 the test allows); fusion is, and a widened pair buys its
 * clearance back through `edx` at about 1:2. Tilt is cheap once the capsule is
 * flat, because a flat capsule sweeps sideways far less per degree — which is
 * what finally makes tilt a real brow rather than the dead channel it was.
 *
 * `sad` and `mad` are still the pair at risk and are still separated by three
 * channels disagreeing at once, never by one: `sad` is small eyes, far apart and
 * low; `mad` is wide flat bars, tilted hard into a V, over a compressed body.
 * Exaggeration makes that separation easier, not harder.
 *
 * Tune these in the tuning grid's paired mode, not in this file.
 */
export const idle: Expression = { p: IDENT, vars: poseVars, bake: bakePose };

/** Wide flat arcs riding high: the universal smiling squint, at full volume. */
export const happy: Expression = {
  p: {
    esx: 1.72,
    esy: 0.3,
    tilt: 8,
    edy: -1.5,
    edx: 1.5,
    // A touch of asymmetry, well short of a wink. The pair reads as *drawn*
    // rather than as stamped twice, which is the whole reason the channel
    // exists — see `docs/references/asymmetric.png` for the loud version.
    esx2: 0.08,
    esy2: 0.05,
    tilt2: -16,
    edy2: 0,
    lock: 1,
    heat: 0,
    shake: 0,
    rock: 0,
    bdy: -2.2,
  },
  vars: poseVars,
  bake: bakePose,
};

/** Small eyes, low and drifted apart, over a body that sinks. */
export const sad: Expression = {
  p: {
    esx: 0.6,
    esy: 0.56,
    tilt: 26,
    edy: 3.6,
    edx: 1.9,
    esx2: -0.05,
    esy2: -0.07,
    tilt2: -7,
    edy2: 0,
    lock: 1,
    heat: 0,
    shake: 0,
    rock: 0,
    bdy: 2.6,
  },
  vars: poseVars,
  bake: bakePose,
};

/**
 * A hard `\ /` of flat bars over a body that compresses, leans and runs hot.
 *
 * The only pose that spends `heat` and `shake`, and the reason both channels
 * exist: `docs/references/angry-red.png` is a brow-down V on a body turned red
 * and vibrating, and two capsule eyes with no brows cannot reach that on
 * geometry alone. `esx` widens the capsules until the tilt has a bar to work on;
 * `edx` stays slightly positive because the widening already closes the inner
 * gap far more than a convergence would, and spending clearance on both fuses
 * the pair.
 */
export const mad: Expression = {
  p: {
    esx: 1.85,
    esy: 0.26,
    tilt: -33,
    edy: 0.4,
    edx: 0.6,
    esx2: 0,
    esy2: -0.03,
    tilt2: 5,
    edy2: 0,
    lock: 1,
    heat: 0.62,
    shake: 0.55,
    rock: 0,
    bdy: 0.8,
  },
  vars: poseVars,
  bake: bakePose,
  tint: heatTint,
};

/**
 * The second roster, added in a minor and additive by construction — every pose
 * below is data on channels that already existed, so `motion.css` does not learn
 * a single new thing and a consumer who imports none of them pays nothing.
 *
 * They were chosen against §2's ceiling rather than against a list of emoji.
 * The rule that picked them: a pose has to differ from its nearest neighbour on
 * **three channels at once**, and the nearest neighbour is usually not the one
 * the name suggests. `sleepy` is not near `sad`, it is near `mad` — both are flat
 * bars — and it survives on tilt, offset and heat all disagreeing.
 *
 * Two of them are only reachable because the `*2` differentials exist. `wink`
 * and `unsure` are *built* on asymmetry rather than seasoned with it, which is
 * the one shape in this vocabulary that no amount of amplitude on a symmetric
 * pose can imitate.
 *
 * Signs, since §2.2 is the trap that ate a whole tuning pass: the capsule's
 * natural aspect is 2.55:1 portrait, so a pose that leaves `esy` near or above 1
 * is still portrait and reads the *opposite* way from a pose that flattens it
 * past square. On a portrait capsule positive `tilt` is the angry direction; on
 * a landscape bar it is the worried one. `surprised` and `scared` are portrait
 * and lean negative; `smug` and `sleepy` are landscape and lean positive.
 */

/**
 * Eyes enlarged rather than squashed — the one direction the first roster never
 * went, and the reason this is the safest addition of the six.
 *
 * Every other pose in the library reduces `esy`; against three poses that all
 * live between 0.26 and 0.56, a pose at 1.34 cannot be mistaken for any of them
 * at any size. It is the antipode of `mad` on the load-bearing channel.
 *
 * Containment is the binding guard here for the first time in this feature —
 * growing the pair and lifting it both push the eye's corners toward the body
 * outline, where every earlier pose was bounded by fusion instead. `edx` is
 * mildly positive for the usual clearance reason and because a wide-set stare
 * reads as startled rather than as focused.
 */
export const surprised: Expression = {
  p: {
    esx: 1.34,
    esy: 1.2,
    tilt: -6,
    edy: -1.05,
    edx: 0.5,
    esx2: 0.05,
    esy2: 0.07,
    tilt2: 3,
    edy2: 0,
    lock: 1,
    heat: 0,
    shake: 0,
    rock: 0,
    bdy: -1.4,
  },
  vars: poseVars,
  bake: bakePose,
};

/**
 * One eye a flat arc, the other open — the pose the per-eye differentials were
 * built for, and the only one in the roster whose meaning *is* the asymmetry.
 *
 * `esy2` does the work at −0.56 against a shared 0.76, which leaves the right
 * eye at 0.2 — closed by this vocabulary's standards — while the left stays
 * nearly its drawn height. `esx2` widens the closed one, because a closing eye
 * spreads as it flattens and a slit that is also narrow reads as a defect.
 *
 * It is the loudest use of the differentials by a wide margin: `happy` sets
 * `esy2` to 0.05 explicitly to stay *short* of a wink, and this is that channel
 * eleven times over.
 */
export const wink: Expression = {
  p: {
    esx: 1.32,
    esy: 0.76,
    tilt: 5,
    edy: -0.6,
    edx: 0.8,
    esx2: 0.26,
    esy2: -0.56,
    tilt2: -11,
    edy2: 0,
    lock: 1,
    heat: 0,
    shake: 0,
    rock: 0,
    bdy: -1.1,
  },
  vars: poseVars,
  bake: bakePose,
};

/**
 * Flat bars with no angle in them, sitting low over a sunk body.
 *
 * The nearest neighbour is `mad`, not `sad` — both are landscape bars, and at
 * 44px the silhouette of a flat capsule is most of what reads. The separation is
 * `tilt` (0 against −33), `edy` (+2.4 against +0.4) and the tint and tremor
 * `mad` carries and this does not. That is three channels, which is the §2 rule
 * and the reason this pose is safe rather than a second angry face.
 *
 * `tilt: 0` under `lock: 1` is not a no-op: it cancels the seeded lean, which is
 * the whole point — a level pair of bars is what reads as lidded, and a seed's
 * 12° lean on that pose reads as suspicion instead.
 */
export const sleepy: Expression = {
  p: {
    esx: 1.14,
    esy: 0.22,
    tilt: 0,
    edy: 2.4,
    edx: 0.3,
    esx2: -0.04,
    esy2: 0.03,
    tilt2: 4,
    edy2: 0,
    lock: 1,
    heat: 0,
    shake: 0,
    rock: 0,
    bdy: 1.2,
  },
  vars: poseVars,
  bake: bakePose,
};

/**
 * Half-lidded, lifted, and leaning in parallel.
 *
 * `tilt2 = −2 × tilt` exactly, which is the `proud-flat.png` trick `happy`
 * already uses: the mirroring makes the left eye `−tilt` and the right
 * `tilt + tilt2`, so setting the differential to `−2t` puts both at `−t` and the
 * pair tilts *together* rather than symmetrically. A symmetric tilt is a brow
 * and reads as an emotion; a parallel tilt is a head cocked and reads as an
 * attitude, which is the entire difference between this pose and `mad`.
 *
 * It sits at `esy: 0.42`, between `happy`'s squint and `sad`'s eye, on purpose:
 * a fully flat smug bar is `sleepy`, and a round one loses the lidded read.
 */
export const smug: Expression = {
  p: {
    esx: 1.3,
    esy: 0.42,
    tilt: 18,
    edy: -0.5,
    edx: 0.5,
    esx2: 0.06,
    esy2: -0.06,
    tilt2: -36,
    edy2: 0,
    lock: 1,
    heat: 0,
    shake: 0,
    rock: 0,
    bdy: -1,
  },
  vars: poseVars,
  bake: bakePose,
};

/**
 * One eye narrowed, the other open — the second differential-first pose, and the
 * quieter one.
 *
 * Where `wink` closes an eye outright, this squashes one to about a third of the
 * other and leaves both clearly eyes. The mismatch is the message: two eyes at
 * different heights on the same face is a thing no symmetric pose can say, and
 * it is what "unsure" looks like without a mouth to twist.
 *
 * The shared `tilt` is small and the differential is large, so the pair comes out
 * near-parallel rather than as a brow — a V here would fight the size mismatch
 * for attention and the mismatch wins anyway.
 */
export const unsure: Expression = {
  p: {
    esx: 0.95,
    esy: 1.02,
    tilt: 4,
    edy: -0.2,
    edx: 0.3,
    esx2: 0.24,
    esy2: -0.44,
    tilt2: -18,
    edy2: 0,
    lock: 1,
    heat: 0,
    shake: 0,
    rock: 0,
    bdy: 0,
  },
  vars: poseVars,
  bake: bakePose,
};

/**
 * Small eyes held high and pulled together, over a body that trembles.
 *
 * The second pose to spend `shake`, and the one that shows the channel is not
 * `mad`'s private property — a tremor is arousal, and arousal is not only anger.
 * It runs at 0.35 against `mad`'s 0.55, which is a shiver rather than a rage.
 *
 * The nearest neighbour is `sad`, and the separation is deliberately the three
 * loudest channels available: the pair goes *up* rather than down (`edy` −1.5
 * against +3.6), *together* rather than apart (`edx` −0.8 against +1.9), and it
 * trembles. Convergence is what costs here — pulling the pair in spends fusion
 * clearance directly — which is why the eyes narrow on the way in and buy most
 * of it back.
 *
 * No tint, which is the point of listing it next to `mad`: fear is not hot, and
 * a consumer who wants this pose should not pay 700 B of colour code for it.
 */
export const scared: Expression = {
  p: {
    esx: 0.78,
    esy: 0.96,
    tilt: -12,
    edy: -1.5,
    edx: -0.8,
    esx2: -0.04,
    esy2: 0.05,
    tilt2: 4,
    edy2: 0,
    lock: 1,
    heat: 0,
    shake: 0.35,
    rock: 0,
    bdy: -0.6,
  },
  vars: poseVars,
  bake: bakePose,
};

/**
 * The tinting targets, and the third roster.
 *
 * `mad` was the only pose that spent colour for two releases, and the reason was
 * never that anger is special — it was that `hot()` was a red with the walk that
 * keeps the contrast guarantee baked into it. Generalising that walk to a `Tint`
 * is the whole enabling change: these three poses are numbers, and the guarantee
 * they inherit is the same one `mad` has always had, verified across every hue,
 * tone, heat *and* target in `test/color.test.ts`.
 *
 * Colour is the loudest channel in the vocabulary (§11) and it is the only one
 * that does not have to fight two capsule eyes for legibility — so the rule for
 * spending it is the rule that already governs every other channel, not a softer
 * one. **A tint is never the only thing separating two poses.** `sick` is not a
 * green `sleepy`, it is `sleepy` with the bars tilted and the body trembling,
 * and it would still be a different pose in greyscale.
 *
 * The price is stated plainly because it is not small: the first tinting pose in
 * a bundle carries the colour path — `tinted`, `mixHex`, `fromHex` and the OKLab
 * matrices, about 700 B. Every one after it is free, since they share the walk.
 * A consumer who wants only cool poses still pays none of it.
 *
 * The targets themselves live in `color.ts` beside `HOT`: `ROSE` is warmth
 * rather than heat, `BLUSH` is pale by construction, and `BILE` is the only one
 * that is not on the warm half of the wheel. They sit there so that the module
 * owning the contrast guarantee owns every endpoint it has to hold for, and so
 * the suite can iterate them.
 */

/**
 * Tall narrow eyes, drawn together, lifted, and rose.
 *
 * The first cut of this pose was wide *and* tall and it failed the bar this
 * whole section sets: rendered in greyscale beside `surprised` it was the same
 * face, and the tint was carrying the entire meaning. A pose that only works in
 * colour is a pose that stops working in a screenshot, a print stylesheet, or
 * anywhere the palette is overridden.
 *
 * So the two large-eyed poses now disagree on *shape* rather than only on
 * direction. `surprised` is wide and spread (`esx` 1.34, `edx` +0.5); this is
 * narrow and drawn together (0.86, −0.35) — startled *by* you against looking
 * *at* you. Three channels plus the tint, and the first three are enough.
 *
 * Convergence is what costs: it spends fusion clearance directly, and the
 * narrowing is what pays for it.
 */
export const love: Expression = {
  p: {
    esx: 0.86,
    esy: 1.28,
    tilt: -14,
    edy: -0.5,
    edx: -0.35,
    esx2: 0.05,
    esy2: 0.06,
    tilt2: 6,
    edy2: 0,
    lock: 1,
    heat: 0.6,
    shake: 0,
    rock: 0,
    bdy: -1.6,
  },
  vars: poseVars,
  bake: bakePose,
  tint: (pal, p) => tintWith(pal, p, ROSE),
};

/**
 * Small squeezed eyes, low and wide apart, over a body that sinks and blushes.
 *
 * The nearest neighbour is `sad`, and the honest reading is that this is the
 * closest pair in the roster: both are small eyes sitting low. They separate on
 * eye *height* (0.45 against 0.56 — squeezed rather than merely small), on how
 * far each sinks, and on the tint. `BLUSH` pulls only 0.4 of the way and lands
 * pale on purpose; a shy blobatar that goes as red as an angry one is an angry
 * one.
 */
export const shy: Expression = {
  p: {
    esx: 0.62,
    esy: 0.5,
    tilt: 10,
    edy: 1.4,
    edx: -0.2,
    esx2: -0.05,
    esy2: -0.04,
    tilt2: -8,
    edy2: 0,
    lock: 1,
    heat: 0.55,
    shake: 0,
    rock: 0,
    bdy: 0.9,
  },
  vars: poseVars,
  bake: bakePose,
  tint: (pal, p) => tintWith(pal, p, BLUSH),
};

/**
 * Flat bars slumped into a `/ \`, over a body that sinks, greens and trembles.
 *
 * The tilt is positive and that is not a typo — on a *landscape* bar the sign
 * inverts (§2.2), so +20 raises the inner ends into the worried `/ \` while
 * `mad`'s −33 drops them into the angry `\ /`. Same channel, opposite reading,
 * and no test in the suite can see the difference.
 *
 * The tremor is a queasy 0.18, well under `scared`'s 0.35 and `mad`'s 0.55. It
 * is the third pose to spend the channel, which is the argument for `shake`
 * being an amplitude on an always-running loop rather than an event: three poses
 * use it at three strengths and none of them had to add anything.
 */
export const sick: Expression = {
  p: {
    esx: 1.25,
    esy: 0.34,
    tilt: 20,
    edy: 1.8,
    edx: 0.8,
    esx2: 0.05,
    esy2: -0.05,
    tilt2: -6,
    edy2: 0,
    lock: 1,
    heat: 0.6,
    shake: 0.18,
    rock: 0,
    bdy: 1.4,
  },
  vars: poseVars,
  bake: bakePose,
  tint: (pal, p) => tintWith(pal, p, BILE),
};

/**
 * The fourth roster, which is one pose, and the first whose message is a
 * *duration* rather than a shape.
 *
 * Every pose before this one is a still frame: `sad` is sad in a screenshot,
 * `mad` is angry in a print stylesheet. "Thinking" is not available on those
 * terms — the whole content of it is *still*, as in still going, and a face
 * holding one shape cannot say still. That is why this pose costs two channels
 * where the second and third rosters cost none: `edy2` for the shape and `rock`
 * for the duration, and neither one carries it alone.
 *
 * **Mid-lidded, level eyes at two different heights, trading places.** The
 * stagger is the statement and the seesaw is the statement changing its mind,
 * which is what being busy looks like on a creature with no mouth and nothing to
 * tap. It is also, not by accident, the two-dot loader every user already reads
 * without being taught — except that a blobatar does not have to *add* two dots,
 * it has exactly two, at eye height, on a face.
 *
 * `tilt: 0` under `lock: 1` for `sleepy`'s reason: a level pair reads as
 * attention held elsewhere, and a seed's 12° lean turns that into suspicion.
 *
 * The nearest neighbour is `unsure`, not `sleepy` — the two asymmetric faces —
 * and the §2 rule clears on four channels: `edy2` (−2.6 against 0), `rock` (0.7
 * against 0), `esy` (0.50 against 1.02) and `esx2` (0.02 against 0.24). The
 * sentences differ too, which is the part the table cannot show: `unsure` is
 * eyes of mismatched *size*, this is eyes at mismatched *height*, and mismatched
 * height is the only one of the two that has anywhere to go.
 *
 * `edy` is positive on a pose that reads as lifted, and that is not a typo. The
 * pair's mean sits at `edy + edy2/2` — +1.0 and −1.3 — so the eyes straddle the
 * idle line 0.3 units high with the stagger hung across it, rather than the
 * whole pair riding low and the right eye alone reaching up.
 */
export const thinking: Expression = {
  p: {
    esx: 1.15,
    esy: 0.62,
    tilt: 0,
    edy: 4.2,
    edx: 0.4,
    esx2: 0.02,
    esy2: 0.06,
    tilt2: 0,
    edy2: -8.4,
    lock: 1,
    heat: 0,
    shake: 0,
    rock: 0.8,
    bdy: -0.4,
  },
  vars: poseVars,
  bake: bakePose,
};
