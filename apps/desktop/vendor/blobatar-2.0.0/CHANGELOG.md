# Changelog

What changed, and — where it matters — what it costs to upgrade.

The thing this file exists to state clearly is churn. A blobatar is derived from
a name, so anything that moves the seed → look mapping changes faces that are
already in production, and no other release note in a package like this one is
as important. Releases that move it say so first.

The mapping itself is frozen per **generation**, and the package major selects
one: `blobatar@1` renders gen1, `blobatar@2` renders gen2. See
[ADR-0006](../../docs/adr/0006-generations.md) and
[ADR-0008](../../docs/adr/0008-package-majors-select-generations.md).

## 2.0.0

**Every seed renders differently.** gen2's ten silhouettes replace gen1's six,
and a new shape is not additive — it takes its share of the band table from the
existing ones. Roughly a third of names come out byte-identical anyway, because
a round body with room for its eyes is drawn by the same arithmetic under both
vocabularies; the rest move. Stay on `blobatar@1` if that is not acceptable
yet, and upgrade when it is.

### Added

- Four silhouettes: `capsule`, `triangle`, `hexagon` and `droplet`, alongside
  `round`, `organic`, `boxy`, `nub`, `cloud` and `sun`. Weighted rather than
  uniform — round and organic stay the everyday shapes and the louder ones stay
  finds.
- Trait keys for what the new shapes read: `capsule.squat`, `poly.round`
  (triangle and hexagon) and `droplet.tip`. `body.rot` is now read on the
  polygons as well as on a boxy body.
- A trait override can be a **list**: `{ shape: [0.11, 0.825, 0.965] }` means
  "round, cloud or sun — whichever this name comes out as". A number narrows a
  key to one outcome and an omitted key leaves it at all of them; a list narrows
  it to what it names and leaves the seed to choose, which is the case a single
  position could not state. The choice rides on that key's own hash, so it is
  per seed, stable, uniform over the list, and independent of every other trait.
  An empty list is the same as omitting the key.
- `thinking` — a fourteenth expression, and the first whose message is a
  *duration* rather than a shape. It holds two eyes at different heights and,
  with `blobatar/motion.css` loaded, seesaws them on a 900ms cycle: the two-dot
  loader, drawn with the two dots a blobatar already has. Set it while you fetch,
  clear it when you are done.
- Two pose channels behind it, both identity on every existing pose: `edy2`, a
  vertical offset on the right eye, and `rock`, a seesaw amplitude built the way
  `shake` is — an amplitude on a loop that always runs, since an expression is
  held and cannot fire.

### Changed

- `Shape` is the union of the ten silhouette names, and `layout` returns it —
  narrow enough that a typo in a bulk filter is a type error.
- `TraitOverrides` widens from `Record<string, number>` to
  `Record<string, number | number[]>`. It accepts every map that was valid
  before; the only callers a widened value type can break are ones reading
  values back out of a map they were handed.
- Core bundle 3.7 KB → 4.4 KB gzipped, measured as `blob only` in
  `scripts/size.ts`. That is what the four silhouettes and the composition seam
  cost; the React and URI entries move by the same amount. Trait lists are +19 B
  of it, and both are inside the budget the file states.
- `motion.css` is ~95 B gz larger, and that lands on every app that imports it
  whether or not it renders a loading face. It buys a channel rather than a
  pose: a future expression that wants a duration is numbers, not stylesheet.
  See §10 of [the expression spec](./docs/expression-spec.md).
- On touch devices the eye loops of a blobatar *wearing an expression* are no
  longer paused. Idle grids are unaffected — that pause is why they are cheap —
  but a loading face that freezes on every phone is the feature not working.
- **The endpoint's unversioned URLs move too.** `blobatar.dev/avatar/<name>`
  follows the current major and now serves gen2. Pin `?gen=1` before upgrading
  on any URL that must keep its old shapes — a pinned generation is never
  retired, and it is the spelling that earns the year-long immutable cache.

### Removed

- **`blobatar/generation`**, and with it the runtime `generation` option. The
  package major is the selector now: pinning a generation is choosing a major
  and letting the lockfile hold it, rather than passing a value at every call
  site. This keeps historical implementations out of the bundle entirely — a
  gen2 consumer no longer carries gen1's layout to pay for a choice it never
  makes — and it is why the endpoint, which does serve both, depends on the
  frozen majors under an alias instead.
- The `droplet.w` and `droplet.n` trait keys. The droplet's taper is drawn as
  the two tangents from its apex to the body, so how far the apex reaches is
  also how wide its base is and how sharp its point comes out: three knobs that
  could disagree became one that cannot. Only reachable through `traits`
  overrides, and only on a droplet.

### Compatibility

Read the headline first: this is a generation change and seeds move. What
follows is about the rest of the release, none of which moves one further.

- Trait lists, `thinking` and the two pose channels are additive. The channels
  are at their identity on every existing pose, and the golden fixture gained
  rows and changed none.
- `thinking` costs +55 B gz in a bundle that already imports any expression, and
  the same as `happy` on its own.

## 1.0.0

- Stabilised the API at 1.0 and added `blobatar/generation`, making gen2
  available as an opt-in value while gen1 stayed the default for the whole
  major. Removed in 2.0.0, where the major became the selector instead.
- Published through npm's trusted publisher: releases are built and signed by
  the tag-driven `release.yml` workflow with provenance, and the repo holds no
  npm token.
- `blobatar.dev/avatar/<name>` went live — the same renderer as an HTTP
  endpoint, for the `<img src>` case that never wanted a dependency.

## 0.2.0

- Nine more expressions, for thirteen: `idle`, `happy`, `sad`, `mad`,
  `surprised`, `wink`, `sleepy`, `smug`, `unsure`, `scared`, `love`, `shy` and
  `sick`. Each is a value imported from `blobatar/expression`, so a consumer
  who uses none carries none.

## 0.1.0

- First release: deterministic blobatars from any string, the six-silhouette
  gen1 vocabulary, `blobatar/react`, `blobatar/uri`, animation through
  `blobatar/motion.css`, and full trait overrides.
