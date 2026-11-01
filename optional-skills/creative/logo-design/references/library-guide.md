# The Reference Library — Guide & Insights

The skill ships with **1,400+ real-world SVG logos** (≈1,200 brands; 233 brands include both a full lockup and a
standalone `-icon` symbol), each visually classified by mark type, technique, geometry, subject, typography, mood and
industry. Use it to learn *how* marks are built and to see what a category already looks like.

> **The logos are trademarks of their respective owners.** They are included only as study material.
> Never copy, trace or lightly modify them for a client. If your concept resembles one, change it.

## Contents
1. What's inside & where
2. How to use it well
3. Data-driven insights
4. Curated lessons by technique (with example files)
5. Catalog schema & extending the library

---

## 1. What's inside & where

```
assets/library/
  svg/                  1,400+ logo files (brand.svg = full logo, brand-icon.svg = standalone symbol)
  catalog.json          one record per file: structure + colours + complexity + visual classification
  classifications.json  the visual labels (source of truth for mark_type, techniques, subject, …)
  stats.json            distributions used by svg_audit.py (anchors, colours, types …)
  gallery.html          filterable visual browser for humans — open locally in a browser
```
The collection skews heavily to technology brands (developer tools 21 %, frameworks/libraries 17 %, cloud 7 %,
databases 6 %, testing/monitoring 6 %…). Keep that bias in mind: its conventions are *tech* conventions.
Categories such as food & drink, fashion, hospitality, health or public services are barely represented: for those,
use `--subject`/`--query` to find marks that share a subject or technique (cups, leaves, arches, letter K…), build
the shelf test from hand-picked files with `preview_sheet.py --refs …`, and rely on your own knowledge of the
category's conventions.

## 2. How to use it well

- **Study a technique before using it**: pull 5–8 exemplary files and *read the SVG* to see how the geometry is
  built (how few anchors, which primitives, how negative space is cut):
  `python3 scripts/search_library.py --technique negative-space --exemplary --format paths`
- **Map a category's conventions** before designing, then decide where to conform and where to depart:
  `python3 scripts/search_library.py --industry security-identity --summary`
- **Check your idea isn't already taken**: `python3 scripts/search_library.py --subject "rocket"`
- **Shelf test your concept** against real marks:
  `python3 scripts/preview_sheet.py concept.svg --refs-industry developer-tools -o shelf.html`
- **Calibrate complexity**: `svg_audit.py` compares your anchor count and colour count with the library.
- For inspiration by feel: `--mood friendly`, `--mood technical`, `--type-style serif`, `--case lowercase`.

Useful filters: `--type`, `--symbol-type`, `--technique`, `--geometry`, `--industry`, `--color`, `--primary-color`,
`--max-colors`, `--aspect`, `--variant icon`, `--type-style`, `--case`, `--mood`, `--subject`, `--query`,
`--exemplary`, `--no-gradient`, `--summary`, `--list-values`.

## 3. Data-driven insights

**Mark types (per file)** — abstract 23.5 %, combination 20.6 %, pictorial 20.2 %, letterform 15.2 %, wordmark 9.9 %,
emblem 3.8 %, mascot 3.8 %, lettermark 3.0 %. (Standalone icon files inflate symbol types.)
Inside combination marks, the symbol is abstract 43 %, pictorial 33 %, letterform 17 %, mascot 5 %.

**Proportions** — 55 % of files are near-square (0.8–1.25 : 1). Of brands with both files, ~9 in 10 lockups are wide
(> 2.5 : 1) and ~3 in 4 icons are near-square: symbol + horizontal lockup is the standard pair.

**Colour** — median 2 colours; ≈ 75 % use ≤ 3; ≈ 47 % of standalone icons are single-colour. Gradients in ≈ 19 % of
files. Dominant hue: blue ≈ 22 %, red ≈ 13 %, monochrome ≈ 13 %, multi-hue ≈ 28 %; yellow and pink are rare.
→ In tech, blue is camouflage; warm or unusual hues are an easy distinction win.

**Geometry** — circle is the most common base (26 %), then organic (14 %), square (14 %), freeform (13 %),
triangle (8 %), hexagon (8 %; 10 % among developer/cloud/data brands — a category cliché), rounded square (7 %).

**Techniques** — containment 32 %, negative space 20 %, monoline 17 %, colour segments 14 %, dimensional shading 10 %,
geometric construction 9 %, modular repetition 7.5 %, overlap/transparency 7 %, hidden meaning 7 %, isometric 6 %,
radial symmetry 6 %, letter substitution 3 %.
→ Containing a symbol in a circle/square is the most common move — useful for app icons, but also generic. Negative
space and hidden meaning appear in most of the strongest (exemplary) marks.

**Typography (541 files containing type)** — geometric sans 49 %, custom/display 19 %, grotesque 13 %, humanist 8 %,
script 5 %, serif 3 %, slab 1 %, rounded 1 %. Case: lowercase 36 %, uppercase 32 %, title 20 %, mixed 13 %.
→ Lowercase geometric sans is the tech default; a serif, slab, humanist or truly custom wordmark stands out.

**Complexity** — square symbols have a median of ~53 anchor points (p75 ≈ 96, p90 ≈ 194). The exemplary marks cluster
at the low end: great marks are built from few, deliberate points.

**Mood labels** most used — technical, friendly, bold, modern, playful. "Friendly" is almost always carried by rounded
geometry and lowercase; "technical" by monoline, grids, brackets, isometric forms.

## 4. Curated lessons by technique (with example files)

Every file below is in `assets/library/svg/`. Read them as SVG to study construction.

**Negative space** — figure and ground both carry meaning.
- `auth0-icon.svg` star carved from a shield · `apache-camel.svg` camel cut from a circle · `doctrine.svg` arrow cut
  from a teardrop · `esdoc.svg` owl face entirely from negative space · `houndci.svg` dog profile in a square ·
  `npm-icon.svg` letter carved from a solid square · `khan_academy-icon.svg` sprout that reads as a person.

**Hidden meaning / double reading** — one form, two ideas.
- `airbnb.svg` one loop = pin + heart + A · `amplitude-icon.svg` A = sound wave · `astro.svg` A = rocket ·
  `gnome-icon.svg` footprint = G · `spidermonkey-icon.svg` monkey = S · `kissmetrics.svg` heart + bar chart ·
  `botanalytics.svg` speech-bubble edge = chart line · `twitch.svg` speech bubble with eyes.

**Letter substitution & custom wordmarks** — make a name ownable.
- `hubspot.svg` sprocket as the "o" · `fastly.svg` stopwatch in a letter · `tor.svg` onion as "o" · `sparkpost.svg`
  flame for "O" · `mozilla.svg` URL syntax in the name · `100tb.svg` double zero as infinity · `adyen.svg` fully
  custom squared letters · `nextjs.svg` extended X stroke · `go.svg` italic + speed lines · `stripe.svg` tight custom
  lowercase.

**Letterform symbols** — one letter, one idea.
- `kotlin-icon.svg` K from one triangular cut · `patreon.svg` P = bar + circle · `pagekit.svg` P from a notched square ·
  `gatsby.svg` G from circle + diagonal · `pinterest.svg` P as a pin · `zendesk-icon.svg` Z from triangles and
  semicircles · `mesos.svg` M from triangles · `metabase.svg` M from a dot grid · `mobx.svg` `]v[` reads as M.

**Geometric construction** — primitives, consistent radii, clean angles.
- `elm.svg` tangram square · `framer.svg` squares and triangles · `google-photos.svg` four half-circles ·
  `circleci.svg` notched ring · `vercel-icon.svg` a single triangle · `twitter.svg` bird from circle arcs ·
  `buck.svg` monoline deer from strict geometry · `tsuru.svg` origami crane · `figma.svg` modular circles.

**Modular repetition & radial symmetry**
- `slack-icon.svg` rotated modules forming a hash · `dropbox.svg` five rhombi · `openai-icon.svg` one module rotated
  six times · `cardano-icon.svg` graduated dots · `centos-icon.svg` pinwheel · `ibm.svg` stripes unifying letters ·
  `tidal-icon.svg` four diamonds · `ubuntu.svg` three figures in a ring.

**Dimension with flat means** — depth without realism.
- `ethereum.svg` faceted octahedron · `sketch.svg` faceted gem · `codesandbox.svg` cube from cuts · `unity.svg` cube
  from negative space · `webpack.svg` cube in cube · `tensorflow.svg` one form reads T and F · `laravel.svg` monoline
  isometric L.

**Overlap & colour segments**
- `mastercard.svg` two overlapping circles · `dreamhost.svg` crescent from two circles · `chrome.svg` three segments +
  core · `google-icon.svg` segmented G · `lit-icon.svg` faceted flame · `playwright.svg` two masks.

**Pictorial reduction** — objects reduced to their most characteristic silhouette.
- `apple.svg` · `redhat-icon.svg` · `couchbase.svg` · `docker-icon.svg` · `swift.svg` · `snowpack.svg` ·
  `stackoverflow-icon.svg` · `trello.svg` · `youtube-icon.svg` · `whatsapp.svg` · `gitlab.svg` (faceted animal).

**Mascots done simply**
- `android-icon.svg` built from rounded primitives · `discord-icon.svg` controller = face · `github-icon.svg` strong
  silhouette in a circle · `giantswarm.svg` terminal glyphs as eyes.

**Emblems & containers**
- `markdown.svg` M + arrow in a frame · `jupyter.svg` orbits framing the name · `lua.svg` moon orbiting a planet.

**Combination systems**
- `soundcloud.svg` cloud built from sound bars · `tableau.svg` plus-sign cluster echoed in the name · `aws.svg` smile
  arrow under plain type · `arduino.svg` infinity loop holding − and + · `microsoft.svg` four squares.

Get more: `python3 scripts/search_library.py --exemplary --type <type>` (141 files are flagged exemplary).

## 5. Catalog schema & extending the library

Each record in `catalog.json`:

| Field | Meaning |
|---|---|
| `file`, `brand`, `variant` (`main`/`icon`), `pair` | identity of the file and its counterpart |
| `width`, `height`, `aspect`, `aspect_class` | canvas proportions |
| `bytes`, `shapes`, `anchors` | size and complexity |
| `colors`, `n_colors`, `color_families`, `primary_family`, `gradients`, `has_mask`, `has_filter` | colour data |
| `mark_type`, `symbol_type` | wordmark · lettermark · letterform · pictorial · abstract · mascot · emblem · combination |
| `subject` | what it depicts, in a few words |
| `geometry`, `techniques` | construction vocabulary (see `--list-values`) |
| `type_style`, `case` | typography, when present |
| `mood`, `industry`, `exemplary`, `note` | tone, sector, teaching flag and why |

To add logos: place SVGs in `assets/library/svg/`, append classification objects with the same keys to
`classifications.json`, run `python3 scripts/build_catalog.py` (and `--check` to verify coverage). Only add marks you
have the right to redistribute for reference.
