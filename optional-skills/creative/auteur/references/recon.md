# recon.md — scouting live references and building a moodboard

Taste is not invented at the desk. Every director watches films before shooting one, and the
reflex table in `taste.md` only tells you what to *avoid* — recon tells you what is currently
*alive*. Two commands do the gathering; you do the reading. Both write into `design/`, both are
cheap, both are bounded: recon is a short pass at the top of phase 0, not a research project.

| | command | answers |
|---|---|---|
| **A** | `node scripts/refscout.mjs` | *how do the best sites in this space actually work?* — live sites, their real stack, their scroll mechanics, screenshots |
| **B** | `node scripts/moodboard.mjs` | *what should this feel like?* — palette, light, composition, type energy, texture |

Run A when the brief has a **mechanic** question (direct register, "wow", scroll choreography).
Run B when the brief has a **look** question (any register with an undecided art direction).
Most briefs want both, in that order, ~5 minutes total. Skip either without guilt when the brief
already fixes that axis (existing brand book → skip B; "boring docs site" → skip A).

Both need `playwright` (`npm i -D playwright && npx playwright install chromium`). No API keys,
no logins, no paid services.

---

## A. refscout — take the references apart

```bash
# harvest what is winning right now, profile it
node scripts/refscout.mjs --from awwwards --limit 8

# same, filtered by the gallery's own category or a text search
node scripts/refscout.mjs --from awwwards:scrolling --limit 6
node scripts/refscout.mjs --from awwwards --search "coffee" --limit 5

# or profile sites you already know / found with WebSearch
node scripts/refscout.mjs https://a.com https://b.com --shots 4
```

Output lands in `design/refs/`: `REFERENCES.md` (read it), `refs.json`, `shots/*.png`.

**Finding candidates.** `--from awwwards` is the only harvester wired in, because it is the only
gallery whose listing *and* detail pages render reliably headless and expose the outbound site
URL. For everything else — godly.website, curated.design, minimal.gallery, land-book, siteinspire,
thefwa, lapa.ninja, mobbin — use WebSearch to find the write-ups, then pass the site URLs to
refscout positionally. Searching for the *page about* a site is more reliable than scraping the
gallery that lists it.

**Naming a technique you can see but can't name.** originkit.dev is a catalogue of ~160 motion
effects, each with a live preview and a name — useful when a reference does something you want to
describe in the commit-sheet and have no word for. Browse it, take the vocabulary, then build the
thing yourself; the components are React/framer-motion behind a signup, and half the catalogue is
the exact drop-in ornament `slopscan` exists to keep out.

### Reading a fingerprint

Each entry reports what the page actually loaded and did, not what its marketing says:

- **stack** — libraries found in the JS the page really fetched. Bundlers hide globals, so this is
  matched against bundle text; treat it as strong evidence, not proof. `GSAP + ScrollTrigger +
  ScrollSmoother + Lenis` is the house style of the entire awwwards top tier — seeing it for the
  fifth time is the useful signal, not a coincidence.
- **mechanics** — the derived read: pinned scenes (counted from `.pin-spacer` elements ScrollTrigger
  leaves in the DOM), WebGL driven by scroll, sticky stacks, CSS `animation-timeline`, mix-blend
  layers, custom cursor, scrubbed video. This is the column you are actually shopping in.
- **page shape** — `22× viewport tall · 14 sections` tells you the scroll budget the reference spends.
  A 22× page with 4 pinned scenes is a different film from a 3× page with one WebGL hero, and the
  difference is a decision you have to make too.
- **type / palette** — the fonts and colours as painted, sampled from computed styles. `LayGrotesk +
  PPNeueMontrealMono`, `Thunder + PP Fraktion Mono` — this is where you learn that the top tier is not
  running Inter, which is exactly ban #8 with receipts. The display face is measured as the largest
  *painted* text (`Thunder, sans-serif 118px/800`), never the first `h1`, because a hidden or
  fallback-styled heading reports whatever the cascade left there and will cheerfully claim an
  awwwards winner ships Inter. The px figure is evidence too: this tier runs display type past 100px.
- **shots** — hero plus scroll stops at 1440. **Look at every hero. Look at the remaining stops only
  for the 2–3 sites you actually take a steal from.** At `--limit 8 --shots 3` that is 23 images and
  reading all of them is not triage, it is a context bonfire. The numbers tell you the machinery;
  your eyes decide whether it is beautiful — but only for the references you are actually using.

### ⚠ "NO CAPTURE" entries

Some sites hand a headless browser the server-rendered HTML and then never deliver their CSS or JS
(edge-streamed pages, bot walls). Anything read off such a page — fonts, colours, stack — would be
Chrome's defaults dressed up as findings, so refscout reports **nothing** for them rather than
something false. Roughly 1 in 8 award sites lands here. When it does: open the URL yourself, or
describe it from the gallery write-up, or drop it. Never quote a fingerprint the tool refused to give.

### The steal rule

Every entry has a `**steal:**` line and it is not decoration — fill it in, one line, before you
close the file:

> "madewithgsap.com — the section title stays pinned while its cards scroll *through* it; we do this
> once, on the proof scene, with the product name instead of a title."

Rules, in order of how often they are broken:

1. **One idea per reference, named as a mechanic** — "pinned title, content scrolls through", not
   "cool scroll effect", and never "the vibe".
2. **Adapted, or it is theft.** A reference is a starting camera position, not a set. Changing the
   colours of a copied layout is not adaptation.
3. **Two references maximum feeding one scene.** Three is a collage, and collages read as slop.
4. **Never cite a reference you did not look at.** A fingerprint without eyes on the shots is half a
   reference.
5. Named brand assets — logos, custom typefaces, photography, illustration — are theirs. Mechanics
   and structure are free; identity is not.

### Feeding the commit-sheet

Recon changes what you write in the commit-sheet, especially field 6:

- The stacks and mechanics you saw five times ARE the first-order reflex (6a) for this category. If
  your peak is the fifth pinned-WebGL-hero of the day, you found the mode, not an idea.
- The type and palette columns give you a real, dated map of what the category currently looks like
  — much sharper than reasoning about it from memory.
- Any mechanic you take goes into the motion budget (field 5) as one of the ≤3 families, not on top
  of it.

In the direct register, put the 2–3 references and their one-line steals in the STORYBOARD header
(`References taken`) so the decision survives into the build.

---

## B. moodboard — decide what it should feel like

```bash
node scripts/moodboard.mjs "editorial brutalist layout dark" "high contrast type poster" --limit 24
node scripts/moodboard.mjs "terracotta ceramic studio light" --source bing,arena --limit 16
```

Output lands in `design/moodboard/`: `contact-sheet*.png` (**look at these**), numbered `img/NN.jpg`,
and `MOODBOARD.md` mapping every tile number to its origin page. Twenty tiles arrive as one image,
so reading a moodboard costs one look, not twenty.

**Sources** (all no-auth, tried in this order and interleaved so none owns the sheet):

- `bing` — the workhorse. Bing's image index reaches into Pinterest, Dribbble and Behance CDNs and
  serves originals that hotlink fine through a browser context.
- `pinterest` — logged-out search renders a partial grid behind the login wall: sometimes ~15 pins,
  sometimes zero, day to day. Genuinely useful when it answers, never load-bearing; when it returns
  nothing the script says so and bing has already covered it.
- `arena` — the public are.na search API. Lower volume, highest curation: these are blocks designers
  saved for themselves.

### Query craft

The quality of a moodboard is decided entirely by the queries. Three rules:

1. **Two or three queries on different axes, never one.** A single query returns near-duplicates of
   one template. Pick axes: *subject/medium* ("ceramic studio photography"), *treatment* ("hard rim
   light, deep shadow"), *graphic language* ("swiss grid poster, red accent").
2. **Search the feeling, not the category.** `"coffee website"` returns other coffee websites — the
   category reflex, delivered to your desk. `"steel and steam, industrial macro, warm shadow"`
   returns material you can actually direct from.
3. **Check the anchor noun is not half of a fixed compound.** `"polished brass instrument macro"`
   returns trumpets and saxophones, because *brass instrument* is one word to a search index. Same
   trap: hard surface, light bulb, glass ceiling, sound board, steel drum. The noun is supposed to
   ground the query, and an idiom hijacks it instead.
4. **Anchor a treatment query with a concrete noun.** Image search collapses an abstract phrase to
   its most commercially indexed substring: `"hard rim light on dark glass, wet stone, near-black
   macro"` came back as *hard surface* — hi-vis workers, gravel, granite pavers. `"wet black stone,
   single hard light"` does not. When one query of three drifts, that is the mechanism.
5. **Never search a brand you intend to resemble.** That is the shortest path to a page that looks
   like a competitor with the logo swapped.

### Reading the sheet

Fill the four lines at the bottom of `MOODBOARD.md` — the moodboard exists to produce these, and an
unfilled index means the sheet was decoration:

- **dominant palette** across the tiles you liked → commit-sheet field 2, converted to OKLCH. Take
  the *relationship* (drenched single hue / near-mono with one accent / earthy midtones), not a
  literal eyedropper.
- **light and contrast character** → the `lighting:` line of every scene-sheet and the literal
  lighting parameter in generated-asset prompts (`assets.md` §1).
- **one composition move worth stealing** → commit-sheet field 4 (grid break).
- **what to explicitly avoid from these** → a real answer, because a sheet always contains the
  category reflex too, and naming it is how you stop drifting into it.

Expect 10–20% junk per sheet. It usually is not spread evenly: one query drifts wholesale while the
others stay clean, so read the junk as a signal about that query, not about the sheet. Ignore the
strays; re-run only if a whole axis came back wrong, and re-run that axis with a concrete noun.

### Reference images are not assets

Downloaded images are direction only. They are other people's work, they are not licensed to you,
and none of them ships:

- never place a downloaded image in the page, not even as a placeholder that "we'll swap later";
- never reproduce one composition — the moodboard sets palette, light and energy, and the actual
  frames get generated per `assets.md`;
- when the art direction is locked, the moodboard's job is done: it informs the generation prompts
  and then stays in `design/`.

`design/moodboard/` and `design/refs/` are working artifacts. Ship them to no one and add them to
`.gitignore` if the repo is public.

---

## When recon fails

| symptom | what it means | do this |
|---|---|---|
| `awwwards listing returned no cards` | the gallery markup moved | fall back to WebSearch + positional URLs |
| most entries are `NO CAPTURE` | network or a broad bot wall | profile fewer, better-known sites; lean on the shots and your own eyes |
| `pinterest returned 0` | login wall, normal | ignore, bing covered it |
| the sheet is 20 variants of one image | single narrow query | re-run with 2–3 queries on different axes |
| one query's tiles are a different subject entirely | search collapsed an abstract phrase to a common substring | re-run that axis anchored to a concrete noun |
| a script looks hung for minutes | you piped it through `tail`/`head` | both stream progress to stderr; drop the pipe to watch it |
| references all look alike | you found the category mode | that IS the finding — write it into reflex-check 6a and deviate |

Recon is bounded. If it has not produced a usable read in one pass, stop and decide from
`taste.md` — a director who cannot find a reference still has to shoot the film.
