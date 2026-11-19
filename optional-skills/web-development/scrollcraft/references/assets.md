# Assets

Generation through kie.ai, then encoding for scrubbing. All of it via
`scripts/kie.mjs` and `scripts/encode.sh`.

```bash
node <skill>/scripts/kie.mjs probe                     # credit check first
node <skill>/scripts/kie.mjs still "<prompt>" out/01.png --ar 16:9 [--ref brand.png]
node <skill>/scripts/kie.mjs shot  "<move>"   out/01.png out/01.mp4 --dur 5 [--tail end.png]
bash  <skill>/scripts/encode.sh out/01.mp4 assets/01.mp4
bash  <skill>/scripts/encode.sh out/01.mp4 assets/01-m.mp4 mobile
```

Models: `seedream/5-pro-text-to-image` (and `-image-to-image` when you pass
`--ref`) for stills, `kling/v2-1-pro` for camera moves. `aspect_ratio`,
`quality` and `output_format` are all required by seedream; omitting any one
returns a bare "This field is required" that does not name the field.

**seedream rejects most aspect ratios, and does not say which it takes.** An
unsupported value fails at `createTask` with `"This aspect_ratio is not within
the range of allowed options"` and no list, which reads like a malformed request
rather than a menu problem. Nothing is charged, so the cost is a wasted round
trip and the time to work out that the string itself was fine.

| `--ar` | Status | Returns |
|---|---|---|
| `16:9` | works | 2736x1520 |
| `9:16` | works | 1520x2736 |
| `3:4` | works | 1776x2352 |
| `4:5` | **rejected** | n/a |

The returned pixel dimensions are near the ratio rather than exactly it, so
derive layout from the file, not from the string you asked for. Treat the table
as the verified set rather than as the whole allowed list: **use a value from it,
and if you need another, send one throwaway call before writing a wave of
prompts around it.** `4:5` is the one that catches people, because it is the
standard portrait social ratio and every other generator takes it. `3:4` is the
portrait to reach for here.

---

## How many assets

Fewer than you think. A six-act page needs roughly:

- **2 clips.** One hero move, one texture or detail move. That is the cap from
  SKILL.md, and it is a quality rule as much as a budget one.
- **4 to 6 stills.** Posters for the clips, plus whatever the flow and rail acts
  show.

Every clip also needs a poster, and the poster must be **the clip's own first
frame**, not a separate generation. Pull it with ffmpeg rather than generating
a lookalike; a poster that does not match causes a visible jump the moment the
video paints.

```bash
ffmpeg -y -i out/01.mp4 -frames:v 1 -q:v 2 assets/01-poster.png
```

**Use the same ffmpeg `encode.sh` resolved, not bare `ffmpeg`.** `encode.sh`
goes looking for a full build precisely because the one on PATH may be stripped,
but this poster line and the PSNR seam check below both call `ffmpeg` directly.
On a stripped build the WebP muxer is missing and you get `Unable to choose an
output format for 'poster.webp'`, which reads like a bad filename rather than a
missing encoder, and `-lavfi psnr` fails with `No such filter: 'psnr'`. Resolve
it once and reuse it:

```bash
FF=$(ls -d "$HOME"/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_*/ffmpeg-*-full_build/bin/ffmpeg.exe 2>/dev/null | head -1)
"$FF" -y -i assets/01.mp4 -frames:v 1 -vf scale=1600:-2 -c:v libwebp -quality 82 assets/p01.webp
```

---

## Stills

1. Pick a world in [worlds.md](worlds.md) and write its preamble once.
2. Every prompt is: **preamble, blank line, scene**. Verbatim preamble, every
   time. This is the single thing that makes separately generated images look
   like one shoot.
3. Name where the empty space goes, because copy sits on these images.
4. **Read every PNG before using it.** Generation is cheap; shipping a bad frame
   is not.

**Pass the real brand object as `--ref`.** For anything with packaging, a logo,
or a product, hand seedream the actual asset. A label that drifts between shots
is the first thing a client notices. It holds up remarkably well: the same can
reads correctly across a dark kitchen, a hard-light flat-lay and a studio
backdrop when the cutout is passed every time.

---

**Published unit costs** (kie.ai, as of August 2026): `seedream/5-pro` still =
**28 credits**, `kling/v2-1-pro` 5s clip = **160 credits**. A seven-still,
two-clip page is therefore around 520 credits. `kie.mjs probe` reports a balance,
not a delta, so plan the spend from these numbers and probe before and after to
confirm. Do not trust a probe delta as this build's cost if anything else is
running against the same key (see [verify.md](verify.md)).

**Plan against those rates; expect the ledger to debit roughly 0.4x of them.**
Two reconciliations against the account with no other consumer put actual debits
at about 38% of the per-call sum: 1447 summed against 530 debited, and 2252
summed against 856 debited. That puts a still nearer ~11 credits and a 5s kling
clip nearer ~61, though those are back-derived from two samples rather than
published, so do not quote them as rates.

The practical consequence: **every budget cap in this skill is conservative, and
a build that comes in at its per-call cap has real headroom.** Keep planning at
28 and 160, because a ceiling that never under-claims is the right one to hold a
build to, and because the ratio is an observation about billing rather than a
published price that will hold. But do not refuse a justified reroll on a cap
computed at the documented rates, and do not report a per-call sum as measured
spend.

---

## Real footage the client already has

Supplied footage is usually **flat**: shot log-ish or picture-profiled, with no
white point. Downscaled straight into the page it produces washed-out acts that
no amount of scrim tuning rescues, and the instinct to fix it with a CSS filter
over full-bleed media is the thing taste.md warns against, because a filter
flattens the whole frame.

Grade it into a pre-encode intermediate instead:

```bash
# 1. measure. YMIN/YMAX/YAVG tell you whether the clip ever reaches white
ffmpeg -i raw.mov -vf signalstats,metadata=print:key=lavfi.signalstats.YMAX -f null -

# 2. expand levels + a small saturation lift, and land the fps you want
ffmpeg -y -i raw.mov -vf "colorlevels=rimin=0.09:gimin=0.09:bimin=0.09:\
rimax=0.64:gimax=0.64:bimax=0.64,eq=saturation=1.08,fps=30,scale=1920:-2" \
  -c:v libx264 -crf 16 -an graded.mp4

# 3. then the normal dense-GOP encode for scrubbing
bash <skill>/scripts/encode.sh graded.mp4 assets/01.mp4
```

Three more things real footage needs that generated clips do not:

- **Trim before anyone enters frame.** "Nothing enters or leaves" is a hard
  requirement for a clip the reader can park anywhere in, and real recordings
  routinely have someone walk through at second nine.
- **Target 24 to 30fps before `encode.sh`.** A 60fps phone clip at a dense GOP is
  several times the size for motion no hand can resolve. Decimate with `fps=30`
  in the intermediate; the dense keyframes then cost half as much.
- **Re-encode, never stream-copy.** Cuts made with `-c copy` decode badly under
  a scrubber.

---

## Camera moves

`shot` takes a still and moves the camera through it.

What makes a clip scrub well is not the same as what makes it watch well:

- **One continuous move, one direction.** A dolly-in, a drift down, a slow
  orbit. Any cut, snap or direction reversal becomes a jolt under the wheel,
  because the reader controls the playhead and will sit on the reversal.
- **Slower than feels right when previewed.** The move is spread over two or
  three viewport-heights of scroll. A move that looks sedate at 24fps feels
  correct under a hand.
- **The subject stays in frame throughout.** The reader may park anywhere.
- **Nothing enters or leaves.** A person walking in is a different shot at
  frame 1 and frame 120, and the poster will match neither.

Prompt shape: what continues, how the camera moves, then the negatives.

> The camera pushes slowly and steadily forward toward the can, a smooth
> continuous dolly-in with a very slight downward tilt. The blurred window
> slides past on the left as parallax. The can stays perfectly still and in
> frame throughout. One single continuous take, no cuts, no camera shake, no
> zoom snap. Slow, cinematic, controlled.

The script already sends a negative prompt covering judder, warping, morphing,
flicker and scene changes, which are the failure modes that specifically wreck a
scrub.

### Seam locking, if you actually need a chain

`--tail` pins the last frame as well as the first. Leg N's tail is leg N+1's
head, so the joint is frame-identical:

```bash
ffmpeg -y -sseof -0.05 -i leg1.mp4 -frames:v 1 -q:v 2 seam1.png
node kie.mjs shot "<move>" seam1.png leg2.mp4 --dur 5
```

Extract the seam frame from the **encoded** clip, not the source: re-encoding
shifts frames slightly, and a seam built from the wrong file is a one-frame pop.

**Chaining on pre-generated anchors makes the legs parallel.** Extracting each
leg's head from the previous leg's *encoded* file forces the whole flight to
generate serially, which is roughly 45 minutes for ten legs. Generating every
anchor still first, then giving leg N `--tail` of anchor N+1 and head of anchor
N, means all ten clips run at once and every joint is still frame-locked to an
image both sides were built from. `orrery` did this for a ten-leg flight and
measured 28.5-39.8 dB across all nine joints, inside the band the serial
`descent` chain shipped at. The cost is that you must author the anchors as a
coherent descent, because the model resolves a head-and-tail conflict by pulling
the camera back.

Most pages should not do this at all. A chain exists only to hide cuts between
scenes, and varying the device between acts removes the cut instead of hiding
it, for free and with no failure mode. Chain only when the brief is literally
"one continuous journey".

---

## Encoding

`encode.sh` sets a dense GOP (`-g 8` desktop, `-g 4` mobile), strips audio, and
adds `+faststart`.

The reason is the whole trick: **a normal web encode places a keyframe every two
to five seconds.** Seeking to an arbitrary time makes the decoder walk forward
from the previous keyframe, so a sparse-GOP file plays perfectly and scrubs like
mud. Dense keyframes cost file size and buy responsiveness.

Expect roughly 3MB for a 5s 1080p desktop clip and 1.5MB for the 720p mobile
one. Two clips is about 9MB of video on the page, and the engine fetches each
only as its act approaches.

**Grain-heavy worlds run about double that.** Documentary grain, moving foliage
and film texture gave 5 to 6MB per desktop clip at the script's `-crf 20`;
`-crf 22` is a reasonable dial if a page needs the megabytes back. Longer real
footage costs proportionally more, so a page carrying 15 seconds of supplied
video will sit above the 9MB guideline. That can be the right trade, but make it
deliberately rather than discovering it at the end.

Audio is stripped because these clips are scrubbed, never played. A muted track
is dead weight and an autoplay-policy hazard.

**A stripped ffmpeg will fail here.** Some toolchains put an ffmpeg on PATH with
about 50 filters and no `scale`, `fps` or `tile`. It reports `No option name
near ...`, which reads like a syntax error in your command rather than a missing
filter. `encode.sh` counts filters and goes looking for a real build; override
with `SCROLLCRAFT_FFMPEG`.

---

## Portrait

A 16:9 clip covering a 9:16 viewport crops to the middle third, and a
composition built around negative space on the left loses exactly that space.

Options, in order of cost:

1. **Compose for both.** Keep the subject in the centre third and the copy in a
   bottom band rather than in side negative space, because the bottom band
   survives a centre crop and side space does not. Cheapest, and usually enough.
2. **Native portrait renders.** Generate a 9:16 still and clip for the hero only,
   wire them as `data-sc-src-mobile`, and swap the poster with `<picture>` so the
   frame-holder matches the clip it is holding for:

   ```html
   <picture>
     <source media="(max-width: 860px)" srcset="assets/01-hero-p.webp">
     <img class="sc-stage__poster" src="assets/01-hero.webp" alt="">
   </picture>
   <video data-sc-scrub data-sc-src="assets/01.mp4"
          data-sc-src-mobile="assets/01-p.mp4" playsinline muted></video>
   ```

   A portrait poster with a landscape clip (or the reverse) jumps visibly the
   moment the video paints, which is the whole failure the poster exists to
   prevent. `encode.sh` has no portrait mode; do it by hand:

   ```bash
   ffmpeg -y -i src.mp4 -vf "scale=720:-2" -c:v libx264 -crf 20 -g 4 \
     -pix_fmt yuv420p -an -movflags +faststart assets/01-p.mp4
   # cropping a 16:9 source to 9:16 instead of rendering native:
   #   -vf "crop=ih*9/16:ih,scale=720:-2"
   ```
3. **Drop the clip on phones.** Serve the poster and let the copy carry the act.
   Under reduced motion the engine already does exactly this, so the layout is
   known to work.

Also step the hero display size down on phones. `--sc-t-4xl` floors at 3.4rem,
which is a desktop floor: it wraps a normal hero headline to six lines at 390px.
See [taste.md](taste.md).

Do not solve it with `object-fit: contain`. Letterboxed video on a landing page
reads as a broken embed.
