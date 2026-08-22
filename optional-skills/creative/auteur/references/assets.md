> **Hermes adaptation note:** upstream auteur generated assets through local agent CLIs (`agy`, `codex`, `grok`). In Hermes, read every such invocation as a call to the built-in `image_generate` tool with the same prompt (then move the returned file into the project's `assets/gen/` path), use the `terminal` tool for `ffmpeg`/`node`/`npx`, and `browser_exec` or Playwright-via-terminal for screenshot loops. The per-CLI routing/strength tables below are upstream reference material — the taste guidance transfers, the CLI names do not.

# assets.md — producing visual assets

The storyboard's `asset:` lines are a shot list. This file turns them into files on disk: generated keyframes, consistent A→B pairs, optimized video/sequences. Two disciplines rule everything: **one frame first** (approve art direction at the cost of one image, then batch) and **cache everything** (generation costs money and minutes; never regenerate what exists).

## 0. The crew — probe once, route by strength

**Probe with a real round trip, never with `--version`.** These are subscription CLIs and the failure
that actually happens is expired auth, not a missing binary — measured: all three passed `--version`
and all three were dead. Worse, they fail dishonestly: **`grok` exits 0 while printing "Not signed
in"** and **`agy` exits 2 while printing nothing at all**. Read the OUTPUT, not the exit code.

```bash
agy -p "reply with the single word: ok"                  # Gemini: fast image gen + edit, writes straight to a path
codex exec --skip-git-repo-check "reply with: ok"        # gpt-image: highest fidelity
grok -p "reply with the single word: ok"                 # grok-4.5: gen + edit + real video, one consistent engine
ffmpeg -version                                          # not a subscription; --version is fine here
```

A tool that does not answer `ok` is unavailable, whatever its version says. Note it as unavailable in
the asset plan and route around it — §0.5 and §4 — rather than discovering it mid-shoot.

Grok reaches **grok-4.5** only through the non-EU proxy (`ALL_PROXY="$GROK_PROXY" HTTPS_PROXY="$GROK_PROXY"`); without it grok still gens/edits/videos on grok-build. MiniMax music (ambient score) needs `MINIMAX_API_KEY` — skip the audio leg if unset.

**Route each asset to its strength** (locked by a shootout, 2026-07):

| Asset | Tool | Why |
|---|---|---|
| Hero / brand-critical stills — peak scene, abstract hero background (needs clean negative space for text), product mockup / UI screen, premium transparent element or icon | **codex** | quality king across every type tested: cleanest UI render, most negative space, best material realism. Weaknesses: palette drifts warm (weak on teal-shadow / cool briefs), and it cannot make video. It *can* edit an existing frame — via stdin only, see §2 |
| Any scene that becomes VIDEO or needs a consistent A→B edit pair; exact brand-COLOR adherence | **grok-4.5** | one engine does gen + edit + video → zero scene drift across the A→B→clip pipeline; best palette adherence when codex drifts warm |
| Volume & CONTEXT — lifestyle/environmental shots (room, hands, props, in-situ), bulk backgrounds, fast iteration | **agy** | fast, natural environmental context, writes direct to file |
| Real video | **grok** `image_to_video` (6 or 10s) | animate an approved keyframe |
| Ambient score | **MiniMax** music | one loopable bed matched to the commit-sheet mood |

All three do transparent PNG (alpha): codex crispest, grok close, agy usable-but-softer. **Match the asset's background to the page** — generate the subject on the SAME ground the page uses (white-on-white, or true alpha) so it melts into the layout with no visible frame; a photographic rectangle floating on a flat page is an instant slop tell.

Missing a tool → don't fake it: descend the ladder (§4), ask the user for assets, or pivot to type-led/CSS scenes (a great film can be shot entirely in typography). Note what's available in the asset plan.

## 0.5 Source before you generate — the routing decision

Generation is not the only tool and for a whole class of assets it is the wrong one. You cannot
generate a glTF mesh, a 16-bit HDRI that actually lights a WebGL scene, or a seamlessly tiling PBR
material with matching normal/rough/AO maps — and CC0 versions of all three exist at production
quality. `scripts/source.mjs` fetches them and writes a licence ledger for every file.

Search first with `--list` (prints a shortlist to stdout, downloads nothing, writes no ledger), then
fetch. **Always pass `--out`** — the default is `assets/sourced` relative to the current directory,
which drops a ledger and a 2.6MB font-metadata cache wherever you happened to be standing.

```bash
O=assets/sourced
node scripts/source.mjs model   "microscope" --list          # read the shortlist, then commit to one
node scripts/source.mjs hdri    "coastal dusk 03" --res 1k --out $O   # numeric suffixes work
node scripts/source.mjs model   "vintage microscope" --res 1k --out $O
node scripts/source.mjs texture "concrete rough"     --res 1k --out $O
node scripts/source.mjs icon    "bottle"             --out $O  # single noun — the index is one keyword
node scripts/source.mjs font    "serif variable"     --out $O  # downloads the woff2 and prints the @font-face
node scripts/source.mjs image   "whisky barrel"      --out $O  # CC — attribution REQUIRED
node scripts/source.mjs video   "snow forest"        --out $O  # stock — ambient only, never the peak
```

**Inspect a mesh before you write the scene it appears in.** `node -e "console.log(JSON.parse(require('fs').readFileSync('x.gltf')).nodes.map(n=>n.name))"` costs nothing and changes films: a mesh whose parts are *named* can come apart, label itself, and be re-assembled on scroll, which is a scene no image model can express at any budget. A mesh that is one welded blob can only spin. Poly Haven's listing also carries `condition` (clean / worn / weathered / rusted) and `material` — a `worn` asset reads as an antique, not as a product someone can buy this week.

**Sourced masters are heavy — budget for the conversion, not the download.** A 1k mesh + HDRI + PBR set is ~7MB of masters, which is most of a page budget. Three moves take that to well under 1MB:

```bash
# HDRI → 512×256 is indistinguishable once PMREM blurs it by roughness anyway
ffmpeg -i env_1k.hdr -vf scale=512:256 -c:v hdr -update 1 -frames:v 1 env_512.hdr
# mesh textures: 1024 for albedo/ARM, 512 for normals, q4
ffmpeg -i tex_diff_1k.jpg -vf scale=1024:1024 -q:v 4 tex_diff.jpg
# drop maps you do not sample: `arm` already carries AO+roughness+metal, so `rough` and `disp` are dead weight
```

**Getting three.js into a no-build page.** Sourcing a mesh means you now need a renderer, and there are three tempting wrong answers: ES modules with an import map (CORS-blocked over `file://`), a CDN (a third-party origin most briefs forbid), and the old UMD build (wrong colour management). Bundle once, commit the output:

```bash
npm i three@latest && npx esbuild entry.js --bundle --format=iife --global-name=THREEX --minify > assets/vendor/three-bundle.js
```
Name the ~20 symbols you actually use in `entry.js` rather than `export * from 'three'` — that alone was 731KB → 560KB. Note `RGBELoader` is a deprecation shim in recent releases; the class is `HDRLoader`.

| The asset is | Route | Why |
|---|---|---|
| a 3D mesh (glTF/GLB) | **source** — Poly Haven, CC0 | no image model produces geometry |
| an HDRI to light a WebGL scene | **source** — Poly Haven, CC0 | a generated "sky picture" is not an IBL; the lighting will look wrong and you won't know why |
| a tiling PBR material (diff/nor/rough/arm) | **source** — Poly Haven, CC0 | seamlessness and matched map sets are not generation outputs |
| an icon set | **source** — Iconify | generated icons drift in weight and stroke across a set (§7) |
| a typeface | **source** — Google Fonts | and check it against ban #8/#18 before falling in love |
| **the peak scene keyframe** | **generate** | it has to be this brand's world and nobody else's — this is the whole point |
| the hero video | **generate** (§3) | the wow moment cannot be a clip three thousand pages already use |
| an environmental / lifestyle still | **generate** (agy) | unless the brief needs a *real, identifiable* place |
| a documentary photo of a real thing or place | **source** — Openverse | generation invents; if it must be true, it must be photographed |
| an ambient background loop or video texture | **source ok** — Coverr | supporting layer only |

**The stock-video rule is not optional.** Stock footage is generic by construction. Auteur exists to
ship committed, specific assets, so sourced video is an ambient loop, a texture, or a
reduced-motion fallback — never the peak. If your wow moment is stock, you do not have a wow moment.

**The ledger ships with the site.** `assets/sourced/ASSETS-SOURCED.md` records the licence of every
downloaded file. CC0 (Poly Haven) and OFL (Google Fonts) need nothing. **Openverse images are
CC-BY / CC-BY-SA: the credit line in the ledger must appear on the page** — a footer credits block is
fine, no credit is a licence violation. Coverr and Mixkit permit use but prohibit redistribution,
which means the clip goes in your page, not in your public asset repo or template. Before shipping,
read the ledger and clear every "attribution required" line.

Do not confuse sourced assets with the moodboard. `design/moodboard/` is other people's work, used
only to decide direction and then thrown away (`recon.md`). `assets/sourced/` is licensed material
that genuinely ships.

**Sourced assets are gitignored by default, which is exactly how the scene 404s in production.** The
fetch script writes into an ignored directory; the deploy builds from the repository; the page arrives
on the host without its textures, HDRI or meshes — and a missing HDRI does not degrade gracefully, it
throws. Decide it once, in writing, before the first deploy: either commit the optimized assets (after
§5 they are small enough to) or run the fetch as a build step. "It works locally" is this bug's
signature, and it always surfaces in front of the client.

## 1. Generating keyframes

Build the prompt FROM the scene-sheet — `subject` + `camera` + `lighting` are literal prompt parameters, plus palette anchors from the commit-sheet:

> "⟨subject⟩, ⟨camera: low-angle close shot / orbital view / macro detail⟩, ⟨lighting: hard rim light at dusk / soft studio / neon-soaked⟩, color palette anchored on ⟨primary OKLCH → describe as human color⟩, photographic, no text, no watermark, 16:9"

**agy (fast, direct to file):**
```bash
agy -p "Generate an image: <prompt>. Save to <ABSOLUTE-PATH>/assets/gen/s3-peak-a.png"
```
Always give an absolute path; verify the file actually landed on disk (agy occasionally reports success without writing — re-run once if missing).

**codex (higher quality, for the peak scene / brand-critical frames):**
```bash
codex exec --skip-git-repo-check "Generate an image: <prompt>"
```
codex cannot write into your project (read-only sandbox). Pick up the newest PNG from its output store and copy it yourself — PowerShell:
```powershell
Get-ChildItem "$env:USERPROFILE\.codex\generated_images" -Recurse -Filter *.png |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
  Copy-Item -Destination "assets/gen/s3-peak-a.png"
```

Default split: agy for volume and iteration speed; codex for the peak scene and anything the viewer will stare at.

## 2. The consistency trick: frame B is an EDIT of frame A, never a second generation

Two independent generations of "the same scene" are never the same scene — lighting, geometry and lens drift. Editing frame A into frame B keeps the world intact and is what makes the two-keyframe cinema moves (displacement morph, before/after scrub) look like camera work instead of a jump cut.

**agy edit — word the change HARSHLY.** agy ignores soft phrasing ("replace X with Y" often returns the original). Use the REQUIRED CHANGE pattern:
```bash
agy -p "Load the image <ABS>/assets/gen/s3-peak-a.png and edit it. REQUIRED CHANGE: the laptop is now open, screen glowing, and the room lights have dimmed. KEEP IDENTICAL: camera angle, framing, composition, every other object, lighting direction, color grade. Save to <ABS>/assets/gen/s3-peak-b.png"
```

**codex edit — prompt via stdin only** (a positional prompt together with `-i` fails with "No prompt provided"):
```bash
printf '%s' "REQUIRED CHANGE: ... KEEP IDENTICAL: camera, composition, lighting." | codex exec --skip-git-repo-check -i assets/gen/s3-peak-a.png -
```
…then pick up from `generated_images` as above.

**Verify the pair eyes-on before building on it:** open A and B side by side. Same camera? Same composition? Only the intended state changed? Small texture drift is fine — the displacement transition tolerates it (it *hides* mid-morph mush). A camera/framing shift is a FAIL: re-edit with harder KEEP IDENTICAL wording, then try codex, then descend the ladder.

**Retry policy:** any generation/edit gets ONE sharpened retry on the same tool, then ONE attempt on the other tool, then descend the ladder. Do not burn ten generations chasing a frame — reshape the scene instead.

### N-frame chains (for the scroll-cinema state-machine engine)

Extend the A→B pair to a chain: **A→B→C→D…, each an EDIT of the previous frame** (never a fresh gen), so
the whole world stays photographically consistent while it ages / opens / transforms / gets crowded. 4–6
frames covers most stories. Verify each link same-camera before editing the next; keep the chain in scroll
order (`s1-a … s1-d`). This chain IS the input to scroll-cinema's Tier-1 scrubber — do the whole chain in
ONE grok session so the image model never drifts.

## 2.5 Depth maps (for the 2.5D composite / rack-focus)

A hero still becomes dimensional with a grayscale depth map (0 = far … 1 = near). Generate it locally —
on this machine (no discrete GPU) **Depth-Anything V2 Small runs on CPU** in seconds per hero image:

```bash
py -3.13 -m pip install -q transformers torch pillow      # one-time (~torch is heavy but CPU-only is fine)
py -3.13 - <<'PY'
from transformers import pipeline; from PIL import Image
dep = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')
dep(Image.open('assets/gen/s1-hero.png'))['depth'].save('assets/gen/s1-hero-depth.png')
PY
```

Alternatives: a **Blender Z-pass** when the scene is a 3D render (Blender CLI; §9); or ask the generator for a
grayscale "depth-style" version (fast, imperfect — ok for subtle pointer-parallax, NOT for rack-focus).
Depth is a cached master like any still. Feed color + depth to scroll-cinema §3 (2.5D composite).

## 3. Video — now local via grok

Grok animates an approved keyframe: `grok image_to_video` (6 or 10 seconds). Run it through the proxy for grok-4.5, save into `assets/gen/`, then optimize (§5). **Spend video like the motion budget spends attention** — one hero clip + at most a couple supporting; a video that isn't the wow peak is usually a still that should have stayed a still.

```bash
ALL_PROXY="$GROK_PROXY" HTTPS_PROXY="$GROK_PROXY" grok -m grok-4.5 --yolo -p \
  "image_to_video on <ABS>/assets/gen/s1-hero-a.png: slow push-in, rising steam, 6s. Save the mp4 to <ABS>/assets/gen/s1-hero.mp4"
```

**Directed A→B state change (before/after, "first+last frame").** ⚠️ grok has NO true first+last-frame interpolator (checked 2026-07): `image_to_video` animates ONE source frame with no end frame; `reference_to_video` takes 2–7 images but treats them as style/content *references*, not strict start/end keyframes — an A+B reference clip is organic drift, not a controlled morph. Routes, best first:
- **Controlled, on the web (preferred):** the WebGL displacement morph between frame A and frame B (scroll-cinema.md) — exact, scroll-scrubbable, no video model, and it's the skill's signature move anyway. This is the real answer to "we have two frames and want the transition".
- **Organic video:** `reference_to_video` with A+B as references for a loose transition, or `image_to_video` on A for pure motion (push-in, steam, drift) — endpoints not guaranteed.
- **A TRUE controlled first→last VIDEO** (hard requirement) still means browser Kling (first+last mode) / Runway / Veo: package frame A + frame B + the motion prompt for the user, continue other scenes, drop the clip in when it arrives.

**Going past 10s — chain segments.** Clips cap at 6–10s: `image_to_video` frame A, generate/edit the next state, animate that, `ffmpeg` concat. Each segment starts on the previous last frame so the seams hide.

**Other honest sources:** user-provided footage (ask at intake — real footage still beats gen for truly photographic hero shots) and **Remotion** (local render) for graphic/typographic motion (kinetic type, animated diagrams, UI mockup motion) — it's code: consistent, revisable, free.

If video still isn't right — the ladder (§4) covers you; scroll-scrubbed *sequences* read as "video" anyway.

## 4. The degradation ladder (per scene, stop at the first rung you can execute)

| Rung | What | Needs | Feels like |
|---|---|---|---|
| 1 | Scroll-scrubbed video | a real clip (§3) | full cinema |
| 2 | Canvas image sequence | a clip to explode into frames, or 6–12 generated in-between edits | Apple-grade product cinema |
| 3 | WebGL displacement morph A→B | just TWO keyframes (§2) | a living transition; the skill's signature move |
| 4 | Layered depth parallax | one keyframe cut into 2–4 layers (subject/bg), or CSS layers | dimensional, quietly premium |
| 5 | Kinetic typography / computed / pure CSS scene | nothing | still cinema, if the type system is strong |

**When NO generator answers the probe**, rungs 1–4 are all unreachable at once — every one of them
needs at least one generated keyframe. Do not treat that as "descend one rung": go back to §0.5 and
re-read the source-vs-generate table as a *fallback* table rather than a spending decision, then land
on rung 5. And drop the idea that rung 5 is a consolation prize: for a brand whose claim is precision,
a scene *computed from the same data the product is about* is more honest than any photograph, because
nothing in it could have been someone else's object. Measured on a real run — three planned
generations became three computed scenes and the page got better.

Rung 3 is the default answer to "we generated two images and want the video feel" — recipe (full GLSL) in scroll-cinema.md.

## 5. Optimization recipes (run for every heavy asset)

```bash
# Hero video → H.264 baseline (plays everywhere incl. iOS), streaming-ready, target ≤2MB
ffmpeg -i src.mp4 -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt yuv420p -movflags +faststart -crf 23 -an hero.mp4
# SCROLL-SCRUBBED video is different — a tiny GOP makes frame-accurate seeking cheap (see scroll-flight.md)
ffmpeg -i src.mp4 -an -vf "unsharp=5:5:0.8:5:5:0.0" -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -g 8 -keyint_min 8 -sc_threshold 0 -movflags +faststart scrub.mp4
# WebM alternative for Chromium (smaller at same quality)
ffmpeg -i src.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -an hero.webm
# Poster (first frame) for instant paint + reduced-motion fallback
ffmpeg -i hero.mp4 -frames:v 1 poster.png && ffmpeg -i poster.png -quality 82 poster.webp
# Explode a clip into a canvas sequence (target 60–240 frames total; ≤150KB/frame at 1440w)
ffmpeg -i hero.mp4 -vf "fps=30,scale=1440:-1" frames/f_%04d.webp
# Any still → WebP for the page (keep PNG originals in assets/gen as masters)
ffmpeg -i in.png -quality 82 out.webp
```

Budgets (verify.md re-checks): hero video ≤2MB · poster ≤300KB · sequence frame ≤150KB @1440w · any static hero image ≤400KB · mobile variants at 720w for every asset >500KB.

**When a 4K texture still looks soft, resolution is not the problem.** Check two things, in this order. First, anisotropic filtering — off by default in three.js, and without it any surface viewed at a grazing angle (ground under a low camera, a floor receding to the horizon) smears no matter how many pixels the map holds: `tex.anisotropy = renderer.capabilities.getMaxAnisotropy()`. Second, the tiling scale, counted as **metres per repeat rather than repeats per plane** — a 260m ground plane with 28 repeats is a 9-metre tile, and at 9 metres the detail is gone at any texture size; ~3m per repeat is a working default for ground. Both mistakes look identical to "the texture is too low-res", which is why the reflex fix (download the 8K version) makes the page heavier and no sharper.

## 6. Cache & bookkeeping

- Names: `assets/gen/s<scene>-<slug>-<a|b>.png`. Before ANY generation, check the path — exists means reuse (iterating on layout must not re-bill image generation).
- Keep `assets/gen/ASSETS.log.md`: one line per asset — file, tool, full prompt, date. Makes retries reproducible and hands the user the recipe to regenerate at higher quality later.
- Masters stay PNG in `assets/gen/`; the page consumes optimized WebP/AVIF/mp4 from `assets/`.
- Rights note for the user (once, in the log header): generated media follows each generator's terms (Gemini / OpenAI / xAI / MiniMax); fine for product marketing, but flag it if the client needs exclusive IP or has legal review.

## 7. Generated elements & mockups (not just full scenes)

The crew also produces the small stuff — but every generated element must survive slopscan; a generated gradient/texture that's just decoration is banned like any other. Spend it, then make it earn its place.

- **Textures / grain / noise / abstract shapes** → agy (fast, transparent where possible). Use as CSS `background`, `mask-image`, or a low-opacity overlay. Generate once, cache, reuse.
- **UI mockups in a scene** (device frame + screen, product-in-hand) → codex or grok for the still; Remotion when the mockup must move.
- **Hero mockup gate** (phase 0) can now be a *generated* frame, not only a hand-built HTML screen — one throwaway, screenshotted, approved before the real build.
- **Iconography / brand marks** → generate a set, then hand-pick: generated icon sets drift in weight/style, so treat them as sketches to redraw in SVG, not final assets.

## 8. Ambient score (MiniMax music)

If `MINIMAX_API_KEY` is set, generate ONE short, loopable ambient bed matched to the commit-sheet mood (tempo, key, tension). Playback recipe lives in scroll-cinema.md; generation rules:

- Off by default; start on a user gesture — never autoplay with sound. Provide an honest, visible mute/unmute.
- Loop seamlessly: generate a phrase that resolves to its own start, then trim on a zero-crossing with ffmpeg.
- Budget it: ≤ ~1MB, mono is fine for ambience, lazy-load after LCP.
- It's set dressing, not content — the page must be complete and comprehensible with sound off.
- No key / not wanted → skip silently. Sound is the least load-bearing layer; never gate meaning on it.

## 9. 3D scenes & camera paths (Blender CLI)

Blender 5.1 is already on `PATH`; `cli-anything-blender` is also available for inspection. Keep the asset build reproducible with one native headless command:

```powershell
blender --background --python tools\build_dolly.py
```

This complete `tools/build_dolly.py` builds a lit scene, moves `PathCamera` along a Bezier curve while tracking an Empty, bakes the camera transform, exports glTF 2.0 with camera + animation, then renders a 16-bit near-white/far-black depth master:

```python
from pathlib import Path
import math
import bpy

ROOT = Path(bpy.path.abspath("//")).resolve()
OUT = ROOT / "assets" / "gen"
OUT.mkdir(parents=True, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
scene.frame_start, scene.frame_end = 1, 180
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.world.color = (0.008, 0.01, 0.012)

def material(name, color, metallic=0.0, roughness=0.45):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.metallic, mat.roughness = metallic, roughness
    return mat

bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
bpy.context.object.data.materials.append(material("Floor", (0.025, 0.03, 0.035), 0.0, 0.28))

bronze = material("Bronze", (0.32, 0.12, 0.035), 0.72, 0.2)
for i, xyz in enumerate(((-4, -1, 1), (-2, 2, 1.6), (0, -2, 1.2), (2, 1, 2.1), (4, -1, 1.4))):
    bpy.ops.mesh.primitive_cube_add(location=xyz, scale=(0.8, 0.8, xyz[2]))
    box = bpy.context.object
    box.name = f"Monolith_{i:02d}"
    box.data.materials.append(bronze)

for name, location, energy, size in (
    ("Key", (-4, -3, 8), 1500, 5),
    ("Rim", (5, 2, 5), 900, 3),
):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.shape, data.size = energy, "DISK", size
    light = bpy.data.objects.new(name, data)
    light.location = location
    scene.collection.objects.link(light)

curve_data = bpy.data.curves.new("DollyPath", "CURVE")
curve_data.dimensions, curve_data.resolution_u = "3D", 32
spline = curve_data.splines.new("BEZIER")
points = ((-7, -7, 2.2), (-4, 2, 3.0), (1, -4, 2.5), (7, 5, 3.8), (2, 8, 4.4))
spline.bezier_points.add(len(points) - 1)
for point, co in zip(spline.bezier_points, points):
    point.co = co
    point.handle_left_type = point.handle_right_type = "AUTO"
path = bpy.data.objects.new("DollyPath", curve_data)
scene.collection.objects.link(path)

target = bpy.data.objects.new("LookTarget", None)
target.empty_display_type = "SPHERE"
target.location = (0, 0, 1.5)
scene.collection.objects.link(target)

camera_data = bpy.data.cameras.new("PathCamera")
camera_data.lens, camera_data.clip_start, camera_data.clip_end = 42, 0.1, 40
camera = bpy.data.objects.new("PathCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

follow = camera.constraints.new("FOLLOW_PATH")
follow.target, follow.use_fixed_location = path, True
follow.forward_axis, follow.up_axis = "FORWARD_NEGATIVE_Z", "UP_Y"
follow.offset_factor = 0.0
follow.keyframe_insert("offset_factor", frame=scene.frame_start)
follow.offset_factor = 1.0
follow.keyframe_insert("offset_factor", frame=scene.frame_end)

track = camera.constraints.new("TRACK_TO")
track.target, track.track_axis, track.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"

bpy.ops.object.select_all(action="DESELECT")
camera.select_set(True)
bpy.context.view_layer.objects.active = camera
bpy.ops.nla.bake(
    frame_start=scene.frame_start,
    frame_end=scene.frame_end,
    step=1,
    only_selected=True,
    visual_keying=True,
    clear_constraints=True,
    bake_types={"OBJECT"},
)
camera.animation_data.action.name = "CameraPath"

bpy.ops.export_scene.gltf(
    filepath=str(OUT / "dolly.glb"),
    export_format="GLB",
    export_cameras=True,
    export_animations=True,
)

scene.view_layers[0].use_pass_z = True
scene.use_nodes = True
nodes, links = scene.node_tree.nodes, scene.node_tree.links
nodes.clear()
layers = nodes.new("CompositorNodeRLayers")
depth_range = nodes.new("CompositorNodeMapRange")
depth_range.inputs["From Min"].default_value = camera_data.clip_start
depth_range.inputs["From Max"].default_value = camera_data.clip_end
depth_range.inputs["To Min"].default_value = 1.0
depth_range.inputs["To Max"].default_value = 0.0
depth_range.use_clamp = True
depth = nodes.new("CompositorNodeOutputFile")
depth.base_path = str(OUT)
depth.format.file_format, depth.format.color_mode, depth.format.color_depth = "PNG", "BW", "16"
depth.file_slots[0].path = "dolly-depth-"
composite = nodes.new("CompositorNodeComposite")
links.new(layers.outputs["Depth"], depth_range.inputs["Value"])
links.new(depth_range.outputs["Value"], depth.inputs[0])
links.new(layers.outputs["Image"], composite.inputs["Image"])

scene.frame_set((scene.frame_start + scene.frame_end) // 2)
scene.render.filepath = str(OUT / "dolly-poster.png")
bpy.ops.render.render(write_still=True)
```

`dolly.glb` is the cached master; Three's `GLTFLoader` handles Blender→Three axis conversion. Keep the GLB ≤3MB, keep `dolly-poster.png` as the no-WebGL/reduced-motion fallback, and feed `dolly-depth-0090.png` to the 2.5D recipe when the live scene is too expensive.

**Rules (or it's slop):** bake constraints before export; export one camera action, not per-shot GLBs; verify first/middle/last frames eyes-on; render depth from the same camera and frame; never rebuild a cached master during layout iteration.
