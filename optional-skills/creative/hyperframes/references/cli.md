# HyperFrames CLI

Everything runs through `npx hyperframes` (or the globally-installed `hyperframes` after `npm install -g hyperframes`). Requires Node.js >= 22 and FFmpeg.

## Workflow

1. **Scaffold** — `npx hyperframes init my-video`
2. **Write** — author HTML composition (see `composition.md`)
3. **Lint** — `npx hyperframes lint`
4. **Preview** — `npx hyperframes preview`
5. **Render** — `npx hyperframes render`

Always lint before preview/render — catches missing `data-composition-id`, overlapping tracks, and unregistered timelines.

## init — Scaffold a Project

```bash
npx hyperframes init my-video                        # interactive wizard
npx hyperframes init my-video --example warm-grain   # pick an example template
npx hyperframes init my-video --video clip.mp4       # seed with a video file
npx hyperframes init my-video --audio track.mp3      # seed with an audio file
npx hyperframes init my-video --non-interactive      # skip prompts (CI / agent use)
```

Templates: `blank`, `warm-grain`, `play-mode`, `swiss-grid`, `vignelli`, `decision-tree`, `kinetic-type`, `product-promo`, `nyt-graph`.

`init` creates the correct file structure, copies media, transcribes audio with Whisper, and installs authoring skills. Use it instead of creating files by hand.

## lint

```bash
npx hyperframes lint                # current directory
npx hyperframes lint ./my-project   # specific project
npx hyperframes lint --verbose      # include info-level findings
npx hyperframes lint --json         # machine-readable output
```

Lints `index.html` and all files in `compositions/`. Reports errors (must fix), warnings (should fix), and info (only with `--verbose`).

## preview

```bash
npx hyperframes preview                # serve current directory (port 3002)
npx hyperframes preview --port 4567    # custom port
```

Hot-reloads on file changes. Opens the Studio in your browser automatically.

## render

```bash
npx hyperframes render                              # standard MP4
npx hyperframes render --output final.mp4           # named output
npx hyperframes render --quality draft              # fast iteration
npx hyperframes render --fps 60 --quality high      # final delivery
npx hyperframes render --format webm                # transparent WebM
npx hyperframes render --docker                     # byte-identical reproducible render
```

| Flag           | Options                 | Default                        | Notes                       |
| -------------- | ----------------------- | ------------------------------ | --------------------------- |
| `--output`     | path                    | `renders/<name>_<timestamp>.mp4` | Output path                 |
| `--fps`        | 24, 30, 60              | 30                             | 60fps doubles render time   |
| `--quality`    | `draft`, `standard`, `high` | standard                   | draft for iterating         |
| `--format`     | `mp4`, `webm`           | mp4                            | WebM supports transparency  |
| `--workers`    | 1–8 or `auto`           | auto                           | Each spawns Chrome          |
| `--docker`     | flag                    | off                            | Reproducible output         |
| `--gpu`        | flag                    | off                            | GPU-accelerated encoding    |
| `--strict`     | flag                    | off                            | Fail on lint errors         |
| `--strict-all` | flag                    | off                            | Fail on errors AND warnings |

**Quality guidance:** `draft` while iterating, `standard` for review, `high` for final delivery.

## transcribe

```bash
npx hyperframes transcribe audio.mp3
npx hyperframes transcribe video.mp4 --model medium.en --language en
npx hyperframes transcribe subtitles.srt     # import existing
npx hyperframes transcribe subtitles.vtt
npx hyperframes transcribe openai-response.json
```

Produces word-level timings suitable for caption components. First run downloads the Whisper model (cached after).

## tts

```bash
npx hyperframes tts "Text here" --voice af_nova --output narration.wav
npx hyperframes tts script.txt --voice bf_emma
npx hyperframes tts --list                    # show all voices
```

Uses Kokoro (local, no API key). Voice prefixes: `af_` (American female), `am_` (American male), `bf_` (British female), `bm_` (British male).

## doctor

```bash
npx hyperframes doctor
```

Verifies environment:
- Node.js >= 22
- FFmpeg present on PATH
- Available RAM (renders are memory-hungry — 4 GB minimum)
- Chrome binary resolution (`chrome-headless-shell` preferred over system Chrome)
- Current `hyperframes` version

Run this **first** when a render fails. See `troubleshooting.md` for interpreting the output.

## browser

```bash
npx hyperframes browser --install      # install the bundled chrome-headless-shell
npx hyperframes browser --path         # print the resolved browser binary path
npx hyperframes browser --clean        # clear the bundled browser cache
```

## info

```bash
npx hyperframes info
```

Prints version, Node version, FFmpeg version, OS, and resolved browser path — useful in bug reports.

## upgrade

```bash
npx hyperframes upgrade -y
```

Check for and install updates. Run this if you hit `HeadlessExperimental.beginFrame` errors — the auto-detect fix shipped in `hyperframes@0.4.2` (commit 4c72ba4, March 2026).

## Other

```bash
npx hyperframes compositions    # list compositions in the project
npx hyperframes docs            # open documentation in browser
npx hyperframes benchmark .     # benchmark render performance
npx hyperframes add <block>     # install a block/component from the catalog
npx hyperframes add --list      # browse the catalog
```

Popular catalog blocks: `flash-through-white` (shader transition), `instagram-follow` (social overlay), `data-chart` (animated chart), `lower-third` (talking-head overlay). See [hyperframes.heygen.com/catalog](https://hyperframes.heygen.com/catalog).
