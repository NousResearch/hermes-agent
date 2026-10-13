# Minimal script-free RML smoke fixture

An original geometric scene for testing the official Rive CLI and a target
runtime. An ochre diamond moves along a neutral horizontal rail. No fonts,
images, scripts, shaders, account IDs or external assets are included.

## Contract

| Item | Value |
| --- | --- |
| Project / output | `minimal-rml` / `build/minimal-rml.riv` |
| Default artboard | `Smoke`, 320 × 200 |
| State machine | `Slide`, Entry → animation state |
| Timeline | `Travel`, 60 FPS, 120-frame loop |
| Marker | `Marker`, 32 × 32 square rotated π/4 radians |
| Motion | x=64 at frame 0 → x=256 at frame 60 → x=64 at frame 120; y=100 |
| Data interface | None; no View Model or legacy input |

The first authored drawable is frontmost. The marker therefore appears before
the rail in the RML. Colors use ARGB hex without `#`; `propertyKey="13"` targets
`Node.x`, as checked with `rive schema Shape --animatable --json`.

## Run

Install/review the official CLI and scope its state using
[the CLI reference](../../references/official-cli.md). Copy this folder to a
fresh task-owned directory, then use `terminal` with that copy as the working
directory and the same isolated environment. Replace `rive` with the discovered
executable when it is not on PATH:

```sh
rive . --verify --format=json
rive inspect . --json
rive . --once --format=json
rive . --screenshot=build/start.png --artboard=Smoke --viewport=320x200 --fit=contain
rive . --screenshot=build/mid.png --artboard=Smoke --viewport=320x200 --fit=contain --advance=30
rive . --screenshot=build/end.png --artboard=Smoke --viewport=320x200 --fit=contain --advance=60
```

Require clean build reports and an empty inspection `problems` list. Check the
actual artboard name, then view the PNGs with `vision_analyze`. The diamond
should move from the left endpoint through the middle toward the right endpoint
and stay in front of the rail. The CLI's entry-state initialization can leave a
capture slightly before the exact authored frame; assert the intended pose and
movement rather than assuming all renderers initialize at the same time.

Finally, load the exported `.riv` independently with your target runtime and
its matching WASM/native library. Check names, render multiple poses, inspect
pixels and dispose resources. No login, publishing or `.rev` export is needed
for this script-free smoke test. Do not substitute this fixture for production
accessibility, lifecycle, interaction or multi-platform tests.

## Provenance

Original source by Chris Mish (cygnostik), [ProDyn](https://prodyn.ai), with
Hermes Agent. MIT licensed under the skill's [LICENSE](../../LICENSE); preserve
its [NOTICE](../../NOTICE.md) when redistributing the source. No private lab
artwork, vendor example artwork, brand graphics or third-party assets were reused.
Generated `build/` output is ignored; only source is distributed.
