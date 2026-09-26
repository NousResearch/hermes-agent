---
name: meshy
description: Generate, texture, rig and 3D-print models with Meshy.
version: 1.0.0
author: Xubin Du (Arlieeee), Meshy
license: MIT
platforms: [macos, linux, windows]
prerequisites:
  commands: [node, npm]
metadata:
  hermes:
    tags: [3d, 3d-generation, text-to-3d, image-to-3d, rigging, 3d-printing, meshy, creative]
    category: creative
    requires_toolsets: [terminal]
---

# Meshy Skill

Turn a text prompt or a photo into a 3D model with [Meshy](https://www.meshy.ai), then texture,
remesh, convert, rig, animate or prepare it for 3D printing. Everything runs through the official
`meshy` CLI ([meshy-cli](https://www.npmjs.com/package/meshy-cli)) via the `terminal` tool; the
skill sends no requests itself. Generation spends the user's Meshy credits; local print
preparation and slicer handoff are free.

## When to Use

- The user wants a 3D model (GLB, FBX, OBJ, USDZ, STL, 3MF) from a description or an image.
- The user wants to retexture, remesh, convert, resize, UV-unwrap, rig or animate a model.
- The user wants to 3D-print something: a white or multicolor print, a Creative Lab figure,
  lamp, keychain or fridge magnet, or scaling an existing OBJ and opening it in a slicer.

Not for 2D-only image editing unrelated to 3D, or for modelling by hand in Blender.

## Prerequisites

- Node.js 22.12 or newer with `npm`. Install the CLI once: `npm install -g meshy-cli`. If a
  global install is refused, prefix every command below with
  `npm exec --yes --package=meshy-cli@0.4.0 --` instead.
- A Meshy account on a paid plan (task creation is not available on the Free plan). Sign-in is
  in the browser (see Procedure); no API key is needed. An existing `MESHY_API_KEY` in the
  environment is honoured instead and takes precedence.

## How to Run

Run every command through the `terminal` tool. Business commands always carry
`--output-schema v1 --format json --no-update-check`, so stdout is one JSON envelope:
`ok`, `result`, `error`, `warnings`. Read IDs and paths from it; never scrape progress text on
stderr. Writes carry `--workspace` so nothing lands outside the user's chosen directory.

Generation is asynchronous: `create --async` returns a task ID immediately, `wait` blocks until
the task finishes (typically 1–5 minutes, longer for textures). Give `wait` a `timeout` of 600 in
the `terminal` call, or run it with `background: true` and `notify: true` and resume on the
notification.

## Quick Reference

| Goal | Command (append `--output-schema v1 --format json --no-update-check`) |
|---|---|
| Plan and estimate, no spend | `meshy make "PROMPT" --dry-run` |
| Text → untextured model | `meshy text-to-3d create --mode preview --prompt "PROMPT" --target-formats glb --async` |
| Add textures to that preview | `meshy text-to-3d create --mode refine --preview-task-id PREVIEW_ID --enable-pbr true --target-formats glb --async` |
| Photo → textured model | `meshy image-to-3d create --image-url "PHOTO" --should-texture true --enable-pbr true --target-formats glb --async` |
| Wait for any task | `meshy RESOURCE wait TASK_ID --timeout 600` |
| Download one format | `meshy download --resource RESOURCE --task-id TASK_ID --model-format glb --output "OUT/model.glb" --workspace "OUT"` |
| List a task's assets | `meshy download --resource RESOURCE --task-id TASK_ID --list` |
| Retexture / remesh / convert / resize | `meshy retexture\|remesh\|convert\|resize create --input-task-id SOURCE_ID … --async` |
| Rig a textured humanoid | `meshy rigging create --input-task-id TEXTURED_ID --height-meters 1.7 --async` |
| Find an animation | `meshy animation-catalog list --search wave` |
| Animate a rig | `meshy animate create --rig-task-id RIG_ID --action-id ACTION_ID --async` |
| Printability check (free) | `meshy analyze-printability create --input-task-id SOURCE_ID --async` |
| Multicolor print (3MF) | `meshy multi-color-print create --input-task-id TEXTURED_ID --max-colors 4 --async` |
| Scale an OBJ for printing (local) | `meshy mesh prepare-print "MODEL.obj" --height-mm 75 --output "OUT/model.print.obj" --workspace "OUT"` |
| Find / open a slicer (local) | `meshy slicer detect`, then `meshy slicer open --slicer SLICER_ID --file "FILE"` |
| Balance | `meshy balance` |

`RESOURCE` is the command that created the task (`text-to-3d`, `image-to-3d`, `rigging`, …).
Each resource documents every option in `meshy RESOURCE create --help`.

## Procedure

1. **Check the session.** `meshy auth status --format json --no-update-check`. With
   `"authenticated": true` and `"verified": true`, skip to step 3. Local-only print work (step 6)
   needs no session at all.
2. **Sign in once.** Start `meshy auth login --device --format json --no-update-check` through
   `terminal` with `background: true` and `notify: true`. Within seconds it prints
   `Enter code XXXX-XXXX at https://www.meshy.ai/device` on stderr; read it with `process`
   (`log`). Send the user that link and code exactly as printed: open the link, sign in to Meshy,
   type the code, approve. Then `process` `wait` on the same session. It exits 0 once the user
   approves; the code expires after 10 minutes, after which the same login is started again.
   Re-run `auth status`, then continue the user's original request.
3. **Agree the plan.** Pick the shortest route: text → preview (→ refine for colour); photo →
   image-to-3d; follow-ups reuse the existing task ID. For a paid chain, quote
   `meshy make "PROMPT" --dry-run` or the price list (https://docs.meshy.ai/en/api/pricing) and
   get the user's go-ahead before the first `create`.
4. **Submit once, then wait.** Run the `create … --async` command, read
   `result.submission.task_id`, then `meshy RESOURCE wait TASK_ID --timeout 600 --output-schema v1
   --format json --no-update-check`. A timeout means the task is still running: wait again on the
   same ID; never create a second task.
5. **Deliver.** Download only the requested format into the user's directory (default
   `./meshy_output`), passing that directory as `--workspace`. Report the file path, the task ID
   for follow-ups and `result.task.consumed_credits`. Download `--asset thumbnail.primary` to show
   a preview image when one exists; a missing preview is stated, never implied.
6. **Printing.** For a white print, generate without textures, download OBJ, then
   `mesh prepare-print` to the requested height. For multicolor, texture first, then
   `multi-color-print` and download its 3MF. Run the free `analyze-printability` when the part
   must be sturdy; `repair-printability` only when analysis reports errors. `slicer detect` /
   `slicer open` only when the user wants the file opened.

## Pitfalls

- **Exit codes carry meaning:** 3 not signed in, 8 wait timed out (task still running), 9 out of
  credits, 10 submission outcome unknown. After 10, check the task list before any retry; a
  blind resubmit can charge twice.
- **Refine only accepts a text-to-3d preview.** Photos, remeshed or uploaded models use
  `retexture` instead.
- **Rigging needs a textured humanoid** with clear limbs, at most 300,000 faces; an untextured
  preview fails. The rig already includes walking and running clips; only a different motion
  needs `animate`.
- **Repair drops textures.** Retexture the repaired model before `multi-color-print`.
- **Assets expire after 3 days** on non-Enterprise plans; download results promptly.
- **Failed tasks are refunded** (`consumed_credits` is 0), so a failure can be retried without a
  new cost approval; an unknown charge is reported as unknown, not zero.
- **The user code in the login line is the only secret-looking value to share.** Never ask for
  an API key in chat, never print tokens, and never use `auth login --no-wait` (its JSON holds
  the device code, a bearer secret). The CLI stores the session in
  `~/.config/meshy/credentials.json` and refreshes it itself.

## Verification

```bash
meshy auth status --format json --no-update-check
```

Prints `"authenticated": true` and `"verified": true` with exit code 0 when the skill is ready.
`meshy doctor --output-schema v1 --format json --no-update-check` diagnoses the environment
without any network call.
