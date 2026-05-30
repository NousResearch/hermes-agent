---
name: photoshop-image-editing-workflow
description: Use when editing or recreating images with Photoshop from Hermes, especially text replacement, font/weight matching, reference-image QA, Windows Photoshop handoff, and Obsidian production logging.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [photoshop, image-editing, design-qa, wsl, windows, typography, creative]
    related_skills: [reference-image-recreation-qa, obsidian]
---

# Photoshop Image Editing Workflow

## Overview

Use this skill when the user asks Hermes to modify an image and wants the result to be inspected or finished in Adobe Photoshop. The workflow is optimized for a WSL Hermes environment controlling Windows Photoshop: Hermes performs measurement, drafting, automation, file generation, QA reporting, and documentation; Photoshop is used as the visual editor/review surface for final design adjustment.

The important principle is: do not edit by eyeballing alone. First extract measurable design criteria from the original image, then create or modify the image against those criteria, generate a side-by-side QA artifact, open the result in Photoshop, and record the production log.

## When to Use

Use this skill when the task includes any of these:

- "Photoshop", "포토샵", "이미지 수정", "디자인 수정", "폰트 맞춰", "두께 맞춰", "원본처럼"
- Text replacement on an existing image or PDP/landing-page asset
- Matching typography, line spacing, stroke weight, color, shadow, placement, or image composition
- A user correction such as "기존과 다르다", "폰트 두께 다시 확인", "원본 기준으로 검증"
- Producing business assets where the final image should be opened in Photoshop for manual confirmation
- Saving a repeatable production note to Obsidian after image work

Do not use this as a substitute for a full Photoshop plug-in development task. If the user wants to build a Photoshop UXP plug-in, MCP server, or custom JSX automation system, treat that as a separate implementation project first.

## Operating Model

Hermes should act as the production operator:

1. Inspect the source image and user request.
2. Define measurable visual targets from the source.
3. Generate or edit candidate images with local tools or image-generation tools.
4. Create QA artifacts and a short report.
5. Open the final candidate in Windows Photoshop.
6. Save a production log to Obsidian when the asset matters beyond a one-off chat.

Photoshop should be treated as the final design surface, not as a place to hide unverified changes.

## Environment Assumptions

Typical WSL + Windows paths:

```text
WSL vault path:      /mnt/c/Users/<WindowsUser>/Documents/Obsidian Vault
Windows vault path:  C:\Users\<WindowsUser>\Documents\Obsidian Vault
Photoshop exe:       C:\Program Files\Adobe\Adobe Photoshop 2025\Photoshop.exe
Windows fonts:       /mnt/c/Windows/Fonts
```

Do not hardcode the Windows username. Discover it if needed:

```bash
find /mnt/c/Users -maxdepth 1 -mindepth 1 -type d
```

Convert paths with `wslpath`:

```bash
wslpath -w "/mnt/c/Users/<user>/Documents/Obsidian Vault/path/to/file.png"
wslpath -u "C:\\Users\\<user>\\Documents\\Obsidian Vault\\path\\to\\file.png"
```

## Step 1 — Intake and Source Preservation

1. Locate the source image.
2. Copy it to a project-local working folder before editing.
3. Create structured output folders:

```text
assets/
  source/
  edited/
  qa/
  reports/
```

For existing project folders, preserve the user's structure and add only the missing subfolders.

Record the source path and working path in the QA report. Never overwrite the original asset.

## Step 2 — Source Measurement Before Editing

Before changing text or layout, measure the original image. For typography tasks, extract at least:

- target text region bounding boxes
- per-line width and height
- approximate baseline/top/bottom position
- text color
- background color or local background sample
- line spacing
- pixel density / filled-pixel ratio for relative weight
- visible stroke/shadow/outline behavior
- candidate font family and weight from the Windows font folder

For a text replacement where two lines have different weights, measure each line separately. Example production logic:

```text
Original line 1: "진짜 혜택이"
- height: measured from source crop
- filled-pixel density: lower
- role: lighter bold headline

Original line 2: "빵빵하다구요?"
- height: measured from source crop
- filled-pixel density: higher
- role: heavier emphasis line

Rule: do not apply the same font weight/stroke to both lines when the original uses different visual weights.
```

A useful relative metric is:

```text
line_weight_ratio = line_2_filled_pixel_density / line_1_filled_pixel_density
```

Use the original ratio as the target. The edited image does not need to match numerically with mathematical perfection, but it should preserve the same hierarchy: lighter first line, heavier second line, similar placement and visual energy.

## Complex GPT Image + Photoshop Composites

When a benchmark or target image is complex, do not force a one-shot GPT Image generation. Use an element-decomposition workflow: generate or source the background, main object, icons, decorations, cards, proof assets, and text-safe blank areas separately, then assemble them in Photoshop as named layers.

Use this path when the design includes many dependent parts, exact Korean text, prices, reviews, proof screenshots, product fidelity, multiple cards, or a benchmark layout that needs close detail control.

Core rule:

```text
GPT Image model = mood, background, visual objects, decorative elements
Photoshop = composition, text, cards, CTA, numbers, proof assets, final alignment
Hermes = orchestration, measurement, prompt assets, QA, documentation
```

For complex assets, Hermes should run a small multi-agent/role workflow:

1. **Director** — define scope, complexity level, deliverables, and pass criteria.
2. **Reference Analyst** — analyze benchmark layout, color, spacing, text zones, and risky elements.
3. **Layout Architect** — create coordinate-based canvas and Photoshop guide plan.
4. **Asset Decomposer** — decide which elements are generated, sourced, Photoshop shapes, SVGs, or editable text.
5. **Prompt Engineer** — write one prompt per generated asset with strict no-text rules.
6. **Image Generation Worker** — generate/regenerate only individual assets, not the whole final design.
7. **Photoshop Compositor** — assemble selected assets into a layered PSD.
8. **Typography Agent** — apply exact Korean copy as editable Photoshop text layers.
9. **QA Agent** — check generated assets and final composite for breakage, readability, product fidelity, and claim risk.
10. **Ops Logger** — save prompts, output paths, QA sheet/report, and production log.

Keep only independent work parallel. Serialize anything that edits the same PSD or final output. See `references/complex-image-element-decomposition.md` for the full decision rule, master-layout template, agent deliverables, folder structure, gates, prompts, and QA checklist.

## Step 3 — Font and Typography Matching

Use installed Windows fonts first. Search under:

```text
/mnt/c/Windows/Fonts
```

Good candidates for Korean UI/PDP work often include:

- `malgun.ttf` / `malgunbd.ttf` — Malgun Gothic regular/bold
- Korean display fonts installed by the user or Adobe/Windows
- Noto or Pretendard families if present

When exact font identity is unknown:

1. Test 2-4 plausible fonts.
2. Compare text height, width, density, and curvature.
3. Keep the candidate that best preserves the source hierarchy.
4. If a line is visually heavier in the original, use a heavier font, stroke, faux-bold layer, or small stroke expansion only for that line.

Do not assume two lines use the same weight just because they share color or style.

## Step 4 — Generate Candidate Edits

Create candidates with explicit versioned filenames:

```text
<source_stem>_edit_v1.png
<source_stem>_edit_v2_weightmatch.png
<source_stem>_edit_v3_final.png
```

For text replacement:

1. Remove or cover the original text area using a background-matched fill or source-aware patch.
2. Render new text with measured position and line spacing.
3. Apply per-line font size, weight, stroke, fill color, and shadow/outline.
4. Export a lossless PNG for QA.

If using Python, prefer Pillow when available. If Pillow is missing, install it only when appropriate for the environment, or use ImageMagick/Photoshop scripting fallback. Report the blocker honestly if the required image library is unavailable.

## Step 5 — QA Sheet and Report

Always create a visual QA artifact for non-trivial edits:

```text
qa/<source_stem>_qa_sheet.png
reports/<source_stem>_qa_report.md
```

The QA sheet should include:

- original image or crop
- edited image or crop
- optional overlay/difference crop
- measured source values
- measured edited values
- final decision and remaining caveats

The report should include:

```markdown
# Image Edit QA Report

## Source
- Original: ...
- Edited: ...

## User Request
- ...

## Measured Source Criteria
| Element | Width | Height | Density | Notes |
|---|---:|---:|---:|---|
| Line 1 | ... | ... | ... | lighter |
| Line 2 | ... | ... | ... | heavier |

## Applied Edit Criteria
| Element | Font | Size | Stroke | Density | Notes |
|---|---|---:|---:|---:|---|
| Line 1 | ... | ... | ... | ... | ... |
| Line 2 | ... | ... | ... | ... | ... |

## Decision
- Pass / needs revision
- What changed from previous version

## Files
- Final image: ...
- QA sheet: ...
```

## Step 6 — Open in Windows Photoshop

After generating the final candidate, open it in Photoshop from WSL via PowerShell.

Use this pattern and verify the process/window afterward:

```bash
WIN_FILE="$(wslpath -w '/mnt/c/Users/<user>/Documents/Obsidian Vault/path/to/final.png')"
powershell.exe -NoProfile -Command "
  \$photoshop = 'C:\\Program Files\\Adobe\\Adobe Photoshop 2025\\Photoshop.exe';
  if (!(Test-Path \$photoshop)) { throw 'Photoshop executable not found: ' + \$photoshop }
  Start-Process -FilePath \$photoshop -ArgumentList @('$WIN_FILE');
  Start-Sleep -Seconds 3;
  Get-Process Photoshop -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,MainWindowTitle
"
```

If the Windows path contains Korean characters, the console may display mojibake even though Photoshop opens the correct file. Verify by checking the Photoshop process/window title or by inspecting the file path and file size separately.

## Step 7 — Photoshop Manual Pass

When Photoshop is open, make manual edits only against the measured criteria:

- compare original and edited crops at the same zoom level
- check line weight hierarchy
- check alignment, optical centering, and baseline
- check edge quality, anti-aliasing, and background patching
- export a new version if manual edits were made

If manual Photoshop edits create a new file, bring it back into the QA loop. Do not call it final until it has been exported and verified on disk.

## Step 8 — Obsidian Production Log

For project/business assets, write an Obsidian log under the relevant project folder. Use a path close to the asset, for example:

```text
Projects/<ProjectName>/YYYY-MM-DD Photoshop 이미지 수정 작업 로그.md
```

Include:

- objective
- source image path
- final image path
- QA sheet/report path
- source measurements
- final applied font/weight/position criteria
- Photoshop handoff status
- remaining manual checks
- reusable lessons

Read the note back before reporting completion.

## Practical Checklist

Use this checklist while working:

- [ ] Source image preserved; no overwrite
- [ ] Output folders exist
- [ ] Original text/image regions measured before editing
- [ ] Font candidate and font file path recorded
- [ ] Different source font weights handled separately
- [ ] Versioned candidate exported
- [ ] QA sheet generated
- [ ] QA report generated
- [ ] Final candidate opened in Photoshop
- [ ] Photoshop process/window verified
- [ ] Obsidian production log written and read back

## Common Pitfalls

1. **Treating visually different lines as the same font weight.** If the source line 2 is heavier than line 1, the edit must preserve that hierarchy.

2. **Skipping measurement because the change seems small.** Small typography differences are exactly where measurement helps. Measure first, then edit.

3. **Overwriting the original file.** Always write to `edited/` or another versioned output path.

4. **Assuming Photoshop opened because `Start-Process` returned.** Verify with `Get-Process Photoshop` and, when possible, the window title.

5. **Ignoring Korean path display issues.** PowerShell output may show mojibake in WSL. Verify by file existence, size, and Photoshop process state.

6. **Stopping at an image file without QA.** For user-facing assets, produce a QA sheet/report so the user can see why the version is correct.

7. **Calling manual Photoshop changes final without re-export verification.** If Photoshop changes the file, verify the exported file exists and has the expected timestamp/size.

## Verification Commands

Check output files:

```bash
python3 - <<'PY'
from pathlib import Path
for p in [
    Path('/mnt/c/Users/<user>/Documents/Obsidian Vault/path/to/final.png'),
    Path('/mnt/c/Users/<user>/Documents/Obsidian Vault/path/to/qa_sheet.png'),
    Path('/mnt/c/Users/<user>/Documents/Obsidian Vault/path/to/qa_report.md'),
]:
    print(p, p.exists(), p.stat().st_size if p.exists() else None)
PY
```

Verify Photoshop:

```bash
powershell.exe -NoProfile -Command "Get-Process Photoshop -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,MainWindowTitle"
```

Verify Obsidian note:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/mnt/c/Users/<user>/Documents/Obsidian Vault/Projects/<ProjectName>/YYYY-MM-DD Photoshop 이미지 수정 작업 로그.md')
print(p.exists(), p.stat().st_size if p.exists() else None)
print(p.read_text(encoding='utf-8')[:1200])
PY
```

## One-Shot Execution Template

When the user says, "이 이미지 포토샵으로 수정해줘":

```text
1. 원본 파일을 확인하고 edited/qa/reports 폴더를 만든다.
2. 원본에서 수정 대상 영역을 crop/measure 한다.
3. 폰트, 두께, 색상, 위치 기준을 표로 잡는다.
4. 후보 이미지를 versioned PNG로 만든다.
5. 원본 대비 QA sheet와 report를 만든다.
6. 최종 후보를 Windows Photoshop에서 연다.
7. Photoshop 상태를 검증한다.
8. 결과 파일과 QA 파일, 기준값을 Obsidian 작업 로그에 기록한다.
9. 사용자에게 최종 파일 경로, QA 경로, Photoshop 열림 상태만 간결히 보고한다.
```

## Reporting Format

Final response should be concise and evidence-based:

```text
완료했습니다.

- 최종 이미지: <Windows path>
- QA 시트: <Windows path>
- QA 리포트: <Windows path>
- Photoshop 확인: Photoshop 프로세스/창 제목 확인됨
- Obsidian 기록: <Vault-relative path>

핵심 기준:
- 1줄: 원본 기준 lighter weight
- 2줄: 원본 기준 heavier weight
- 두 줄을 동일 두께로 처리하지 않음
```

Do not invent Photoshop completion if the app did not open or if the exported file was not verified.
