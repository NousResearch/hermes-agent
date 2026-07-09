# Seedance Script Helpers

This folder contains optional lightweight helpers for turning production Markdown into machine-readable handoff files.

## Helpers

### `convert-keyframe-table.mjs`

Extracts 12-column keyframe tables from a Markdown file and writes CSV or JSON.

Usage:

```powershell
node .\scripts\convert-keyframe-table.mjs input.md --format csv --out keyframes.csv
node .\scripts\convert-keyframe-table.mjs input.md --format json --out keyframes.json
```

If `--out` is omitted, the converted content is printed to stdout.

The parser expects this header:

```md
| 序号 | 关键帧 | 时间 | 镜头 | 运镜 | 转场 | 动作 | 情绪/细节 | 台词 | 旁白 | 状态/音效 | 英文 |
```

If a cell needs a literal vertical bar, escape it as `\|`.

### `validate-keyframe-table.mjs`

Checks Markdown 12-column keyframe tables before handoff or packaging.

Usage:

```powershell
node .\scripts\validate-keyframe-table.mjs input.md
```

It reports structural errors for wrong column counts, invalid time ranges, and mixed dialogue/narration/state channels. It reports warnings for partial 3x3 groups, missing English, and unusual keyframe marker counts.

### `check-dist-freshness.mjs`

Checks whether the packaged `.skill` file is older than the source skill folder.

Usage from the project root:

```powershell
node .\seedance-script-designer\scripts\check-dist-freshness.mjs .\seedance-script-designer .\dist\seedance-script-designer.skill
```

If the package is stale, rebuild it with the current packaging tool before sharing. Do not hand-edit the `.skill` package.

### `verify-package-contents.mjs`

Checks whether the packaged `.skill` file contains the expected source, reference, script, asset, and example files.

Usage from the project root:

```powershell
node .\seedance-script-designer\scripts\verify-package-contents.mjs .\dist\seedance-script-designer.skill
```

Run this after rebuilding the package. It fails if required files are missing or unexpected files are included.
