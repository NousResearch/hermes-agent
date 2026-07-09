# Release Checklist

Use this checklist before distributing or packaging `seedance-script-designer`.

## Source Consistency

- Keep the project copy `seedance-script-designer/` and the installed global copy `C:\Users\ruoyu\.codex\skills\seedance-script-designer\` in sync.
- Treat the project copy as the editable source when both versions differ, then sync outward to the global skill.
- Confirm `SKILL.md` frontmatter still has valid `name` and `description`.
- Confirm `references/quick-modes.md` exists if `SKILL.md` or `output-schema.md` mentions quick output modes.
- Confirm `references/dialogue-voice-method.md` exists if `SKILL.md` or `output-schema.md` mentions dialogue/voice output.

## Table and Example Validation

Run the helper checks from the project root:

```powershell
node .\seedance-script-designer\scripts\validate-keyframe-table.mjs .\seedance-script-designer\examples\minimal-12col-keyframes.md
node .\seedance-script-designer\scripts\convert-keyframe-table.mjs .\seedance-script-designer\examples\minimal-12col-keyframes.md --format json
```

Validation should fail on structural problems such as wrong column counts, invalid time ranges, or mixed dialogue/narration/state channels. Warnings are acceptable when the output is intentionally partial, such as a revision-only table.

## Dist Freshness

If `dist/seedance-script-designer.skill` will be shared, check whether the package is older than the source files:

```powershell
node .\seedance-script-designer\scripts\check-dist-freshness.mjs .\seedance-script-designer .\dist\seedance-script-designer.skill
node .\seedance-script-designer\scripts\verify-package-contents.mjs .\dist\seedance-script-designer.skill
```

If the freshness check reports stale, rebuild the package with the current project packaging tool before distribution. If the contents check fails, rebuild the package from the expected file list. Do not hand-edit the `.skill` package.

## Packaging Contents

The package should include:

- `README.md`
- `SKILL.md`
- `references/story-structure-method.md`
- `references/dialogue-voice-method.md`
- `references/shot-breakdown-method.md`
- `references/output-schema.md`
- `references/storyboard-annotation-rules.md`
- `references/quick-modes.md`
- `references/release-checklist.md`
- `assets/README.md`
- `assets/asset-manifest-template.md`
- `scripts/README.md`
- `scripts/convert-keyframe-table.mjs`
- `scripts/validate-keyframe-table.mjs`
- `scripts/check-dist-freshness.mjs`
- `scripts/verify-package-contents.mjs`
- `examples/minimal-12col-keyframes.md`
- `examples/asset-manifest-example.md`
- `examples/audio-sync-example.md`

## Final Manual Review

- The 12-column table stays consistent: `序号 | 关键帧 | 时间 | 镜头 | 运镜 | 转场 | 动作 | 情绪/细节 | 台词 | 旁白 | 状态/音效 | 英文`.
- The keyframe table does not contain a per-frame `参考资产` column by default.
- Dialogue, narration, and state/sound are not merged into one prefixed cell.
- English is present for dialogue/narration, and sound rows use `[SFX: ...]`, `[Ambience: ...]`, or `[N/A]`.
- 3x3 storyboard image prompts keep text metadata in the table unless the user explicitly asks for annotated images.
- Script-specific Markdown deliverables are stored under `00剧本\<剧本项目>\`.
- Generated storyboard/keyframe images are copied into `00剧本\<剧本项目>\image\<剧本名>\分镜图关键帧\` with semantic names instead of random generated IDs.
- Generated character, scene, prop, and style assets are copied into `00剧本\<剧本项目>\image\<剧本名>\资产\` with semantic names.
- New script/project image batches ask for the target `00剧本\<剧本项目>` folder unless the user already provides an exact folder or file path.
