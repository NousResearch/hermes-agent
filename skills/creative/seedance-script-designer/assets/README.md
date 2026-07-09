# Seedance Asset Reference Guide

Use this folder for lightweight asset documentation and manifests. Do not store large production binaries here unless the user explicitly asks for a portable skill bundle.

## Naming Pattern

Use stable asset IDs in local planning documents and prompts:

| Prefix | Use | Example |
|---|---|---|
| `@图` | character, scene, prop, style, first/last frame images | `@图1_艾莉娅三视图` |
| `@视频` | camera, motion, action, rhythm, extension references | `@视频1_史诗预告节奏` |
| `@音频` | dialogue, narration, music, ambience, sound effects | `@音频1_预告片配乐` |

Local shorthand can use `@图1`, but Seedance/即梦 platform prompts should use the official uploaded reference name, usually `@图片1` with the character `片`. Convert local `@图1` to platform `@图片1` when pasting into Seedance if the platform displays that name.

## Reference Rules

- Keep IDs stable across all tables and prompts.
- Bind each asset to a concrete role: character identity, scene layout, prop shape, camera motion, music rhythm, voice, sound effect, first frame, or last frame.
- Prefer concise names that include the target object and use, such as `@图2_凯兰黑袍三视图` or `@音频3_低钟与圣咏`.
- Do not put asset IDs into the default 12-column 3x3 keyframe table. The user's normal workflow uses the generated nine-panel keyframe storyboard image as one visual reference.
- When asset references are needed, keep them in the separate asset manifest, the素材引用规划 section, or the final video prompt.
- In Seedance prompts, describe how each asset is used instead of re-describing details already defined by the asset.

## Project Asset Storage

Production assets belong inside the matching script project under `00剧本`, not in a workspace-root image folder by default.

Default structure:

```text
00剧本\<剧本项目>\
  <剧本名>_分镜包或生产包.md
  image\
    <剧本名>\
      分镜图关键帧\
      资产\
```

Example:

```text
00剧本\02灰烬恋人_\
  灰烬恋人_3分钟Seedance分镜包.md
  image\
    灰烬恋人\
      分镜图关键帧\
      资产\
```

Use `分镜图关键帧` for 3x3 storyboard sheets, clean keyframe boards, annotated storyboard boards, first-frame/last-frame boards, and any image whose primary purpose is shot continuity.

Use `资产` for reusable production assets, including character sheets, character emotion sheets, scene/environment concepts, prop images, style references, costume references, and other images the user may upload separately as @ references.

Before starting a new script/project image batch, ask the user which `00剧本\<剧本项目>` folder should receive the outputs unless the user already provides an exact folder or file path. If the user provides a path such as `00剧本\02灰烬恋人_\image\灰烬恋人\分镜图关键帧`, use it exactly.

Do not keep random generated filenames as final deliverables. Copy generated files into the project folder with semantic names:

| Type | Filename pattern | Example |
|---|---|---|
| Clean 3x3 storyboard | `序号_时间_场景名_无标注九宫格.png` | `01_00-00-00-15_灰烬山路_无标注九宫格.png` |
| Annotated 3x3 storyboard | `序号_时间_场景名_标注九宫格.png` | `01_00-00-00-15_灰烬山路_标注九宫格.png` |
| Character asset | `角色_角色名_用途_v01.png` | `角色_艾莉娅_三视图_v01.png` |
| Scene asset | `场景_场景名_用途_v01.png` | `场景_灰烬神庙_内殿_v01.png` |
| Prop asset | `道具_道具名_v01.png` | `道具_黑铁王冠_v01.png` |

Historical folders such as renamed root-level image directories may exist, but they are not the default production destination. Use them only if the user explicitly points to them.

## Practical Reference Limits

- Do not overload a single shot with too many references. For most shots, use 1-3 primary image references plus optional video/audio references only when they control motion, extension, rhythm, voice, or sound.
- If a project has more than 12 assets, mark priority in the manifest: `P0` for identity-critical assets, `P1` for quality/rhythm assets, and `P2` for optional ambience or polish.
- Use the smallest sufficient reference set per shot. Character identity, scene layout, key prop, and audio rhythm should each have a clear reason to be present.
- For first/last-frame or video-extension tasks, explicitly mark which image or video is the start frame, end frame, or base video.
- Keep heavy binary assets outside the skill folder unless the user explicitly asks for a portable bundle. Store stable file paths or URLs in the manifest instead.
