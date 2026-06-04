# Session note — Honor of Kings male batches, publish/XHS loop, and image-generation retry

## Context

Nick requested repeated `生成王者荣耀4个男性热门角色` batches in the glossy neon sticker-bomb style. The workflow needed to preserve numbered selection mapping for later commands such as `发布1234` and optional Xiaohongshu copy generation.

## Useful batch choices

To avoid repeating earlier male characters (`李白`, `韩信`, `赵云`, `百里守约`), the next male batch used:

1. `诸葛亮 / Zhuge Liang` — side-profile star-map split layout; blue-white strategist mage; star-map tablet/fan.
2. `澜 / Lan` — dutch-angle shark blade dash; black-blue assassin; twin blades and wave/shark motifs.
3. `马可波罗 / Marco Polo` — foreground twin-pistol depth; red-gold explorer marksman; maps/compass motifs.
4. `孙策 / Sun Ce` — low-angle wave-crash diagonal; naval warrior; heavy weapon, cyan waves, rope/ship motifs.

Earlier male batch:

1. `李白 / Li Bai` — diagonal sword calligraphy arc.
2. `韩信 / Han Xin` — low-angle spear diagonal.
3. `赵云 / Zhao Yun` — circular dragon spear ring.
4. `百里守约 / Baili Shouyue` — foreground scope lens + split sniper poster.

## Retry lesson

One `image_generate` call for Sun Ce failed with:

```text
CLIProxyAPI image generation failed (400): Tool choice 'image_generation' not found in 'tools' parameter.
```

The successful recovery was to retry only the failed character with a shorter prompt, preserving:

- same character identity cues;
- same composition archetype;
- same `NickZag` in-scene placement;
- same safety/non-explicit constraints;
- same palette/style mechanics in compact form.

Do not regenerate successful batch images after this provider/tool-choice failure; keep their paths and patch/write the manifest after the retry succeeds.

## Publication + Xiaohongshu lesson

Nick used:

```text
发布1234
其中1234发布小红书
```

Interpretation used successfully:

- publish all selected numbers `1,2,3,4` to Eagle-backed image2skill as usual;
- also generate Xiaohongshu-ready Chinese copy for **all four** selected images;
- send the XHS copy to Discord channel `discord:1502172364987830393` when that is the active configured Xiaohongshu target.

The copy style that fit: short, character/story-oriented, not prompt/parameter-oriented, one compact paragraph or 3–5 lines per image, with light hashtags.
