# Honor of Kings neon batch continuity — 2026-05

## Trigger

Use this when Nick asks for repeated 王者荣耀 / Honor of Kings female-character neon sticker-bomb batches, especially after interruptions or commands like `继续`, `发布12`, or `发布王者荣耀5个结果`.

## What happened

A four-character generation request was interrupted after two successful images:

1. Lady Sun / 孙尚香
2. Shangguan Wan'er / 上官婉儿

Nick later said `继续`; two more images were generated:

3. Gongsun Li / 公孙离
4. Da Qiao / 大乔

Because the first two were generated before interruption and the continuation manifest initially only contained items 3–4, later `发布12` still correctly referred to the delivered numbering across the combined batch: Sun Shangxiang and Shangguan Wan'er, not the newer continuation-only manifest.

## Operational rule

For interrupted batches:

1. Preserve successful pre-interruption generations as part of the same logical batch.
2. Write or patch a manifest as soon as any generated image succeeds.
3. If Nick says `继续`, keep original numbering and append later results; do not renumber continuation images starting at 1 unless explicitly starting a fresh batch.
4. For `发布12`, resolve against the latest delivered numbered list, not only the most recent manifest file.
5. If the manifest is split across interruption boundaries, create a consolidated mapping before importing to Eagle.

## Retry rule

If one requested character times out:

- Retry only the failed character.
- Use a shorter prompt preserving identity cues, composition, safety constraints, and in-scene `NickZag` placement.
- Do not rerun successful images or change their numbering.

## Worked examples

Generated/published examples from the session:

- `Lady-Sun_孙尚香_neon-stickerbomb-cyber-pop-foreground-cannon-poster`
- `Shangguan-Waner_上官婉儿_neon-stickerbomb-cyber-pop-calligraphy-brush-poster`
- `Gongsun-Li_公孙离_neon-stickerbomb-cyber-pop-umbrella-petal-leap-poster`
- `Da-Qiao_大乔_neon-stickerbomb-cyber-pop-water-portal-ribbon-poster`

A later seven-character batch included:

- Daji / 妲己
- Garo / 伽罗 — first long prompt timed out, shorter retry succeeded
- Nakoruru / 娜可露露
- Angela / 安琪拉
- Charlotte / 夏洛特
- Nuwa / 女娲
- Mai Shiranui / 不知火舞

## Pitfall

Conversation recency can be misleading after tool interruption. The newest manifest may contain only the continuation images, while the user-visible numbering spans messages before and after the interruption. Always reconcile against the delivered numbered list before publishing selected numbers.