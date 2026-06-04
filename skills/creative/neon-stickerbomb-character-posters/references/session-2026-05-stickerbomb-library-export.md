# Session note: exporting historical neon stickerbomb outputs by IP

When Nick asks to collect or reorganize historical neon stickerbomb images, treat it as an artifact-library export problem, not as image generation.

## Desired destination shape

- Default target used in this session: `/Users/nick/Desktop/stickerbomb`.
- Organize by work/IP folder, not by generation date, manifest name, character name, or batch.
- Stable folder names used successfully: `Pokémon`, `Crayon Shin-chan`, `One Piece`, `Valorant`, `Saint Seiya`, `Honor of Kings`, `Onmyoji`, `Demon Slayer`, `Overlord`, `The Boys`, `Dragon Ball`, `Gintama`, `Evangelion`, `Naruto`.
- If Nick writes `overload` in this context, map it to `Overlord` unless he clarifies otherwise.

## Working method

1. Discover manifests/state JSON under the active Hermes profile, especially `state/`, `cache/`, and image manifest locations.
2. Extract only real image paths from JSON/state/session artifacts; do not treat failed/empty-response paths as valid.
3. Classify by IP using metadata fields first (`source`, prompt/title/manifest name), then keyword fallback.
4. Copy only files that actually exist; never create placeholders for missing historical references.
5. Write/update an `_index.json` in the destination with source path, destination path, title/character if known, classification, and source manifest/session.
6. Keep a `missing referenced files` count/sample in the index or final report so Nick knows why some historical images were not copied.
7. Verify with real filesystem counts and size before reporting: image file count, category directories, and `du -sh`.

## Pitfalls from this session

- A first-pass manifest-only scan found only 74 images; expanding to state/cache/session JSON raised the verified export to 951 images. Do not stop at the first manifest glob if Nick asks for “过往所有/历史生成”.
- Some historical manifests point to local files that no longer exist. Count those as missing references and report plainly; do not imply they were copied.
- Broad full-home `find` searches can time out or wander into huge dependency trees. Prefer targeted profile directories and bounded JSON parsing. Exclude `node_modules`-like trees when scanning.
- If a parser accidentally treats multi-line command output as one path, it can produce `File name too long`; split lines and validate candidate paths before filesystem operations.

## Reporting pattern

Report the destination path, verified total image count, directory size, category counts, and missing-reference count. For user-named categories with no copied real files, say “未找到可复制的本地原图/源路径已失效” rather than inventing a count.