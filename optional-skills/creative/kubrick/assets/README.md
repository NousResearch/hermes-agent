# Kubrick visual assets

## Canonical hero (README + social)

The README hero and GitHub repository social preview must use the **same artwork**:
cinematic shattered-glass KUBRICK wordmark with cyan/amber beams.

| File | Role |
|---|---|
| `hero.jpg` | README hero banner (`README.md` top image). Full frame (~1168×784). |
| `social-preview.jpg` | 1280×640 crop for **GitHub Settings → General → Social preview**. |

Both are derived from the same photoreal master. Do not use the geometric SVG for either surface unless intentionally redesigning both together.

## Optional vector

| File | Role |
|---|---|
| `kubrick-hero.svg` | Geometric vector interpretation (not used for README or social). |

## Updating the brand image

1. Replace `hero.jpg` with the new full-frame master (JPEG, wide cinematic frame).
2. Re-export `social-preview.jpg` as a 1280×640 center crop of that master.
3. Update GitHub **Settings → General → Social preview** by uploading `social-preview.jpg`.
4. Confirm link unfurls and the README hero match.
