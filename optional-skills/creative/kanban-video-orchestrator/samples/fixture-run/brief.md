# Video Brief — Fixture Product Teaser

> Slug: `fixture-product-teaser` · Tenant: `fixture-product-teaser` · Project workspace: `~/projects/video-pipeline/fixture-product-teaser`

## 1. Concept

**One-line pitch.** A tiny deterministic teaser used to validate the video orchestration skill.

**Emotional north star.** clear, mechanical, safe
*(What should the viewer feel walking away?)*

## 2. Scope

| | |
|---|---|
| Duration | 12 seconds |
| Aspect ratio | 16:9 |
| Resolution | 1280x720 |
| Frame rate | 24 fps |
| Target platforms | internal test fixture |
| Deadline | none |
| Quality bar | fixture-valid *(rough draft / polished / archival)* |

## 3. Style

**Visual references.** monospace title card

**Tone.** crisp

**Brand constraints.** none
*(colors, typography, motion language; or "n/a")*

**Aesthetic rules.**
deterministic text outputs only

## 4. Scenes

Beat-by-beat breakdown. Each scene gets a row.

| # | Time | Content | Target tool / skill | Audio | Notes |
|---|------|---------|---------------------|-------|-------|
| 1 | 0:00-0:06 | ASCII title card: Fixture Product Teaser | ascii-renderer | soft launch sting | deterministic fixture scene |
| 2 | 0:06-0:12 | Product silhouette resolves into call to action | ascii-renderer | resolve sting | same renderer for fixture simplicity |

## 5. Audio

**Approach.** silent fixture with placeholder cues
*(narration / music-only / synced to track / silent / mixed)*

**Voiceover.** n/a
*(provider, voice, language, script source — "n/a" if no VO)*

**Music.** n/a
*(provided track path / commission via Suno / commission via heartmula /
license-free / "n/a")*

**SFX.** text cues only
*(generated, library, or "n/a")*

## 6. Deliverables

| Format | Resolution | Notes |
|--------|-----------|-------|
| mp4 | 1280x720 | The main output |
| _(none)_ |  |  |

**Final filename.** `output/final.mp4`
*(plus optional `output/final-9x16.mp4`, `output/captions.srt`, etc.)*

## 7. Constraints

- API keys required: none
- External dependencies: Python 3.11+
- Source assets to incorporate: none

---

**This brief is the contract. The director and every downstream profile read
it. If the brief changes, the kanban must be re-fired — don't edit live.**
