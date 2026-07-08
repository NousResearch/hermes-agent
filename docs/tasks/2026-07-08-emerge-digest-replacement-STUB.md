# BU spec STUB — emerge-digest-replacement (B3 cron lane, blocked on skill + design)

Status: STUB — not validated, not scheduled. Written 2026-07-08 during B3 assessment: no backing
Hermes skill exists for an emerge-style digest, and the C3 advisor-synthesis design
(`~/brain/wiki/synthesis/hermes-c3-advisor-synthesis-design.md`, grill-ratified 2026-07-07)
overlaps this lane heavily — C3's threshold-triggered connection pass over the same corpora may
SUBSUME emerge-digest entirely (the step-4f/brain-heal subsumption lesson: upgrade, don't
duplicate).

Decision gate when picked up: FIRST determine whether C3's advisor pass makes a separate
emerge-digest cron redundant. Only if a distinct need survives (e.g. a lightweight weekly
"what's emerging" Telegram digest vs C3's durable advisor notes) does this become its own skill BU:
- New `bop/skills/emerge-digest/SKILL.md`: read-only pass over recent `~/brain/raw/` +
  wiki additions; short themes digest into `~/.hermes/workspace/reports/emerge/`; no vault writes.
- Cron weekly, `--workdir ~/.hermes/workspace`; registry + eval-log same-unit.

Do not build before the C3 subsumption question is answered.
