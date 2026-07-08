# BU spec STUB — pipeline-hygiene-replacement (B3 cron lane, blocked on skill)

Status: STUB — not validated, not scheduled. Written 2026-07-08 during B3 assessment: the
Paperclip `pipeline-hygiene` agent (daily 08:00 PT, revenue-ops) has NO backing Hermes skill among
the 14 deployed, and B3's rule is "any NEW skill code a cron needs rides a BU first."

Unit shape when picked up:
- New `bop/skills/pipeline-hygiene/SKILL.md`: read-only audit of `~/ai-agency/clients/_pipeline.md`
  + `_capability-matrix.md` (stale rows, missing next-touch, matrix/pipeline drift); report into
  `~/.hermes/workspace/reports/pipeline/`; ZERO writes to ai-agency (reporting lane, not intake —
  no fence change needed).
- Port the Paperclip agent's mission text as source canon (cite it); meet the dsm-mirror
  retirement-procedure bar: final supervised Paperclip run → grace → diff → retire row.
- Cron: daily 08:00 PT via `hermes cron create --workdir ~/.hermes/workspace` (the cwd lesson from
  dsm-mirror-replacement fire 1, eval-log 2026-07-07).
- Registry + eval-log lanes same-unit.

Acceptance: report matches a manual pipeline read; 3 verified fires; Paperclip pipeline-hygiene
retired per procedure.
