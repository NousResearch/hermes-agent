# Generated / noise exclusions

Excluded intentionally from this review branch:

- `docs/brand-design-docs/the-system-20260626/_pdf_pages/`
  - Generated page raster intermediates used only to create the contact sheet.
  - Contact sheet is included instead as the compact QA artifact.

- Profile-local runtime installs:
  - `~/.hermes/profiles/darin/home/brand-design-docs/the-system-default/`
  - `~/.hermes/profiles/sergio/home/brand-design-docs/the-system-default/`
  - `~/.hermes/profiles/jack-system/home/brand-design-docs/the-system-default/`
  - These are runtime state, not repo source. The repo carries the canonical files and registry guidance.

- Public proof docroot copies:
  - `~/agent-lab/state/public-proofs/the-system-brand-design-doc-20260626/`
  - These are review-serving copies only; repo source is under `docs/brand-design-docs/the-system-20260626/`.

- Uploaded/private lesson packet PDFs and cache files.
  - Student-facing packets are not source for this default brand doc branch and should not be committed.

- Existing unrelated dirty/untracked repo files shown by `git status` before this branch.
  - This branch is path-bounded to brand-doc, skill, and review packet artifacts only.


## v2 scope correction

- `skills/creative/coach-brand-ingestion/` and `skills/creative/coach-lesson-recap-artifacts/` are intentionally excluded from v2.
- This avoids creating duplicate canonical skill truth. James should merge the precedence/model language into the existing canonical coach-brand lane.
