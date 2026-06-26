# Risks / blockers for James review

## Risks

1. Canonical skill home
   - v2 intentionally does not promote profile-local skill directories into the repo.
   - James should merge the precedence/model language into the existing canonical coach-brand lane after review.

2. Binary artifact size / repo policy
   - The branch includes a PDF, PNG contact sheet, and three PNG logo assets.
   - James should confirm the main repo should carry these proof/source artifacts, or move large binaries to a release/artifact store and keep only hashes/links in repo.

3. Runtime registry integration
   - Local profile registries were installed on Jack for Darin/Sergio/Jack-system, but repo code may not yet automatically install or resolve this registry on fresh profiles.
   - If main-repo behavior should auto-provision The System default BDD for every new coach profile, additional installer/runtime code may be needed.

4. Existing unrelated worktree noise
   - The repository had unrelated modified/untracked files before this branch. This commit stages only the explicit allowlist; reviewers should not infer the branch contains all local work.

## Blockers

- No known content blocker for review.
- Main integration decision needed: whether to commit the PDF/PNG artifacts directly or keep them as external proof artifacts with repo-side manifests/hashes.

## v2 fixes applied

- Clean branch based on current `origin/main`.
- No unrelated commits before the brand-doc commit.
- PNG-byte logo assets use `.png` extensions.
- Color page includes visible hex values and usage notes.
- Typography page includes sizes, weights, line heights, and usage specs.
- Cover logo card enlarged.
- Source notes contrast/readability improved.
- Skill directories excluded to avoid duplicate skill truth.
