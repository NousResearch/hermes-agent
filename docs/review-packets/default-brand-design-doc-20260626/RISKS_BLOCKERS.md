# Risks / blockers for James review

## Risks

1. Skill promotion scope
   - `coach-brand-ingestion` and `coach-lesson-recap-artifacts` existed as profile-local skills on Jack.
   - This branch promotes them into repo `skills/creative/` for review. James should confirm this is the desired in-repo skill location and there is no duplicate canonical source elsewhere.

2. Binary artifact size / repo policy
   - The branch includes a PDF, PNG contact sheet, and three WEBP logo assets.
   - James should confirm the main repo should carry these proof/source artifacts, or move large binaries to a release/artifact store and keep only hashes/links in repo.

3. Runtime registry integration
   - Local profile registries were installed on Jack for Darin/Sergio/Jack-system, but repo code may not yet automatically install or resolve this registry on fresh profiles.
   - If main-repo behavior should auto-provision The System default BDD for every new coach profile, additional installer/runtime code may be needed.

4. Existing unrelated worktree noise
   - The repository had unrelated modified/untracked files before this branch. This commit stages only the explicit allowlist; reviewers should not infer the branch contains all local work.

## Blockers

- No known content blocker for review.
- Main integration decision needed: whether to commit the PDF/PNG/WEBP artifacts directly or keep them as external proof artifacts with repo-side manifests/hashes.
