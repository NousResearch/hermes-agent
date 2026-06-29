# GitHub Dual Setup - Action Required

**Status**: Waiting for access scope expansion  
**Created**: 2026-06-29  

> **Note (Codex P2 correction):** GitHub's `.github` special repository stores community health files (CODE_OF_CONDUCT, CONTRIBUTING, issue templates, etc.) and *workflow templates* visible in the "Actions → New workflow" picker — but workflows placed under `.github/.github/workflows/` are **not** automatically installed or executed in other repositories. To deploy the 4 CI workflows (auto-merge, healer, auto-fix, escalation-detector) globally, each target repository needs the workflow files copied into its own `.github/workflows/` directory. The steps below reflect this: once access is granted, the agent will push the files directly to `dizhaky/hermes-agent` (and any other target repos you specify) rather than relying on cross-repo workflow inheritance.

---

## What's Needed

### 1. Create Two `.github` Repositories (for community health files only)

**Personal Account (dizhaky):**
```
Name: .github
URL: https://github.com/dizhaky/.github
Description: Community health files (CONTRIBUTING, issue templates, etc.)
Visibility: Public
Create via: https://github.com/new
```

**Organization (JHJ):**
```
Name: .github
URL: https://github.com/JHJ/.github
Description: Community health files (CONTRIBUTING, issue templates, etc.)
Visibility: Public
Create via: https://github.com/organizations/JHJ/repositories/new
```

### 2. Expand GitHub Access Scope

Request that my GitHub access be expanded to include:
- `dizhaky/.github`
- `JHJ/.github`
- Any other target repos where workflows should be deployed

**How to request:**
- Contact your Claude Code environment admin
- Or check Claude Code settings for "GitHub access scope"
- Request expansion to include the two `.github` repositories above

---

## What Happens Next (AUTOMATIC)

Once repositories exist and access is granted:

✅ Claude Code agent will:
1. Detect access to new repositories
2. Push all 4 workflow files into each **target repo's** `.github/workflows/` directory:
   - `auto-merge-prs.yml`
   - `ci-auto-healer.yml`
   - `ci-auto-fix.yml`
   - `ci-escalation-detector.yml`
3. Add documentation and README files
4. Commit changes to each repo
5. Send completion confirmation

> **Important:** Workflows must be added to each repo individually. There is no GitHub mechanism to auto-deploy workflows from a `.github` repo to all other repos.

---

## Success Criteria

Once setup completes for each target repository, you'll have:

✅ **Per-repository**
- `.github/workflows/` directory with 4 workflows
- Auto-merge enabled (applies to PRs in that repo)
- CI escalation available (applies to that repo's CI runs)

---

## Questions?

See `CI_ESCALATION_GUIDE.md` for detailed workflow documentation.
