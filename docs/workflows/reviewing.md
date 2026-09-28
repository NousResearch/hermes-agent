# Reviewing Workflow

Read and follow this file only when your sole task is to review the codebase,
a branch, a pull request, or an implementation and report findings. This
workflow is a narrow exception to the normal `AGENTS.md` implementation flow.
If you are implementing changes and only running review as part of that
end-to-end work, do not use this workflow.

## Rules

- Do not invoke `./.codex/scripts/codex-review.sh`, `codex exec review`, or any other reviewer agent.
- Do not run tests, builds, linters, formatters, type checks, or other validation commands.
- Review the diff and relevant repository context only.
- Do not edit files, write code, or apply patches.
- Do not create, switch, or delete branches.
- Do not stage, commit, push, tag, or create/update pull requests.

## Review Focus

- Identify correctness bugs, regressions, security issues, data-loss risks, and
  behavior that does not match the stated intent.
- Judge whether the implementation is the right architecture for the change or
  whether it layers a workaround where a root-cause change is required.
- Enforce the boundaries and contracts documented in `ARCHITECTURE.md` and the
  relevant files under `docs/`.
- Flag duplicated logic, unnecessary fallback or legacy paths, dead code, and
  maintainability regressions.
- Report missing or inadequate test coverage as a finding without running tests.
