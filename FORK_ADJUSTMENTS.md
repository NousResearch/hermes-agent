# Fork Adjustments

This fork follows `NousResearch/hermes-agent` by default. The convergence tree
is based on upstream commit
`98105f31f46d3de58a8f69a2a439cee3f7a5e389` and intentionally differs only in
the four paths listed below.

| Path | Retained behavior | Removal condition |
|---|---|---|
| `.github/workflows/hermes-pr-tag-listener.yml` | Calls the organization-owned `@hermes` listener using an immutable reusable-workflow revision and only the permissions it requires. | Remove if the listener is no longer used by this repository. |
| `.gitignore` | Excludes AO-managed `.claude` settings and metadata files that are machine-local and may contain credentials. | Remove individual patterns only when the corresponding files can no longer contain machine-local secrets. |
| `.coderabbit.yaml` | Enables automatic CodeRabbit review without fork-specific approval or merge gates. | Remove if review automation moves to organization-level configuration. |
| `FORK_ADJUSTMENTS.md` | Records the complete intentional divergence from upstream. | Keep while this repository remains a fork. |

No fork-only application Python, bundled user plugins, Green Gate, Skeptic, or
merge-train files are retained in this tree.

The following historical changes are not carried forward. Each requires a
current-upstream reproduction and a separate minimal change before it can be
reintroduced:

- cross-workspace Slack channel-name disambiguation (`4485fc71f`);
- outbound destination binding (`95213093e`), which must be proven superseded
  or replaced before production convergence;
- bounded memory rendering (`b5240afe1`);
- RTK rewriting as a supported user-installed plugin only (`d49d54c9e`); and
- Python 3.14 daemon-pool compatibility from the superseded fork PR.

Historical fork baseline:
`047662e56e5510378adf1fe9da3a684c4784571d`.
