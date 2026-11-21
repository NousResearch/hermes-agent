# Contributor local HTTP rerun — 2026-10-30

These files are unedited collector output supplied by the contributor from a separate local machine, running code commit `bdd3ac64af2363ab73f7e78e694d23b0060f229b`. They supplement the original cloud HTTP/browser evidence rather than replacing it.

- [Healthy complete report](healthy-complete/report.md): 252 HTTP samples, zero incorrect samples, errors or timeouts.
- [Injected report](injected/report.md): 252 HTTP samples, 36 expected incorrect samples from deliberately truncated exports, zero HTTP errors or timeouts.
- Both runs confirmed 100, 1,000 and 10,000 products, eight categories and zero orphans; cleanup verified zero run-owned residuals and an intact unrelated sentinel.
- Each scenario retains its original plan, manifest, samples and result JSON. Report aggregates and embedded samples/plans were checked for consistency before publication.

Python 3.14.4 and SQLite 3.46.1 were recorded. Exact platform and UTC timestamps remain in each result. These are sequential HTTP measurements against a synthetic loopback fixture, with no UI/browser or real Hermes agent execution. No production capacity or direct cross-machine performance comparison is claimed. The PR remains draft pending real Hermes agent E2E.

The uploaded archive also contained an earlier `healthy` run interrupted by KeyboardInterrupt. It is excluded from this completed-run selection; the later `healthy-complete` output is preserved under its original name. No incomplete run is presented as complete. No databases or full catalog dumps are included. Pattern review found no credentials or personal paths in the uploaded files.
