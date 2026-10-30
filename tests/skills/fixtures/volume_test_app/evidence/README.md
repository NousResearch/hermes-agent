# Recorded technical validation

The healthy and injected directories contain unedited raw collector output from code commit `102dee9ef7a4a8bfbf6b6e0e7a2ed25dfc8d8b56`. No generated databases or full catalog dumps are committed. Each scenario contains 252 HTTP samples (first request + warmup + ten warmed measurements for seven operations at three absolute volumes) and 180 UI samples (five UI operations, same populations). Reports link to six actual browser screenshots in total.

- Healthy: all HTTP and UI samples correct; zero console/page/network errors. Confirmed 100, 1,000 and 10,000 products, eight categories, zero orphans.
- Injected: 36 HTTP samples detect the deliberately truncated full/filtered CSV exports; visible listing/filter/sort/pagination remains correct. This is not a naturally discovered defect. First failure: full export at 1,000 products returns 50 instead of 1,000. Raw first-failure CSV excerpts are retained in samples.json/result.json. A faster incomplete export is not a performance improvement.
- Both runs: cleanup re-read confirmed zero run-owned products/categories and an intact unrelated sentinel before the disposable database was removed.
- Median warmed healthy export request time was 0.935 ms at 100 products and 25.419 ms at 10,000. Healthy first-page listing medians were 1.220 ms and 2.647 ms. These are local observations without an SLA or infrastructure-causality claim. UI timings are separate in each report.

## Reproduce

From a checkout containing the code commit above, use the fixture README commands with fresh output directories, seed 42, and optionally a local Chromium path. This machine used Linux, Python 3.12.14 for the standalone fixture and existing Playwright/Chromium; exact kernel, SQLite, browser and UTC start/end values are recorded in result.json. Hermes and the canonical tests used the PM-managed Python 3.14.9. No model credentials or professional materials were used.

## Checks actually run

- `scripts/run_tests.sh tests/skills/test_volume_test_skill.py -q`: 9 passed, hermetic/in-process, no live network.
- `scripts/run_tests.sh tests/skills/test_authoring_standards.py -k volume-test -q`: 7 passed for this skill; other skills/full suite were not executed.
- `python scripts/check --staged` (and final explicit base check): all 14 checks passed, including ruff and code health.
- The canonical runner required execution permission for its own `/var/tmp` scratch directory in this cloud sandbox. No runner or repository configuration was changed to bypass it.
- Browser DOM checks and screenshots are real technical Playwright runs against the loopback fixture, not a claim that Hermes invoked browser_exec.

## Real Hermes validation: blocked, not passed

An isolated `HERMES_HOME` and `HERMES_RUNTIME_DIR` were selected before running the official `source ./activate` workflow. PM installed the hash-verified runtime and test environment. The PM-managed optional Chromium download returned `403 Forbidden`; the separately available system Chromium was used only for the technical UI runs above.

`hermes skills install official/software-development/volume-test --yes` successfully installed the working-tree optional skill. The official scanner reported `SAFE`, decision `ALLOWED`, and installed SKILL.md plus its references/template/helper. The current chat entry point was then attempted:

```sh
hermes chat --skills volume-test --toolsets skills --max-turns 3 --run-budget 30 --oneshot -q 'Use the volume-test skill to inspect its procedure and report prerequisites. Do not write data or contact a target.'
```

It exited with status 1 before an agent session: `Hermes isn't configured yet -- no API keys or providers found`, followed by noninteractive setup instructions. No keys, personal Hermes profiles or professional credentials were copied. Installation is not proof of agent usage. **A fresh-session Hermes run of the full healthy/injected workflow remains required**, using a securely configured model and the local fixture; the PR is draft for this reason. Do not mark the agent E2E checkbox complete based on the technical scripts.

## Upstream and review scope

Normative files were read at base `2d2808ee84549f9836b0056750265058e5488a8a`; the applicable root/skills/tests instructions, contribution guide and PR template were rechecked against subsequently observed main `bff22bde93db8d9fd57b63ee1bcb47947ed04fd2`, with no differences in those files. PR #138936 was also inspected: it changes the internal execute_code batching contract, not a volume-testing skill. No equivalent exact-name PR was found in the final duplicate search. Maintainer placement/acceptance remains open; no merge is requested or performed.

## Contributor local HTTP rerun

A separate [local rerun on 2026-10-30](local-rerun-2026-10-30/README.md) adds unedited completed healthy/injected HTTP evidence from code commit `bdd3ac64af2363ab73f7e78e694d23b0060f229b`. Each has 252 HTTP samples; healthy has zero incorrect samples and injected has 36 expected incorrect export samples. Both verified scoped cleanup. This rerun contains no browser or real Hermes agent evidence and does not resolve the agent E2E blocker above.
