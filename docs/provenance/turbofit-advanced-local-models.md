# TurboFit reuse ledger: advanced local models

This feature selectively adapts ideas and data conventions from
[`SouthpawIN/turbofit`](https://github.com/SouthpawIN/turbofit), created by
[sovthpaw](https://github.com/sovthpaw) (SouthpawIN), under the MIT license,
without importing its runtime controller or Turbohaul lifecycle.

| Source snapshot | Source paths inspected | Hermes use | Treatment |
| --- | --- | --- | --- |
| `98b45598785c4ca8efe5a5d5ea0835782f4ee007` | `src/turbofit_runtime/recipes.py` | MTP/draft capability vocabulary; server capacity is distinct from context | Adapted concept; no source code copied |
| `13e1ac200732beef99f048d248ae1fd6fffe00b5` | `src/turbofit_runtime/evidence.py` | Candidate/evidence distinction and atomic publication discipline | Adapted concept; no source code copied |
| `13e1ac200732beef99f048d248ae1fd6fffe00b5` | `src/turbofit_runtime/routes.py` | Explicit typed route state and atomic route publication pattern | Adapted concept; gateway implementation remains Hermes-native |
| `13e1ac200732beef99f048d248ae1fd6fffe00b5` | `runtime-profiles/acquisitions.json`, `hybrid-models.json` | Future catalog import candidates | Not imported until artifacts and engine compatibility are independently verified |

The feature was implemented with Hermes' own estimator, supervisor, configuration
writer, router preset generation, and gateway route mechanisms. This ledger is
not an endorsement by TurboFit's authors.
