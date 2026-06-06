# Progress: Fix Mac-persona auto-dispatch: repo inference + packet-landing
Card: t_cea89616
Branch: bld-cea89616-macroute
Started: 2026-06-06T13:25Z

## Checklist
- [x] Read card and estimated
- [x] Orient: git log, test suite, file structure
- [x] Study existing bld-5650-f856 Mac dispatch code (source branch)
- [ ] Add MAC_PERSONAS constant + detect_crashed_workers skip
- [ ] Add _mac_session_name, _set_mac_worker_info, _infer_mac_repo_path
- [ ] Add fixed _spawn_mac_session (repo inference, preflight, launch verify)
- [ ] Patch dispatch_once (ready + review) for Mac preclaim
- [ ] Patch _default_spawn for Mac routing fallback
- [ ] Add TestMacPersonaDispatch tests from bld-5650-f856
- [ ] Add new tests: repo inference, t_2ad4b03f life-engine regression, t_d80003fa ccat-guru regression
- [ ] Run tests
- [ ] Push to branch
- [ ] Open PR

## Notes
- MAC_PERSONAS = builder, reviewer, scout, designer
- origin/main does NOT have _spawn_mac_session; porting from bld-5650-f856 + fixing
- Key fixes: repo_path default "ccat" -> infer via ordered map; wait for Claude idle before send
- HERMES_MAC_SYNTHETIC=1 suppresses actual bridge calls in tests
- claim_task/claim_review_task both accept claimer= kwarg (already in origin/main)
