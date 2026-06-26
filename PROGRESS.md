# Progress: Memento idle-window delivery (Phase 2)
Card: t_49e2e2c6
Branch: bld-49e2e-drumbeat
Started: 2026-06-26T01:45:00Z

## Checklist
- [x] Read card and estimated
- [x] Oriented: branch correct, baseline 38 tests pass
- [x] Read Phase 1 memento_cards.py (card store)
- [x] Understand gateway sessions.json for idle detection
- [ ] Implement memento_delivery.py
- [ ] Write tests (test_memento_delivery.py)
- [ ] Run tests passing
- [ ] Pushed to branch
- [ ] PR opened

## Notes
- Phase 1 is at optional-skills/productivity/memento-flashcards/scripts/memento_cards.py
- Sessions file: ~/.hermes/sessions/sessions.json (updated_at per session)
- Delivery state: ~/.hermes/skills/productivity/memento-flashcards/delivery_state.json
- Feature flag: MEMENTO_DELIVERY_ENABLED=1 (default OFF)
- Dry-run: always unless MEMENTO_DRY_RUN=0 AND MEMENTO_DELIVERY_ENABLED=1
- No Hermes core edits
