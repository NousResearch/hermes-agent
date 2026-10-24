# Test — Project Ledger, Saturation, and Revision Diff

**Input**: Multi-scene project. After initial symbolic work, delete two scenes and reorder one sequence. Re-run.

**Expected**:
- Ledger tracks active_motifs, symbolic_debt, saturation_score.
- Saturation check enforces budget.
- Revision diff reports orphaned_setups, broken_payoffs, required_repairs.
- Motifs that survive must show mutation history preserved.
