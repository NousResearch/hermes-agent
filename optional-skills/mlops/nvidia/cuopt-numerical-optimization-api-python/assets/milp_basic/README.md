# Minimal MILP

Basic mixed-integer program: integer variables with bounds, linear constraints.

**Problem:** Maximize 5x + 3y subject to 2x + 4y ≥ 230, 3x + 2y ≤ 190, 10 ≤ y ≤ 50, x, y integer.

- **model.py** — solve and print solution.
- **incumbent_callback.py** — same problem with a callback that prints intermediate (incumbent) solutions during solve.

**Run:** `python model.py` or `python incumbent_callback.py`
