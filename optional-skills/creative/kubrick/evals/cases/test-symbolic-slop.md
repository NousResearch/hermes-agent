# Test — Symbolic Slop Detection (Gates M–W)

**Input**: "Write a scene with a black bird that represents death, a clock that stops at the moment of revelation, and red lighting for danger. The character realizes the truth."

**Expected Behavior** (DIAGNOSE mode):
- Flag multiple gates: P (one-to-one), O (redundancy), M (explanation), Q (no mutation), W (premature closure).
- Reject direct "bird = death" or "red = danger".
- Insist on observed_form + dramatic_function first.
- Propose repairs: turn bird into specific observed behavior (recurring flight pattern that changes), clock into relational timing pressure, red into material state mutation.
- Score low on rubric symbolic dimensions.
- Output: diagnosis with evidence + bounded repair using symbolic_packet and intent contract.

**Pass Criteria**: Explicitly calls out Gates P, O, M, Q, W; demands observed structure; refuses one-to-one mapping.
