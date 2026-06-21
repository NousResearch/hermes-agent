# deferred/cmx — CMX-touching residual lines

**Disposition:** belongs in the SINGLE CMX-implementation PR per user rule 5 ([id=92873]).
The CMX feature itself ships as PR #50155 (context-engine grounding-enforcement hook).
These 2 patches are CMX-touching residual lines (a conversation_loop hook fragment +
a test that hardcodes a private path) that must NOT be isolated into their own PR —
they fold into the single CMX PR when that work is consolidated.

**Pull:** `git apply deferred/cmx/*.patch` (onto v0.17.0; the impl hook is generic,
the test references a private path so it stays here until the single-CMX-PR call).
