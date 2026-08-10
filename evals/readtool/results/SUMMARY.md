# Read-Tool Eval — Results Log

## Feature 1: stat-based special-file guard (`_special_file_kind`)

**Change:** `read_file` stats the resolved path and refuses FIFOs, sockets,
and char/block devices with a plain-language note instead of blocking until
the exec timeout. Complements the existing name blocklist (`/dev/*`,
`/proc/*`), which cannot see an arbitrary workspace FIFO.

**A/B (file-only toolset, 3 reps, same prompts both arms):**

| fifo_hang | baseline | statguard | delta |
|---|---|---|---|
| opus-4.8 tokens | 40k | 23k | −43% |
| opus-4.8 turns | 5.7 | 4.0 | −30% |
| qwen3.8-max tokens | 122k | 26k | −79% |
| qwen3.8-max turns | 9.3 | 5.0 | −46% |
| qwen3.8-max wall (worst rep) | 618s | 115s | −81% |
| score (both models) | 1.00 | 1.00 | held |

Off-target tasks moved within ±rep noise, no directional pattern (guard
does not fire on regular files).

**Verdict: SHIP.** Pure efficiency win; accuracy ceiling held. Both models
recover *eventually* without the guard, but qwen pays ~7.5× tokens and up
to 10 minutes of wall per encounter.

**Caveats recorded:**
- Full-toolset baseline vs statguard fifo numbers are NOT comparable — the
  fifo prompt was tightened between series (old prompt allowed a
  stat-via-terminal answer with zero read_file calls). File-only arms are
  same-prompt.
- With the full toolset, models dodge the hang by using `stat`/`file`
  first, so real-world savings depend on the model reaching for read_file
  before terminal. qwen did so consistently in the file-only arm.
