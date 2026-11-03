# Autoreason Experimental Results

Quick reference for the results reported in
[NousResearch/autoreason](https://github.com/NousResearch/autoreason).

## Code scaling

- **Haiku 3.5**: Autoreason reaches ~40% private-test accuracy on CodeContests.
- **Haiku 4.5**: ~60% private-test accuracy; this is the transition point where
  the held-out gain from Autoreason vanishes.
- **Sonnet 4**: ~64% private-test accuracy with Autoreason.
- **Sonnet 4.6**: ~77% private-test accuracy with Autoreason.
- **Sonnet 4.6**: 77% Autoreason vs 73% single-pass on 150 CodeContests
  problems.
- **Haiku 3.5**: 40% Autoreason vs 31% best-of-6 sampling at matched compute
  (150 problems).

## Writing / ablations

- **Haiku 3.5 + Autoreason**: 42/42 perfect Borda across 3 tasks; all
  baselines degraded below single-pass.
- **Judge count**: 7 judges converge about 3× faster than 3 judges; 1 judge
  is noisy and slow.
- **Component necessity**: removing either B or AB collapses the tournament /
  causes premature convergence.
- **Design choices**: incumbent A as a first-class option and a conservative
  A-favored tiebreak are central to the method.
