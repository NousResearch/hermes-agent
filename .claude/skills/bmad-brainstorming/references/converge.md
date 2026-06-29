# Converging: Narrow & Decide

Load this when divergence is spent and the user wants to narrow the field — or asks to "decide," "prioritize," "pick," or "make it real." The whole catalog is *divergent* by design (it generates); this is the deliberate opposite phase, and keeping the two apart is the point. Never run convergence while ideas are still flowing, and never let it leak into a generating batch — premature judgment is what kills good ideas. `{doc_workspace}/.memlog.md` is the canonical record; everything here works from it. Communicate in `{communication_language}`.

**Mode holds.** In **Facilitator** you run the convergence *on the user's verdicts* — you structure and prompt, they judge; never rank for them. In **Creative Partner** you weigh in too, each call logged by author. In **Ideate for me** you converge yourself and show the result, then offer to keep going.

## How to run it

First, reflect the field back: pull the live candidates from the memlog (include the odd and buried ones, not just the recent obvious ones) so there's a concrete set to work on. Then pick **one** convergence move that fits the goal — don't hand the user a menu of methods; choose the one that suits *this* decision and name it. Run it to a result, log the outcome, and stop when a clear short-list or single direction emerges.

Pick by what the decision needs:

- **Affinity Clustering** — when there are many scattered ideas: group them into themes, name each cluster, and surface the through-line. Often the right *first* move, to turn a pile into a handful.
- **Impact–Effort** — when the goal is action: place each candidate on impact vs effort; harvest high-impact / low-effort first, park the rest.
- **NUF Test** — when novelty matters: score each New, Useful, Feasible (1–10 each); the totals expose the quiet winners and the dazzling-but-doomed.
- **Forced Ranking / Dot Vote** — when you just need a ranked top-N: make the ideas compete, no ties; (a literal dot-vote when it's genuinely a group).
- **PMI (Plus / Minus / Interesting)** — when one strong candidate needs pressure-testing before commitment: list its pluses, minuses, and the merely-interesting, then judge.
- **MoSCoW** — when scoping a build: sort into Must / Should / Could / Won't-this-time.

Log the surviving directions and the reasoning with `uv run {project-root}/_bmad/scripts/memlog.py append --workspace {doc_workspace} --type decision --text "<one-line gist>"` (use `--by` in Creative Partner mode). Two or three convergence moves chained is fine (e.g. cluster → score the clusters); more than that is usually over-processing.

## Then finalize

Once a short-list or direction is settled, **load `references/finalize.md`** and run it last — synthesis, `status: complete`, and artifacts build on the decisions you just logged. Convergence narrows; finalize captures and ships. Do not set `status: complete` here — that belongs to finalize.
