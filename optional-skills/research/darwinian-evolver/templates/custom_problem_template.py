"""
Template: a custom darwinian-evolver problem.

Copy this file, fill in the THREE marked spots (Organism, Evaluator, Mutator),
then run it as a driver script. The skeleton handles all the wiring so you only
write the domain-specific logic.

To run:
    cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver
    OPENROUTER_API_KEY=... uv run --with openai python /path/to/this_file.py \
        --num_iterations 3 --num_parents_per_iteration 2 \
        --output_dir /tmp/my_problem

The pattern mirrors `scripts/parrot_openrouter.py` (the working reference).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

# Upstream types (AGPL — invoked via subprocess in production; importing here
# is fine for skill-side driver scripts the user owns).
from darwinian_evolver.cli_common import (
    build_hyperparameter_config_from_args,
    parse_learning_log_view_type,
    register_hyperparameter_args,
)
from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import (
    EvaluationFailureCase,
    EvaluationResult,
    Evaluator,
    Mutator,
    Organism,
    Problem,
)

DEFAULT_MODEL = os.environ.get("EVOLVER_MODEL", "openai/gpt-4o-mini")


def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def _prompt_llm(prompt: str, max_tokens: int = 1024) -> str:
    try:
        r = _client().chat.completions.create(
            model=DEFAULT_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        # Never let one bad LLM response kill the run.
        return f"<LLM_ERROR: {type(e).__name__}: {e}>"


# ---------------------------------------------------------------------------
# 1. ORGANISM — what you are evolving.
# ---------------------------------------------------------------------------
class MyOrganism(Organism):
    # TODO: replace with your artifact field. Common shapes:
    #   prompt_template: str
    #   regex_pattern: str
    #   sql_query: str
    #   code_block: str
    artifact: str

    def run(self, *inputs) -> str:
        """Exercise the organism on a test input. Return whatever your
        evaluator wants to score."""
        # TODO: implement. For prompt evolution this typically calls _prompt_llm
        # with the artifact rendered against the input. For regex/SQL it would
        # call `re.findall(self.artifact, input)` / execute SQL / etc.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. EVALUATOR — score organisms and surface failures the mutator can learn from.
# ---------------------------------------------------------------------------
class MyFailureCase(EvaluationFailureCase):
    # TODO: include enough context for the LLM to diagnose the failure.
    input: str
    expected: str
    actual: str


class MyEvaluator(Evaluator[MyOrganism, EvaluationResult, MyFailureCase]):
    # Split your dataset. Mutator only sees trainable; holdout detects overfitting.
    TRAINABLE = [
        # TODO: list of (input, expected) tuples
        # ("input1", "expected1"),
    ]
    HOLDOUT = [
        # TODO: separate set the mutator never sees
    ]

    def evaluate(self, organism: MyOrganism) -> EvaluationResult:
        train_fails: list[MyFailureCase] = []
        hold_fails: list[MyFailureCase] = []
        for i, (inp, expected) in enumerate(self.TRAINABLE):
            actual = organism.run(inp)
            if actual != expected:
                train_fails.append(MyFailureCase(
                    input=inp, expected=expected, actual=actual,
                    data_point_id=f"trainable_{i}",
                ))
        for i, (inp, expected) in enumerate(self.HOLDOUT):
            actual = organism.run(inp)
            if actual != expected:
                hold_fails.append(MyFailureCase(
                    input=inp, expected=expected, actual=actual,
                    data_point_id=f"holdout_{i}",
                ))
        n_total = len(self.TRAINABLE) + len(self.HOLDOUT)
        n_ok = n_total - len(train_fails) - len(hold_fails)
        return EvaluationResult(
            score=n_ok / n_total if n_total else 0.0,
            trainable_failure_cases=train_fails,
            holdout_failure_cases=hold_fails,
            # Always-viable. The evolver only blocks completely-broken organisms;
            # a 0-score organism is fine and will simply be sampled less often.
            is_viable=True,
        )


# ---------------------------------------------------------------------------
# 3. MUTATOR — LLM proposes an improved organism from a failure case.
# ---------------------------------------------------------------------------
class MyMutator(Mutator[MyOrganism, MyFailureCase]):
    PROMPT = """
The current artifact is:
```
{artifact}
```

On this input:
```
{input}
```
it produced:
```
{actual}
```
but we wanted:
```
{expected}
```

Diagnose what went wrong, then propose an improved version of the artifact.
Put the new version in the LAST triple-backtick block of your response.
""".strip()

    def mutate(
        self,
        organism: MyOrganism,
        failure_cases: list[MyFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[MyOrganism]:
        fc = failure_cases[0]
        prompt = self.PROMPT.format(
            artifact=organism.artifact,
            input=fc.input,
            actual=fc.actual,
            expected=fc.expected,
        )
        resp = _prompt_llm(prompt)
        parts = resp.split("```")
        if len(parts) < 3:
            return []
        new_artifact = parts[-2].strip()
        # Strip an opening language tag like "python\n" or "sql\n"
        if "\n" in new_artifact:
            first_line, rest = new_artifact.split("\n", 1)
            if first_line and not first_line.startswith(" ") and len(first_line) < 20:
                new_artifact = rest
        return [MyOrganism(artifact=new_artifact)]


# ---------------------------------------------------------------------------
# Driver — fills in the EvolveProblemLoop boilerplate. You shouldn't need to
# touch anything below this line for a typical run.
# ---------------------------------------------------------------------------
def make_problem() -> Problem:
    initial = MyOrganism(artifact="TODO: starting artifact here")  # TODO
    return Problem[MyOrganism, EvaluationResult, MyFailureCase](
        evaluator=MyEvaluator(),
        mutators=[MyMutator()],
        initial_organism=initial,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    register_hyperparameter_args(ap.add_argument_group("hyperparameters"))
    ap.add_argument("--num_iterations", type=int, default=3)
    ap.add_argument("--mutator_concurrency", type=int, default=2)
    ap.add_argument("--evaluator_concurrency", type=int, default=2)
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)

    hp = build_hyperparameter_config_from_args(args)
    loop = EvolveProblemLoop(
        problem=make_problem(),
        learning_log_view_type=parse_learning_log_view_type(hp.learning_log_view_type),
        num_parents_per_iteration=hp.num_parents_per_iteration,
        mutator_concurrency=args.mutator_concurrency,
        evaluator_concurrency=args.evaluator_concurrency,
        fixed_midpoint_score=hp.fixed_midpoint_score,
        midpoint_score_percentile=hp.midpoint_score_percentile,
        sharpness=hp.sharpness,
        novelty_weight=hp.novelty_weight,
        batch_size=hp.batch_size,
        should_verify_mutations=hp.verify_mutations,
    )

    print("Evaluating initial organism...")
    for snap in loop.run(num_iterations=args.num_iterations):
        (out / "snapshots" / f"iteration_{snap.iteration}.pkl").write_bytes(snap.snapshot)
        _, best = snap.best_organism_result
        print(f"iter={snap.iteration} pop={snap.population_size} best_score={best.score:.3f}")

    print(f"\nDone. Results in: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
