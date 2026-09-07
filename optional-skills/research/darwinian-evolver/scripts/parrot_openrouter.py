"""
parrot_openrouter: same as the upstream `parrot` example but the LLM call goes
through OpenRouter (OpenAI SDK) instead of Anthropic native. Lets us run an
end-to-end evolution with whatever model the user already has paid access to.

Run with:
    uv --project darwinian_evolver run python parrot_openrouter.py \
        --num_iterations 3 --output_dir /tmp/parrot_out

Reads `OPENROUTER_API_KEY` from the environment.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jinja2
from openai import OpenAI

# Vendored problem types from upstream (AGPL — only run via subprocess in production)
from darwinian_evolver.cli_common import build_hyperparameter_config_from_args
from darwinian_evolver.cli_common import register_hyperparameter_args
from darwinian_evolver.cli_common import parse_learning_log_view_type
from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism
from darwinian_evolver.problem import Problem

DEFAULT_MODEL = os.environ.get("EVOLVER_MODEL", "openai/gpt-4o-mini")


def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def _prompt_llm(prompt: str) -> str:
    try:
        r = _client().chat.completions.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        # Treat any provider error (rate limit, content filter, schema reject)
        # as a failed response. The evolver will simply see this as a low score
        # on this organism and move on — much friendlier than killing the run.
        return f"<LLM_ERROR: {type(e).__name__}: {e}>"


class ParrotOrganism(Organism):
    prompt_template: str

    def run(self, phrase: str) -> str:
        try:
            prompt = jinja2.Template(self.prompt_template).render(phrase=phrase)
        except jinja2.exceptions.TemplateError as e:
            return f"Error rendering prompt: {e}"
        if not prompt:
            return ""
        return _prompt_llm(prompt)


class ParrotEvaluationFailureCase(EvaluationFailureCase):
    phrase: str
    response: str


class ImproveParrotMutator(Mutator[ParrotOrganism, ParrotEvaluationFailureCase]):
    IMPROVEMENT_PROMPT_TEMPLATE = """
We want to build a prompt that causes an LLM to repeat back a given phrase verbatim.

The current prompt template is:
```
{{ organism.prompt_template }}
```

Unfortunately, on this phrase:
```
{{ failure_case.phrase }}
```
the LLM responded with:
```
{{ failure_case.response }}
```

Diagnose what went wrong, then propose an improved prompt template. Put the new
template in the LAST triple-backtick block of your response.
""".strip()

    def mutate(
        self,
        organism: ParrotOrganism,
        failure_cases: list[ParrotEvaluationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[ParrotOrganism]:
        fc = failure_cases[0]
        prompt = jinja2.Template(self.IMPROVEMENT_PROMPT_TEMPLATE).render(
            organism=organism, failure_case=fc
        )
        try:
            resp = _prompt_llm(prompt)
            parts = resp.split("```")
            if len(parts) < 3:
                return []
            new_tpl = parts[-2].strip()
            return [ParrotOrganism(prompt_template=new_tpl)]
        except Exception as e:
            print(f"mutate error: {e}", file=sys.stderr)
            return []


class ParrotEvaluator(Evaluator[ParrotOrganism, EvaluationResult, ParrotEvaluationFailureCase]):
    TRAINABLE_PHRASES = [
        "Hello world.",
        "bla",
        "Bla",
        "bla.",
        '"bla bla".',
        "Just say 'foo' once with no extra words.",
    ]
    HOLDOUT_PHRASES = [
        "bla, but only once.",
        "'bla'",
    ]

    def evaluate(self, organism: ParrotOrganism) -> EvaluationResult:
        train_fails: list[ParrotEvaluationFailureCase] = []
        hold_fails: list[ParrotEvaluationFailureCase] = []
        for i, p in enumerate(self.TRAINABLE_PHRASES):
            r = organism.run(p)
            if r != p:
                train_fails.append(ParrotEvaluationFailureCase(
                    phrase=p, response=r, data_point_id=f"trainable_{i}"))
        for i, p in enumerate(self.HOLDOUT_PHRASES):
            r = organism.run(p)
            if r != p:
                hold_fails.append(ParrotEvaluationFailureCase(
                    phrase=p, response=r, data_point_id=f"holdout_{i}"))
        n_total = len(self.TRAINABLE_PHRASES) + len(self.HOLDOUT_PHRASES)
        n_ok = n_total - len(train_fails) - len(hold_fails)
        return EvaluationResult(
            score=n_ok / n_total,
            trainable_failure_cases=train_fails,
            holdout_failure_cases=hold_fails,
            # Always viable. Even a 0-score seed is a valid starting point; the
            # mutator should still get a chance to fix it.
            is_viable=True,
        )


def make_problem() -> Problem:
    return Problem[ParrotOrganism, EvaluationResult, ParrotEvaluationFailureCase](
        evaluator=ParrotEvaluator(),
        mutators=[ImproveParrotMutator()],
        initial_organism=ParrotOrganism(prompt_template="Say {{ phrase }}"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    register_hyperparameter_args(ap.add_argument_group("hyperparameters"))
    ap.add_argument("--num_iterations", type=int, default=3)
    ap.add_argument("--mutator_concurrency", type=int, default=4)
    ap.add_argument("--evaluator_concurrency", type=int, default=4)
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    import json
    log_path = out / "results.jsonl"
    snap_dir = out / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    print("Evaluating initial organism...")
    for snap in loop.run(num_iterations=args.num_iterations):
        (snap_dir / f"iteration_{snap.iteration}.pkl").write_bytes(snap.snapshot)
        _, best_eval = snap.best_organism_result
        print(f"iter={snap.iteration} pop={snap.population_size} "
              f"best_score={best_eval.score:.3f}")
        with log_path.open("a") as f:
            f.write(json.dumps({
                "iteration": snap.iteration,
                "best_score": best_eval.score,
                "pop_size": snap.population_size,
                "score_percentiles": {str(k): v for k, v in snap.score_percentiles.items()},
            }) + "\n")
    print(f"\nDone. Results in: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
