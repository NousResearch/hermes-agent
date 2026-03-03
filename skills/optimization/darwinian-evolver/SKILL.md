---
name: darwinian-evolver
description: Use LLM-driven evolutionary optimization to improve code, prompts, and algorithms. Achieves 2-3x performance improvements by evolving populations of solutions through mutation and selection.
version: 1.0.0
metadata:
  hermes:
    tags: [optimization, evolution, prompts, code, llm]
    category: optimization
---

# Darwinian Evolver — Evolutionary Code & Prompt Optimization

## When to Use
Use this skill when you need to:
- Optimize a prompt for better LLM performance
- Evolve code to improve accuracy or efficiency
- Discover better algorithms through iterative search
- Achieve significant performance gains (2-3x) over one-shot generation

Do NOT use for simple one-shot tasks — evolution is expensive and designed for hard optimization problems.

## Setup

### 1. Install the Darwinian Evolver
```bash
uv pip install darwinian-evolver
# or
pip install darwinian-evolver
```

### 2. Verify installation
```bash
python3 -c "import darwinian_evolver; print('OK')"
```

### 3. Set your LLM API key
```bash
export ANTHROPIC_API_KEY=your_key   # for Claude
export OPENAI_API_KEY=your_key      # for OpenAI
export GOOGLE_API_KEY=your_key      # for Gemini
```

## Core Concepts

- **Organism** — the thing being evolved (a prompt, a function, an algorithm)
- **Evaluator** — scores each organism and identifies failure cases
- **Mutator** — LLM-powered agent that proposes improvements based on failures
- **Population** — collection of organisms tracked across generations

## Procedure

### Prompt Optimization

1. Create a problem file `problem.py`:
```python
from darwinian_evolver import Organism, Evaluator, Mutator, EvaluationResult, EvolveProblemLoop

class PromptOrganism(Organism):
    prompt: str

class MyEvaluator(Evaluator):
    def evaluate(self, organism):
        # Test the prompt and return a score 0-1
        score = test_your_prompt(organism.prompt)
        return EvaluationResult(score=score, trainable_failure_cases=[])

class MyMutator(Mutator):
    def mutate(self, organism, failure_cases, learning_log):
        # LLM improves the prompt based on failures
        improved = call_llm_to_improve(organism.prompt, failure_cases)
        return [PromptOrganism(prompt=improved)]

# Run evolution
loop = EvolveProblemLoop(
    evaluator=MyEvaluator(),
    mutator=MyMutator(),
    initial_organism=PromptOrganism(prompt="Your starting prompt here"),
    population_size=10,
    n_mutations_per_iteration=3,
)
loop.run(n_iterations=20)
print(f"Best score: {loop.best_organism.score}")
print(f"Best prompt: {loop.best_organism.prompt}")
```

2. Run evolution:
```bash
python3 problem.py
```

3. Monitor progress — the evolver prints generation stats and saves snapshots automatically.

### Code Evolution (using GitBasedOrganism)
```python
from darwinian_evolver import GitBasedOrganism

class CodeOrganism(GitBasedOrganism):
    # Evolves code tracked in a Git repo
    repo_path: str = "./my_project"
    entry_file: str = "solution.py"
```

### Resume an interrupted run
```python
loop = EvolveProblemLoop.from_snapshot("snapshots/latest.pkl")
loop.run(n_iterations=10)
```

## Pitfalls
- **AGPL v3 license** — use as external CLI tool only, do not import into MIT-licensed projects
- **Cost** — each iteration consumes API credits ($2-9/task for complex problems), set a budget limit
- **Bad evaluators** → bad evolution — spend time defining a good scoring function
- **Overfitting** — use separate train/test datasets to avoid evolving to specific failure cases
- **Snapshots are pickle files** — not portable across Python versions

## Verification
After setup, run a quick sanity check:
```bash
python3 -c "
from darwinian_evolver import Organism
print('Darwinian Evolver ready')
"
```

## References
- GitHub: https://github.com/imbue-ai/darwinian_evolver
- Blog post: https://imbue.com/research/2026-02-27-darwinian-evolver/
- ARC-AGI-2 results: https://imbue.com/research/2026-02-27-arc-agi-2-evolution/
