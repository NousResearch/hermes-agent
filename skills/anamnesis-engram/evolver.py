"""
Module 3: GA Slice Optimizer
Uses DEAP to evolve memory slices for maximum information density
within character constraints.

Each individual = a candidate memory slice (list of fact indices to include)
Fitness = composite score from scorer module
"""

import random
from deap import base, creator, tools, algorithms
from scorer import score_slice


def evolve_slices(
    facts: list[str],
    max_chars: int = 300,
    population_size: int = 50,
    generations: int = 40,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.3,
    separators: list[str] = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Evolve optimal memory slices from a pool of candidate facts.
    
    Args:
        facts: list of candidate fact strings
        max_chars: character limit per slice
        population_size: GA population size
        generations: number of GA generations
        crossover_prob: crossover probability
        mutation_prob: mutation probability per bit
        separators: list of separators to try between facts [". ", "; ", " — ", ", "]
        verbose: print generation stats
    
    Returns:
        Top 5 evolved slices as scored dicts
    """
    if not facts:
        return []

    if separators is None:
        separators = [". ", "; ", " — ", ", "]

    n_facts = len(facts)

    # Create fitness and individual classes
    # Using try/except to handle re-creation in same process
    try:
        creator.create("SliceFitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.SliceFitness)
    except RuntimeError:
        pass

    toolbox = base.Toolbox()

    # Each gene = 0 or 1 (include this fact or not)
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_facts,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        """Build a slice from selected facts, score it."""
        selected = [facts[i] for i, bit in enumerate(individual) if bit == 1]
        if not selected:
            return (0.0,)

        # Try each separator, pick best
        best_score = 0
        best_text = ""
        for sep in separators:
            text = sep.join(selected)
            if len(text) > max_chars:
                # Trim facts until it fits
                trimmed = list(selected)
                while len(sep.join(trimmed)) > max_chars and trimmed:
                    trimmed.pop()
                text = sep.join(trimmed)
                if not trimmed:
                    continue

            result = score_slice(text, max_chars)
            if result["composite"] > best_score:
                best_score = result["composite"]
                best_text = text

        if not best_text:
            return (0.0,)

        # Heavy bonus for including more facts — we want dense, not short
        fact_coverage = len(selected) / n_facts
        final_score = best_score + fact_coverage * 0.5

        # Penalty for being too short — we want to USE the space
        utilization = len(best_text) / max_chars
        if utilization < 0.3:
            final_score *= 0.5  # punish underuse of space
        elif utilization < 0.5:
            final_score *= 0.75

        return (final_score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run evolution
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(v[0] for v in x) / len(x))
    stats.register("max", lambda x: max(v[0] for v in x))

    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=generations,
        stats=stats if verbose else None,
        halloffame=hof,
        verbose=verbose,
    )

    # Build final slices from hall of fame
    results = []
    seen = set()
    for ind in hof:
        selected = [facts[i] for i, bit in enumerate(ind) if bit == 1]
        if not selected:
            continue

        # Find best separator
        best_text = ""
        best_score = 0
        for sep in separators:
            text = sep.join(selected)
            if len(text) <= max_chars:
                result = score_slice(text, max_chars)
                if result["composite"] > best_score:
                    best_score = result["composite"]
                    best_text = text

        if best_text and best_text not in seen:
            seen.add(best_text)
            scored = score_slice(best_text, max_chars)
            scored["facts_used"] = selected
            scored["facts_count"] = len(selected)
            results.append(scored)

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results[:5]


if __name__ == "__main__":
    facts = [
        "Ardeshir writes fiction, standup, screenplay",
        "Unpublished but has performed writing",
        "Biggest challenge: time and unwillingness to commit",
        "The free nature of writing can be exhausting",
        "Humanoid; probably a necromancer",
        "Heard about Writing Room from Duncan Trussel podcast",
        "Verbatim means verbatim — don't correct spelling or spacing",
        "Uses intentional spacing and word choices",
        "Don't ask for approval after every step",
        "Prefers direct communication",
        "Member of A Writing Room on mn.co",
    ]

    print("Evolving optimal memory slices...\n")
    results = evolve_slices(facts, max_chars=550, verbose=True)
    
    print("\n=== TOP EVOLVED SLICES ===\n")
    for i, r in enumerate(results):
        print(f"Rank {i+1} (score: {r['composite']}, chars: {r['char_count']}, facts: {r['facts_count']})")
        print(f"  {r['text']}")
        print()
