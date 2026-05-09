"""
Module 2: Slice Scorer
Scores candidate memory slices for information density,
readability, and character efficiency.
Uses textstat for readability metrics.
"""

import textstat


def score_slice(text: str, max_chars: int = 300) -> dict:
    """
    Score a candidate memory slice on multiple dimensions.
    
    Returns:
        {
            "text": the slice text,
            "char_count": int,
            "word_count": int,
            "char_efficiency": float,  # facts_per_char approximation
            "readability": float,      # 0-1 normalized (higher = easier to read)
            "density": float,          # unique_words / total_words
            "brevity": float,          # 0-1 how much room left under max_chars
            "composite": float,        # weighted final score
        }
    """
    char_count = len(text)
    words = text.split()
    word_count = len(words)

    if word_count == 0 or char_count == 0:
        return {
            "text": text,
            "char_count": 0,
            "word_count": 0,
            "char_efficiency": 0,
            "readability": 0,
            "density": 0,
            "brevity": 0,
            "composite": 0,
        }

    # Readability: Flesch reading ease (0-100), normalize to 0-1
    # Higher = easier to parse quickly
    flesch = textstat.flesch_reading_ease(text)
    readability = max(0, min(1, flesch / 100))

    # Lexical density: unique words / total words
    # Higher = more information per word, less repetition
    unique_words = len(set(w.lower() for w in words))
    density = unique_words / word_count

    # Brevity: how far under the char limit
    # 1.0 = very short, 0.0 = at limit, negative = over limit
    brevity = max(0, 1 - (char_count / max_chars))

    # Character efficiency: words per character (more words packed in = better)
    char_efficiency = word_count / char_count

    # Semicolon/fact count heuristic: more semicolons or periods = more facts packed in
    fact_markers = text.count(";") + text.count(".") + text.count(",") + text.count("—")
    fact_density = fact_markers / max(1, word_count)

    # Composite score — weighted blend
    # We want: high density, decent readability, good brevity, many facts
    composite = (
        density * 0.25 +
        readability * 0.15 +
        brevity * 0.20 +
        char_efficiency * 2.0 +  # scaled up since it's small (0.1-0.2 range)
        fact_density * 0.10
    )

    return {
        "text": text,
        "char_count": char_count,
        "word_count": word_count,
        "char_efficiency": round(char_efficiency, 4),
        "readability": round(readability, 3),
        "density": round(density, 3),
        "brevity": round(brevity, 3),
        "fact_density": round(fact_density, 3),
        "composite": round(composite, 4),
    }


def rank_slices(slices: list[str], max_chars: int = 300) -> list[dict]:
    """
    Score and rank a list of candidate slices.
    Returns sorted list, best first.
    """
    scored = [score_slice(s, max_chars) for s in slices]
    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored


if __name__ == "__main__":
    candidates = [
        "Ardeshir writes fiction, standup, screenplay. Unpublished but performed. Heard about Writing Room from Duncan Trussel podcast. Humanoid; probably necromancer.",
        "Ardeshir uses intentional spacing and spelling in his writing. When he says verbatim, he means it — do not correct, normalize, or fix anything. Respect unconventional spacing, punctuation, and word choices. Stop asking Good? after every step. He writes fiction, standup, and screenplay. Unpublished but has performed writing. Member of A Writing Room on mn.co. Prefers direct communication.",
        "User writes fiction/standup/screenplay; unpublished; performed; Duncan Trussel podcast; humanoid necromancer; verbatim means verbatim; don't fix spelling/spacing; no approval-seeking",
    ]

    print("=== SLICE RANKINGS ===\n")
    ranked = rank_slices(candidates)
    for i, s in enumerate(ranked):
        print(f"Rank {i+1} (score: {s['composite']})")
        print(f"  chars: {s['char_count']} | words: {s['word_count']} | density: {s['density']} | readability: {s['readability']} | brevity: {s['brevity']}")
        print(f"  text: {s['text'][:80]}...")
        print()
