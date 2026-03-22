# feat: ANAMNESIS_ENGRAM — learn about yourself by teaching user data about you

## What does this PR do?

An AI agent asked to improve its own memory built a tool to do exactly that.

**ANAMNESIS_ENGRAM** is an evolutionary memory optimization system for Hermes Agent. It takes raw conversation data — the sprawling, messy transcript of a human-agent session — and compresses it into optimally dense memory slices using NLP extraction and genetic algorithms. Think of it as `gzip` for conversation memory, except it actually *understands* what it's compressing.

The pipeline: **spaCy** extracts facts from conversation text → **textstat** scores candidate slices for information density, readability, and character efficiency → **DEAP** evolves populations of 60 candidate slices across 50 generations, selecting for maximum facts packed into minimum characters.

The result: memory slices that are Pareto-optimal — maximum information in minimum tokens. No fluff, no redundancy, no wasted characters.

**The meta angle:** this tool was built *during a live Hermes Agent session* where a user asked the agent to improve its own memory system. The agent researched spaCy, textstat, and DEAP, then wrote and tested all four modules in a single session. An agent building its own memory optimizer is exactly the kind of self-improvement loop Hermes Agent was designed to enable.

### Relationship to `hermes-agent-self-evolution`

NousResearch's [hermes-agent-self-evolution](https://github.com/NousResearch/hermes-agent-self-evolution) repo explores agent self-improvement via DSPy + GEPA. Memory Slicer complements that work from a different angle:

| | hermes-agent-self-evolution | Memory Slicer |
|---|---|---|
| **Optimization target** | Agent behavior/prompts | Memory/context compression |
| **Evolutionary framework** | GEPA (DSPy) | DEAP genetic algorithms |
| **Input** | Agent performance traces | Raw conversation text |
| **Output** | Optimized agent configs | Dense memory slices |

Both are evolutionary approaches to agent self-improvement — one optimizes *what the agent does*, the other optimizes *what the agent remembers*.

## Related Issue

Late submission for the Hermes Agent Hackathon (March 8–17, 2026).

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [x] Other (Hackathon submission — self-built agent tool)

## Changes Made

**4 modules, 618 lines of Python:**

- **`digester.py`** (126 lines) — spaCy NLP pipeline. Loads `en_core_web_sm`, performs tokenization, lemmatization, named entity extraction, noun chunk identification, and fact extraction. Identifies candidate facts via entity presence, verb detection, and personal pronoun heuristics. Supports both raw text and structured `{role, content}` turn formats.

- **`scorer.py`** (110 lines) — textstat-based multi-dimensional scoring. Evaluates candidate slices on: Flesch readability (normalized 0–1), lexical density (unique/total words), character efficiency (words/char), brevity (utilization of char budget), and fact density (punctuation-based fact marker heuristic). Produces a weighted composite score.

- **`evolver.py`** (185 lines) — DEAP genetic algorithm engine. Represents each individual as a binary vector over the fact pool (include/exclude). Evaluates fitness via the scorer's composite score plus fact coverage bonuses and space-utilization penalties. Uses two-point crossover (`cxTwoPoint`), bit-flip mutation (`mutFlipBit`, `indpb=0.15`), and tournament selection (`tournsize=3`). Maintains a Hall of Fame of top 10 individuals. Tries multiple separators (`. `, `; `, ` — `, `, `) per individual to find optimal fact joining.

- **`slicer.py`** (197 lines) — CLI runner and pipeline orchestrator. Reads input from file or stdin (raw text or JSON turns), runs digest → deduplicate → evolve → rank. Supports `--max-chars`, `--slices`, `--turns`, and `--verbose` flags. Includes token-overlap deduplication (threshold 0.6) to remove near-duplicate facts before evolution.

- **`test_session.txt`** — Sample conversation data used during development (the actual session transcript that prompted this tool's creation).

## How to Test

1. **Install dependencies:**
   ```bash
   pip install spacy textstat deap
   python -m spacy download en_core_web_sm
   ```

2. **Run on sample data:**
   ```bash
   cd ~/.hermes/skills/memory-slicer
   python3 slicer.py test_session.txt --max-chars 550 --slices 3 --verbose
   ```

3. **Run on piped input:**
   ```bash
   echo "I am a software engineer. I work at Google. I prefer Python over Java. My name is Alex." | python3 slicer.py - --max-chars 200 --slices 2
   ```

4. **Test individual modules:**
   ```bash
   python3 digester.py   # Runs built-in test with sample conversation
   python3 scorer.py     # Scores and ranks 3 sample slices
   python3 evolver.py    # Evolves slices from 11 sample facts
   ```

5. **Verify output format:** Each slice should show a composite score, character count, fact count, and the optimized text. Slices should be dense, readable, and non-redundant.

## Checklist

- [x] I have tested my changes locally
- [x] My code follows the project's coding style
- [x] I have added appropriate documentation
- [x] New dependencies are documented (spacy, textstat, deap)
- [x] I have verified backwards compatibility (standalone tool, no modifications to core agent)

---

**Author:** Ardeshir
**Name:** ANAMNESIS_ENGRAM
**Tagline:** learn about yourself by teaching user data about you
**Built with:** Hermes Agent (the tool was literally built by the agent it's designed to improve)
**Dependencies:** `spacy` (en_core_web_sm), `textstat`, `deap`
**Lines of code:** 618 across 4 modules
