# Pólya's Heuristics

George Pólya, *How to Solve It* (Princeton UP, 1945). Four-phase problem-solving framework + dictionary of heuristic moves. Written for math but applies to any well-defined "find X such that..." problem.

## When to use

- Math, physics, theoretical problems
- Algorithm design, debugging
- Any problem with a clear target (find X such that...)
- Teaching problem-solving

## Don't use when

- Open-ended creative problems with no defined target
- Difficulty is *understanding the problem space*, not solving within it (use dérive or compression-progress first)
- Solution is more about taste than analysis
- Real-world problems where data is incomplete and conditions vague

## The four phases

### 1. Understand the problem
- What is the **unknown**?
- What are the **data**?
- What is the **condition** linking them?
- Is the condition sufficient? Insufficient? Redundant? Contradictory?
- State in your own words.
- Draw a figure. Introduce notation.

This phase is most often skipped. **Most problem-solving failures are upstream of method** — they're failures to understand the problem precisely.

### 2. Devise a plan
Find the connection between data and unknown. Heuristic moves:
- **Have you seen this problem before?** Or in slightly different form?
- **Do you know a related problem?**
- **Look at the unknown** — find a familiar problem with the same or similar unknown.
- **Could you use a related problem's result? Its method?**
- **Restate.**
- If you can't solve the proposed problem, solve a related one:
  - More general
  - More specific
  - Analogous
  - A part of the problem
  - With a condition relaxed
- **Did you use all the data?** All the conditions?

### 3. Carry out the plan
- Can you see clearly that each step is correct?
- Can you prove it?

### 4. Look back
- Check the result. Check the argument.
- Can you derive it differently? See it at a glance?
- Can you use the result, or the method, for some other problem?

The looking-back phase is the *learning* phase — what makes Pólya's method an *educational* method, not just a problem-solving one.

## Key heuristics from the dictionary

- **Decompose and recombine.** Break into parts; solve each; combine.
- **Generalization.** The general case is sometimes easier than the specific because it forces you to identify essential structure.
- **Specialization.** Try the smallest case, the simplest case, the case where one parameter is zero. Look for pattern.
- **Analogy.** Find a related problem with same structure, different surface.
- **Auxiliary problem.** Solve a related problem first; use its result.
- **Working backwards.** Start from the unknown and work back. Forward direction often has too many branches; backward is more constrained.
- **Setting up an equation.** Most word-problem failure is in translation, not algebra.
- **Reductio ad absurdum.** Assume the conclusion is false; derive contradiction.
- **Pattern recognition.** Small cases → conjecture → prove.
- **Symmetry.** Where there's symmetry in the problem, there's usually symmetry in the solution.

## Anti-slop notes

- Reciting the four phases without doing them = slop. The structure is fine; the value is in actually executing each phase.
- Don't pretend you've understood when you haven't. State the unknown, the data, the condition concretely.
- Don't claim "Pólya'd it" without consulting specific heuristics.
- Don't apply to fuzzy problems. Pólya assumes clear problem statements.

Source: Pólya, *How to Solve It* (Princeton UP, 1945; current edition 2014).
