# OuLiPo

*Ouvroir de Littérature Potentielle*, founded 1960 by Raymond Queneau and François Le Lionnais. Members: Perec, Calvino, Roubaud, Mathews, Garréta. "Rats who construct the labyrinth from which they plan to escape" (Queneau). Constraint as generative engine.

## When to use

- Writing — fiction, poetry, copy, lyrics, anything text
- Writing feels samey; constraint suppresses your default sentence shape
- Generating titles, names, taglines (short forms benefit most)
- Software constraint by analogy (code golf, no-dependency, single-file)

## Don't use when

- You want the prose invisible (constraints are usually visible in the result)
- Blocked because you don't know what to say (constraint gives you *how*, not *what*)
- The constraint will compensate for not having a subject (Perec's *La Disparition* works because the missing E is the subject)

## The constraints

### Lipogram
Exclude one or more letters. Perec's *La Disparition* (1969): 300 pages without E. The previous sentence is a lipogram in B, F, J, K, Q, V, Y, Z.

### Univocalism
Only one vowel letter. (Letter, not phoneme — "born" and "cot" both qualify in English.)

### Snowball / Rhopalism
Each line one word; each word one letter longer than the previous.

### S+7 (or N+7)
Replace every noun with the 7th noun after it in a dictionary. "Call me Ishmael. Some years ago..." → "Call me Ishmael. Some yes-men ago..."

Generalizes: V+7, Adj+7, N+k for any k.

### Stile
Each new sentence stems from the last word/phrase of the previous: "I descend the long ladder brings me to the ground floor is spacious..."

### Palindrome
Sonnets, paragraphs, or longer constructed palindromically. Perec wrote a 5,566-letter palindrome.

### Prisoner's constraint (Macao)
Lipogram excluding letters with ascenders or descenders (b, d, f, g, h, j, k, l, p, q, t, y).

### Pilish
Word lengths follow the digits of π: "How I want a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

### Sonnet machine (Queneau)
Fixed structure with interchangeable line-strips. Queneau's *Cent Mille Milliards de Poèmes* (1961): 10 sonnets cut into 14 strips each → 10^14 combinations.

### Antonymy
Replace each word with its antonym. Reveals what the text is *about* by what it would mean if reversed.

## Procedure

### For openings
1. Pick a constraint that fits your domain.
2. Write 200 words under it.
3. Note what the constraint forced you to say.
4. Decide: keep the constraint for the whole piece, or use the opening then unconstrain.

### For unblocking
Apply S+7 to the stuck paragraph. The dislocation surfaces what the original was about.

### Software analogues
- Lipogram → no `e` in identifiers
- N+7 → replace each function with the 7th in a library; describe what the result does
- Snowball → each commit one line longer
- Univocalism → variable names use one vowel
- Pilish → comment word counts follow π

## Anti-slop notes

- Constrained-without-subject = exercise, not work. *La Disparition* works because the missing E *is* the subject.
- Apply strictly. Half-constrained is worse than unconstrained.
- Don't fake "Calvino-style" surface qualities. Use the actual constraints.
- Acrostics are not OuLiPo (centuries older). Use a real constraint or call an acrostic an acrostic.
