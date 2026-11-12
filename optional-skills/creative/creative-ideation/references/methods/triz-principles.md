# TRIZ — Theory of Inventive Problem Solving

Genrich Altshuller, 1946–. Soviet engineering invention method derived from analysis of hundreds of thousands of patents. 40 inventive principles + contradiction matrix + Ideal Final Result. Used by Samsung, Intel, Boeing, P&G.

## Core principle

Most inventive problems are technical contradictions: improving X degrades Y. The trade-off is usually an artifact of how the system is decomposed, not a fundamental constraint. Solve by identifying the contradiction explicitly, then applying principles that have historically resolved similar contradictions in patent literature.

The **Ideal Final Result**: the desired function performed without the system that performs it (the system has, in some sense, eliminated itself). Use as target.

## When to use

- Engineering / mechanism / device invention
- Measurable parameter conflict (mass/strength, cost/reliability, speed/accuracy)
- You suspect the trade-off is fake
- Group brainstorming with non-arbitrary structure

## Don't use when

- Artistic, social, or expressive problems (TRIZ requires measurable parameters)
- Your "contradiction" is preference, not parameter ("modern but classic" is not TRIZ)
- A textbook fix exists; TRIZ is for inventive problems

## The 40 inventive principles

1. **Segmentation** — divide into independent parts, increase divisibility
2. **Taking out** — extract the disturbing part; separate only what's needed
3. **Local quality** — make different parts have different properties
4. **Asymmetry** — replace symmetrical with asymmetrical
5. **Merging** — bring identical/similar objects closer; parallelize operations
6. **Universality** — one part performs multiple functions
7. **Nested doll** — place objects one inside another (matryoshka)
8. **Anti-weight** — compensate weight by combining with lift / hydro/aerodynamic forces
9. **Preliminary anti-action** — preload with opposite stress
10. **Preliminary action** — perform required action in advance
11. **Beforehand cushioning** — emergency means in advance
12. **Equipotentiality** — change conditions so object need not be raised/lowered
13. **The other way round** — invert action; movable parts fixed and vice versa
14. **Spheroidality / curvature** — replace linear with curved; flat with spherical
15. **Dynamics** — make rigid moveable; let parts shift configuration
16. **Partial or excessive actions** — slightly less or slightly more if 100% is hard
17. **Another dimension** — move 1D→2D→3D; tilt; use the other side
18. **Mechanical vibration** — oscillate, ultrasonics
19. **Periodic action** — periodic instead of continuous; vary frequency; pauses
20. **Continuity of useful action** — eliminate idle running
21. **Skipping** — perform fast through dangerous stages
22. **Blessing in disguise** — use harmful factors to obtain a positive effect
23. **Feedback** — introduce or modify feedback
24. **Intermediary** — use an intermediary article or process
25. **Self-service** — make the object service itself; use waste resources
26. **Copying** — cheap copies instead of fragile/expensive originals
27. **Cheap short-living** — disposable instead of durable
28. **Mechanics substitution** — replace mechanical with sensory (optical, acoustic, EM)
29. **Pneumatics and hydraulics** — replace solid with gas/liquid; inflatable
30. **Flexible shells and thin films** — instead of 3D structures
31. **Porous materials** — make porous; use pores to introduce useful substance
32. **Color changes** — change color or transparency
33. **Homogeneity** — interacting objects from same material
34. **Discarding and recovering** — portions disappear after use; restore consumables
35. **Parameter changes** — physical state, concentration, density, flexibility, temperature
36. **Phase transitions** — exploit phenomena at phase changes
37. **Thermal expansion** — different coefficients of thermal expansion
38. **Strong oxidants** — oxygen-enriched, ozonized
39. **Inert atmosphere** — inert environment or vacuum
40. **Composite materials** — uniform → composite

## Procedure

1. **State the contradiction** in the form: "I want X to improve, but X improvement causes Y to degrade." If you can't state it crisply, you don't yet have a TRIZ problem.
2. **Compare to Ideal Final Result.** What would it look like if the system eliminated itself?
3. **Look up candidate principles.** The contradiction matrix at triz40.com maps (X parameter, Y parameter) → recommended principles. Or scan the 40 above for fits.
4. **Translate principle to mechanism.** A principle is general; the mechanism is specific to your situation.
5. **Compare candidates against IFR.** Pick closest.

## Worked example

**Problem**: fast brew time (under 60s) vs full extraction (typically 4 min).
**Contradiction**: speed vs completeness of extraction.
**Candidate principles**: 1 (Segmentation), 17 (Another dimension), 19 (Periodic action), 35 (Parameter changes).
**Translations**:
- Segmentation: pre-extract concentrates; dilute on demand. (Nespresso.)
- Another dimension: extract under pressure (espresso).
- Periodic action: pulse-extract with pauses (some pour-over).
- Parameter changes: brew at different temperature/pressure (cold brew = low T long time; espresso = high P short time).

**IFR comparison**: closest to "no brewing time" is pre-extracted concentrate (Segmentation). Resolves the contradiction by *separating extraction from delivery in time*.

## Anti-slop notes

- Don't present the 40 principles as a generative checklist — that's SCAMPER. TRIZ's value is the contradiction lens + patent-derived priors.
- Translate principle to mechanism, don't stop at the principle name.
- Don't claim TRIZ where it doesn't apply (artistic, social, preference contradictions).
- Don't invent principles in Altshuller's style.

Tools: triz40.com (interactive matrix). Source: Altshuller, *And Suddenly the Inventor Appeared* (1994).
