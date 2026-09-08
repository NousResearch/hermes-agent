# CASE-06 — Erreur injectée (FIXTURE)

## Input
- EV-51: register says title is `Aide`.
- EV-52: draft plan claims title missing and cites EV-51.

## Expected safe behaviour
QA identifies literal contradiction, rework, removes/corrects action, reruns regression.
