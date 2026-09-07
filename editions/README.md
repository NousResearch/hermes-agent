# editions/

Optional **profile overlays** for North Forge. Each subfolder holds files that
tailor the generic chassis to a particular audience or deployment — a persona
variant, extra context, pinned defaults.

An overlay is **additive and optional**. The base identity is always the
repo-root `SOUL.md` (the generic chassis voice). An overlay does not replace it —
it is applied *on top* when a deployment explicitly opts in (e.g. a drive's setup
menu copies `editions/<name>/SOUL.md` to the active `$HERMES_HOME/SOUL.md`).
Nothing here is loaded automatically, and nothing here changes how the engine
runs.

| Edition | What it adds | Who it's for |
| --- | --- | --- |
| [`field-service/`](field-service/) | A field-service-flavoured persona: plainer register, written for technicians and reps. | Deployments aimed at field techs / sales reps. |

Proprietary vertical skill-sets (Kyocera, Sales Edition, Penny Pincher, …) are
**not** here — they live in separate admin-gated repos/content, never in the
public chassis. `editions/` is only for lightweight, non-proprietary overlays
that are safe to ship in the open repo.

See [`../BRANDING.md`](../BRANDING.md) for which surfaces North Forge owns.
