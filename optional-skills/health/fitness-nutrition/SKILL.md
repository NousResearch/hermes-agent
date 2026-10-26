---
name: fitness-nutrition
description: "Exercise/food reference and body metrics via wger/USDA."
platforms: [linux, macos, windows]
version: 1.1.0
author: Hailey Marshall (haileymarshall), Hermes Agent
authors:
  - haileymarshall
license: MIT
metadata:
  hermes:
    tags: [health, fitness, nutrition, gym, workout, diet, exercise]
    category: health
    prerequisites:
      tools: [terminal, web_extract]
required_environment_variables:
  - name: USDA_API_KEY
    prompt: "USDA FoodData Central API key (free)"
    help: "Get one free at https://fdc.nal.usda.gov/api-key-signup/ — or skip to use DEMO_KEY with lower rate limits"
    required_for: "higher rate limits on food/nutrition lookups (DEMO_KEY works without signup)"
    optional: true
---

# Fitness & Nutrition Skill

Exercise and food-reference data plus offline calculators. Use this skill for
lookups and mechanical calculations, not individualized medical advice, dietetic
care, progressive workout programming, or individualized meal planning.

**Data sources:** wger (open exercise data) and USDA FoodData Central (food
composition data). The bundled helpers use only the Python standard library.

## When to Use

Trigger this skill when the user asks about:
- Exercise reference by movement name, muscle, equipment, or category
- Food energy, macro, micronutrient, or ingredient composition
- BMI, body-fat, TDEE, or calorie-target calculations
- One-rep-max estimates and derived training percentages
- Generic macro templates for cutting, bulking, or maintenance

Do not use this as the primary skill for progressive workout programming, workout
tracking, individualized meal planning, or medical nutrition. Route those tasks
to the relevant planning, tracking, or clinical workflow, and use this skill only
for needed reference data or calculations.

## Prerequisites

- The native `terminal` tool for invoking the bundled helper scripts.
- The native `web_extract` tool for read-only direct lookups when a helper does
  not cover the request.
- For higher USDA FoodData Central rate limits, configure the optional
  `USDA_API_KEY`; `DEMO_KEY` works without signup.
- Network access is needed for wger and USDA lookups; calculators are offline.

## How to Run

Use `terminal` with a helper path relative to this skill directory:

- `scripts/exercise_search.py` searches and filters wger exercises.
- `scripts/body_calc.py` performs BMI, TDEE, one-rep-max, macro, and body-fat
  calculations.

For direct read-only wger or USDA requests, use `web_extract` with the endpoint
URL. Keep returned JSON intact while inspecting it, and present only the fields
needed by the user. Do not use an unrelated unfiltered response as a search
result.

## Quick Reference

| Task | Native tool or helper | Request |
|------|------------------------|---------|
| Search exercises by name | `terminal` + `scripts/exercise_search.py` | `name__search`, `language__code=en` |
| Exercise details | `web_extract` | `https://wger.de/api/v2/exerciseinfo/{id}/` |
| Filter by muscle | `web_extract` | `/api/v2/exerciseinfo/?muscles={id}&language__code=en` |
| Filter by equipment | `web_extract` | `/api/v2/exerciseinfo/?equipment={id}&language__code=en` |
| List categories | `web_extract` | `/api/v2/exercisecategory/` |
| List muscles | `web_extract` | `/api/v2/muscle/` |
| Search foods | `web_extract` | `/fdc/v1/foods/search?query=&dataType=Foundation,SR Legacy` |
| Food details | `web_extract` | `/fdc/v1/food/{fdcId}` |
| BMI / TDEE / 1RM / macros | `terminal` + `scripts/body_calc.py` | Offline calculator arguments |

## Procedure

### Exercise Lookup (wger API)

Use `terminal` to invoke the helper for searches. It calls the supported
`/api/v2/exerciseinfo/` endpoint, requests English data, fetches at most 100
candidates, reranks exact and all-token name matches, and emits at most the
requested 1–20 records as compact JSON.

Examples of helper arguments:

- Name: `scripts/exercise_search.py "pull up" --limit 10`
- Muscle: `scripts/exercise_search.py --muscle 4 --limit 20`
- Category: `scripts/exercise_search.py --category 11 --limit 20`
- Equipment: `scripts/exercise_search.py --equipment 3 --limit 20`
- Combined filters: `scripts/exercise_search.py --muscle 4 --equipment 1 --limit 20`

For a detail or catalog request outside the helper, use `web_extract` on the
read-only wger endpoint and request `language__code=en` and `format=json`.
Relevant paths are `exerciseinfo/{exercise_id}/`,
`exerciseinfo/?category={id}`, `exercisecategory/`, and `muscle/`.

Reference IDs commonly used by the API:

| ID | Category | ID | Muscle | ID | Equipment |
|----|----------|----|--------|----|-----------|
| 8 | Arms | 1 | Biceps brachii | 1 | Barbell |
| 9 | Legs | 2 | Anterior deltoid | 3 | Dumbbell |
| 10 | Abs | 4 | Pectoralis major | 6 | Pull-up bar |
| 11 | Chest | 8 | Gluteus maximus | 7 | Bodyweight |
| 12 | Back | 10 | Quadriceps femoris | 8 | Bench |
| 13 | Shoulders | 12 | Latissimus dorsi | 10 | Kettlebell |
| 14 | Calves | 14 | Triceps brachii | | |
| 15 | Cardio | 15 | Soleus | | |

Present exercise names, IDs, muscles, equipment, descriptions, and images when
available. Treat community-maintained exercise data as reference material, not
individualized technique or medical advice.

### Nutrition Lookup (USDA FoodData Central)

Use `web_extract` for the read-only USDA endpoints. Use `USDA_API_KEY` when
configured, otherwise `DEMO_KEY`. Search with
`/fdc/v1/foods/search?api_key={key}&query={food}&pageSize=5&dataType=Foundation,SR%20Legacy`.
For a selected record, fetch `/fdc/v1/food/{fdcId}?api_key={key}`.

Inspect descriptions and FDC IDs before choosing a result; search rank alone is
not an identity match. For packaged food, prefer the product label when
available. Treat generic USDA values as estimates and state the per-100 g basis
before scaling to a serving. Report kcal, protein, fat, and carbohydrate when
those nutrients are present.

### Offline Calculators

Use `terminal` to invoke `scripts/body_calc.py` with one of these argument forms:

- `bmi <weight_kg> <height_cm>`
- `tdee <weight_kg> <height_cm> <age> <M|F> <activity 1-5>`
- `1rm <weight> <reps>`
- `macros <tdee_kcal> <cut|maintain|bulk>`
- `bodyfat <M|F> <neck_cm> <waist_cm> [hip_cm] <height_cm>`

See `references/FORMULAS.md` for the science behind each formula. The helper
rejects unsupported categories, non-positive measurements, and one-rep-max sets
above 10 repetitions instead of silently selecting defaults.

## Pitfalls

- wger has multiple translations; use the helper or request English language ID
  `2` when inspecting translations.
- USDA `DEMO_KEY` has 30 requests per hour; use a configured key for higher
  limits and avoid unneeded repeated requests.
- USDA data is per 100 g; scale explicitly to the user's portion size.
- USDA search results can be approximate; select the intended FDC ID, and let a
  current product label override generic database values.
- BMI does not distinguish muscle from fat.
- Body-fat formulas are estimates (±3–5%); recommend DEXA scans for precision.
- One-rep-max formulas lose accuracy above 10 repetitions; sets of 3–5 are best.
- Macro percentages are templates, not individualized prescriptions.

## Verification

After exercise search, confirm the JSON includes names, IDs, and relevant muscle
or equipment fields; an unrelated unfiltered list is not a successful search.
After nutrition lookup, confirm the selected food identity and per-100 g kcal,
protein, fat, and carbohydrate values before scaling.
After calculations, confirm the helper succeeded and sanity-check outputs (for
example, TDEE is commonly 1500–3500 kcal for adults, but individual values vary).
