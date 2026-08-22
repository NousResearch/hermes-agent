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
      commands: [curl, python]
required_environment_variables:
  - name: USDA_API_KEY
    prompt: "USDA FoodData Central API key (free)"
    help: "Get one free at https://fdc.nal.usda.gov/api-key-signup/ — or skip to use DEMO_KEY with lower rate limits"
    required_for: "higher rate limits on food/nutrition lookups (DEMO_KEY works without signup)"
    optional: true
---

# Fitness & Nutrition

Exercise and food-reference data plus offline calculators. Use this skill for
lookups and mechanical calculations, not as a substitute for individualized
medical advice, dietetic care, or a progressive training plan.

**Data sources (all free, no pip dependencies):**

- **wger** (https://wger.de/api/v2/) — open exercise database with muscles, equipment, and images. Public endpoints need zero authentication.
- **USDA FoodData Central** (https://api.nal.usda.gov/fdc/v1/) — US government food-composition database. `DEMO_KEY` works instantly; free signup for higher limits.

**Offline calculators (pure stdlib Python):**

- BMI, TDEE (Mifflin-St Jeor), one-rep max (Epley/Brzycki/Lombardi), macro splits, body fat % (US Navy method)

---

## When to Use

Trigger this skill when the user asks about:
- Exercise reference by movement name, muscle, equipment, or category
- Food energy, macro, micronutrient, or ingredient composition
- BMI, body-fat, TDEE, or calorie-target calculations
- One-rep-max estimates and derived training percentages
- Generic macro templates for cutting, bulking, or maintenance

Do not use this as the primary skill for progressive workout programming, workout tracking, individualized meal planning, or medical nutrition. Route those tasks to the relevant planning, tracking, or clinical workflow, and use this skill only for the needed reference data or calculation.

---

## Procedure

### Exercise Lookup (wger API)

All wger public endpoints return JSON and require no auth. Use the read-only
`exerciseinfo` endpoint, add `language__code=en`, and request `format=json`.

**Step 1 — Identify what the user wants:**

- By muscle → use `/api/v2/exerciseinfo/?muscles={id}&language__code=en&format=json`
- By category → use `/api/v2/exerciseinfo/?category={id}&language__code=en&format=json`
- By equipment → use `/api/v2/exerciseinfo/?equipment={id}&language__code=en&format=json`
- By name → run `python scripts/exercise_search.py "{query}"`; it uses
  `/api/v2/exerciseinfo/?name__search={query}&language__code=en&format=json`
- Full details → use `/api/v2/exerciseinfo/{exercise_id}/?format=json`

**Step 2 — Reference IDs (so you don't need extra API calls):**

Exercise categories:

| ID | Category    |
|----|-------------|
| 8  | Arms        |
| 9  | Legs        |
| 10 | Abs         |
| 11 | Chest       |
| 12 | Back        |
| 13 | Shoulders   |
| 14 | Calves      |
| 15 | Cardio      |

Muscles:

| ID | Muscle                    | ID | Muscle                  |
|----|---------------------------|----|-------------------------|
| 1  | Biceps brachii            | 2  | Anterior deltoid        |
| 3  | Serratus anterior         | 4  | Pectoralis major        |
| 5  | Obliquus externus         | 6  | Gastrocnemius           |
| 7  | Rectus abdominis          | 8  | Gluteus maximus         |
| 9  | Trapezius                 | 10 | Quadriceps femoris      |
| 11 | Biceps femoris            | 12 | Latissimus dorsi        |
| 13 | Brachialis                | 14 | Triceps brachii         |
| 15 | Soleus                    |    |                         |

Equipment:

| ID | Equipment      |
|----|----------------|
| 1  | Barbell        |
| 3  | Dumbbell       |
| 4  | Gym mat        |
| 5  | Swiss Ball     |
| 6  | Pull-up bar    |
| 7  | none (bodyweight) |
| 8  | Bench          |
| 9  | Incline bench  |
| 10 | Kettlebell     |

**Step 3 — Fetch and present results:**

```bash
# Search exercises by name
python scripts/exercise_search.py "pull up" --limit 10
```

The helper fetches at most 100 wger candidates, reranks exact and all-token name matches, and returns only the requested 1–20 records.

```bash
# Get full details for a specific exercise
EXERCISE_ID="$1"
curl -s "https://wger.de/api/v2/exerciseinfo/${EXERCISE_ID}/?format=json" \
  | python -c "
import json,sys,html,re
data=json.load(sys.stdin)
trans=[t for t in data.get('translations',[]) if t.get('language')==2]
t=trans[0] if trans else data.get('translations',[{}])[0]
desc=re.sub('<[^>]+>','',html.unescape(t.get('description','N/A')))
print(f\"Exercise  : {t.get('name','N/A')}\")
print(f\"Category  : {data.get('category',{}).get('name','N/A')}\")
print(f\"Primary   : {', '.join(m.get('name_en','') for m in data.get('muscles',[])) or 'N/A'}\")
print(f\"Secondary : {', '.join(m.get('name_en','') for m in data.get('muscles_secondary',[])) or 'none'}\")
print(f\"Equipment : {', '.join(e.get('name','') for e in data.get('equipment',[])) or 'bodyweight'}\")
print(f\"How to    : {desc[:500]}\")
imgs=data.get('images',[])
if imgs: print(f\"Image     : {imgs[0].get('image','')}\")
"
```

```bash
# List exercises by one filter
python scripts/exercise_search.py --muscle 4 --limit 20
python scripts/exercise_search.py --category 11 --limit 20
python scripts/exercise_search.py --equipment 3 --limit 20

# Combine filters as needed
python scripts/exercise_search.py --muscle 4 --equipment 1 --limit 20
```

The helper resolves English names from each record's `translations`, then emits
compact JSON with IDs, names, muscles, equipment, descriptions, and images.

### Nutrition Lookup (USDA FoodData Central)

Uses `USDA_API_KEY` env var if set, otherwise falls back to `DEMO_KEY`.
DEMO_KEY = 30 requests/hour. Free signup key = 1,000 requests/hour.

Inspect the returned descriptions and FDC IDs before choosing a result; search
rank alone is not an identity match. For packaged food, use the product label
when available. Treat generic USDA values as estimates and state the per-100 g
basis before scaling to a serving.

```bash
# Search foods by name
FOOD="$1"
API_KEY="${USDA_API_KEY:-DEMO_KEY}"
ENCODED=$(python -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$FOOD")
curl -s "https://api.nal.usda.gov/fdc/v1/foods/search?api_key=${API_KEY}&query=${ENCODED}&pageSize=5&dataType=Foundation,SR%20Legacy" \
  | python -c "
import json,sys
data=json.load(sys.stdin)
foods=data.get('foods',[])
if not foods: print('No foods found.'); sys.exit()
for f in foods:
    n={x['nutrientName']:x.get('value','?') for x in f.get('foodNutrients',[])}
    cal=n.get('Energy','?'); prot=n.get('Protein','?')
    fat=n.get('Total lipid (fat)','?'); carb=n.get('Carbohydrate, by difference','?')
    print(f\"{f.get('description','N/A')}\")
    print(f\"  Per 100g: {cal} kcal | {prot}g protein | {fat}g fat | {carb}g carbs\")
    print(f\"  FDC ID: {f.get('fdcId','N/A')}\")
    print()
"
```

```bash
# Detailed nutrient profile by FDC ID
FDC_ID="$1"
API_KEY="${USDA_API_KEY:-DEMO_KEY}"
curl -s "https://api.nal.usda.gov/fdc/v1/food/${FDC_ID}?api_key=${API_KEY}" \
  | python -c "
import json,sys
d=json.load(sys.stdin)
print(f\"Food: {d.get('description','N/A')}\")
print(f\"{'Nutrient':<40} {'Amount':>8} {'Unit'}\")
print('-'*56)
for x in sorted(d.get('foodNutrients',[]),key=lambda x:x.get('nutrient',{}).get('rank',9999)):
    nut=x.get('nutrient',{}); amt=x.get('amount',0)
    if amt and float(amt)>0:
        print(f\"  {nut.get('name',''):<38} {amt:>8} {nut.get('unitName','')}\")
"
```

### Offline Calculators

Use the helper scripts in `scripts/` for batch operations,
or run inline for single calculations:

- `python scripts/body_calc.py bmi <weight_kg> <height_cm>`
- `python scripts/body_calc.py tdee <weight_kg> <height_cm> <age> <M|F> <activity 1-5>`
- `python scripts/body_calc.py 1rm <weight> <reps>`
- `python scripts/body_calc.py macros <tdee_kcal> <cut|maintain|bulk>`
- `python scripts/body_calc.py bodyfat <M|F> <neck_cm> <waist_cm> [hip_cm] <height_cm>`

See `references/FORMULAS.md` for the science behind each formula.
The CLI rejects unsupported categories, non-positive measurements, and 1RM
sets above 10 repetitions instead of silently selecting defaults.

---

## Pitfalls

- wger exercise data includes multiple translations — use the name-search
  helper or select translation language ID `2` for English
- wger is community-maintained reference data — verify exercise details instead
  of treating a database entry as individualized technique or medical advice
- USDA `DEMO_KEY` has **30 req/hour** — add `sleep 2` between batch requests or get a free key
- USDA data is **per 100g** — remind users to scale to their actual portion size
- USDA search results can be approximate — select the intended FDC ID, and let
  a current product label override generic database values
- BMI does not distinguish muscle from fat — high BMI in muscular people is not necessarily unhealthy
- Body fat formulas are **estimates** (±3-5%) — recommend DEXA scans for precision
- 1RM formulas lose accuracy above 10 reps — use sets of 3-5 for best estimates
- Macro percentages are templates, not individualized prescriptions; use a
  user-approved plan when one exists

---

## Verification

After running exercise search: confirm the JSON includes exercise names,
muscle groups, and equipment; an HTTP 200 with an unrelated unfiltered list is
not a successful search.
After nutrition lookup: confirm per-100g macros are returned with kcal, protein, fat, carbs.
After calculators: sanity-check outputs (e.g. TDEE should be 1500-3500 for most adults).

---

## Quick Reference

| Task | Source | Endpoint |
|------|--------|----------|
| Search exercises by name | wger | `python scripts/exercise_search.py "query"` (`name__search`, `language__code=en`) |
| Exercise details | wger | `GET /api/v2/exerciseinfo/{id}/` |
| Filter by muscle | wger | `GET /api/v2/exerciseinfo/?muscles={id}&language__code=en` |
| Filter by equipment | wger | `GET /api/v2/exerciseinfo/?equipment={id}&language__code=en` |
| List categories | wger | `GET /api/v2/exercisecategory/` |
| List muscles | wger | `GET /api/v2/muscle/` |
| Search foods | USDA | `GET /fdc/v1/foods/search?query=&dataType=Foundation,SR Legacy` |
| Food details | USDA | `GET /fdc/v1/food/{fdcId}` |
| BMI / TDEE / 1RM / macros | offline | `python scripts/body_calc.py` |
