# Formulas Reference

Scientific references for all calculators used in the fitness-nutrition skill.

## BMI (Body Mass Index)

**Formula:** BMI = weight (kg) / height (m)²

| Category     | BMI Range  |
|-------------|------------|
| Underweight | < 18.5     |
| Normal      | 18.5 – 24.9 |
| Overweight  | 25.0 – 29.9 |
| Obese       | 30.0+      |

**Limitation:** BMI does not distinguish muscle from fat. A muscular person
can have a high BMI while being lean. Use body fat % for a better picture.

Reference: Quetelet, A. (1832). Keys et al., Int J Obes (1972).

## TDEE (Total Daily Energy Expenditure)

Uses the **Mifflin-St Jeor equation** — the most accurate BMR predictor for
the general population according to the ADA (2005).

**BMR formulas:**

- Male: BMR = 10 × weight(kg) + 6.25 × height(cm) − 5 × age + 5
- Female: BMR = 10 × weight(kg) + 6.25 × height(cm) − 5 × age − 161

**Activity multipliers:**

| Level | Description                    | Multiplier |
|-------|--------------------------------|------------|
| 1     | Sedentary (desk job)           | 1.200      |
| 2     | Lightly active (1-3 days/wk)   | 1.375      |
| 3     | Moderately active (3-5 days)   | 1.550      |
| 4     | Very active (6-7 days)         | 1.725      |
| 5     | Extremely active (2x/day)      | 1.900      |

Reference: Mifflin et al., Am J Clin Nutr 51, 241-247 (1990).

## One-Rep Max (1RM)

Three validated formulas. Average of all three is most reliable.

- **Epley:** 1RM = w × (1 + r/30)
- **Brzycki:** 1RM = w × 36 / (37 − r)
- **Lombardi:** 1RM = w × r^0.1

All formulas are most accurate for r ≤ 10. Above 10 reps, error increases.

Reference: LeSuer et al., J Strength Cond Res 11(4), 211-213 (1997).

## Macro Splits

Recommended splits based on goal:

| Goal         | Protein | Fat  | Carbs | Calorie Offset |
|-------------|---------|------|-------|----------------|
| Fat loss    | 40%     | 30%  | 30%   | −500 kcal      |
| Maintenance | 30%     | 30%  | 40%   | 0              |
| Lean bulk   | 30%     | 25%  | 45%   | +400 kcal      |

Protein targets for muscle growth: 1.6–2.2 g/kg body weight per day.
Minimum fat intake: 0.5 g/kg to support hormone production.

Conversion: Protein = 4 kcal/g, Fat = 9 kcal/g, Carbs = 4 kcal/g.

Reference: Morton et al., Br J Sports Med 52, 376–384 (2018).

## Body Fat % (US Navy Method)

**Male:**

BF% = 86.010 × log₁₀(waist − neck) − 70.041 × log₁₀(height) + 36.76

**Female:**

BF% = 163.205 × log₁₀(waist + hip − neck) − 97.684 × log₁₀(height) − 78.387

All measurements in centimeters.

| Category      | Male   | Female |
|--------------|--------|--------|
| Essential    | 2-5%   | 10-13% |
| Athletic     | 6-13%  | 14-20% |
| Fitness      | 14-17% | 21-24% |
| Average      | 18-24% | 25-31% |
| Obese        | 25%+   | 32%+   |

Accuracy: ±3-5% compared to DEXA. Measure at the navel (waist),
at the Adam's apple (neck), and widest point (hip, females only).

Reference: Hodgdon & Beckett, Naval Health Research Center (1984).

## APIs

- wger: https://wger.de/api/v2/ — AGPL-3.0, exercise data is CC-BY-SA 3.0
- USDA FoodData Central: https://api.nal.usda.gov/fdc/v1/ — public domain (CC0 1.0)