#!/usr/bin/env python3
"""
body_calc.py — All-in-one fitness calculator.

Subcommands:
  bmi      <weight_kg> <height_cm>
  tdee     <weight_kg> <height_cm> <age> <M|F> <activity 1-5>
  1rm      <weight> <reps>
  macros   <tdee_kcal> <cut|maintain|bulk>
  bodyfat  <M|F> <neck_cm> <waist_cm> [hip_cm] <height_cm>

No external dependencies — stdlib only.
"""
import sys
import math


def bmi(weight_kg, height_cm):
    h = height_cm / 100
    val = weight_kg / (h * h)
    if val < 18.5:
        cat = "Underweight"
    elif val < 25:
        cat = "Normal weight"
    elif val < 30:
        cat = "Overweight"
    else:
        cat = "Obese"
    print(f"BMI: {val:.1f} — {cat}")
    print()
    print("Ranges:")
    print(f"  Underweight : < 18.5")
    print(f"  Normal      : 18.5 – 24.9")
    print(f"  Overweight  : 25.0 – 29.9")
    print(f"  Obese       : 30.0+")


def tdee(weight_kg, height_cm, age, sex, activity):
    if sex.upper() == "M":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    multipliers = {
        1: ("Sedentary (desk job, no exercise)", 1.2),
        2: ("Lightly active (1-3 days/week)", 1.375),
        3: ("Moderately active (3-5 days/week)", 1.55),
        4: ("Very active (6-7 days/week)", 1.725),
        5: ("Extremely active (athlete + physical job)", 1.9),
    }

    label, mult = multipliers.get(activity, ("Moderate", 1.55))
    total = bmr * mult

    print(f"BMR (Mifflin-St Jeor): {bmr:.0f} kcal/day")
    print(f"Activity: {label} (x{mult})")
    print(f"TDEE: {total:.0f} kcal/day")
    print()
    print("Calorie targets:")
    print(f"  Aggressive cut (-750): {total - 750:.0f} kcal/day")
    print(f"  Fat loss       (-500): {total - 500:.0f} kcal/day")
    print(f"  Mild cut       (-250): {total - 250:.0f} kcal/day")
    print(f"  Maintenance          : {total:.0f} kcal/day")
    print(f"  Lean bulk      (+250): {total + 250:.0f} kcal/day")
    print(f"  Bulk           (+500): {total + 500:.0f} kcal/day")


def one_rep_max(weight, reps):
    if reps < 1:
        print("Error: reps must be at least 1.")
        sys.exit(1)
    if reps == 1:
        print(f"1RM = {weight:.1f} (actual single)")
        return

    epley = weight * (1 + reps / 30)
    brzycki = weight * (36 / (37 - reps)) if reps < 37 else 0
    lombardi = weight * (reps ** 0.1)
    avg = (epley + brzycki + lombardi) / 3

    print(f"Estimated 1RM ({weight} x {reps} reps):")
    print(f"  Epley    : {epley:.1f}")
    print(f"  Brzycki  : {brzycki:.1f}")
    print(f"  Lombardi : {lombardi:.1f}")
    print(f"  Average  : {avg:.1f}")
    print()
    print("Training percentages off average 1RM:")
    for pct, rep_range in [
        (100, "1"), (95, "1-2"), (90, "3-4"), (85, "4-6"),
        (80, "6-8"), (75, "8-10"), (70, "10-12"),
        (65, "12-15"), (60, "15-20"),
    ]:
        print(f"  {pct:>3}% = {avg * pct / 100:>7.1f}  (~{rep_range} reps)")


def macros(tdee_kcal, goal):
    goal = goal.lower()
    if goal in ("cut", "lose", "deficit"):
        cals = tdee_kcal - 500
        p, f, c = 0.40, 0.30, 0.30
        label = "Fat Loss (-500 kcal)"
    elif goal in ("bulk", "gain", "surplus"):
        cals = tdee_kcal + 400
        p, f, c = 0.30, 0.25, 0.45
        label = "Lean Bulk (+400 kcal)"
    else:
        cals = tdee_kcal
        p, f, c = 0.30, 0.30, 0.40
        label = "Maintenance"

    prot_g = cals * p / 4
    fat_g = cals * f / 9
    carb_g = cals * c / 4

    print(f"Goal: {label}")
    print(f"Daily calories: {cals:.0f} kcal")
    print()
    print(f"  Protein : {prot_g:>6.0f}g ({p * 100:.0f}%)  = {prot_g * 4:.0f} kcal")
    print(f"  Fat     : {fat_g:>6.0f}g ({f * 100:.0f}%)  = {fat_g * 9:.0f} kcal")
    print(f"  Carbs   : {carb_g:>6.0f}g ({c * 100:.0f}%)  = {carb_g * 4:.0f} kcal")
    print()
    print(f"Per meal (3 meals): P {prot_g / 3:.0f}g | F {fat_g / 3:.0f}g | C {carb_g / 3:.0f}g")
    print(f"Per meal (4 meals): P {prot_g / 4:.0f}g | F {fat_g / 4:.0f}g | C {carb_g / 4:.0f}g")


def bodyfat(sex, neck_cm, waist_cm, hip_cm, height_cm):
    sex = sex.upper()
    if sex == "M":
        if waist_cm <= neck_cm:
            print("Error: waist must be larger than neck."); sys.exit(1)
        bf = 86.010 * math.log10(waist_cm - neck_cm) - 70.041 * math.log10(height_cm) + 36.76
    else:
        if (waist_cm + hip_cm) <= neck_cm:
            print("Error: waist + hip must be larger than neck."); sys.exit(1)
        bf = 163.205 * math.log10(waist_cm + hip_cm - neck_cm) - 97.684 * math.log10(height_cm) - 78.387

    print(f"Estimated body fat: {bf:.1f}%")

    if sex == "M":
        ranges = [
            (6, "Essential fat (2-5%)"),
            (14, "Athletic (6-13%)"),
            (18, "Fitness (14-17%)"),
            (25, "Average (18-24%)"),
        ]
        default = "Obese (25%+)"
    else:
        ranges = [
            (14, "Essential fat (10-13%)"),
            (21, "Athletic (14-20%)"),
            (25, "Fitness (21-24%)"),
            (32, "Average (25-31%)"),
        ]
        default = "Obese (32%+)"

    cat = default
    for threshold, label in ranges:
        if bf < threshold:
            cat = label
            break

    print(f"Category: {cat}")
    print(f"Method: US Navy circumference formula")


def usage():
    print(__doc__)
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        usage()

    cmd = sys.argv[1].lower()

    try:
        if cmd == "bmi":
            bmi(float(sys.argv[2]), float(sys.argv[3]))

        elif cmd == "tdee":
            tdee(
                float(sys.argv[2]), float(sys.argv[3]),
                int(sys.argv[4]), sys.argv[5], int(sys.argv[6]),
            )

        elif cmd in ("1rm", "orm"):
            one_rep_max(float(sys.argv[2]), int(sys.argv[3]))

        elif cmd == "macros":
            macros(float(sys.argv[2]), sys.argv[3])

        elif cmd == "bodyfat":
            sex = sys.argv[2]
            if sex.upper() == "M":
                bodyfat(sex, float(sys.argv[3]), float(sys.argv[4]), 0, float(sys.argv[5]))
            else:
                bodyfat(sex, float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))

        else:
            print(f"Unknown command: {cmd}")
            usage()

    except (IndexError, ValueError) as e:
        print(f"Error: {e}")
        usage()


if __name__ == "__main__":
    main()