#!/usr/bin/env python3
"""
nutrition_search.py — Search USDA FoodData Central for nutrition info.

Usage:
  python3 nutrition_search.py "chicken breast"
  python3 nutrition_search.py "rice" "eggs" "broccoli"
  echo -e "oats\\nbanana\\nwhey protein" | python3 nutrition_search.py -

Reads USDA_API_KEY from environment, falls back to DEMO_KEY.
No external dependencies.
"""
import sys
import os
import json
import time
import urllib.request
import urllib.parse

API_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
BASE = "https://api.nal.usda.gov/fdc/v1"


def search(query, max_results=3):
    encoded = urllib.parse.quote(query)
    url = (
        f"{BASE}/foods/search?api_key={API_KEY}"
        f"&query={encoded}&pageSize={max_results}"
        f"&dataType=Foundation,SR%20Legacy"
    )
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return None


# USDA FoodData Central nutrient IDs (stable across SR Legacy and Foundation).
# Display names collide (1008 kcal and 1062 kJ are both "Energy") and vary by
# data type (Atwater General/Specific energy and Foundation macro IDs).
_ENERGY_KCAL_IDS = (1008, 2047, 2048)  # Energy; Atwater General; Atwater Specific
_PROTEIN_IDS = (1003,)
_FAT_IDS = (1004, 1085)                # Total lipid (fat); Total fat (NLEA)
_CARB_IDS = (1005, 1050)               # Carbohydrate, by difference; by summation
_FIBER_IDS = (1079, 2033)              # Fiber, total dietary; AOAC 2011.25
_SUGAR_IDS = (2000, 1063)              # Total Sugars (SR Legacy); Sugars, Total (Foundation)


def extract_macros(food):
    by_id_unit = {}
    for item in food.get("foodNutrients") or []:
        nid = item.get("nutrientId")
        val = item.get("value")
        unit = item.get("unitName")
        if nid is None or val is None or not isinstance(unit, str):
            continue
        by_id_unit.setdefault((nid, unit.upper()), val)

    def pick(ids, unit="G"):
        for nid in ids:
            key = (nid, unit)
            if key in by_id_unit:
                return by_id_unit[key]
        return "?"

    return {
        "calories": pick(_ENERGY_KCAL_IDS, unit="KCAL"),
        "protein": pick(_PROTEIN_IDS),
        "fat": pick(_FAT_IDS),
        "carbs": pick(_CARB_IDS),
        "fiber": pick(_FIBER_IDS),
        "sugar": pick(_SUGAR_IDS),
    }


def display(food):
    macros = extract_macros(food)

    print(f"  {food.get('description', 'N/A')}")
    print(f"    Calories : {macros['calories']} kcal")
    print(f"    Protein  : {macros['protein']}g")
    print(f"    Fat      : {macros['fat']}g")
    print(f"    Carbs    : {macros['carbs']}g (fiber: {macros['fiber']}g, sugar: {macros['sugar']}g)")
    print(f"    FDC ID   : {food.get('fdcId', 'N/A')}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "-":
        queries = [line.strip() for line in sys.stdin if line.strip()]
    else:
        queries = sys.argv[1:]

    for query in queries:
        print(f"\n--- {query.upper()} (per 100g) ---")
        data = search(query, max_results=2)
        if not data or not data.get("foods"):
            print("  No results found.")
        else:
            for food in data["foods"]:
                display(food)
                print()
        if len(queries) > 1:
            time.sleep(1)  # respect rate limits

    if API_KEY == "DEMO_KEY":
        print("\nTip: using DEMO_KEY (30 req/hr). Set USDA_API_KEY for 1000 req/hr.")
        print("Free signup: https://fdc.nal.usda.gov/api-key-signup/")


if __name__ == "__main__":
    main()