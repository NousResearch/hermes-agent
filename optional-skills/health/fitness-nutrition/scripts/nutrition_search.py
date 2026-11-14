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


def display(food):
    # Names collide (Energy in kcal/kJ) and vary between USDA data types.
    nutrients = {
        (n["nutrientId"], n["unitName"].upper()): n["value"]
        for n in food.get("foodNutrients", [])
        if n.get("value") is not None
    }

    def value(*ids, unit="G"):
        for nutrient_id in ids:
            if (nutrient_id, unit) in nutrients:
                return nutrients[nutrient_id, unit]
        return "?"

    # Prefer legacy Energy, then Atwater general, then Atwater specific.
    cal = value(1008, 2047, 2048, unit="KCAL")
    prot = value(1003)
    fat = value(1004)
    carb = value(1005)
    fib = value(1079)
    sug = value(2000)

    print(f"  {food.get('description', 'N/A')}")
    print(f"    Calories : {cal} kcal")
    print(f"    Protein  : {prot}g")
    print(f"    Fat      : {fat}g")
    print(f"    Carbs    : {carb}g (fiber: {fib}g, sugar: {sug}g)")
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