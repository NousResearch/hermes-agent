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
import urllib.error

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
    nutrients = {n["nutrientName"]: n.get("value", "?") for n in food.get("foodNutrients", [])}
    cal = nutrients.get("Energy", "?")
    prot = nutrients.get("Protein", "?")
    fat = nutrients.get("Total lipid (fat)", "?")
    carb = nutrients.get("Carbohydrate, by difference", "?")
    fib = nutrients.get("Fiber, total dietary", "?")
    sug = nutrients.get("Sugars, total including NLEA", "?")

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