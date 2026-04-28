#!/usr/bin/env python3
"""
chembl_target.py — Search ChEMBL for a target and retrieve top active compounds.
Usage: python3 chembl_target.py "EGFR" --min-pchembl 7 --limit 20
No external dependencies.
"""
import sys, json, time, argparse
import urllib.request, urllib.parse, urllib.error

BASE = "https://www.ebi.ac.uk/chembl/api/data"

def get(endpoint):
    try:
        req = urllib.request.Request(f"{BASE}{endpoint}", headers={"Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"API error: {e}", file=sys.stderr); return None

def main():
    parser = argparse.ArgumentParser(description="ChEMBL target → active compounds")
    parser.add_argument("target")
    parser.add_argument("--min-pchembl", type=float, default=6.0)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    enc = urllib.parse.quote(args.target)
    data = get(f"/target/search?q={enc}&limit=5&format=json")
    if not data or not data.get("targets"):
        print("No targets found."); sys.exit(1)

    t = data["targets"][0]
    tid = t.get("target_chembl_id","")
    print(f"\nTarget: {t.get('pref_name')} ({tid})")
    print(f"Type: {t.get('target_type')} | Organism: {t.get('organism','N/A')}")
    print(f"\nFetching compounds with pChEMBL ≥ {args.min_pchembl}...\n")

    acts = get(f"/activity?target_chembl_id={tid}&pchembl_value__gte={args.min_pchembl}&assay_type=B&limit={args.limit}&order_by=-pchembl_value&format=json")
    if not acts or not acts.get("activities"):
        print("No activities found."); sys.exit(0)

    print(f"{'Molecule':<18} {'pChEMBL':>8} {'Type':<12} {'Value':<10} {'Units'}")
    print("-"*65)
    seen = set()
    for a in acts["activities"]:
        mid = a.get("molecule_chembl_id","N/A")
        if mid in seen: continue
        seen.add(mid)
        print(f"{mid:<18} {str(a.get('pchembl_value','N/A')):>8} {str(a.get('standard_type','N/A')):<12} {str(a.get('standard_value','N/A')):<10} {a.get('standard_units','N/A')}")
        time.sleep(0.1)
    print(f"\nTotal: {len(seen)} unique molecules")

if __name__ == "__main__": main()
