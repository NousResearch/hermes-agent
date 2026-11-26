#!/usr/bin/env python3
"""
ntd_utils.py — Utility scripts for NTD drug discovery workflows.

Outbound network calls (read-only GET requests only):
  - https://ghoapi.azureedge.net    (WHO Global Health Observatory — NTD burden)
  - https://www.ebi.ac.uk/chembl    (ChEMBL — NTD bioactivity and drug data)

No data is transmitted outbound. All requests fetch public read-only data.
No API keys. No authentication required.

Commands:
    python3 ntd_utils.py burden NGA
    python3 ntd_utils.py burden-compare NGA GHA KEN TZA ETH
    python3 ntd_utils.py compounds "Plasmodium falciparum"
    python3 ntd_utils.py drug praziquantel
    python3 ntd_utils.py pathogen-box
    python3 ntd_utils.py repurpose CHEMBL364

Author: Bennytimz
"""

import sys
import os
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error

GHO_API    = "https://ghoapi.azureedge.net/api"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

AFRICAN_COUNTRIES = {
    "NGA": "Nigeria", "GHA": "Ghana", "KEN": "Kenya", "TZA": "Tanzania",
    "ETH": "Ethiopia", "MOZ": "Mozambique", "COD": "DR Congo", "UGA": "Uganda",
    "ZAF": "South Africa", "ZMB": "Zambia", "ZWE": "Zimbabwe", "MWI": "Malawi",
    "MDG": "Madagascar", "CMR": "Cameroon", "SEN": "Senegal", "MLI": "Mali",
    "BFA": "Burkina Faso", "NER": "Niger", "TCD": "Chad", "SSD": "South Sudan",
    "SDN": "Sudan", "CAF": "Central African Republic", "COG": "Republic of Congo",
    "RWA": "Rwanda", "BDI": "Burundi", "SOM": "Somalia", "AGO": "Angola",
    "NAM": "Namibia", "BWA": "Botswana", "LSO": "Lesotho", "SWZ": "Eswatini",
}

ENDEMIC_NTDS = {
    "NGA": ["Lymphatic Filariasis", "Onchocerciasis", "Schistosomiasis",
            "Soil-transmitted Helminthiases", "Trachoma", "Leprosy", "Buruli Ulcer"],
    "GHA": ["Lymphatic Filariasis", "Onchocerciasis", "Schistosomiasis",
            "Soil-transmitted Helminthiases", "Trachoma", "Buruli Ulcer"],
    "KEN": ["Lymphatic Filariasis", "Onchocerciasis", "Schistosomiasis",
            "Soil-transmitted Helminthiases", "Trachoma", "Leishmaniasis"],
    "TZA": ["Lymphatic Filariasis", "Schistosomiasis", "Trachoma",
            "Soil-transmitted Helminthiases", "Onchocerciasis"],
    "ETH": ["Lymphatic Filariasis", "Trachoma", "Leishmaniasis",
            "Soil-transmitted Helminthiases", "Schistosomiasis", "Podoconiosis"],
    "COD": ["HAT (sleeping sickness)", "Lymphatic Filariasis", "Onchocerciasis",
            "Schistosomiasis", "Soil-transmitted Helminthiases", "Yaws", "Buruli Ulcer"],
    "UGA": ["HAT (sleeping sickness)", "Lymphatic Filariasis", "Onchocerciasis",
            "Schistosomiasis", "Soil-transmitted Helminthiases", "Trachoma"],
    "CMR": ["Lymphatic Filariasis", "Onchocerciasis", "Schistosomiasis",
            "Soil-transmitted Helminthiases", "Buruli Ulcer"],
}


def http_get(url, timeout=15):
    """GET request to public read-only API — no data transmitted outbound."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"HTTP {e.code}: {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def cmd_burden(country_code):
    """Get NTD burden data for a specific country from WHO GHO."""
    code = country_code.upper()
    name = AFRICAN_COUNTRIES.get(code, code)
    print(f"\n{'='*60}")
    print(f"  NTD Burden Report: {name} ({code})")
    print(f"{'='*60}\n")
    url  = f"{GHO_API}/NTD_PEOPLE?$filter=SpatialDim eq '{code}'&$orderby=TimeDim desc&$top=5"
    data = http_get(url)
    if not data or not data.get("value"):
        print(f"No WHO GHO data found for: {code}")
        print("Valid codes: NGA, GHA, KEN, TZA, ETH, MOZ, COD, UGA, ZAF, CMR")
    else:
        print("People requiring NTD treatment (WHO):")
        print(f"  {'Year':<6}  {'Count':>22}")
        print("  " + "-"*32)
        for v in data["value"]:
            year = v.get("TimeDim", "N/A")
            val  = v.get("NumericValue")
            if val:
                try:
                    print(f"  {year:<6}  {int(float(val)):>22,}")
                except Exception:
                    print(f"  {year:<6}  {str(val):>22}")
    print()
    endemic = ENDEMIC_NTDS.get(code)
    if endemic:
        print(f"Endemic NTDs in {name}:")
        for d in endemic:
            print(f"  - {d}")
    else:
        print("Endemic profile: https://www.who.int/teams/control-of-neglected-tropical-diseases")
    print()
    print("Resources:")
    print("  ChEMBL-NTD: https://chembl.gitbook.io/chembl-ntd")
    print("  Pathogen Box: https://www.pathogenbox.org")


def cmd_burden_compare(country_codes):
    """Compare NTD burden across multiple countries."""
    print(f"\nNTD Burden Comparison — {len(country_codes)} countries\n")
    print(f"{'Country':<25}  {'Code':<5}  {'Year':<6}  {'People requiring treatment':>25}")
    print("-" * 70)
    for code in country_codes:
        code = code.upper()
        name = AFRICAN_COUNTRIES.get(code, code)
        url  = f"{GHO_API}/NTD_PEOPLE?$filter=SpatialDim eq '{code}'&$orderby=TimeDim desc&$top=1"
        data = http_get(url)
        if data and data.get("value"):
            v    = data["value"][0]
            year = v.get("TimeDim", "N/A")
            val  = v.get("NumericValue")
            try:
                count = f"{int(float(val)):,}" if val else "N/A"
            except Exception:
                count = str(val) if val else "N/A"
        else:
            year  = "N/A"
            count = "No data"
        print(f"{name:<25}  {code:<5}  {year:<6}  {count:>25}")
        time.sleep(0.3)
    print("\nSource: WHO Global Health Observatory — Indicator: NTD_PEOPLE")


def cmd_compounds(organism, min_pchembl=6.0, limit=10):
    """Search ChEMBL for active compounds against an NTD pathogen."""
    encoded = urllib.parse.quote(organism)
    print(f"\nSearching ChEMBL for: {organism}\n")
    url  = f"{CHEMBL_API}/target/search?q={encoded}&limit=3&format=json"
    data = http_get(url)
    if not data or not data.get("targets"):
        print(f"No ChEMBL targets found for: {organism}")
        return
    for target in data["targets"][:2]:
        tid  = target.get("target_chembl_id", "N/A")
        name = target.get("pref_name", "N/A")
        print(f"Target: {name} ({tid})")
        act_url  = (f"{CHEMBL_API}/activity?target_chembl_id={tid}"
                    f"&pchembl_value__gte={min_pchembl}&limit={limit}"
                    f"&order_by=-pchembl_value&format=json")
        act_data = http_get(act_url)
        if not act_data or not act_data.get("activities"):
            print(f"  No activities found (pIC50 >= {min_pchembl})\n")
            continue
        print(f"  {'Molecule':<20} {'pIC50':>6}  {'IC50 (nM)':>10}  Assay")
        print("  " + "-"*55)
        seen = set()
        for a in act_data["activities"]:
            mid = a.get("molecule_chembl_id", "N/A")
            if mid in seen:
                continue
            seen.add(mid)
            print(f"  {mid:<20} {str(a.get('pchembl_value','N/A')):>6}  "
                  f"{str(a.get('standard_value','N/A')):>10}  {a.get('assay_chembl_id','N/A')}")
        print(f"\n  Total: {len(seen)} unique molecules\n")
        time.sleep(0.5)


def cmd_drug(drug_name):
    """Get full profile for an NTD drug from ChEMBL."""
    encoded = urllib.parse.quote(drug_name)
    data    = http_get(f"{CHEMBL_API}/molecule/search?q={encoded}&limit=1&format=json")
    if not data or not data.get("molecules"):
        print(f"Drug not found: {drug_name}")
        return
    m       = data["molecules"][0]
    props   = m.get("molecule_properties", {}) or {}
    structs = m.get("molecule_structures", {}) or {}
    phase   = m.get("max_phase", "N/A")
    labels  = {4:"APPROVED", 3:"Phase 3", 2:"Phase 2", 1:"Phase 1", 0:"Preclinical"}
    label   = labels.get(phase, str(phase))
    print(f"\n{'='*60}")
    print(f"  NTD Drug: {m.get('pref_name', drug_name)}")
    print(f"{'='*60}")
    print(f"  ChEMBL ID  : {m.get('molecule_chembl_id', 'N/A')}")
    print(f"  Status     : {label}")
    print(f"  Type       : {m.get('molecule_type', 'N/A')}")
    print(f"  SMILES     : {structs.get('canonical_smiles', 'N/A')}")
    print(f"\n  Properties:")
    print(f"    MW       : {props.get('full_mwt', 'N/A')} Da")
    print(f"    LogP     : {props.get('alogp', 'N/A')}")
    print(f"    HBD      : {props.get('hbd', 'N/A')}")
    print(f"    HBA      : {props.get('hba', 'N/A')}")
    print(f"    TPSA     : {props.get('psa', 'N/A')} Angstrom2")
    print(f"    Ro5 viol : {props.get('num_ro5_violations', 'N/A')}")
    print(f"    QED      : {props.get('qed_weighted', 'N/A')}")


def cmd_pathogen_box():
    """Display MMV Pathogen Box summary."""
    print("\nMMV Pathogen Box — ChEMBL-NTD (CHEMBL3301361)\n")
    print("400 diverse drug-like molecules active against 12 NTD disease areas.\n")
    data  = http_get(f"{CHEMBL_API}/activity?document_chembl_id=CHEMBL3301361&limit=5&format=json")
    if data:
        total = data.get("page_meta", {}).get("total_count", "?")
        print(f"Total activity records in ChEMBL: {total}\n")
    diseases = [
        "Malaria", "Tuberculosis", "Chagas Disease", "Leishmaniasis",
        "HAT (Sleeping Sickness)", "Cryptosporidiosis", "Lymphatic Filariasis",
        "Onchocerciasis", "Schistosomiasis", "Dengue", "Chikungunya", "Toxoplasmosis"
    ]
    print("Disease areas:")
    for d in diseases:
        print(f"  - {d}")
    print()
    print("Access full dataset : https://chembl.gitbook.io/chembl-ntd")
    print("Request free samples: https://www.pathogenbox.org")


def cmd_repurpose(target_id, min_pchembl=7.0):
    """Screen for approved drugs with activity against an NTD target."""
    print(f"\nDrug Repurposing Screen — Target: {target_id}")
    print(f"Criteria: pIC50 >= {min_pchembl}\n")
    url  = (f"{CHEMBL_API}/activity?target_chembl_id={target_id}"
            f"&pchembl_value__gte={min_pchembl}&limit=30&format=json")
    data = http_get(url)
    if not data or not data.get("activities"):
        print(f"No activities found for {target_id} at pIC50 >= {min_pchembl}")
        return
    acts       = data["activities"]
    candidates = []
    seen       = set()
    print(f"Checking {len(acts)} activity records for approved status...\n")
    for a in acts:
        mid = a.get("molecule_chembl_id", "N/A")
        if mid in seen:
            continue
        seen.add(mid)
        mol = http_get(f"{CHEMBL_API}/molecule/{mid}?format=json")
        if mol:
            phase = int(mol.get("max_phase") or 0)
            if phase >= 3:
                candidates.append({
                    "id":      mid,
                    "name":    mol.get("pref_name", mid),
                    "phase":   phase,
                    "pchembl": float(a.get("pchembl_value") or 0),
                })
        time.sleep(0.2)
    if not candidates:
        print(f"No approved/Phase 3+ drugs found. Screened {len(seen)} molecules.")
        print("Try lowering threshold with --min-pchembl 6")
    else:
        candidates.sort(key=lambda x: x["pchembl"], reverse=True)
        print(f"Repurposing candidates ({len(candidates)} approved/late-stage drugs):")
        print(f"  {'Drug':<30} {'Status':<12}  {'pIC50':>6}")
        print("  " + "-"*55)
        for c in candidates:
            label = "APPROVED" if c["phase"] == 4 else f"Phase {c['phase']}"
            print(f"  {c['name']:<30} {label:<12}  {c['pchembl']:>6.1f}")
        print()
        print("These drugs have existing safety and PK data.")
        print("Repurposing bypasses Phase 1 tox studies — faster to patients.")


def main():
    parser = argparse.ArgumentParser(
        description="NTD drug discovery utility scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  burden          <COUNTRY_CODE>         WHO NTD burden for one country
  burden-compare  <CODE1> <CODE2> ...    Compare burden across countries
  compounds       <ORGANISM>             ChEMBL active compounds vs pathogen
  drug            <DRUG_NAME>            Full NTD drug profile
  pathogen-box                           MMV Pathogen Box dataset summary
  repurpose       <CHEMBL_TARGET_ID>     Screen approved drugs vs NTD target

Examples:
  python3 ntd_utils.py burden NGA
  python3 ntd_utils.py burden-compare NGA GHA KEN TZA ETH
  python3 ntd_utils.py compounds "Plasmodium falciparum"
  python3 ntd_utils.py compounds "Schistosoma mansoni"
  python3 ntd_utils.py drug praziquantel
  python3 ntd_utils.py pathogen-box
  python3 ntd_utils.py repurpose CHEMBL364

Country codes: NGA=Nigeria, GHA=Ghana, KEN=Kenya, TZA=Tanzania,
               ETH=Ethiopia, COD=DR Congo, UGA=Uganda, ZAF=South Africa
        """
    )
    parser.add_argument("command", choices=[
        "burden", "burden-compare", "compounds", "drug", "pathogen-box", "repurpose"
    ])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--min-pchembl", type=float, default=6.0)
    args = parser.parse_args()

    if args.command == "burden":
        if not args.args:
            print("Usage: burden <COUNTRY_CODE>"); return
        cmd_burden(args.args[0])
    elif args.command == "burden-compare":
        if not args.args:
            print("Usage: burden-compare <CODE1> <CODE2> ..."); return
        cmd_burden_compare(args.args)
    elif args.command == "compounds":
        if not args.args:
            print('Usage: compounds "Plasmodium falciparum"'); return
        cmd_compounds(" ".join(args.args), min_pchembl=args.min_pchembl)
    elif args.command == "drug":
        if not args.args:
            print("Usage: drug <drug_name>"); return
        cmd_drug(" ".join(args.args))
    elif args.command == "pathogen-box":
        cmd_pathogen_box()
    elif args.command == "repurpose":
        if not args.args:
            print("Usage: repurpose <CHEMBL_TARGET_ID>"); return
        cmd_repurpose(args.args[0], min_pchembl=args.min_pchembl)


if __name__ == "__main__":
    main()
