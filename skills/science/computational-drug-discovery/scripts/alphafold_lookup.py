#!/usr/bin/env python3
"""
alphafold_lookup.py — Batch AlphaFold structure lookup via UniProt IDs or gene names.
Usage:
    python3 alphafold_lookup.py P00533 P69905 P04637
    python3 alphafold_lookup.py --gene EGFR BRAF TP53
No external dependencies.
"""
import sys, json, time, argparse
import urllib.request, urllib.parse, urllib.error

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
UNIPROT_API   = "https://rest.uniprot.org/uniprotkb/search"

def fetch(url):
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404: return None
        print(f"HTTP {e.code} for {url}", file=sys.stderr); return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); return None

def gene_to_uniprot(gene):
    enc = urllib.parse.quote(f"{gene} AND organism_id:9606 AND reviewed:true")
    data = fetch(f"{UNIPROT_API}?query={enc}&fields=accession&format=json&size=1")
    if not data or not data.get("results"): return None
    return data["results"][0].get("primaryAccession")

def confidence_label(score):
    try:
        s = float(score)
        if s >= 90: return "Very high (reliable for docking)"
        if s >= 70: return "High (backbone reliable)"
        if s >= 50: return "Low (use with caution)"
        return "Very low (likely disordered - avoid docking)"
    except: return "N/A"

def print_entry(identifier, entry, resolved=""):
    label = identifier + (f" -> {resolved}" if resolved else "")
    if entry is None:
        print(f"Not found: {label}"); return
    score = entry.get("confidenceAvgLocalScore", "N/A")
    print(f"\n{'='*60}")
    print(f"  Protein   : {entry.get('uniprotDescription', 'N/A')}")
    print(f"  Gene      : {entry.get('gene', 'N/A')}")
    print(f"  Organism  : {entry.get('organismScientificName', 'N/A')}")
    print(f"  UniProt   : {entry.get('uniprotAccession', 'N/A')}")
    print(f"  Length    : {entry.get('uniprotSequenceLength', 'N/A')} aa")
    print(f"  pLDDT avg : {score} — {confidence_label(score)}")
    print(f"  PDB file  : {entry.get('pdbUrl', 'N/A')}")
    print(f"  View      : https://alphafold.ebi.ac.uk/entry/{entry.get('uniprotAccession', '')}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Batch AlphaFold lookup")
    parser.add_argument("identifiers", nargs="+")
    parser.add_argument("--gene", action="store_true", help="Resolve gene names to UniProt IDs")
    args = parser.parse_args()
    print(f"\nAlphaFold lookup — {len(args.identifiers)} queries\n")
    for ident in args.identifiers:
        uniprot_id = ident
        resolved = ""
        if args.gene:
            resolved = gene_to_uniprot(ident)
            if not resolved:
                print(f"Gene not found in UniProt (human): {ident}"); continue
            uniprot_id = resolved
        entry_data = fetch(f"{ALPHAFOLD_API}/{uniprot_id}")
        entry = entry_data[0] if isinstance(entry_data, list) and entry_data else None
        print_entry(ident, entry, resolved if args.gene else "")
        time.sleep(0.3)
    print()

if __name__ == "__main__": main()
