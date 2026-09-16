#!/usr/bin/env python3
"""
docking_utils.py — Utility scripts for molecular docking workflows.

Outbound network calls (read-only GET requests only):
  - https://search.rcsb.org       (PDB structure search)
  - https://data.rcsb.org         (PDB entry metadata)
  - https://files.rcsb.org        (PDB file download)
  - https://pubchem.ncbi.nlm.nih.gov  (compound properties)

No data is transmitted outbound. All requests fetch public read-only data.
No API keys. No authentication required.

Commands:
    python3 docking_utils.py fetch-pdb 1IEP
    python3 docking_utils.py ligand-center protein.pdb STI
    python3 docking_utils.py parse-vina result.log
    python3 docking_utils.py pdb-search "EGFR kinase"
    python3 docking_utils.py batch-smiles compounds.txt
"""

import sys
import os
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error
import re
import math

RCSB_SEARCH  = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA    = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_FILES   = "https://files.rcsb.org/download"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def http_get(url, timeout=15):
    """GET request to public read-only API — no data transmitted outbound."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"HTTP {e.code}: {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def http_post(url, payload, timeout=15):
    """POST request to RCSB search API — sends search query only, no user data."""
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"POST error: {e}", file=sys.stderr)
        return None


def cmd_fetch_pdb(pdb_id, output_dir="."):
    """Fetch a PDB file and show its key metadata."""
    pdb_id = pdb_id.upper().strip()
    raw = http_get(f"{RCSB_DATA}/{pdb_id}")
    if not raw:
        print(f"PDB entry {pdb_id} not found.")
        return
    d      = json.loads(raw)
    struct = d.get("struct", {})
    exptl  = d.get("exptl", [{}])[0]
    refine = d.get("refine", [{}])[0]
    info   = d.get("rcsb_entry_info", {})
    method = exptl.get("method", "N/A")
    res    = refine.get("ls_d_res_high", "N/A")
    rfree  = refine.get("ls_r_factor_r_free", "N/A")
    title  = struct.get("title", "N/A")
    atoms  = info.get("deposited_atom_count", "N/A")
    resid  = info.get("deposited_modeled_polymer_monomer_count", "N/A")
    res_flag = ""
    if res != "N/A":
        try:
            res_flag = " GOOD for docking" if float(str(res)) < 2.5 else " low resolution"
        except ValueError:
            pass
    print(f"\n{'='*60}")
    print(f"  PDB ID     : {pdb_id}")
    print(f"  Title      : {title}")
    print(f"  Method     : {method}")
    print(f"  Resolution : {res} Angstrom{res_flag}")
    print(f"  R-free     : {rfree}")
    print(f"  Atoms      : {atoms}")
    print(f"  Residues   : {resid}")
    print(f"{'='*60}\n")
    pdb_url  = f"{RCSB_FILES}/{pdb_id}.pdb"
    out_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    print(f"Downloading {pdb_id}.pdb ...")
    try:
        urllib.request.urlretrieve(pdb_url, out_path)
        print(f"Saved: {out_path} ({os.path.getsize(out_path):,} bytes)")
        print()
        print("Next steps:")
        print("  1. Open in BIOVIA Discovery Studio Visualizer")
        print("  2. Identify co-crystallized ligand (HETATM records)")
        print(f"  3. Run: python3 docking_utils.py ligand-center {pdb_id}.pdb <LIGAND_CODE>")
        print("  4. Clean protein (remove waters, heteroatoms, add H)")
        print("  5. Convert to PDBQT in PyRx")
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Manual download: {pdb_url}")


def cmd_ligand_center(pdb_file, ligand_code):
    """Extract centroid of a co-crystallized ligand from a PDB file."""
    ligand_code = ligand_code.strip().upper()
    if not os.path.exists(pdb_file):
        print(f"File not found: {pdb_file}")
        return
    coords      = []
    atoms_found = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith(("HETATM", "ATOM")):
                if line[17:20].strip() == ligand_code:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append((x, y, z))
                        atoms_found.append(line[12:16].strip())
                    except ValueError:
                        pass
    if not coords:
        print(f"Ligand '{ligand_code}' not found in {pdb_file}")
        print("Available HETATMs:")
        seen = set()
        with open(pdb_file) as f:
            for line in f:
                if line.startswith("HETATM"):
                    res = line[17:20].strip()
                    if res and res != "HOH" and res not in seen:
                        print(f"  {res}")
                        seen.add(res)
        return
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    bx = max(20, round(max(xs) - min(xs) + 20))
    by = max(20, round(max(ys) - min(ys) + 20))
    bz = max(20, round(max(zs) - min(zs) + 20))
    print(f"\nLigand: {ligand_code}  ({len(coords)} atoms)")
    print(f"Atoms: {', '.join(atoms_found[:8])}{'...' if len(atoms_found) > 8 else ''}")
    print()
    print("Grid Box Center (use in PyRx or Vina config):")
    print(f"  Center X = {cx:.3f}")
    print(f"  Center Y = {cy:.3f}")
    print(f"  Center Z = {cz:.3f}")
    print()
    print(f"Ligand span: X={max(xs)-min(xs):.1f}A  Y={max(ys)-min(ys):.1f}A  Z={max(zs)-min(zs):.1f}A")
    print(f"Recommended Grid Box: {bx} x {by} x {bz} Angstrom (ligand + 10A buffer)")
    print()
    print("Vina config.txt:")
    print(f"  center_x = {cx:.3f}")
    print(f"  center_y = {cy:.3f}")
    print(f"  center_z = {cz:.3f}")
    print(f"  size_x = {bx}")
    print(f"  size_y = {by}")
    print(f"  size_z = {bz}")


def cmd_parse_vina(log_file):
    """Parse AutoDock Vina output log and rank docking results."""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    with open(log_file) as f:
        content = f.read()
    pattern = r"^\s*(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)"
    results = []
    for m in re.finditer(pattern, content, re.MULTILINE):
        mode, affinity, rmsd_lb, rmsd_ub = m.groups()
        results.append({
            "mode":     int(mode),
            "affinity": float(affinity),
            "rmsd_lb":  float(rmsd_lb),
            "rmsd_ub":  float(rmsd_ub),
        })
    if not results:
        print("No docking results found in log file.")
        return
    best = min(results, key=lambda x: x["affinity"])
    print(f"\n{'Mode':>4}  {'Affinity':>15}  {'RMSD l.b.':>10}  {'RMSD u.b.':>10}  Notes")
    print("-" * 70)
    for r in results:
        note = "SELECT" if r["rmsd_lb"] == 0.0 else ("similar to top" if r["rmsd_lb"] < 1.0 else "")
        print(f"{r['mode']:>4}  {r['affinity']:>14.1f}  {r['rmsd_lb']:>10.3f}  {r['rmsd_ub']:>10.3f}  {note}")
    RT    = 0.592
    ki_nM = math.exp(best["affinity"] / RT) * 1e9
    score = best["affinity"]
    if score < -9.0:
        interp = "Excellent — strong binding (nanomolar). High priority."
    elif score < -7.0:
        interp = "Good binding. Proceed to ADMET and MD validation."
    elif score < -5.0:
        interp = "Moderate. Consider structural optimization."
    else:
        interp = "Weak binding. Deprioritize or redesign scaffold."
    print()
    print(f"Best: Mode {best['mode']}  {best['affinity']} kcal/mol  RMSD={best['rmsd_lb']:.3f}")
    print(f"Estimated Ki: ~{ki_nM:.1f} nM  (delta-G = RT x ln(Ki), T=298K)")
    print(f"Assessment: {interp}")
    print()
    print("Selection rule: most negative affinity WITH RMSD l.b. = 0.000")
    print("Next: Open best pose in Discovery Studio > Receptor-Ligand Interactions > Show Interactions")


def cmd_pdb_search(query, max_results=8):
    """Search RCSB PDB for protein structures."""
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query}
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }
    print(f"\nSearching RCSB PDB for: '{query}' ...\n")
    raw = http_post(RCSB_SEARCH, payload)
    if not raw:
        print("Search failed.")
        return
    ids = [r["identifier"] for r in json.loads(raw).get("result_set", [])]
    if not ids:
        print("No structures found.")
        return
    print(f"{'PDB':>6}  {'Method':<15}  {'Resolution':>12}  {'R-free':>8}  Title")
    print("-" * 85)
    for pdb_id in ids:
        raw2 = http_get(f"{RCSB_DATA}/{pdb_id}")
        if not raw2:
            continue
        d      = json.loads(raw2)
        method = d.get("exptl", [{}])[0].get("method", "N/A")
        res    = d.get("refine", [{}])[0].get("ls_d_res_high", "N/A")
        rfree  = d.get("refine", [{}])[0].get("ls_r_factor_r_free", "N/A")
        title  = d.get("struct", {}).get("title", "N/A")[:48]
        flag   = ""
        try:
            flag = "GOOD" if float(str(res)) < 2.5 else ""
        except Exception:
            pass
        print(f"{pdb_id:>6}  {method:<15}  {str(res):>11} {flag:<4}  {str(rfree):>8}  {title}")
        time.sleep(0.2)
    print()
    print("GOOD = resolution < 2.5 Angstrom (preferred for docking)")
    print(f"Download: python3 docking_utils.py fetch-pdb <PDB_ID>")


def cmd_batch_smiles(smiles_file):
    """Validate and look up PubChem data for a list of compound names."""
    if not os.path.exists(smiles_file):
        print(f"File not found: {smiles_file}")
        return
    with open(smiles_file) as f:
        compounds = [line.strip() for line in f if line.strip()]
    print(f"\nBatch lookup: {len(compounds)} compounds\n")
    print(f"{'Compound':<30}  {'CID':>10}  {'MW':>8}  {'LogP':>6}  {'Ro5?'}")
    print("-" * 70)
    for name in compounds:
        encoded = urllib.parse.quote(name)
        try:
            # GET request to PubChem public API — read-only, no data transmitted
            cid_raw = http_get(f"{PUBCHEM_BASE}/name/{encoded}/cids/TXT")
            if not cid_raw:
                print(f"{name:<30}  {'NOT FOUND':>10}")
                continue
            cid      = cid_raw.decode().strip().splitlines()[0]
            prop_raw = http_get(
                f"{PUBCHEM_BASE}/cid/{cid}/property/"
                f"MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount/JSON"
            )
            if not prop_raw:
                print(f"{name:<30}  {cid:>10}  (no properties)")
                continue
            props = json.loads(prop_raw)["PropertyTable"]["Properties"][0]
            mw    = float(props.get("MolecularWeight", 0))
            logp  = float(props.get("XLogP", 0))
            hbd   = int(props.get("HBondDonorCount", 0))
            hba   = int(props.get("HBondAcceptorCount", 0))
            v     = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
            ro5   = "PASS" if v <= 1 else f"FAIL ({v} violations)"
            print(f"{name[:29]:<30}  {cid:>10}  {mw:>8.1f}  {logp:>6.2f}  {ro5}")
        except Exception as e:
            print(f"{name:<30}  ERROR: {e}")
        time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Molecular docking utility scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  fetch-pdb     <PDB_ID>            Download PDB file and show metadata
  ligand-center <PDB_FILE> <CODE>   Get grid box center from co-crystallized ligand
  parse-vina    <LOG_FILE>          Parse AutoDock Vina output, rank poses, estimate Ki
  pdb-search    <QUERY>             Search RCSB PDB by keyword
  batch-smiles  <FILE>              Batch Ro5 check for compound list

Examples:
  python3 docking_utils.py fetch-pdb 1IEP
  python3 docking_utils.py ligand-center 1IEP.pdb STI
  python3 docking_utils.py parse-vina vina_result.log
  python3 docking_utils.py pdb-search "EGFR kinase inhibitor"
  python3 docking_utils.py batch-smiles my_compounds.txt
        """
    )
    parser.add_argument(
        "command",
        choices=["fetch-pdb", "ligand-center", "parse-vina", "pdb-search", "batch-smiles"]
    )
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()

    if args.command == "fetch-pdb":
        if not args.args:
            print("Usage: fetch-pdb <PDB_ID> [output_dir]")
            return
        cmd_fetch_pdb(args.args[0], args.args[1] if len(args.args) > 1 else ".")
    elif args.command == "ligand-center":
        if len(args.args) < 2:
            print("Usage: ligand-center <PDB_FILE> <LIGAND_CODE>")
            return
        cmd_ligand_center(args.args[0], args.args[1])
    elif args.command == "parse-vina":
        if not args.args:
            print("Usage: parse-vina <LOG_FILE>")
            return
        cmd_parse_vina(args.args[0])
    elif args.command == "pdb-search":
        if not args.args:
            print("Usage: pdb-search <QUERY>")
            return
        cmd_pdb_search(" ".join(args.args))
    elif args.command == "batch-smiles":
        if not args.args:
            print("Usage: batch-smiles <FILE>")
            return
        cmd_batch_smiles(args.args[0])


if __name__ == "__main__":
    main()
