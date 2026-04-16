#!/usr/bin/env python3
"""
ro5_screen.py — Batch Lipinski Ro5 + Veber screening via PubChem API.
Usage: python3 ro5_screen.py aspirin ibuprofen paracetamol
No external dependencies beyond stdlib.
"""
import sys, json, time, argparse
import urllib.request, urllib.parse, urllib.error

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
PROPS = "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA"

def fetch(name):
    url = f"{BASE}/{urllib.parse.quote(name)}/property/{PROPS}/JSON"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())["PropertyTable"]["Properties"][0]
    except Exception:
        return None

def check(p):
    mw,logp,hbd,hba,rot,tpsa = float(p.get("MolecularWeight",0)),float(p.get("XLogP",0)),int(p.get("HBondDonorCount",0)),int(p.get("HBondAcceptorCount",0)),int(p.get("RotatableBondCount",0)),float(p.get("TPSA",0))
    v = sum([mw>500,logp>5,hbd>5,hba>10])
    return dict(mw=mw,logp=logp,hbd=hbd,hba=hba,rot=rot,tpsa=tpsa,violations=v,ro5=v<=1,veber=tpsa<=140 and rot<=10,ok=v<=1 and tpsa<=140 and rot<=10)

def report(name, r):
    if not r: print(f"✗ {name:30s} — not found"); return
    s = "✓ PASS" if r["ok"] else "✗ FAIL"
    flags = (f" [Ro5 violations:{r['violations']}]" if not r["ro5"] else "") + (" [Veber fail]" if not r["veber"] else "")
    print(f"{s}  {name:28s} MW={r['mw']:.0f} LogP={r['logp']:.2f} HBD={r['hbd']} HBA={r['hba']} TPSA={r['tpsa']:.0f} RotB={r['rot']}{flags}")

def main():
    compounds = sys.stdin.read().splitlines() if len(sys.argv)<2 or sys.argv[1]=="-" else sys.argv[1:]
    print(f"\n{'Status':<8} {'Compound':<30} Properties\n" + "-"*85)
    passed = 0
    for name in compounds:
        props = fetch(name.strip())
        result = check(props) if props else None
        report(name.strip(), result)
        if result and result["ok"]: passed += 1
        time.sleep(0.3)
    print(f"\nSummary: {passed}/{len(compounds)} passed Ro5 + Veber.\n")

if __name__ == "__main__": main()
