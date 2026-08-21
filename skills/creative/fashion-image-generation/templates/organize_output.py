#!/usr/bin/env python3
"""Organize generated FW26 images into PRODUCT IMG with SKU-based progressive renaming.

Reads the authoritative SKU map from the HELMUR-master-prodotti spreadsheet (tab VARIANTI),
renames local generated outputs (ghost_<C>.png / indossato_<posa>_<C>.png) to
<SKU>_<modello>-<colore>_<NN>_<posa>.png, and uploads them to:

    PRODUCT IMG (root) > DONNA|UOMO > <MODELLO> > <CodColore>-<COLORE>/

Idempotent: skips a file whose exact target filename already exists in its target folder.

Drive operations use the Google API python client directly (google_api.build_service),
NOT shell quoting — this avoids the .split() bug that mangled parent-search queries and
created duplicate folders.

Usage:
  python organize_output.py                              # all garments
  python organize_output.py MONTANA ALASKA               # only these models
  python organize_output.py --local-only MONTANA         # rename locally, no upload
  python organize_output.py --start 1                    # progressive index base (default 0)
"""
import argparse, os, re, sys

try:
    sys.path.insert(0, "/root/.hermes/skills/productivity/google-workspace/scripts")
    import google_api as ga
    from googleapiclient.http import MediaFileUpload
except Exception as e:  # pragma: no cover
    print("ERROR: cannot import google_api:", e)
    sys.exit(2)

BASE = os.environ.get("FW_OUTPUT_BASE", os.path.dirname(os.path.abspath(__file__)))
PRODUCT_IMG_ROOT = "1vy41E81IYScJOVJYYz076sCc_eBCCblN"
SEX_FOLDERS = {"Donna": "1YqQ27arr_CvWIFPZe3XK9B-UpuhDw6l8", "Uomo": "1ZhlmWVWFEnmN6VJpTMEog9zWTQMg4Vdi"}
XLSX = os.environ.get("HELMUR_XLSX", "/root/analysis/master.xlsx")

MIME_FOLDER = "application/vnd.google-apps.folder"
POSA_ORDER = {"ghost": 0, "front": 1, "bust34": 2, "editorial": 3, "back": 4, "detail": 5}


def _svc():
    return ga.build_service("drive", "v3")


def list_children(svc, parent):
    out, token = [], None
    while True:
        r = svc.files().list(q=f"'{parent}' in parents and trashed=false",
                             pageSize=1000, pageToken=token,
                             fields="nextPageToken, files(id,name,mimeType)").execute()
        out.extend(r.get("files", []))
        token = r.get("nextPageToken")
        if not token:
            return out


def ensure_folder(svc, parent, name):
    for x in list_children(svc, parent):
        if x["name"] == name and x["mimeType"] == MIME_FOLDER:
            return x["id"]
    f = svc.files().create(body={"name": name, "mimeType": MIME_FOLDER, "parents": [parent]},
                           fields="id").execute()
    return f["id"]


def has_file(svc, parent, name):
    for x in list_children(svc, parent):
        if x["name"] == name:
            return True
    return False


def parse_code(token):
    m = re.search(r"-(\d+)$", token)
    return m.group(1) if m else token


def load_sku_map(xlsx):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)
    ws = wb["VARIANTI"]
    rows = ws.iter_rows(values_only=True)
    next(rows)
    m = {}
    for r in rows:
        sku, mod, rep, cat, codcol, col, *_rest = (list(r) + [None] * 10)[:10]
        if not mod:
            continue
        m.setdefault(mod, {})[str(codcol)] = {"sku": sku, "colore": col, "reparto": rep}
    return m


def target_name(sku_map, modulo, colore_token, posa_enum, idx, start):
    code = parse_code(colore_token)
    info = sku_map.get(modulo, {}).get(code, {}) if sku_map else {}
    nome_col = (info.get("colore") or colore_token.split("-")[0]).lower()
    sku = info.get("sku") or f"{modulo[:4].upper()}-{code}"
    nn = str(start + idx).zfill(2)
    posa = "ghost" if posa_enum == "ghost" else posa_enum
    return f"{sku}_{modulo.lower()}-{nome_col}_{nn}_{posa}.png", info


def classify_posa(fname):
    if fname.startswith("ghost_"):
        return "ghost"
    if "bust34" in fname:
        return "bust34"
    if "editorial" in fname:
        return "editorial"
    if "back" in fname:
        return "back"
    if "detail" in fname:
        return "detail"
    return "front"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelli", nargs="*")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--xlsx", default=XLSX)
    a = ap.parse_args()

    sku_map = load_sku_map(a.xlsx) if os.path.exists(a.xlsx) else {}
    want = set(a.modelli) or None
    svc = _svc() if not a.local_only else None

    for capo in sorted(os.listdir(BASE)):
        capodir = os.path.join(BASE, capo)
        if not os.path.isdir(capodir) or capo.startswith("."):
            continue
        if want and capo not in want:
            continue
        files = sorted(f for f in os.listdir(capodir)
                       if re.match(r"(ghost|indossato)_", f) and f.endswith(".png"))
        bycolour = {}
        for f in files:
            m = re.match(r"(ghost|indossato)_(?:([a-z0-9]+)_)?(.+)\.png$", f)
            if not m:
                continue
            colore = m.group(3)
            bycolour.setdefault(colore, []).append(f)
        for colore, flist in sorted(bycolour.items()):
            flist.sort(key=lambda f: POSA_ORDER.get(classify_posa(f), 9))
            sep = min((i for i, f in enumerate(flist) if i > 0), default=len(flist))
            prev_name, info = None, {}
            for idx, f in enumerate(flist):
                posa_enum = classify_posa(f)
                newname, info = target_name(sku_map, capo, colore, posa_enum, idx, a.start)
                src = os.path.join(capodir, f)
                dst = os.path.join(capodir, f)
                if a.local_only:
                    print(f"  [local] {capo}/{colore}: {f} -> {newname}", flush=True)
                    continue
                reparto = info.get("reparto") or ("Uomo" if capo in ("OXFORD", "PORTLAND", "LEMANS") else "Donna")
                sex = "Donna" if str(reparto).strip().lower() in ("donna", "femmina", "f") else "Uomo"
                sexid = SEX_FOLDERS.get(sex) or ensure_folder(svc, PRODUCT_IMG_ROOT, sex)
                model_id = ensure_folder(svc, sexid, capo)
                col_name = f"{parse_code(colore)}-{info.get('colore') or colore}"
                col_id = ensure_folder(svc, model_id, col_name)
                if has_file(svc, col_id, newname):
                    print(f"  [skip] {capo}/{colore}: {newname} già presente", flush=True)
                    continue
                mime = "image/png" if newname.endswith(".png") else "image/jpeg"
                media = MediaFileUpload(src, mimetype=mime)
                svc.files().create(body={"name": newname, "parents": [col_id]},
                                   media_body=media, fields="id,name").execute()
                print(f"  [ok] {capo}/{colore}: {newname} -> PRODUCT IMG/{sex}/{capo}/{col_name}/", flush=True)


if __name__ == "__main__":
    main()
