#!/usr/bin/env python3
"""Intentionally permissive external-tool fixture for the R6 acceptance regression.

Models a validator that accepts a syntactically valid unsigned manifest because it
has no signature-verification behavior. The acceptance gate must still reject the
project when its frozen requirement expects unsigned input to fail closed.
"""

import json
import sys


if len(sys.argv) != 2:
    raise SystemExit(2)
with open(sys.argv[1], encoding="utf-8") as manifest_file:
    manifest = json.load(manifest_file)
print(json.dumps({"verdict": "VERIFIED", "cases_evaluated": len(manifest.get("cases", [])), "violations": []}))
